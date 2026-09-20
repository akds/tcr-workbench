"""Resource planning uses synthetic OS metrics; no tensor imports or hardware jobs."""
import json
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from tcr_workbench import resources
from tcr_workbench.resources import GIB, MIB, HardwareMetrics, plan_resources


def host(total=32 * GIB, available=24 * GIB, **gpu):
    return HardwareMetrics(host_total_bytes=total, host_available_bytes=available, **gpu)


@pytest.mark.parametrize("parameters", [300_000_000, 600_000_000])
@pytest.mark.parametrize("device,backend", [("cpu", "torch"), ("apple", "mlx")])
def test_small_models_fit_with_explicit_estimated_byte_counts(parameters, device, backend):
    plan = plan_resources(parameters * 4, device, backend=backend, hardware=host())
    assert plan.risk == "within_budget" and not plan.blocked
    assert plan.minimum_required_bytes < plan.estimated_peak_bytes < plan.available_bytes
    assert plan.device == device and plan.backend == backend
    assert json.loads(plan.model_dump_json())["parameter_bytes"] == parameters * 4
    assert any("not measured" in item for item in plan.assumptions)


@pytest.mark.parametrize("device", ["cpu", "apple"])
def test_six_billion_fp32_weights_block_on_32_gib_host_with_24_gib_available(device):
    plan = plan_resources(6_000_000_000 * 4, device, hardware=host())
    assert plan.blocked and plan.risk == "over_budget"
    assert plan.estimated_peak_bytes > plan.available_bytes
    assert plan.suggestions and plan.device == device


def test_apple_plan_uses_one_shared_pool_and_requires_separate_cpu_preparation():
    memory = host(gpu_total_bytes=32 * GIB, gpu_available_bytes=32 * GIB, gpu_device="cuda:0")
    apple = plan_resources(3 * GIB, "apple", backend="mlx", hardware=memory)
    assert apple.memory_pool == "apple_shared"
    assert apple.total_bytes == apple.host_total_bytes == 32 * GIB
    assert apple.available_bytes == apple.host_available_bytes == 24 * GIB
    assert apple.gpu_total_bytes is None and apple.gpu_estimated_peak_bytes is None
    assert apple.estimated_peak_bytes == apple.host_estimated_peak_bytes
    assert any("Plan CPU reference/conversion separately" in item for item in apple.assumptions)
    large_apple = plan_resources(12 * GIB, "apple", backend="mlx", hardware=memory)
    preparation = plan_resources(12 * GIB, "cpu", hardware=memory)
    assert not large_apple.blocked and preparation.blocked


def test_memory_override_preserves_over_budget_classification_and_requested_model_bytes():
    plan = plan_resources(24 * GIB, "cpu", allow_memory_risk=True, hardware=host())
    assert not plan.blocked and plan.risk == "over_budget" and plan.allow_memory_risk
    assert plan.parameter_bytes == 24 * GIB and plan.device == "cpu"
    assert any("override" in warning for warning in plan.warnings)
    assert any("very slow" in warning for warning in plan.warnings)


def test_eighty_percent_threshold_warns_without_blocking():
    initial = plan_resources(GIB, "cpu", hardware=host())
    memory = host(available=initial.estimated_peak_bytes * 5 // 4)
    plan = plan_resources(GIB, "cpu", hardware=memory)
    assert not plan.blocked and plan.risk == "memory_pressure"
    assert any("80%" in warning for warning in plan.warnings)


def test_missing_metrics_never_claim_fit_or_invent_memory():
    plan = plan_resources(GIB, "cpu", hardware={})
    assert plan.risk == "unknown" and not plan.blocked
    assert plan.available_bytes is None and plan.total_bytes is None
    assert any("cannot be established" in warning for warning in plan.warnings)


def test_total_memory_can_block_when_available_metric_is_missing():
    plan = plan_resources(4 * GIB, "cpu", hardware={"host_total_bytes": 8 * GIB})
    assert plan.minimum_required_bytes < 8 * GIB < plan.estimated_peak_bytes
    assert plan.blocked and any("exceeds total" in warning for warning in plan.warnings)


def test_zero_available_memory_is_measured_exhaustion_not_missing_information():
    plan = plan_resources(GIB, "cpu", hardware=host(available=0))
    assert plan.blocked and plan.available_bytes == 0


def test_gpu_vram_shortage_suggests_cpu_only_if_full_cpu_estimate_fits():
    memory = host(gpu_device="cuda:1", gpu_total_bytes=2 * GIB, gpu_available_bytes=GIB)
    plan = plan_resources(2 * GIB, "cuda:1", hardware=memory)
    assert plan.blocked and plan.memory_pool == "gpu" and plan.available_bytes == GIB
    assert plan.host_estimated_peak_bytes == 4 * GIB + 256 * MIB
    assert any("--device cpu" in suggestion for suggestion in plan.suggestions)
    limited = memory.model_copy(update={"host_available_bytes": 3 * GIB})
    assert not any("--device cpu" in item for item in plan_resources(2 * GIB, "cuda:1", hardware=limited).suggestions)


def test_gpu_vram_fit_does_not_hide_host_staging_over_budget():
    memory = host(available=2 * GIB, gpu_device="cuda:0", gpu_total_bytes=32 * GIB,
                  gpu_available_bytes=30 * GIB)
    plan = plan_resources(2 * GIB, "gpu", hardware=memory)
    assert plan.estimated_peak_bytes < plan.available_bytes and plan.blocked
    assert any("host estimated peak" in warning for warning in plan.warnings)


@pytest.mark.parametrize("gpu_device", [None, "cuda:1"])
def test_supplied_gpu_metrics_must_match_logical_device(gpu_device):
    memory = host(gpu_device=gpu_device, gpu_total_bytes=GIB, gpu_available_bytes=GIB)
    plan = plan_resources(2 * GIB, "cuda:0", hardware=memory)
    assert plan.risk == "unknown" and not plan.blocked
    assert plan.gpu_available_bytes is None and plan.gpu_total_bytes is None
    assert any("no matching" in warning for warning in plan.warnings)


def test_batch_and_length_increase_workspace_without_changing_parameter_identity():
    first = plan_resources(GIB, "cpu", batch_size=1, sequence_length=128, hardware=host())
    second = plan_resources(GIB, "cpu", batch_size=4, sequence_length=2048, hardware=host())
    assert second.workspace_bytes > first.workspace_bytes
    assert second.parameter_bytes == first.parameter_bytes


def test_full_lightning_storage_is_reserved_with_cpu_runtime_weights():
    memory = host(available=5 * GIB)
    inference_only = plan_resources(GIB, "cpu", hardware=memory)
    full_checkpoint = plan_resources(GIB, "cpu", checkpoint_storage_bytes=3 * GIB, hardware=memory)
    assert full_checkpoint.checkpoint_storage_bytes == 3 * GIB
    assert full_checkpoint.estimated_peak_bytes == 4 * GIB + full_checkpoint.workspace_bytes
    assert full_checkpoint.estimated_peak_bytes > inference_only.estimated_peak_bytes
    assert full_checkpoint.minimum_required_bytes == inference_only.minimum_required_bytes
    assert any("optimizer state" in assumption for assumption in full_checkpoint.assumptions)


def test_gpu_checkpoint_loading_can_exceed_both_device_and_host_budgets():
    memory = host(available=3 * GIB, gpu_device="cuda:0", gpu_total_bytes=8 * GIB,
                  gpu_available_bytes=4 * GIB)
    plan = plan_resources(GIB, "cuda", checkpoint_storage_bytes=3 * GIB, hardware=memory)
    assert plan.host_estimated_peak_bytes == 4 * GIB + 256 * MIB
    assert plan.gpu_estimated_peak_bytes == 4 * GIB + plan.workspace_bytes
    assert plan.blocked and any("host estimated" in warning for warning in plan.warnings)
    assert any("GPU estimated" in warning for warning in plan.warnings)


def test_apple_bundle_inference_does_not_load_optimizer_storage_in_shared_pool():
    memory = host()
    bundle = plan_resources(GIB, "apple", backend="mlx", hardware=memory)
    checkpoint = plan_resources(GIB, "apple", backend="mlx", checkpoint_storage_bytes=3 * GIB, hardware=memory)
    assert checkpoint.estimated_peak_bytes == bundle.estimated_peak_bytes
    assert checkpoint.host_estimated_peak_bytes == checkpoint.estimated_peak_bytes
    assert checkpoint.gpu_estimated_peak_bytes is None
    preparation = plan_resources(GIB, "cpu", checkpoint_storage_bytes=3 * GIB, hardware=memory)
    assert preparation.estimated_peak_bytes > checkpoint.estimated_peak_bytes


@pytest.mark.parametrize("storage", [True, 0, -1, 1.5, "1000"])
def test_checkpoint_storage_must_be_positive_integer_bytes(storage):
    with pytest.raises(ValidationError):
        plan_resources(GIB, "cpu", checkpoint_storage_bytes=storage, hardware={})


@pytest.mark.parametrize("kwargs", [{"parameter_bytes": True}, {"parameter_bytes": 0},
    {"parameter_bytes": 1.5}, {"batch_size": False}, {"batch_size": 0}, {"sequence_length": 0},
    {"sequence_length": 32769}, {"backend": "automatic"}, {"allow_memory_risk": "yes"},
    {"device": "cuda:-1"}])
def test_invalid_resource_request_fails_loudly(kwargs):
    options = dict(parameter_bytes=GIB, device="cpu", hardware={})
    options.update(kwargs)
    with pytest.raises((ValidationError, ValueError)):
        plan_resources(**options)


def test_inconsistent_hardware_metrics_fail_loudly():
    with pytest.raises(ValueError, match="exceeds total"):
        plan_resources(GIB, "cpu", hardware={"host_total_bytes": GIB, "host_available_bytes": 2 * GIB})
    with pytest.raises(ValidationError):
        HardwareMetrics(host_available_bytes=True)


def test_linux_memory_uses_memavailable_and_kib_units(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Linux")
    def read(path):
        assert path == Path("/proc/meminfo")
        return "MemTotal: 32768000 kB\nMemFree: 10 kB\nMemAvailable: 16777216 kB\nCached: 999 kB\n"
    monkeypatch.setattr(Path, "read_text", read)
    metrics = resources.detect_hardware()
    assert metrics.host_total_bytes == 32768000 * 1024
    assert metrics.host_available_bytes == 16 * GIB
    assert metrics.gpu_total_bytes is None


def test_linux_old_kernel_memfree_fallback_and_missing_metrics():
    assert resources._linux_memory("MemTotal: 10 kB\nMemFree: 2 kB\n") == (10240, 2048)
    assert resources._linux_memory("unexpected output") == (None, None)


def test_mac_memory_uses_native_page_size_and_never_double_counts_purgeable(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Darwin")
    vm = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
          'Pages free: 100.\nPages inactive: 200.\nPages speculative: 50.\n'
          'Pages purgeable: 99999.\nPages occupied by compressor: 20000.\n')
    calls = []
    def query(argv):
        calls.append(argv)
        return str(32 * GIB) if argv[0] == "sysctl" else vm
    monkeypatch.setattr(resources, "_command", query)
    metrics = resources.detect_hardware("apple")
    assert metrics.host_total_bytes == 32 * GIB
    assert metrics.host_available_bytes == 350 * 16384
    assert calls == [["sysctl", "-n", "hw.memsize"], ["vm_stat"]]


@pytest.mark.parametrize("text", ["Pages free: 10.", "page size of 4096 bytes\nPages free: 10.\n",
                                  "page size of 0 bytes"])
def test_incomplete_mac_metrics_remain_unknown(text):
    assert resources._mac_available(text) is None


def test_failed_query_preserves_total_but_warns_about_unknown_available(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Darwin")
    def query(argv):
        if argv[0] == "sysctl":
            return str(32 * GIB)
        raise subprocess.TimeoutExpired(argv, 3)
    monkeypatch.setattr(resources, "_command", query)
    metrics = resources.detect_hardware("apple")
    assert metrics.host_total_bytes == 32 * GIB and metrics.host_available_bytes is None
    assert "TimeoutExpired" in metrics.warnings[0]


@pytest.mark.parametrize("device,environment,output,expected", [
    ("cuda:0", {}, "GPU-aaa, 10240, 8192\n", (10 * GIB, 8 * GIB)),
    ("cuda:0", {"CUDA_VISIBLE_DEVICES": "0"}, "GPU-aaa, 10240, 8192\n", (10 * GIB, 8 * GIB)),
    ("cuda:1", {"CUDA_VISIBLE_DEVICES": "GPU-bbb,GPU-aaa"},
     "GPU-aaa, 10240, 8192\nGPU-bbb, 20480, 16384\n", (10 * GIB, 8 * GIB)),
    ("cuda:0", {}, "GPU-aaa, 10240, 8192\nGPU-bbb, 20480, 16384\n", (None, None)),
    ("cuda:0", {"CUDA_VISIBLE_DEVICES": "1,0"},
     "GPU-aaa, 10240, 8192\nGPU-bbb, 20480, 16384\n", (None, None)),
    ("cuda:0", {"CUDA_VISIBLE_DEVICES": ""}, "GPU-aaa, 10240, 8192\n", (None, None)),
    ("cuda:0", {"CUDA_VISIBLE_DEVICES": "MIG-aaa"}, "GPU-aaa, 10240, 8192\n", (None, None)),
    ("cuda:0", {"CUDA_VISIBLE_DEVICES": "GPU-aa"}, "GPU-aaa, 10240, 8192\n", (None, None)),
    ("cuda:1", {}, "GPU-aaa, 10240, 8192\n", (None, None)),
    ("cuda:0", {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-aaa"},
     "GPU-aaa, 10240, 8192\n", (None, None)),
])
def test_nvidia_mapping_never_guesses_physical_index(monkeypatch, device, environment, output, expected):
    def query(argv):
        assert argv == ["nvidia-smi", "--query-gpu=uuid,memory.total,memory.free,mig.mode.current", "--format=csv,noheader,nounits"]
        return "\n".join(line + ", Disabled" for line in output.splitlines())
    monkeypatch.setattr(resources, "_command", query)
    total, available, warning = resources._nvidia_memory(device, environment)
    assert (total, available) == expected
    assert bool(warning) == (total is None)


@pytest.mark.parametrize("mig_mode,known", [("Enabled", False), ("unknown", False),
                                         ("N/A", True), ("[N/A]", True)])
def test_nvidia_physical_capacity_does_not_count_as_mig_partition_capacity(monkeypatch, mig_mode, known):
    monkeypatch.setattr(resources, "_command", lambda argv: f"GPU-aaa, 10240, 8192, {mig_mode}\n")
    total, available, warning = resources._nvidia_memory("cuda:0", {})
    assert (total is not None) == known and (available is not None) == known
    assert bool(warning) != known


def test_nvidia_missing_driver_produces_unknown_not_fake_zero_memory(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Windows")
    def missing(argv):
        raise FileNotFoundError("synthetic missing nvidia-smi")
    monkeypatch.setattr(resources, "_command", missing)
    metrics = resources.detect_hardware("gpu", environment={})
    assert metrics.gpu_available_bytes is None and metrics.gpu_device is None
    assert any("GPU memory query unavailable" in item for item in metrics.warnings)


def test_hardware_query_uses_argument_vector_and_bounded_timeout(monkeypatch):
    def run(argv, **kwargs):
        assert argv == ["nvidia-smi", "--help"]
        assert kwargs == dict(check=True, text=True, capture_output=True, timeout=3)
        return subprocess.CompletedProcess(argv, 0, stdout="synthetic")
    monkeypatch.setattr(resources.subprocess, "run", run)
    assert resources._command(["nvidia-smi", "--help"]) == "synthetic"
