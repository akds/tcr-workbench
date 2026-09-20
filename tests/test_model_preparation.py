"""Preparation cache contracts with synthetic arrays and no model/framework calls."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tcr_workbench import model_preparation as preparation
from tcr_workbench.resources import GIB, HardwareMetrics


@pytest.fixture
def storage_counter():
    worker = Path(preparation.__file__).parent / "backends/preparation_worker.py"
    spec = importlib.util.spec_from_file_location("synthetic_preparation_worker", worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.checkpoint_storage_bytes


def test_checkpoint_storage_counts_optimizer_and_shared_views_once_without_reading_values(
    storage_counter,
):
    class Storage:
        def __init__(self, pointer, size):
            self.pointer, self.size = pointer, size

        def data_ptr(self):
            return self.pointer

        def nbytes(self):
            return self.size

    class Tensor:
        def __init__(self, pointer, size):
            self.storage = Storage(pointer, size)

        def untyped_storage(self):
            return self.storage

        def __getattr__(self, name):
            pytest.fail(f"Storage inspection must not read tensor values via {name}")

    weights, view = Tensor(1000, 4096), Tensor(1000, 4096)
    optimizer = Tensor(2000, 8192)
    checkpoint = {
        "state_dict": {"weights": weights, "view": view},
        "optimizer_states": [{"state": {0: (optimizer, weights)}}],
        "epoch": 8,
        "empty": Tensor(0, 0),
    }
    checkpoint["cycle"] = checkpoint
    assert storage_counter(checkpoint, SimpleNamespace(Tensor=Tensor)) == 12288


def test_checkpoint_storage_scan_handles_deep_container_nesting_without_recursion(storage_counter):
    class Tensor:
        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: 1000, nbytes=lambda: 4096)

    checkpoint = Tensor()
    for _ in range(2000):
        checkpoint = [checkpoint]
    assert storage_counter(checkpoint, SimpleNamespace(Tensor=Tensor)) == 4096


@pytest.fixture
def harness(tmp_path, monkeypatch):
    source = tmp_path / "weights.ckpt"
    source.write_bytes(b"synthetic checkpoint v1")
    upstream = tmp_path / "decoder"
    upstream.mkdir()
    options = dict(
        decoder_dir=str(upstream),
        python_executable="decoder-python",
        model="esmc-300m",
        device="cpu",
        precision="float32",
        checkpoint=str(source),
        mlx_python=None,
        timeout=10,
    )
    identity = dict(
        workbench_adapter_sha256="core-v1",
        environment_sha256="python-v1",
        germline_sha256="germlines-v1",
        source_sha256="decoder-v1",
    )
    mlx_identity = dict(
        mlx_environment_sha256="mlx-environment-v1", mlx_source_sha256="mlx-source-v1"
    )
    calls = []
    h = SimpleNamespace(
        source=source,
        options=options,
        state=tmp_path / "prepared",
        calls=calls,
        identity=identity,
        mlx_identity=mlx_identity,
        parameter_bytes=GIB,
        checkpoint_storage_bytes=3 * GIB,
        tensor_count=308,
        inspections=[],
        hardware={"host_total_bytes": 64 * GIB, "host_available_bytes": 48 * GIB},
    )
    monkeypatch.setattr(preparation, "_decoder_fingerprint", lambda *a: dict(identity))
    monkeypatch.setattr(
        preparation.mlx_decoder, "runtime_fingerprint", lambda *a: dict(mlx_identity)
    )

    def fixture(options, checkpoint, output, *, backend, reference=None):
        calls.append((backend, Path(checkpoint)))
        tokens = np.array([[0, 5, 2, 1], [0, 6, 7, 2]], dtype=np.int32)
        logits = np.ones((2, 4, 64), dtype=np.float32)
        np.savez(output, tokens=tokens, logits=logits)
        if backend == "torch":
            info = {
                "backend": "torch",
                "model": options["model"],
                "tensor_count": h.tensor_count,
                "parameter_bytes": h.parameter_bytes,
                "checkpoint_storage_bytes": h.checkpoint_storage_bytes,
                "device": "cpu" if options["device"] == "apple" else options["device"],
            }
        else:
            info = {
                "backend": "mlx",
                "model": options["model"],
                "device": "apple",
                "precision": options["precision"],
                "source_sha256": preparation.file_sha256(source),
            }
        info.update(load_seconds=1.0, forward_seconds=0.5, batch_size=2, sequence_length=4)
        Path(str(output) + ".json").write_text(json.dumps(info))

    def inspect(options, checkpoint, output):
        spec = preparation.resolve_model(options["model"])
        proof = preparation.InventoryProof(
            model=spec.name,
            tensor_count=h.tensor_count,
            parameter_bytes=h.parameter_bytes,
            checkpoint_storage_bytes=h.checkpoint_storage_bytes,
            backbone=spec.backbone,
            architecture=spec.arch,
            finite_values_checked=False,
            forward_checked=False,
        )
        output.write_text(proof.model_dump_json())
        h.inspections.append(Path(checkpoint))
        return proof

    def convert(command, *args):
        assert command[1] == "-c" and "convert_checkpoint" in command[2]
        target = Path(command[4])
        target.mkdir()
        (target / "model.safetensors").write_bytes(b"synthetic converted tensors")
        (target / "config.json").write_text('{"dtype":"float32"}')

    h.fixture = fixture
    monkeypatch.setattr(preparation, "_fixture", fixture)
    monkeypatch.setattr(preparation, "_inspect", inspect)
    monkeypatch.setattr(
        preparation, "_hardware", lambda options: HardwareMetrics.model_validate(h.hardware)
    )
    monkeypatch.setattr(preparation, "_execute", convert)
    return h


def prepare(h, **kwargs):
    return preparation.prepare_model(h.options, h.state, **kwargs)


def assert_unpublished(h):
    assert not list(h.state.glob("*/prepared.json"))
    assert not list(h.state.glob("*.lock"))
    assert not list(h.state.glob("preparing-*"))


def test_identical_checkpoint_runtime_reuses_preparation_without_model_call(harness):
    h = harness
    options, first = prepare(h)
    assert options["checkpoint"] == str(h.source)
    assert options["model"] == "DecoderTCR-ESMC_300M"
    assert not first["cache_hit"]
    card = json.loads(Path(first["record"]).read_text())
    assert card["checks"]["strict_tensor_inventory"] and card["checks"]["finite_forward"]
    assert card["artifacts"]
    _, second = prepare(h)
    assert second["cache_hit"] and second["record"] == first["record"]
    assert len(h.calls) == 1


def test_plan_only_inspects_metadata_but_never_runs_fixture_or_publishes(harness):
    h = harness
    effective, result = prepare(h, plan_only=True, estimate_forwards=100)
    assert effective["checkpoint"] == str(h.source)
    assert result["plan_only"] and not result["prepared"] and not result["cache_hit"]
    assert result["inventory"]["parameter_bytes"] == h.parameter_bytes
    assert result["inventory"]["checkpoint_storage_bytes"] == h.checkpoint_storage_bytes
    assert not result["inventory"]["finite_values_checked"]
    assert not result["inventory"]["forward_checked"]
    assert result["time_estimate"]["status"] == "unavailable"
    assert h.inspections == [h.source] and not h.calls
    assert set(result["resources"]) == {"preparation", "inference"}
    assert_unpublished(h)
    assert not list(h.state.iterdir())


def test_memory_over_budget_stops_before_allocating_model_and_plan_remains_available(harness):
    h = harness
    h.hardware["host_available_bytes"] = GIB
    with pytest.raises(ValueError, match="Memory check stopped inference"):
        prepare(h)
    assert h.inspections == [h.source] and not h.calls
    assert_unpublished(h)
    _, result = prepare(h, plan_only=True)
    assert result["resources"]["inference"]["blocked"] and not h.calls


def test_cached_preparation_rechecks_current_memory_without_rerunning_model(harness):
    h = harness
    _, first = prepare(h)
    old_card = Path(first["record"]).read_bytes()
    h.hardware["host_available_bytes"] = GIB
    with pytest.raises(ValueError, match="Memory check stopped inference"):
        prepare(h)
    _, planning = prepare(h, plan_only=True, estimate_forwards=8, sequence_length=8)
    assert planning["cache_hit"] and planning["plan_only"]
    assert planning["resources"]["inference"]["blocked"]
    assert planning["time_estimate"]["status"] == "rough_extrapolation"
    assert set(planning["resources"]) == {"inference"}
    assert h.inspections == [h.source] and len(h.calls) == 1
    assert Path(first["record"]).read_bytes() == old_card


def test_apple_cpu_reference_phase_can_block_even_when_apple_inference_fits(harness):
    h = harness
    apple_options(h)
    h.parameter_bytes = 12 * GIB
    h.hardware.update(host_total_bytes=32 * GIB, host_available_bytes=24 * GIB)
    _, planning = prepare(h, plan_only=True)
    plans = planning["resources"]
    assert plans["inference"]["device"] == "apple" and not plans["inference"]["blocked"]
    assert plans["preparation"]["device"] == "cpu" and plans["preparation"]["blocked"]
    assert plans["preparation"]["checkpoint_storage_bytes"] == h.checkpoint_storage_bytes
    assert plans["inference"]["checkpoint_storage_bytes"] is None
    with pytest.raises(ValueError, match="Memory check stopped preparation"):
        prepare(h)
    assert not h.calls
    assert_unpublished(h)


def test_explicit_memory_risk_override_allows_attempt_and_reports_risk(harness, capsys):
    h = harness
    h.hardware["host_available_bytes"] = GIB
    _, record = prepare(h, allow_memory_risk=True)
    assert len(h.calls) == 1 and Path(record["record"]).is_file()
    assert record["resources"]["inference"]["risk"] == "over_budget"
    assert not record["resources"]["inference"]["blocked"]
    assert "override is enabled" in capsys.readouterr().err
    # The override is one invocation's choice, not a permanent cached bypass.
    with pytest.raises(ValueError, match="Memory check stopped"):
        prepare(h)


def test_timing_projection_is_optional_explicitly_rough_and_scales_with_rows_and_length(harness):
    h = harness
    _, first = prepare(h)
    assert first["time_estimate"] is None
    card = json.loads(Path(first["record"]).read_text())
    assert card["checks"]["parameter_bytes"] == GIB
    assert card["timing"] == dict(
        load_seconds=1.0, forward_seconds=0.5, batch_size=2, sequence_length=4
    )
    _, projected = prepare(h, estimate_forwards=8, sequence_length=8)
    estimate = projected["time_estimate"]
    assert estimate["status"] == "rough_extrapolation"
    assert estimate["lower_seconds"] == 2 and estimate["upper_seconds"] == 33
    assert estimate["context_rows"] == 8 and "Not a confidence interval" in estimate["limitations"]
    _, larger = prepare(h, estimate_forwards=16, sequence_length=16)
    assert larger["time_estimate"]["lower_seconds"] > estimate["lower_seconds"]
    assert larger["time_estimate"]["upper_seconds"] > estimate["upper_seconds"]
    assert len(h.calls) == 1


@pytest.mark.parametrize(
    "argument,value",
    [
        ("sequence_length", 0),
        ("sequence_length", 2049),
        ("sequence_length", True),
        ("sequence_length", 1.5),
        ("estimate_forwards", 0),
        ("estimate_forwards", -1),
        ("estimate_forwards", True),
        ("estimate_forwards", 1.5),
    ],
)
def test_invalid_timing_scaling_arguments_reject_before_inspection(harness, argument, value):
    h = harness
    with pytest.raises(ValueError):
        prepare(h, **{argument: value})
    assert not h.calls and not h.inspections and not h.state.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_count", 307),
        ("parameter_bytes", 1000),
        ("checkpoint_storage_bytes", 1000),
        ("batch_size", 1),
        ("sequence_length", 3),
        ("forward_seconds", 0.0),
        ("forward_seconds", float("nan")),
        ("load_seconds", -1.0),
    ],
)
def test_inspection_forward_inventory_and_timing_must_agree(harness, monkeypatch, field, value):
    h = harness

    def corrupt(options, checkpoint, output, **kwargs):
        h.fixture(options, checkpoint, output, **kwargs)
        metadata = Path(str(output) + ".json")
        proof = json.loads(metadata.read_text())
        proof[field] = value
        metadata.write_text(json.dumps(proof))

    monkeypatch.setattr(preparation, "_fixture", corrupt)
    with pytest.raises(ValueError):
        prepare(h)
    assert_unpublished(h)


def test_apple_report_keeps_selected_backend_timing(harness, monkeypatch):
    h = harness
    apple_options(h)

    def timed(options, checkpoint, output, **kwargs):
        h.fixture(options, checkpoint, output, **kwargs)
        if kwargs["backend"] == "mlx":
            metadata = Path(str(output) + ".json")
            proof = json.loads(metadata.read_text())
            proof.update(load_seconds=2.0, forward_seconds=0.125)
            metadata.write_text(json.dumps(proof))

    monkeypatch.setattr(preparation, "_fixture", timed)
    _, result = prepare(h, estimate_forwards=20)
    assert result["time_estimate"]["calibration"]["forward_seconds"] == 0.125
    assert result["time_estimate"]["calibration"]["load_seconds"] == 2.0


@pytest.mark.parametrize(
    "field,value", [("batch_size", 3), ("sequence_length", 5), ("forward_seconds", 0.0)]
)
def test_apple_timing_contract_must_match_returned_arrays(harness, monkeypatch, field, value):
    h = harness
    apple_options(h)

    def corrupt(options, checkpoint, output, **kwargs):
        h.fixture(options, checkpoint, output, **kwargs)
        if kwargs["backend"] == "mlx":
            metadata = Path(str(output) + ".json")
            proof = json.loads(metadata.read_text())
            proof[field] = value
            metadata.write_text(json.dumps(proof))

    monkeypatch.setattr(preparation, "_fixture", corrupt)
    with pytest.raises(ValueError):
        prepare(h)
    assert_unpublished(h)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"model": "different"},
        {"backbone": "esm2"},
        {"architecture": "600m"},
        {"tensor_count": True},
        {"parameter_bytes": 0},
        {"finite_values_checked": True},
        {"forward_checked": True},
    ],
)
def test_inventory_inspection_contract_is_metadata_only_and_architecture_bound(
    tmp_path, monkeypatch, change
):
    spec = preparation.resolve_model("esmc-300m")
    proof = dict(
        model=spec.name,
        tensor_count=308,
        parameter_bytes=GIB,
        checkpoint_storage_bytes=3 * GIB,
        backbone=spec.backbone,
        architecture=spec.arch,
        finite_values_checked=False,
        forward_checked=False,
    )
    proof.update(change)
    options = dict(model=spec.name, python_executable="synthetic-python", decoder_dir=str(tmp_path))
    checkpoint, output = tmp_path / "weights.ckpt", tmp_path / "inventory.json"

    def execute(command, cwd, timeout):
        assert command[0] == options["python_executable"]
        assert "--inspect-only" in command and cwd == tmp_path
        assert command[command.index("--checkpoint") + 1] == str(checkpoint)
        Path(command[command.index("--output") + 1]).write_text(json.dumps(proof))

    monkeypatch.setattr(preparation, "_execute", execute)
    if change:
        with pytest.raises(ValueError):
            preparation._inspect(options, checkpoint, output)
    else:
        result = preparation._inspect(options, checkpoint, output)
        assert result.parameter_bytes == GIB and not result.forward_checked


@pytest.mark.parametrize(
    "outcome", ["valid", "negative", "bool", "missing", "malformed", "list", "failed"]
)
def test_selected_torch_gpu_memory_fallback_validates_metrics(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(
        preparation,
        "detect_hardware",
        lambda device: HardwareMetrics(host_total_bytes=32 * GIB, host_available_bytes=24 * GIB),
    )
    options = dict(
        device="cuda:1",
        python_executable="selected-env-python",
        decoder_dir=str(tmp_path),
        timeout=90,
    )

    def execute(command, cwd, timeout):
        assert command[0] == "selected-env-python" and command[-2] == "cuda:1"
        assert "mem_get_info" in command[2] and cwd == tmp_path and timeout == 30
        if outcome == "failed":
            raise RuntimeError("CUDA query failed")
        payload = dict(free=8 * GIB, total=12 * GIB)
        if outcome == "negative":
            payload["free"] = -1
        elif outcome == "bool":
            payload["free"] = True
        elif outcome == "missing":
            del payload["total"]
        elif outcome == "list":
            payload = []
        Path(command[-1]).write_text(
            "invalid JSON" if outcome == "malformed" else json.dumps(payload)
        )

    monkeypatch.setattr(preparation, "_execute", execute)
    metrics = preparation._hardware(options)
    assert metrics.host_available_bytes == 24 * GIB
    if outcome == "valid":
        assert metrics.gpu_device == "cuda:1" and metrics.gpu_available_bytes == 8 * GIB
        assert metrics.gpu_total_bytes == 12 * GIB
        assert "selected Torch runtime cuda.mem_get_info" in metrics.sources
    else:
        assert metrics.gpu_available_bytes is None and metrics.gpu_total_bytes is None
        assert any("query failed" in warning for warning in metrics.warnings)


def test_unambiguous_gpu_metrics_skip_selected_runtime_subprocess(monkeypatch):
    metrics = HardwareMetrics(
        host_available_bytes=24 * GIB,
        gpu_total_bytes=12 * GIB,
        gpu_available_bytes=8 * GIB,
        gpu_device="cuda:0",
    )
    monkeypatch.setattr(preparation, "detect_hardware", lambda device: metrics)
    monkeypatch.setattr(
        preparation, "_execute", lambda *a: pytest.fail("unnecessary second hardware query")
    )
    assert preparation._hardware({"device": "cuda"}) == metrics


@pytest.mark.parametrize(
    "key", ["workbench_adapter_sha256", "environment_sha256", "germline_sha256", "source_sha256"]
)
def test_source_environment_and_germline_changes_invalidate_cache(harness, key):
    h = harness
    _, first = prepare(h)
    original = Path(first["record"]).read_bytes()
    h.identity[key] = "changed"
    _, second = prepare(h)
    assert not second["cache_hit"] and first["record"] != second["record"]
    assert len(h.calls) == 2 and Path(first["record"]).read_bytes() == original


def test_checkpoint_content_change_invalidates_even_with_same_size_and_mtime(harness):
    h = harness
    _, first = prepare(h)
    stat = h.source.stat()
    h.source.write_bytes(b"synthetic checkpoint v2")
    os.utime(h.source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    _, second = prepare(h)
    assert not second["cache_hit"] and first["record"] != second["record"]
    assert len(h.calls) == 2


def test_explicit_checksum_mismatch_fails_before_preparing(harness):
    h = harness
    with pytest.raises(ValueError, match="SHA-256"):
        prepare(h, expected_sha256="0" * 64)
    assert not h.calls and not h.state.exists()


def test_user_label_does_not_repeat_numerical_preparation(harness):
    h = harness
    _, first = prepare(h, model_id="lab-release-a")
    _, second = prepare(h, model_id="lab-release-b")
    assert first["record"] == second["record"] and len(h.calls) == 1
    assert second["cache_hit"]
    assert json.loads(Path(second["record"]).read_text())["model_id"] == "lab-release-a"


def test_tampered_array_artifact_preserves_corrupt_record_and_refuses_cache_hit(harness):
    h = harness
    _, result = prepare(h)
    record = Path(result["record"])
    original = record.read_bytes()
    (record.parent / "reference.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        prepare(h)
    assert len(h.calls) == 1 and record.read_bytes() == original
    assert not list(h.state.glob("*.lock"))


@pytest.mark.parametrize(
    "mutation", ["redirect_checkpoint", "path_traversal", "empty_artifacts", "false_check"]
)
def test_edited_preparation_card_cannot_be_reused(harness, mutation):
    h = harness
    _, result = prepare(h)
    record = Path(result["record"])
    card = json.loads(record.read_text())
    if mutation == "redirect_checkpoint":
        other = h.source.with_name("different.ckpt")
        other.write_bytes(b"other weights")
        card["checkpoint"] = str(other)
    elif mutation == "path_traversal":
        card["artifacts"] = {"../outside": hashlib.sha256(b"nothing").hexdigest()}
    elif mutation == "empty_artifacts":
        card["artifacts"] = {}
    else:
        card["checks"]["strict_tensor_inventory"] = False
    record.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="integrity"):
        prepare(h)
    assert len(h.calls) == 1


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("checks", "parameter_bytes", 1),
        ("checks", "tensor_count", 1),
        ("checks", "checkpoint_storage_bytes", 1),
        ("timing", "forward_seconds", 0.001),
    ],
)
def test_preparation_card_memory_and_timing_cannot_disagree_with_worker_artifacts(
    harness, section, field, value
):
    h = harness
    _, result = prepare(h)
    record = Path(result["record"])
    card = json.loads(record.read_text())
    card[section][field] = value
    record.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="integrity"):
        prepare(h)
    assert len(h.calls) == 1


def test_worker_failure_releases_lock_and_leaves_no_prepared_record(harness, monkeypatch):
    h = harness

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(preparation, "_fixture", fail)
    with pytest.raises(RuntimeError, match="worker failure"):
        prepare(h)
    assert_unpublished(h)


def test_identity_change_during_worker_cannot_publish(harness, monkeypatch):
    h = harness

    def changing(*args, **kwargs):
        h.fixture(*args, **kwargs)
        h.identity["germline_sha256"] = "changed during run"

    monkeypatch.setattr(preparation, "_fixture", changing)
    with pytest.raises(ValueError, match="changed during preparation"):
        prepare(h)
    assert_unpublished(h)


@pytest.mark.parametrize(
    "corruption", ["nan", "wrong_vocab", "float_tokens", "shape", "device", "backend", "count_bool"]
)
def test_invalid_reference_worker_result_is_not_registered(harness, monkeypatch, corruption):
    h = harness

    def corrupt(options, checkpoint, output, **kwargs):
        h.fixture(options, checkpoint, output, **kwargs)
        with np.load(output) as archive:
            tokens, logits = archive["tokens"], archive["logits"]
        info_path = Path(str(output) + ".json")
        info = json.loads(info_path.read_text())
        if corruption == "nan":
            logits[0, 0, 0] = np.nan
        elif corruption == "wrong_vocab":
            logits = logits[:, :, :33]
        elif corruption == "float_tokens":
            tokens = tokens.astype(np.float32)
        elif corruption == "shape":
            logits = logits[:, :2]
        elif corruption == "device":
            info["device"] = "cuda:0"
        elif corruption == "backend":
            info["backend"] = "mlx"
        else:
            info["tensor_count"] = True
        np.savez(output, tokens=tokens, logits=logits)
        info_path.write_text(json.dumps(info))

    monkeypatch.setattr(preparation, "_fixture", corrupt)
    with pytest.raises(ValueError):
        prepare(h)
    assert_unpublished(h)


def apple_options(h):
    h.options.update(device="apple", mlx_python="mlx-python")


def test_raw_apple_checkpoint_converts_once_and_preserves_original(harness):
    h = harness
    apple_options(h)
    original = h.source.read_bytes()
    effective, result = prepare(h)
    converted = Path(effective["checkpoint"])
    assert converted == Path(result["record"]).parent / "bundle"
    assert (converted / "model.safetensors").is_file()
    assert h.source.read_bytes() == original
    assert result["checks"]["parity_passed"] is True
    assert [backend for backend, _ in h.calls] == ["torch", "mlx"]
    cached, cached_result = prepare(h)
    assert cached_result["cache_hit"] and cached["checkpoint"] == str(converted)
    assert len(h.calls) == 2


@pytest.mark.parametrize("key", ["mlx_environment_sha256", "mlx_source_sha256"])
def test_apple_runtime_change_invalidates_preparation(harness, key):
    h = harness
    apple_options(h)
    _, first = prepare(h)
    h.mlx_identity[key] = "changed"
    _, second = prepare(h)
    assert second["record"] != first["record"] and not second["cache_hit"]
    assert len(h.calls) == 4


def test_apple_precision_modes_have_separate_preparation_records(harness):
    h = harness
    apple_options(h)
    _, first = prepare(h)
    h.options["precision"] = "float16"
    _, second = prepare(h)
    assert first["record"] != second["record"] and len(h.calls) == 4


@pytest.mark.parametrize("corruption", ["source_hash", "backend", "device", "precision", "parity"])
def test_invalid_apple_result_cannot_publish_converted_bundle(harness, monkeypatch, corruption):
    h = harness
    apple_options(h)

    def corrupt(options, checkpoint, output, **kwargs):
        h.fixture(options, checkpoint, output, **kwargs)
        if kwargs["backend"] == "mlx":
            metadata = Path(str(output) + ".json")
            info = json.loads(metadata.read_text())
            if corruption == "source_hash":
                info["source_sha256"] = "0" * 64
            elif corruption == "backend":
                info["backend"] = "torch"
            elif corruption == "device":
                info["device"] = "cpu"
            elif corruption == "precision":
                info["precision"] = "float16"
            else:
                with np.load(output) as archive:
                    tokens, logits = archive["tokens"], archive["logits"]
                np.savez(output, tokens=tokens, logits=logits + 1)
            metadata.write_text(json.dumps(info))

    monkeypatch.setattr(preparation, "_fixture", corrupt)
    with pytest.raises(ValueError):
        prepare(h)
    assert_unpublished(h)


def test_cache_is_rechecked_after_acquiring_lock(harness, monkeypatch):
    h = harness
    prepare(h)
    original = preparation._cached
    checks = []

    def delayed(path, fingerprint):
        checks.append(path)
        return None if len(checks) == 1 else original(path, fingerprint)

    monkeypatch.setattr(preparation, "_cached", delayed)
    _, result = prepare(h)
    assert result["cache_hit"] and len(checks) == 2 and len(h.calls) == 1
    assert not list(h.state.glob("*.lock"))


def test_lock_exclusion_preserves_owner_lock_and_cleans_up(tmp_path):
    path = tmp_path / "preparation.lock"
    with preparation._lock(path):
        original = path.read_bytes()
        with pytest.raises(ValueError, match="already running"):
            with preparation._lock(path):
                pytest.fail("second worker acquired lock")
        assert path.read_bytes() == original
    assert not path.exists()


def existing_bundle(h):
    bundle = h.source.parent / "existing-bundle"
    bundle.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (bundle / name).write_bytes(b"synthetic bundle component")
    (bundle / "manifest.json").write_text(
        json.dumps({"source_sha256": preparation.file_sha256(h.source)})
    )
    apple_options(h)
    h.options["checkpoint"] = str(bundle)
    return bundle


def test_existing_bundle_requires_exact_matching_original_for_parity(harness):
    h = harness
    bundle = existing_bundle(h)
    with pytest.raises(ValueError, match="matching original PyTorch checkpoint"):
        prepare(h)
    assert_unpublished(h)
    assert not h.calls
    effective, record = prepare(h, reference_checkpoint=h.source)
    assert effective["checkpoint"] == str(bundle)
    assert h.calls == [("torch", h.source), ("mlx", bundle)]
    assert record["checks"]["parity_passed"]


def test_existing_bundle_can_find_relative_cpu_configuration_reference(harness):
    h = harness
    bundle = existing_bundle(h)
    (h.state.parent / "runtime-cpu.json").write_text(
        json.dumps(
            {"decoder_dir": "decoder", "python_executable": "python", "checkpoint": h.source.name}
        )
    )
    effective, _ = prepare(h)
    assert effective["checkpoint"] == str(bundle) and h.calls[0] == ("torch", h.source)


@pytest.mark.parametrize("damaged_original", [False, True])
def test_prepared_apple_bundle_retains_its_original_after_cpu_release_changes(harness, monkeypatch, damaged_original):
    h = harness
    apple_options(h)
    convert = preparation._execute

    def complete_bundle(command, *args):
        convert(command, *args)
        bundle = Path(command[4])
        (bundle / "tokenizer.json").write_text("{}")
        (bundle / "manifest.json").write_text(json.dumps({"source_sha256": preparation.file_sha256(h.source)}))

    monkeypatch.setattr(preparation, "_execute", complete_bundle)
    effective, _ = prepare(h)
    newer = h.source.with_name("newer-release.ckpt")
    newer.write_bytes(b"different checkpoint for a later CPU setup")
    (h.state.parent / "runtime-cpu.json").write_text(json.dumps(
        {"decoder_dir": "decoder", "python_executable": "python", "checkpoint": str(newer)}))
    h.options["checkpoint"] = effective["checkpoint"]
    h.calls.clear()
    if damaged_original:
        h.source.write_bytes(b"damaged original")
        with pytest.raises(ValueError, match="matching original PyTorch checkpoint"):
            prepare(h)
        assert not h.calls
    else:
        restored, record = prepare(h)
        assert restored["checkpoint"] == effective["checkpoint"]
        assert record["checks"]["parity_passed"]
        assert h.calls == [("torch", h.source), ("mlx", Path(effective["checkpoint"]))]


def test_converted_weights_tampering_cannot_reuse_preparation(harness):
    h = harness
    apple_options(h)
    effective, _ = prepare(h)
    (Path(effective["checkpoint"]) / "model.safetensors").write_bytes(b"altered")
    with pytest.raises(ValueError, match="integrity"):
        prepare(h)
    assert len(h.calls) == 2


def test_parity_ignores_finite_pad_query_difference_but_checks_real_tokens(tmp_path):
    tokens = np.array([[0, 5, 2, 1], [0, 5, 6, 2]], dtype=np.int32)
    expected = np.ones((2, 4, 64), dtype=np.float32)
    observed = expected.copy()
    observed[0, 3] = 1000
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    np.savez(reference, tokens=tokens, logits=expected)
    np.savez(candidate, tokens=tokens, logits=observed)
    result = preparation._compare(reference, candidate, "float32")
    assert result["max_absolute_logit_error"] == 0
    observed[0, 2, 1] += 0.01
    np.savez(candidate, tokens=tokens, logits=observed)
    with pytest.raises(ValueError, match="parity failed"):
        preparation._compare(reference, candidate, "float32")


def test_float16_parity_rejects_one_collapsed_token_despite_small_global_error(tmp_path):
    tokens = np.array([[0, 5, 2, 1], [0, 5, 6, 2]], dtype=np.int32)
    expected = np.full((2, 4, 64), 1000, dtype=np.float32)
    expected[0, 1] = 1
    observed = expected.copy()
    observed[0, 1] = 0
    valid = tokens != 1
    assert (
        np.linalg.norm(observed[valid] - expected[valid]) / np.linalg.norm(expected[valid]) < 0.01
    )
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    np.savez(reference, tokens=tokens, logits=expected)
    np.savez(candidate, tokens=tokens, logits=observed)
    with pytest.raises(ValueError, match="parity failed"):
        preparation._compare(reference, candidate, "float16")


@pytest.mark.parametrize("corruption", ["float_tokens", "float64_logits", "different_token", "nan"])
def test_parity_rejects_invalid_candidate_contract(tmp_path, corruption):
    tokens = np.array([[0, 5, 2, 1], [0, 5, 6, 2]], dtype=np.int32)
    logits = np.ones((2, 4, 64), dtype=np.float32)
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    np.savez(reference, tokens=tokens, logits=logits)
    if corruption == "float_tokens":
        tokens = tokens.astype(np.float32)
    elif corruption == "float64_logits":
        logits = logits.astype(np.float64)
    elif corruption == "different_token":
        tokens[0, 1] = 6
    else:
        logits[0, 3, 0] = np.nan
    np.savez(candidate, tokens=tokens, logits=logits)
    with pytest.raises(ValueError):
        preparation._compare(reference, candidate, "float32")


def test_explicit_prepare_cli_prints_parseable_record_without_network(harness, capsys):
    from tcr_workbench.cli import main

    h = harness
    assert (
        main(
            [
                "prepare-model",
                "--decoder-dir",
                h.options["decoder_dir"],
                "--python",
                h.options["python_executable"],
                "--checkpoint",
                str(h.source),
                "--state-dir",
                str(h.state),
                "--model-id",
                "my-local-release",
                "--expected-sha256",
                preparation.file_sha256(h.source),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert record["cache_hit"] is False and Path(record["record"]).is_file()
    assert "Preparing" in captured.err


@pytest.mark.parametrize("enabled", [False, True])
def test_auto_preparation_is_opt_in_and_passes_effective_checkpoint(tmp_path, monkeypatch, enabled):
    import polars as pl
    from tcr_workbench import decoder_pmhc
    from tcr_workbench.cli import main

    panel = tmp_path / "panel.csv"
    panel.write_text("peptide,hla\nAC,A*02:01\n")
    source, converted, state = tmp_path / "raw.ckpt", tmp_path / "converted", tmp_path / "prepared"
    calls = []
    if enabled:
        monkeypatch.setenv("TCR_WORKBENCH_PREPARE_DIR", str(state))
    else:
        monkeypatch.delenv("TCR_WORKBENCH_PREPARE_DIR", raising=False)

    def prepare_options(options, state_dir, **kwargs):
        calls.append((options, state_dir, kwargs))
        return {**options, "checkpoint": str(converted)}, {"cache_hit": False}

    def score(panel_path, output, **kwargs):
        assert Path(kwargs["checkpoint"]) == (converted if enabled else source)
        return pl.DataFrame({"score": [-1.0]}), {}

    monkeypatch.setattr(preparation, "prepare_model", prepare_options)
    monkeypatch.setattr(decoder_pmhc, "score_pmhc", score)
    assert (
        main(
            [
                "pmhc-score",
                "--panel",
                str(panel),
                "--out",
                str(tmp_path / "out"),
                "--decoder-dir",
                str(tmp_path),
                "--python",
                "decoder-python",
                "--device",
                "apple",
                "--checkpoint",
                str(source),
                "--mlx-python",
                "mlx-python",
            ]
        )
        == 0
    )
    assert len(calls) == int(enabled)
    if enabled:
        assert Path(calls[0][1]) == state


def test_missing_cli_input_fails_before_automatic_preparation(tmp_path, monkeypatch):
    from tcr_workbench.cli import main

    monkeypatch.setenv("TCR_WORKBENCH_PREPARE_DIR", str(tmp_path / "prepared"))
    monkeypatch.setattr(
        preparation,
        "prepare_model",
        lambda *a, **k: pytest.fail("prepared before missing input validation"),
    )
    assert (
        main(
            [
                "pmhc-score",
                "--panel",
                str(tmp_path / "missing.csv"),
                "--out",
                str(tmp_path / "out"),
                "--decoder-dir",
                str(tmp_path),
                "--python",
                "unused",
            ]
        )
        == 2
    )
