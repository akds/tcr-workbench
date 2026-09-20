"""6B setup dispatch and resource gates, without downloads or model allocation."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tcr_workbench import resources


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    root = tmp_path / "relocated project"
    root.mkdir()
    for name in ("tcr", "bootstrap"):
        path = root / f"{name}.py"
        shutil.copyfile(Path(__file__).resolve().parents[1] / path.name, path)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module, root


def fake_runtime(module, root, monkeypatch):
    decoder = root / ".tcr/DecoderTCR"
    decoder.mkdir(parents=True)
    (decoder / ".workbench-revision").write_text(module.DECODER_REV)
    calls = []
    monkeypatch.setattr(module, "ensure_uv", lambda *args: ["uv"])
    monkeypatch.setattr(module, "run", lambda command, **kwargs: calls.append(list(command)))
    monkeypatch.setattr(module, "install_decoder", lambda *args: calls.append(["install_decoder"]))
    monkeypatch.setattr(module, "probe", lambda settings, *args, **kwargs: calls.append(["probe", settings]))
    monkeypatch.setattr(module, "download", lambda *args: pytest.fail("6B setup downloaded model bytes"))
    return calls


def local_fixture(module, root, monkeypatch):
    checkpoint = root / "arbitrary checkpoint filename.ckpt"
    checkpoint.write_bytes(b"small synthetic fixture; never used for inference")
    model, artifact, _, _ = module.SETUP_MODELS["esmc-6b"]
    monkeypatch.setitem(module.SETUP_MODELS, "esmc-6b", (
        model, artifact, hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint.stat().st_size))
    return checkpoint


def test_6b_rejects_missing_local_file_before_installation(bootstrap, monkeypatch, capsys):
    module, root = bootstrap
    monkeypatch.setattr(module, "ensure_uv", lambda *args: pytest.fail("installation started"))
    flags = ["setup", "--model", "esmc-6b", "--checkpoint", str(root / "missing.ckpt")]
    assert module.main(flags, root=root) == 1
    assert not (root / ".tcr").exists()
    assert "checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("device", ["cpu", "apple"])
def test_6b_setup_uses_pinned_content_and_correct_backend(bootstrap, monkeypatch, device):
    module, root = bootstrap
    calls = fake_runtime(module, root, monkeypatch)
    checkpoint = local_fixture(module, root, monkeypatch)
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs:
                        calls.append(["resources", args[2], args[3], kwargs["allow_memory_risk"]]))
    flags = ["setup", "--device", device, "--model", "esmc-6b", "--checkpoint", str(checkpoint),
             "--allow-memory-risk"]
    assert module.main(flags, root=root) == 0
    guard = ["resources", "DecoderTCR-ESMC_6B", device, True]
    assert calls.index(guard) < calls.index(["install_decoder"])
    cpu = json.loads((root / ".tcr/runtime-cpu.json").read_text())
    assert (cpu["model"], cpu["device"], cpu["checkpoint"]) == (
        "DecoderTCR-ESMC_6B", "cpu", str(checkpoint))
    selected = json.loads((root / ".tcr/runtime.json").read_text())
    assert selected["device"] == device and selected["precision"] == "float32"
    conversion = [command for command in calls if any("convert_weights.py" in str(x) for x in command)]
    if device == "apple":
        assert len(conversion) == 1
        command = conversion[0]
        assert command[command.index("--model-size") + 1] == "6b"
        assert command[command.index("--expected-sha256") + 1] == module.SETUP_MODELS["esmc-6b"][2]
        assert command[command.index("--output") + 1] == root / ".tcr/models/decoder-6b-fp32"
        assert Path(selected["checkpoint"]).name == "decoder-6b-fp32"
    else:
        assert conversion == []


@pytest.mark.parametrize("corruption", [b"different size", b"X"])
def test_6b_setup_rejects_unpinned_local_bytes(bootstrap, monkeypatch, capsys, corruption):
    module, root = bootstrap
    fake_runtime(module, root, monkeypatch)
    checkpoint = local_fixture(module, root, monkeypatch)
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(corruption if len(corruption) > 1 else corruption * len(original))
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs: None)
    assert module.main(["setup", "--model", "esmc-6b", "--checkpoint", str(checkpoint)], root=root) == 1
    assert "pinned DecoderTCR-ESMC_6B" in capsys.readouterr().err
    assert not list((root / ".tcr").glob("runtime*.json"))


def test_blocked_setup_preserves_configuration_and_stops_before_model_install(bootstrap, monkeypatch):
    module, root = bootstrap
    calls = fake_runtime(module, root, monkeypatch)
    checkpoint = local_fixture(module, root, monkeypatch)
    config = root / ".tcr/runtime.json"
    config.write_text("previous configuration")

    def blocked(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["resource-check"])

    monkeypatch.setattr(module, "check_setup_resources", blocked)
    assert module.main(["setup", "--model", "esmc-6b", "--checkpoint", str(checkpoint)], root=root) == 1
    assert ["install_decoder"] not in calls
    assert not any("convert_weights.py" in str(command) for command in calls)
    assert config.read_text() == "previous configuration"
    assert not (root / ".tcr/setup.lock").exists()


def test_core_only_6b_selection_does_not_require_a_checkpoint_or_check_model_memory(bootstrap, monkeypatch):
    module, root = bootstrap
    fake_runtime(module, root, monkeypatch)
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs: pytest.fail("model resource check"))
    assert module.main(["setup", "--core-only", "--model", "esmc-6b"], root=root) == 0
    assert not list((root / ".tcr").glob("runtime*.json"))


@pytest.mark.parametrize("flags", [
    ["--checkpoint-url", ""],
    ["--checkpoint", "", "--expected-sha256", "a" * 64],
    ["--checkpoint-url", "https://example.invalid/release.ckpt"],
    ["--checkpoint-url", "https://example.invalid/release.ckpt", "--expected-sha256", "invalid"],
    ["--checkpoint-url", "http://example.invalid/release.ckpt", "--expected-sha256", "a" * 64],
    ["--checkpoint-url", "https://user:secret@example.invalid/release.ckpt", "--expected-sha256", "a" * 64],
    ["--expected-sha256", "a" * 64],
    ["--core-only", "--checkpoint-url", "https://example.invalid/release.ckpt", "--expected-sha256", "a" * 64],
    ["--reuse-config", "runtime.json", "--checkpoint-url", "https://example.invalid/release.ckpt",
     "--expected-sha256", "a" * 64],
])
def test_custom_checkpoint_options_fail_before_installation(bootstrap, monkeypatch, flags):
    module, root = bootstrap
    monkeypatch.setattr(module, "ensure_uv", lambda *args: pytest.fail("installation started"))
    assert module.main(["setup", *flags], root=root) == 1
    assert not (root / ".tcr").exists()


@pytest.mark.parametrize("model", ["esmc-300m", "esmc-600m", "esmc-6b"])
@pytest.mark.parametrize("device", ["cpu", "gpu", "apple"])
def test_future_releases_download_prepare_and_save_selected_defaults(bootstrap, monkeypatch, model, device):
    module, root = bootstrap
    real_download = module.download
    calls = fake_runtime(module, root, monkeypatch)
    monkeypatch.setattr(module, "download", real_download)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux" if device == "gpu" else "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs:
                        calls.append(["resources", kwargs]))
    downloads = []
    body = b"synthetic V1.5 checkpoint"

    def open_url(request, **kwargs):
        downloads.append(request.full_url)
        return io.BytesIO(body)

    monkeypatch.setattr(module.urllib.request, "urlopen", open_url)
    preparations = []

    def prepare(settings, state, core, env, digest, allow):
        assert Path(settings["checkpoint"]).read_bytes() == body
        preparations.append((dict(settings), digest, allow))
        if device == "apple":
            return dict(settings, checkpoint=str(state / "prepared" / digest / "bundle"))
        return settings

    monkeypatch.setattr(module, "prepare_setup_model", prepare)
    checkpoints = []
    for version in ("v1.5", "v1.6"):
        body = f"synthetic {version} {model} checkpoint".encode()
        digest = hashlib.sha256(body).hexdigest()
        url = f"https://github.com/Biohub/DecoderTCR/releases/download/{version}/{model}.ckpt"
        flags = ["setup", "--device", device, "--model", model, "--checkpoint-url", url,
                 "--expected-sha256", digest.upper(), "--allow-memory-risk"]
        assert module.main(flags, root=root) == 0
        checkpoint = root / ".tcr/checkpoints" / model / digest / "weights.ckpt"
        assert checkpoint.read_bytes() == body
        checkpoints.append(checkpoint)
        saved = json.loads((root / ".tcr/runtime.json").read_text())
        assert saved["model"] == module.SETUP_MODELS[model][0]
        assert saved["device"] == ("cuda" if device == "gpu" else device)
        assert saved["checkpoint"] == (str(root / ".tcr/prepared" / digest / "bundle")
                                       if device == "apple" else str(checkpoint))
        cpu = json.loads((root / ".tcr/runtime-cpu.json").read_text())
        assert cpu["device"] == "cpu" and cpu["checkpoint"] == str(checkpoint)
        assert preparations[-1][1:] == (digest, True)
        assert module.main(flags, root=root) == 0  # Reuse verified bytes without network.
    assert len(downloads) == 2
    assert checkpoints[0] != checkpoints[1] and all(path.is_file() for path in checkpoints)
    assert next(i for i, call in enumerate(calls) if call[0] == "resources") < calls.index(["install_decoder"])
    assert not any("convert_weights.py" in str(command) for command in calls)
    previous = (root / ".tcr/runtime.json").read_bytes()
    checkpoints[-1].write_bytes(b"corrupted cached download")
    assert module.main(flags, root=root) == 1
    assert (root / ".tcr/runtime.json").read_bytes() == previous and len(downloads) == 2


@pytest.mark.parametrize("failure", ["checksum", "compatibility"])
def test_future_release_failure_preserves_saved_defaults(bootstrap, monkeypatch, failure):
    module, root = bootstrap
    fake_runtime(module, root, monkeypatch)
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs: None)
    checkpoint = root / "release.ckpt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for name in ("runtime.json", "runtime-cpu.json", "runtime-apple.json", "runtime-gpu.json"):
        (root / ".tcr" / name).write_text("previous configuration")

    def prepare(*args):
        assert failure == "compatibility"
        raise ValueError("Checkpoint architecture does not match selected model")

    monkeypatch.setattr(module, "prepare_setup_model", prepare)
    assert module.main(["setup", "--checkpoint", str(checkpoint), "--expected-sha256",
                        digest if failure == "compatibility" else "0" * 64], root=root) == 1
    assert all(path.read_text() == "previous configuration" for path in (root / ".tcr").glob("runtime*.json"))
    assert not (root / ".tcr/setup.lock").exists()


def test_local_future_checkpoint_becomes_default_without_copying_or_download(bootstrap, monkeypatch):
    module, root = bootstrap
    fake_runtime(module, root, monkeypatch)
    checkpoint = root / "new release.ckpt"
    checkpoint.write_bytes(b"synthetic newer weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "prepare_setup_model", lambda settings, *args: settings)
    assert module.main(["setup", "--checkpoint", str(checkpoint), "--expected-sha256", digest], root=root) == 0
    saved = json.loads((root / ".tcr/runtime.json").read_text())
    assert saved["checkpoint"] == str(checkpoint)
    assert not (root / ".tcr/checkpoints").exists()


def test_setup_preparation_uses_existing_model_checks(bootstrap, monkeypatch):
    from tcr_workbench import model_preparation

    module, root = bootstrap
    seen = []
    settings = {"checkpoint": "raw.ckpt", "model": "DecoderTCR-ESMC_600M", "device": "apple"}

    def prepare(options, state, **kwargs):
        seen.append((options, state, kwargs))
        return dict(options, checkpoint="prepared/bundle"), {}

    monkeypatch.setattr(model_preparation, "prepare_model", prepare)

    def execute(command, **kwargs):
        monkeypatch.setattr(sys, "argv", ["-c", *[str(arg) for arg in command[3:]]])
        exec(compile(command[2], "setup-preparation", "exec"), {})

    monkeypatch.setattr(module, "run", execute)
    prepared = module.prepare_setup_model(settings, root, Path(sys.executable), {}, "a" * 64, True)
    assert prepared == dict(settings, checkpoint="prepared/bundle")
    assert seen == [(settings, str(root / "prepared"), dict(expected_sha256="a" * 64,
                                                           allow_memory_risk=True, sequence_length=64))]
    assert not list(root.glob("model-setup-*"))


@pytest.mark.parametrize("device,expected", [
    ("cpu", [("cpu", "torch")]), ("gpu", [("cuda", "torch")]),
    ("apple", [("cpu", "torch"), ("apple", "mlx")]),
])
@pytest.mark.parametrize("allow", [False, True])
def test_setup_resource_estimates_use_resident_parameters_and_separate_phases(
        bootstrap, monkeypatch, device, expected, allow):
    module, _ = bootstrap
    seen = []
    real_plan = resources.plan_resources

    def plan(parameter_bytes, selected, backend, **kwargs):
        seen.append((parameter_bytes, selected, backend, kwargs))
        return real_plan(parameter_bytes, selected, backend, hardware=resources.HardwareMetrics(), **kwargs)

    monkeypatch.setattr(resources, "plan_resources", plan)

    def execute(command, **kwargs):
        monkeypatch.setattr(sys, "argv", ["-c", *command[3:]])
        exec(compile(command[2], "setup-resource-check", "exec"), {})

    monkeypatch.setattr(module, "run", execute)
    module.check_setup_resources(Path(sys.executable), {}, "DecoderTCR-ESMC_6B", device,
                                 allow_memory_risk=allow)
    assert [(item[1], item[2]) for item in seen] == expected
    assert all(item[0] == 25_408_020_736 for item in seen)
    assert all(item[0] != module.SETUP_MODELS["esmc-6b"][3] for item in seen)
    for _, _, backend, kwargs in seen:
        expected_kwargs = dict(batch_size=2, sequence_length=64, allow_memory_risk=allow)
        if backend == "torch":
            expected_kwargs["checkpoint_storage_bytes"] = module.SETUP_MODELS["esmc-6b"][3]
        assert kwargs == expected_kwargs


def test_300m_setup_budgets_optimizer_checkpoint_storage_separately(bootstrap, monkeypatch):
    module, _ = bootstrap
    real_plan = resources.plan_resources
    hardware = resources.HardwareMetrics(host_total_bytes=16 * resources.GIB,
                                         host_available_bytes=5 * resources.GIB)
    plans = []

    def plan(*args, **kwargs):
        result = real_plan(*args, hardware=hardware, **kwargs)
        plans.append(result)
        return result

    monkeypatch.setattr(resources, "plan_resources", plan)

    def execute(command, **kwargs):
        monkeypatch.setattr(sys, "argv", ["-c", *command[3:]])
        exec(compile(command[2], "setup-resource-check", "exec"), {})

    monkeypatch.setattr(module, "run", execute)
    parameter_bytes = module.SETUP_PARAMETER_COUNTS["DecoderTCR-ESMC_300M"] * 4
    without_storage = real_plan(parameter_bytes, "cpu", batch_size=2, sequence_length=64,
                                hardware=hardware)
    assert not without_storage.blocked
    with pytest.raises(SystemExit, match="stopped before model installation/conversion"):
        module.check_setup_resources(Path(sys.executable), {}, "DecoderTCR-ESMC_300M", "apple")
    cpu, apple = plans
    assert cpu.blocked and cpu.estimated_peak_bytes > without_storage.estimated_peak_bytes
    assert cpu.parameter_bytes == apple.parameter_bytes == parameter_bytes
    assert cpu.checkpoint_storage_bytes == module.MODEL_SIZE
    assert not apple.blocked
    assert apple.estimated_peak_bytes == real_plan(parameter_bytes, "apple", "mlx", batch_size=2,
                                                  sequence_length=64, hardware=hardware).estimated_peak_bytes


@pytest.mark.parametrize("allow", [False, True])
def test_setup_resource_gate_honors_explicit_memory_risk_override(bootstrap, monkeypatch, allow):
    module, _ = bootstrap
    real_plan = resources.plan_resources
    hardware = resources.HardwareMetrics(host_total_bytes=32 * resources.GIB,
                                         host_available_bytes=24 * resources.GIB)
    monkeypatch.setattr(resources, "plan_resources", lambda *args, **kwargs:
                        real_plan(*args, hardware=hardware, **kwargs))

    def execute(command, **kwargs):
        monkeypatch.setattr(sys, "argv", ["-c", *command[3:]])
        exec(compile(command[2], "setup-resource-check", "exec"), {})

    monkeypatch.setattr(module, "run", execute)
    if allow:
        module.check_setup_resources(Path(sys.executable), {}, "DecoderTCR-ESMC_6B", "apple",
                                     allow_memory_risk=True)
    else:
        with pytest.raises(SystemExit, match="stopped before model installation/conversion"):
            module.check_setup_resources(Path(sys.executable), {}, "DecoderTCR-ESMC_6B", "apple")


def test_deep_apple_doctor_validates_6b_architecture_with_shared_contract(bootstrap, monkeypatch):
    module, _ = bootstrap
    commands = []
    monkeypatch.setattr(module, "run", lambda command, **kwargs: commands.append(command))
    module.probe(dict(python_executable="torch-python", device="apple", mlx_python="mlx-python",
                      checkpoint="bundle", model="DecoderTCR-ESMC_6B"), {}, deep=True)
    command = commands[-1]
    assert command[0] == "mlx-python"
    assert "validate_decoder_config(c,sys.argv[2])" in command[2]
    assert command[3:] == ["bundle", "DecoderTCR-ESMC_6B"]
