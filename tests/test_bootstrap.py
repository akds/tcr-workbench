"""Setup dispatch and resource gates, without downloads or model allocation."""
import importlib.util
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
    """Record setup subprocess/install steps; model weights are never downloaded.

    There is no download function anymore: weights are resolved from the shared
    registry on first use, so a fresh install only creates the model env, connects
    it to the registry, and probes it.
    """
    calls = []
    monkeypatch.setattr(module, "ensure_uv", lambda *args: ["uv"])
    monkeypatch.setattr(module, "run", lambda command, **kwargs: calls.append(list(command)))
    monkeypatch.setattr(module, "install_decoder",
                        lambda uv, env_dir, env, source, device="cpu":
                        calls.append(["install_decoder", str(env_dir), source, device])
                        or module.python_in(env_dir))
    monkeypatch.setattr(module, "configure_registry",
                        lambda python, registry, env: calls.append(["configure_registry", str(python), registry]))
    monkeypatch.setattr(module, "ensure_germlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "probe", lambda settings, *args, **kwargs: calls.append(["probe", settings]))
    return calls


@pytest.mark.parametrize("device", ["cpu", "apple"])
def test_full_setup_installs_model_env_and_connects_registry_by_backend(bootstrap, monkeypatch, device):
    module, root = bootstrap
    registry = root / "shared registry"
    registry.mkdir()
    calls = fake_runtime(module, root, monkeypatch)
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs:
                        calls.append(["resources", args[2], args[3], kwargs["allow_memory_risk"]]))
    monkeypatch.setattr(module, "prepare_setup_model", lambda settings, *args:
                        dict(settings, checkpoint=str(root / ".tcr/prepared/bundle")))
    flags = ["setup", "--device", device, "--model", "esmc-6b", "--registry", str(registry),
             "--allow-memory-risk"]
    assert module.main(flags, root=root) == 0
    # The memory gate runs before any model environment is installed.
    guard = ["resources", "DecoderTCR-ESMC_6B", device, True]
    install = ["install_decoder", str(root / ".tcr/envs/model"), module.DECODER_PACKAGE,
               "cpu" if device == "apple" else device]
    assert calls.index(guard) < calls.index(install)
    # The freshly installed model interpreter is pointed at the shared registry.
    model_python = str(module.python_in(root / ".tcr/envs/model"))
    assert ["configure_registry", model_python, str(registry)] in calls
    # Torch CPU settings carry no local checkpoint; weights are registry-resolved.
    cpu = json.loads((root / ".tcr/runtime-cpu.json").read_text())
    assert (cpu["model"], cpu["device"], cpu["checkpoint"]) == ("DecoderTCR-ESMC_6B", "cpu", None)
    assert cpu["decoder_dir"] == str(root / ".tcr/envs/model")
    selected = json.loads((root / ".tcr/runtime.json").read_text())
    assert selected["device"] == device and selected["precision"] == "float32"
    if device == "apple":
        # Apple converts and parity-checks a local MLX bundle from the registry artifact.
        assert Path(selected["checkpoint"]).name == "bundle"
        assert (root / ".tcr/runtime-apple.json").exists()
    else:
        assert selected["checkpoint"] is None


def test_full_setup_requires_a_registry_before_installation(bootstrap, monkeypatch, capsys):
    module, root = bootstrap
    monkeypatch.delenv("DECODERTCR_REGISTRY", raising=False)
    monkeypatch.setattr(module, "ensure_uv", lambda *args: pytest.fail("installation before registry validation"))
    assert module.main(["setup", "--model", "esmc-6b", "--device", "cpu"], root=root) == 1
    assert "registry" in capsys.readouterr().err.lower()
    assert not (root / ".tcr").exists()


def test_registry_from_environment_is_accepted(bootstrap, monkeypatch):
    module, root = bootstrap
    registry = root / "env registry"
    registry.mkdir()
    calls = fake_runtime(module, root, monkeypatch)
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs: None)
    monkeypatch.setenv("DECODERTCR_REGISTRY", str(registry))
    assert module.main(["setup", "--device", "cpu"], root=root) == 0
    assert any(call[0] == "configure_registry" and call[2] == str(registry) for call in calls)


def test_blocked_setup_preserves_configuration_and_stops_before_model_install(bootstrap, monkeypatch):
    module, root = bootstrap
    registry = root / "registry"
    registry.mkdir()
    calls = fake_runtime(module, root, monkeypatch)
    config = root / ".tcr/runtime.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("previous configuration")

    def blocked(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["resource-check"])

    monkeypatch.setattr(module, "check_setup_resources", blocked)
    assert module.main(["setup", "--model", "esmc-6b", "--registry", str(registry)], root=root) == 1
    assert not any(call[0] == "install_decoder" for call in calls)
    assert not any(call[0] == "configure_registry" for call in calls)
    assert config.read_text() == "previous configuration"
    assert not (root / ".tcr/setup.lock").exists()


def test_core_only_6b_selection_does_not_install_model_or_check_model_memory(bootstrap, monkeypatch):
    module, root = bootstrap
    calls = fake_runtime(module, root, monkeypatch)
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs: pytest.fail("model resource check"))
    assert module.main(["setup", "--core-only", "--model", "esmc-6b"], root=root) == 0
    assert not any(call[0] in ("install_decoder", "configure_registry") for call in calls)
    assert not list((root / ".tcr").glob("runtime*.json"))


def test_removed_checkpoint_options_are_rejected_by_the_parser(bootstrap):
    # Downloaded/local checkpoints were replaced by the shared registry; the old
    # checkpoint flags no longer exist and argparse rejects them (exit code 2).
    module, root = bootstrap
    for flags in (["--checkpoint", "weights.ckpt"], ["--checkpoint-url", "https://example.invalid/w.ckpt"],
                  ["--expected-sha256", "a" * 64]):
        with pytest.raises(SystemExit) as excinfo:
            module.main(["setup", *flags], root=root)
        assert excinfo.value.code == 2
    assert not (root / ".tcr").exists()


def test_full_setup_failure_preserves_saved_defaults(bootstrap, monkeypatch):
    module, root = bootstrap
    registry = root / "registry"
    registry.mkdir()
    fake_runtime(module, root, monkeypatch)
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "check_setup_resources", lambda *args, **kwargs: None)
    (root / ".tcr").mkdir(parents=True, exist_ok=True)
    for name in ("runtime.json", "runtime-cpu.json", "runtime-apple.json", "runtime-gpu.json"):
        (root / ".tcr" / name).write_text("previous configuration")

    def prepare(*args):
        raise ValueError("Apple bundle architecture/tokenizer does not match")

    monkeypatch.setattr(module, "prepare_setup_model", prepare)
    assert module.main(["setup", "--device", "apple", "--model", "esmc-6b", "--registry", str(registry)],
                       root=root) == 1
    assert all(path.read_text() == "previous configuration" for path in (root / ".tcr").glob("runtime*.json"))
    assert not (root / ".tcr/setup.lock").exists()


def test_setup_preparation_uses_existing_model_checks(bootstrap, monkeypatch):
    from tcr_workbench import model_preparation

    module, root = bootstrap
    seen = []
    settings = {"checkpoint": None, "model": "DecoderTCR-ESMC_600M", "device": "apple"}

    def prepare(options, prepared, **kwargs):
        seen.append((options, prepared, kwargs))
        return dict(options, checkpoint="prepared/bundle"), {}

    monkeypatch.setattr(model_preparation, "prepare_model", prepare)

    def execute(command, **kwargs):
        monkeypatch.setattr(sys, "argv", ["-c", *[str(arg) for arg in command[3:]]])
        exec(compile(command[2], "setup-preparation", "exec"), {})

    monkeypatch.setattr(module, "run", execute)
    prepared = module.prepare_setup_model(settings, root, Path(sys.executable), {}, True)
    assert prepared == dict(settings, checkpoint="prepared/bundle")
    assert seen == [(settings, str(root / "prepared"),
                     dict(allow_memory_risk=True, sequence_length=64))]
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
    parameter_bytes = module.SETUP_PARAMETER_COUNTS["DecoderTCR-ESMC_6B"] * 4
    assert parameter_bytes == 25_408_020_736
    assert all(item[0] == parameter_bytes for item in seen)
    for _, _, backend, kwargs in seen:
        expected_kwargs = dict(batch_size=2, sequence_length=64, allow_memory_risk=allow)
        if backend == "torch":
            # The fp32 registry artifact bytes are estimated as parameters * 4.
            expected_kwargs["checkpoint_storage_bytes"] = parameter_bytes
        assert kwargs == expected_kwargs


def test_300m_apple_setup_budgets_cpu_and_apple_phases_separately(bootstrap, monkeypatch):
    # Apple setup plans the CPU reference/conversion phase and the Apple inference phase
    # independently. The heavier CPU reference phase can block on capacity that still
    # fits the Apple shared-memory inference phase; the gate stops before installation.
    module, _ = bootstrap
    real_plan = resources.plan_resources
    hardware = resources.HardwareMetrics(host_total_bytes=16 * resources.GIB,
                                         host_available_bytes=3221225472)  # 3 GiB
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
    with pytest.raises(SystemExit, match="stopped before model installation/conversion"):
        module.check_setup_resources(Path(sys.executable), {}, "DecoderTCR-ESMC_300M", "apple")
    cpu, apple = plans
    assert cpu.blocked and not apple.blocked
    assert cpu.parameter_bytes == apple.parameter_bytes == parameter_bytes
    # The CPU/torch phase reserves the fp32 registry artifact bytes (parameters * 4).
    assert cpu.backend == "torch" and cpu.checkpoint_storage_bytes == parameter_bytes
    assert apple.backend == "mlx" and cpu.estimated_peak_bytes > apple.estimated_peak_bytes


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
