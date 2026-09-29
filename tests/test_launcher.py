"""Portable launcher/setup checks; no installation, network or pretrained models."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    # Loading copied launchers tests relocation and avoids touching real setup state.
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    modules = {}
    for name in ("tcr", "bootstrap"):
        source = root / (name + ".py")
        shutil.copyfile(PROJECT / source.name, source)
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["tcr"], modules["bootstrap"], root


@pytest.mark.parametrize("args,code,phrase", [
    (["--help"], 0, "First use"), (["setup", "--help"], 0, "--core-only"),
    (["doctor", "--help"], 0, "--deep"), (["doctor"], 1, "Core environment missing"),
    (["screen", "--help"], 2, "Setup is needed"),
])
def test_clean_launcher_uses_only_standard_library(launcher, args, code, phrase):
    _, _, root = launcher
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-S", str(root/"tcr.py"), *args], cwd=root.parent,
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == code, result.stderr
    assert phrase in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert not (root / ".tcr").exists()


def config_files(root):
    state = root/".tcr"
    state.mkdir()
    for name in ("runtime", "runtime-cpu", "runtime-apple", "runtime-gpu"):
        (state/(name+".json")).write_text("{}")
    return state


@pytest.mark.parametrize("device,expected", [
    ("cpu", "cpu"), ("apple", "apple"), ("gpu", "gpu"), ("cuda:2", "gpu"),
    ("GPU", "gpu"), ("MLX", "apple"), ("CUDA:0", "gpu"),
])
def test_device_selects_matching_saved_runtime(launcher, device, expected):
    module, _, root = launcher
    state = config_files(root)
    for flags in (["--device",device], ["--device="+device]):
        args=["pmhc-score","--panel","relative panel.csv",*flags]
        command=module.command_line(args)
        assert command[3:5] == ["--config",str(state/f"runtime-{expected}.json")]
        assert command[5:] == args


@pytest.mark.parametrize("flags", [["--config","relative.json"], ["--config=relative.json"]])
def test_explicit_config_is_preserved(launcher, flags):
    module, _, root=launcher
    config_files(root)
    args=[*flags,"pmhc-score","--device","cpu","--panel","relative.csv"]
    assert module.command_line(args)[3:] == args


def test_default_config_and_symlink_interpreter_preserve_relative_paths(launcher, monkeypatch):
    module, _, root=launcher
    state=config_files(root)
    interpreter=module.python_in(state/"envs/core")
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    captured=[]
    monkeypatch.setenv("PYTHONPATH","do-not-inherit")
    monkeypatch.setenv("PYTHONHOME","do-not-inherit")
    monkeypatch.setattr(module.subprocess,"call",lambda command,**kwargs: captured.append((command,kwargs)) or 0)
    args=["screen","--input","../relative.csv","--out","./my report"]
    assert module.main(args) == 0
    command,kwargs=captured[0]
    assert command[0] == str(interpreter) and command[0] != str(interpreter.resolve())
    assert command[3:5] == ["--config",str(state/"runtime.json")]
    assert command[5:] == args
    assert "cwd" not in kwargs
    assert "PYTHONPATH" not in kwargs["env"] and "PYTHONHOME" not in kwargs["env"]


def settings_fixture(root):
    # Torch CPU/CUDA weights are registry-resolved (checkpoint=None); an Apple bundle
    # directory still exercises checkpoint-path resolution and interpreter symlinks.
    directory=root/"settings"
    directory.mkdir()
    (directory/"decoder").mkdir()
    (directory/"bundle").mkdir()
    interpreter=directory/"runtime/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    mlx=directory/"mlx/bin/python"
    mlx.parent.mkdir(parents=True)
    mlx.symlink_to(sys.executable)
    values=dict(decoder_dir="decoder",python_executable="runtime/bin/python",checkpoint="bundle",
                mlx_python="mlx/bin/python",device="apple",model="esmc-300m",precision="float32")
    path=directory/"config.json"
    path.write_text(json.dumps(values))
    return path,values,interpreter


def validation_environment(bootstrap,root):
    # The temporary core interpreter uses the real copied package's schema; only
    # the validation subprocess runs, never the configured model interpreter.
    return dict(bootstrap.clean_env(root/".tcr"),PYTHONPATH=str(PROJECT/"src"))


def test_config_paths_resolve_relative_to_config_without_resolving_python_symlink(launcher):
    _,bootstrap,root=launcher
    path,_,interpreter=settings_fixture(root)
    values=bootstrap.read_settings(path,Path(sys.executable),validation_environment(bootstrap,root))
    assert values["python_executable"] == str(interpreter)
    assert values["python_executable"] != str(interpreter.resolve())
    assert values["checkpoint"] == str(path.parent/"bundle")
    assert values["model"] == "DecoderTCR-ESMC_300M"


@pytest.mark.parametrize("changes", [
    {"device":"cpu","precision":"float16"}, {"device":"invented"}, {"model":"invented"},
    {"mlx_python":None}, {"checkpoint":None}, {"unknown":True}, {"batch_size":True},
])
def test_reused_settings_validate_schema_and_backend_combinations(launcher,changes):
    _,bootstrap,root=launcher
    path,values,_=settings_fixture(root)
    path.write_text(json.dumps({**values,**changes}))
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        bootstrap.read_settings(path,Path(sys.executable),validation_environment(bootstrap,root))


def test_reused_settings_reject_duplicate_keys(launcher):
    _,bootstrap,root=launcher
    path,_,_=settings_fixture(root)
    path.write_text('{"precision":"float16",'+path.read_text()[1:])
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        bootstrap.read_settings(path,Path(sys.executable),validation_environment(bootstrap,root))


@pytest.mark.parametrize("explicit", [False, True])
def test_doctor_never_installs_or_rewrites_configuration(launcher,monkeypatch,capsys,explicit):
    module,bootstrap,root=launcher
    core=module.python_in(root/".tcr/envs/core")
    core.parent.mkdir(parents=True)
    core.touch()
    cfg=root/("chosen.json" if explicit else ".tcr/runtime.json")
    cfg.write_text("unchanged config")
    seen=[]
    monkeypatch.setattr(bootstrap,"run",lambda *a,**k:None)
    def read_settings(path,*args):
        assert path == cfg
        return dict(decoder_dir=str(root/"decoder"),model="DecoderTCR-ESMC_300M",
                    device="cpu",precision="float32")
    monkeypatch.setattr(bootstrap,"read_settings",read_settings)
    monkeypatch.setattr(bootstrap,"probe",lambda *a,**k:seen.append(k["deep"]))
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:pytest.fail("doctor installed dependencies"))
    monkeypatch.setattr(bootstrap,"install_decoder",lambda *a:pytest.fail("doctor installed the model env"))
    flags=["doctor","--config",str(cfg)] if explicit else ["doctor"]
    assert bootstrap.main(flags,root=root) == 0
    assert seen == [False] and cfg.read_text() == "unchanged config"
    assert "Use doctor --deep" in capsys.readouterr().out
    assert not (root/".tcr/setup.lock").exists()


@pytest.mark.parametrize("exception",[RuntimeError,KeyboardInterrupt])
def test_setup_lock_cleanup_and_exclusion(launcher,exception):
    _,bootstrap,root=launcher
    state=root/".tcr"
    with pytest.raises(exception):
        with bootstrap.setup_lock(state):
            original=(state/"setup.lock").read_text()
            with pytest.raises(ValueError,match="Another setup"):
                with bootstrap.setup_lock(state):
                    pytest.fail("concurrent setup acquired lock")
            assert (state/"setup.lock").read_text() == original
            raise exception("interrupted")
    assert not (state/"setup.lock").exists()


@pytest.mark.parametrize("failure_phase", ["validation", "probe"])
def test_failed_reuse_probe_preserves_all_existing_configurations(launcher,monkeypatch,failure_phase):
    _,bootstrap,root=launcher
    state=config_files(root)
    originals={path:path.read_bytes() for path in state.glob("*.json")}
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv"])
    monkeypatch.setattr(bootstrap,"run",lambda *a,**k:None)
    values=iter([{"device":"cpu"},{"device":"apple"}])
    def read_settings(*args):
        value=next(values)
        if failure_phase == "validation" and value["device"] == "apple":
            raise ValueError("synthetic invalid configuration")
        return value
    monkeypatch.setattr(bootstrap,"read_settings",read_settings)
    def probe(value,*args,**kwargs):
        if value["device"] == "apple":
            raise ValueError("synthetic bundle verification failure")
    monkeypatch.setattr(bootstrap,"probe",probe)
    assert bootstrap.main(["setup","--reuse-config","one.json","--reuse-config","two.json"],root=root) == 1
    assert {path:path.read_bytes() for path in state.glob("*.json")} == originals
    assert not (state/"setup.lock").exists()


def test_core_only_setup_never_installs_the_model_environment(launcher,monkeypatch):
    _,bootstrap,root=launcher
    calls=[]
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv"])
    monkeypatch.setattr(bootstrap,"run",lambda cmd,**k:calls.append([str(x) for x in cmd]))
    monkeypatch.setattr(bootstrap,"install_decoder",lambda *a:pytest.fail("model env install during core-only setup"))
    monkeypatch.setattr(bootstrap,"configure_registry",lambda *a:pytest.fail("registry configured during core-only setup"))
    monkeypatch.setattr(bootstrap,"probe",lambda *a,**k:pytest.fail("model probe during core-only setup"))
    assert bootstrap.main(["setup","--core-only"],root=root) == 0
    assert any("pip" in command for command in calls)
    # Core-only installs only the core env with pinned deps; no registry configure call.
    assert not any("configure" in command for command in calls)
    assert not any("decodertcr_internal" in " ".join(command) for command in calls if "install" not in command)
    assert not (root/".tcr/runtime.json").exists()


@pytest.mark.parametrize("selected,canonical",[(None,"DecoderTCR-ESMC_300M"),
    ("esmc-600m","DecoderTCR-ESMC_600M"),("esmc-6b","DecoderTCR-ESMC_6B")])
def test_cpu_setup_installs_model_env_and_configures_registry_without_download(launcher,monkeypatch,selected,canonical):
    # Full CPU setup installs decodertcr_internal into an isolated model env, connects
    # it to the shared registry, and saves registry-resolved settings (checkpoint=None).
    _,bootstrap,root=launcher
    registry=root/"shared registry"
    registry.mkdir()
    calls=[]
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv","--no-config"])
    monkeypatch.setattr(bootstrap,"run",lambda cmd,**k:calls.append([str(x) for x in cmd]))
    installs=[]
    def install_decoder(uv,env_dir,env,source,device="cpu"):
        installs.append((Path(env_dir),source,device))
        python=bootstrap.python_in(env_dir)
        python.parent.mkdir(parents=True,exist_ok=True)
        python.touch()
        return python
    monkeypatch.setattr(bootstrap,"install_decoder",install_decoder)
    configured=[]
    monkeypatch.setattr(bootstrap,"configure_registry",lambda python,reg,env:configured.append((Path(python),reg)))
    monkeypatch.setattr(bootstrap,"ensure_germlines",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"check_setup_resources",lambda *a,**k:None)
    probes=[]
    monkeypatch.setattr(bootstrap,"probe",lambda settings,env,**k:probes.append((settings,k.get("deep"))))
    args=["setup","--device","cpu","--registry",str(registry)]
    if selected:
        args += ["--model",selected]
    assert bootstrap.main(args,root=root) == 0
    model_env=root/".tcr/envs/model"
    # The model env was installed from the pinned package source at cpu backend.
    assert installs == [(model_env,bootstrap.DECODER_PACKAGE,"cpu")]
    # The registry was configured against the freshly installed model interpreter.
    assert configured == [(bootstrap.python_in(model_env),str(registry))]
    # No download function exists; nothing is fetched over the network.
    assert not hasattr(bootstrap,"download")
    saved=json.loads((root/".tcr/runtime.json").read_text())
    assert saved["model"] == canonical and saved["checkpoint"] is None
    assert saved["device"] == "cpu" and saved["precision"] == "float32"
    assert saved["decoder_dir"] == str(model_env)
    cpu=json.loads((root/".tcr/runtime-cpu.json").read_text())
    assert cpu == saved and cpu["checkpoint"] is None
    assert probes and probes[-1][1] is True


def test_full_setup_requires_a_registry(launcher,monkeypatch,capsys):
    _,bootstrap,root=launcher
    monkeypatch.delenv("DECODERTCR_REGISTRY",raising=False)
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:pytest.fail("installation before registry validation"))
    assert bootstrap.main(["setup","--device","cpu"],root=root) == 1
    assert "registry" in capsys.readouterr().err.lower() and not (root/".tcr").exists()


def test_huggingface_setup_installs_hf_hub_and_skips_registry(launcher,monkeypatch):
    # A HuggingFace CPU setup needs no --registry: it installs huggingface_hub,
    # never calls configure_registry, and saves weight_source=huggingface settings.
    _,bootstrap,root=launcher
    monkeypatch.delenv("DECODERTCR_REGISTRY",raising=False)
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv","--no-config"])
    monkeypatch.setattr(bootstrap,"run",lambda cmd,**k:None)
    def install_decoder(uv,env_dir,env,source,device="cpu"):
        python=bootstrap.python_in(env_dir)
        python.parent.mkdir(parents=True,exist_ok=True)
        python.touch()
        return python
    monkeypatch.setattr(bootstrap,"install_decoder",install_decoder)
    hf_installs=[]
    monkeypatch.setattr(bootstrap,"ensure_huggingface",lambda uv,python,env:hf_installs.append(Path(python)))
    monkeypatch.setattr(bootstrap,"configure_registry",lambda *a,**k:pytest.fail("registry must not be configured for a HuggingFace source"))
    monkeypatch.setattr(bootstrap,"ensure_germlines",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"check_setup_resources",lambda *a,**k:None)
    probes=[]
    monkeypatch.setattr(bootstrap,"probe",lambda settings,env,**k:probes.append(settings))
    assert bootstrap.main(["setup","--device","cpu","--weight-source","huggingface"],root=root) == 0
    model_env=root/".tcr/envs/model"
    assert hf_installs == [bootstrap.python_in(model_env)]
    saved=json.loads((root/".tcr/runtime.json").read_text())
    assert saved["weight_source"] == "huggingface"
    assert saved["hf_repo"] == bootstrap.MODEL_HF_REPOS["DecoderTCR-ESMC_300M"]
    assert saved["checkpoint"] is None and saved["device"] == "cpu"
    assert probes and probes[-1]["weight_source"] == "huggingface"


def test_huggingface_setup_rejects_apple(launcher,monkeypatch,capsys):
    _,bootstrap,root=launcher
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:pytest.fail("no install before the apple/HuggingFace guard"))
    assert bootstrap.main(["setup","--device","apple","--weight-source","huggingface"],root=root) == 1
    assert "huggingface" in capsys.readouterr().err.lower() and not (root/".tcr").exists()


def test_atomic_config_write_keeps_old_config_when_replace_fails(launcher,monkeypatch):
    _,bootstrap,root=launcher
    path=root/"runtime.json"
    path.write_text("old")
    def fail(*args):
        raise OSError("synthetic replace failure")
    monkeypatch.setattr(Path,"replace",fail)
    with pytest.raises(OSError):
        bootstrap.atomic_json(path,{"new":True})
    assert path.read_text() == "old" and not path.with_suffix(".json.tmp").exists()


def test_cpu_reuse_keeps_registry_resolved_torch_settings_without_local_checkpoint(launcher):
    # Torch weights are registry-resolved by model ID; a CPU config has no local .ckpt.
    _,bootstrap,root=launcher
    path,values,_=settings_fixture(root)
    values.update(device="cpu",checkpoint=None,mlx_python=None)
    path.write_text(json.dumps(values))
    checked=bootstrap.read_settings(path,Path(sys.executable),validation_environment(bootstrap,root))
    assert checked["checkpoint"] is None
    assert checked["device"] == "cpu" and checked["model"] == "DecoderTCR-ESMC_300M"


@pytest.mark.parametrize("device,filename",[("cpu","cpu"),("apple","apple"),("cuda:2","gpu")])
def test_successful_reuse_saves_device_specific_and_default_configuration(launcher,monkeypatch,device,filename):
    _,bootstrap,root=launcher
    value={"device":device,"precision":"float32"}
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv"])
    monkeypatch.setattr(bootstrap,"run",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"read_settings",lambda *a:value)
    monkeypatch.setattr(bootstrap,"probe",lambda *a,**k:None)
    assert bootstrap.main(["setup","--reuse-config","chosen.json"],root=root) == 0
    state=root/".tcr"
    assert json.loads((state/f"runtime-{filename}.json").read_text()) == value
    assert json.loads((state/"runtime.json").read_text()) == value
    assert not (state/"setup.lock").exists()


@pytest.mark.parametrize("available,count,device,passes",[(False,2,"cuda:0",False),
    (True,1,"cuda:1",False),(True,1,"cuda",True),(True,2,"cuda:1",True)])
def test_deep_doctor_checks_cuda_availability_and_selected_index(launcher,monkeypatch,available,count,device,passes):
    from types import ModuleType, SimpleNamespace

    _,bootstrap,_=launcher
    fake=ModuleType("torch")
    fake.__version__="synthetic"
    fake.device=lambda name:SimpleNamespace(index=int(name.partition(":")[2]) if ":" in name else None)
    fake.cuda=SimpleNamespace(is_available=lambda:available,device_count=lambda:count)
    monkeypatch.setitem(sys.modules,"torch",fake)
    # Only the CUDA availability/index probe is exercised here; the germline, registry
    # reachability and registry-release probes are treated as already satisfied.
    def run(command,**kwargs):
        if "torch.cuda.is_available" in command[2]:
            monkeypatch.setattr(sys,"argv",["-c",*command[3:]])
            exec(command[2],{})  # Execute the real CUDA probe against a fake Torch runtime.
    monkeypatch.setattr(bootstrap,"run",run)
    settings=dict(device=device,python_executable="unused",model="DecoderTCR-ESMC_300M",checkpoint="unused")
    if passes:
        bootstrap.probe(settings,{},deep=True)
    else:
        with pytest.raises(AssertionError,match="CUDA.*unavailable"):
            bootstrap.probe(settings,{},deep=True)


@pytest.mark.parametrize("system,device",[("Linux","cpu"),("Linux","gpu"),("Darwin","cpu")])
def test_decoder_install_uses_cpu_torch_on_linux_and_pinned_source(launcher,monkeypatch,system,device):
    # install_decoder creates an isolated Python 3.12 env and pip-installs the decoder
    # source; a Linux CPU install pins the CPU Torch wheel to avoid multi-GB NVIDIA wheels.
    module,bootstrap,root=launcher
    env_dir=root/"envs/model"
    source="decodertcr-internal==0.5.0"
    calls=[]
    monkeypatch.setattr(bootstrap.platform,"system",lambda:system)
    monkeypatch.setattr(bootstrap,"run",lambda command,**kwargs:calls.append([str(x) for x in command]))
    returned=bootstrap.install_decoder(["uv","--no-config"],env_dir,{},source,device=device)
    assert returned == module.python_in(env_dir)
    # Missing interpreter forces a fresh venv creation before installation.
    created=next(command for command in calls if "venv" in command)
    assert created[created.index("--python")+1] == "3.12" and str(env_dir) in created
    installed=next(command for command in calls if "install" in command)
    assert installed[installed.index("--python")+1] == str(module.python_in(env_dir))
    assert source in installed
    if system == "Linux" and device == "cpu":
        assert installed[installed.index("--torch-backend")+1] == "cpu"
        assert "torch==2.10.0" in installed
    else:
        assert "--torch-backend" not in installed
        assert not any("nvidia" in value.lower() for command in calls for value in command)


@pytest.mark.parametrize("passes",[True,False])
def test_linux_gpu_setup_publishes_only_after_successful_cuda_probe(launcher,monkeypatch,passes):
    _,bootstrap,root=launcher
    registry=root/"registry"
    registry.mkdir()
    state=config_files(root)
    original={path:path.read_bytes() for path in state.glob("*.json")}
    devices=[]
    probes=[]
    monkeypatch.setattr(bootstrap.platform,"system",lambda:"Linux")
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv"])
    monkeypatch.setattr(bootstrap,"run",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"install_decoder",
                        lambda uv,env_dir,env,source,device:devices.append(device) or bootstrap.python_in(env_dir))
    monkeypatch.setattr(bootstrap,"configure_registry",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"ensure_germlines",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"check_setup_resources",lambda *a,**k:None)
    def probe(settings,env,*,deep):
        assert {path:path.read_bytes() for path in state.glob("*.json")} == original
        assert deep and settings["device"] == "cuda" and settings["precision"] == "float32"
        probes.append(settings)
        if not passes:
            raise subprocess.CalledProcessError(1,["cuda-probe"],stderr="CUDA device unavailable")
    monkeypatch.setattr(bootstrap,"probe",probe)
    assert bootstrap.main(["setup","--device","gpu","--registry",str(registry)],root=root) == (0 if passes else 1)
    assert devices == ["gpu"] and len(probes) == 1
    assert not (state/"setup.lock").exists()
    if passes:
        cpu=json.loads((state/"runtime-cpu.json").read_text())
        gpu=json.loads((state/"runtime-gpu.json").read_text())
        assert gpu == probes[0] and cpu == dict(gpu,device="cpu")
        assert json.loads((state/"runtime.json").read_text()) == gpu
        assert (state/"runtime-apple.json").read_bytes() == original[state/"runtime-apple.json"]
    else:
        assert {path:path.read_bytes() for path in state.glob("*.json")} == original


@pytest.mark.parametrize("system",["Darwin","Windows"])
def test_non_linux_automatic_gpu_setup_rejects_before_state_or_install(launcher,monkeypatch,capsys,system):
    _,bootstrap,root=launcher
    monkeypatch.setattr(bootstrap.platform,"system",lambda:system)
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:pytest.fail("installation before platform validation"))
    assert bootstrap.main(["setup","--device","gpu"],root=root) == 1
    assert "targets Linux" in capsys.readouterr().err and not (root/".tcr").exists()


@pytest.mark.parametrize("model",["DecoderTCR-ESMC_300M","DecoderTCR-ESMC_600M"])
@pytest.mark.parametrize("changes,metal,passes",[({},True,True),({},False,False),
    ({"source_variant":"biohub-esmc-published-v1"},True,False),
    ({"tokenizer_variant":"biohub-esmc"},True,False),({"hidden_size":1536},True,False),
    ({"num_hidden_layers":31},True,False),({"num_attention_heads":16},True,False),
    ({"vocab_size":33},True,False),({"dtype":"float16"},True,False)])
def test_deep_apple_probe_checks_decoder_bundle_identity(launcher,monkeypatch,model,changes,metal,passes):
    from types import ModuleType, SimpleNamespace

    _,bootstrap,_=launcher
    spec=importlib.util.spec_from_file_location("esmc_mlx.config",PROJECT/"esmc-mlx/esmc_mlx/config.py")
    config_module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,"esmc_mlx.config",config_module)
    spec.loader.exec_module(config_module)
    factory=config_module.config_300m if model.endswith("300M") else config_module.config_600m
    config=factory("decodertcr-lightning-v03").model_copy(update=changes)
    mlx=ModuleType("mlx")
    mlx.core=ModuleType("mlx.core")
    mlx.core.metal=SimpleNamespace(is_available=lambda:metal)
    esmc=ModuleType("esmc_mlx")
    esmc.config=config_module
    esmc.weights=ModuleType("esmc_mlx.weights")
    esmc.weights.verify_bundle=lambda path:(config,{"checkpoint_identity":"synthetic"},None)
    for name,module in (("mlx",mlx),("mlx.core",mlx.core),("esmc_mlx",esmc),("esmc_mlx.weights",esmc.weights)):
        monkeypatch.setitem(sys.modules,name,module)
    def run(command,**kwargs):
        if command[0] == "mlx-test":
            monkeypatch.setattr(sys,"argv",["-c",*command[3:]])
            exec(command[2],{})
    monkeypatch.setattr(bootstrap,"run",run)
    settings=dict(device="apple",python_executable="decoder-test",mlx_python="mlx-test",checkpoint="bundle",
                  model=model)
    if passes:
        bootstrap.probe(settings,{},deep=True)
    else:
        with pytest.raises((AssertionError,ValueError),match="Metal|not a DecoderTCR variant|architecture/tokenizer"):
            bootstrap.probe(settings,{},deep=True)


def test_deep_cpu_probe_resolves_the_registry_release_identity(launcher,monkeypatch):
    # The CPU/CUDA deep probe confirms the exact registry release resolves via the
    # decodertcr_internal registry by model ID; no local checkpoint file is inspected.
    from types import ModuleType

    _,bootstrap,_=launcher
    torch=ModuleType("torch")
    torch.__version__="synthetic"
    monkeypatch.setitem(sys.modules,"torch",torch)
    resolved=[]
    dt=ModuleType("decodertcr_internal")
    dt.__version__="synthetic"
    dt.models=lambda:["release"]
    dt.info=lambda model_id:(resolved.append(model_id) or {
        "model_id":model_id,
        "member":{"sequence_convention":"esmc","weights":{"sha256":"f"*64}}})
    dt.normalize_gene=lambda gene:gene
    monkeypatch.setitem(sys.modules,"decodertcr_internal",dt)
    def run(command,**kwargs):
        if "info=dt.info" in command[2]:
            monkeypatch.setattr(sys,"argv",["-c",*command[3:]])
            exec(command[2],{})
    monkeypatch.setattr(bootstrap,"run",run)
    settings=dict(device="cpu",python_executable="unused",model="DecoderTCR-ESMC_300M",checkpoint=None)
    bootstrap.probe(settings,{},deep=True)
    assert resolved == [bootstrap.MODEL_IDS["DecoderTCR-ESMC_300M"]]
