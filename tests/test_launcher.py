"""Portable launcher/setup checks; no installation, network or pretrained models."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

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
    directory=root/"settings"
    directory.mkdir()
    (directory/"decoder").mkdir()
    (directory/"weights.ckpt").write_bytes(b"synthetic")
    interpreter=directory/"runtime/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    values=dict(decoder_dir="decoder",python_executable="runtime/bin/python",checkpoint="weights.ckpt",
                device="cpu",model="esmc-300m",precision="float32")
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
    assert values["checkpoint"] == str(path.parent/"weights.ckpt")
    assert values["model"] == "DecoderTCR-ESMC_300M"


@pytest.mark.parametrize("changes", [
    {"precision":"float16"}, {"device":"invented"}, {"model":"invented"},
    {"device":"apple","mlx_python":None}, {"unknown":True}, {"batch_size":True},
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
    source=root/"decoder/src/DecoderTCR/utils/predict_from_genes.py"
    source.parent.mkdir(parents=True)
    source.touch()
    seen=[]
    monkeypatch.setattr(bootstrap,"run",lambda *a,**k:None)
    def read_settings(path,*args):
        assert path == cfg
        return dict(decoder_dir=str(root/"decoder"),model="DecoderTCR-ESMC_300M",
                    device="cpu",precision="float32")
    monkeypatch.setattr(bootstrap,"read_settings",read_settings)
    monkeypatch.setattr(bootstrap,"probe",lambda *a,**k:seen.append(k["deep"]))
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:pytest.fail("doctor installed dependencies"))
    monkeypatch.setattr(bootstrap,"download",lambda *a:pytest.fail("doctor downloaded data"))
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


def test_core_only_setup_never_touches_model_downloads(launcher,monkeypatch):
    _,bootstrap,root=launcher
    calls=[]
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv"])
    monkeypatch.setattr(bootstrap,"run",lambda cmd,**k:calls.append([str(x) for x in cmd]))
    monkeypatch.setattr(bootstrap,"download",lambda *a:pytest.fail("model download during core-only setup"))
    monkeypatch.setattr(bootstrap,"probe",lambda *a,**k:pytest.fail("model probe during core-only setup"))
    assert bootstrap.main(["setup","--core-only"],root=root) == 0
    assert any("pip" in command for command in calls)
    assert not (root/".tcr/runtime.json").exists()


@pytest.mark.parametrize("selected",[None,"esmc-600m","esmc-6b"])
def test_setup_model_selection_uses_immutable_checkpoint_identity(launcher,monkeypatch,selected):
    _,bootstrap,root=launcher
    decoder=root/".tcr/DecoderTCR"
    decoder.mkdir(parents=True)
    (decoder/".workbench-revision").write_text(bootstrap.DECODER_REV)
    downloads=[]
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv"])
    monkeypatch.setattr(bootstrap,"run",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"install_decoder",lambda *a:None)
    monkeypatch.setattr(bootstrap,"probe",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"download",lambda *args:downloads.append(args))
    args=["setup","--device","cpu"]
    if selected:
        args += ["--model",selected]
    assert bootstrap.main(args,root=root) == 0
    model,filename,digest,size=bootstrap.SETUP_MODELS[selected or "esmc-300m"]
    checkpoint=decoder/"checkpoints"/filename
    assert downloads == [(f"https://huggingface.co/biohub/DecoderTCR/resolve/{bootstrap.MODEL_REV}/{filename}",
                          checkpoint,digest,size)]
    saved=json.loads((root/".tcr/runtime.json").read_text())
    assert saved["model"] == model and saved["checkpoint"] == str(checkpoint)
    assert saved["device"] == "cpu" and saved["precision"] == "float32"


def test_checksum_download_is_atomic_and_reuses_verified_artifact(launcher,monkeypatch):
    _,bootstrap,root=launcher
    body=b"synthetic checkpoint bytes"
    expected=hashlib.sha256(body).hexdigest()
    calls=[]
    def open_url(*args,**kwargs):
        calls.append(args)
        return io.BytesIO(body)
    monkeypatch.setattr(bootstrap.urllib.request,"urlopen",open_url)
    dest=root/"downloads/weights.ckpt"
    bootstrap.download("https://example.invalid/model",dest,expected,len(body))
    assert dest.read_bytes() == body and not dest.with_suffix(".ckpt.part").exists()
    bootstrap.download("https://example.invalid/model",dest,expected,len(body))
    assert len(calls) == 1
    with pytest.raises(ValueError,match="Checksum"):
        bootstrap.download("https://example.invalid/model",dest,"0"*64,len(body))
    assert dest.read_bytes() == body and len(calls) == 1


@pytest.mark.parametrize("error",["checksum","short","long","transfer"])
def test_failed_download_cleans_partial_file_without_publishing(launcher,monkeypatch,error):
    _,bootstrap,root=launcher
    body=b"synthetic"
    class Interrupted(io.BytesIO):
        def read(self,size=-1):
            if self.tell():
                raise OSError("transfer failed")
            return super().read(size)
    source=Interrupted(body) if error == "transfer" else io.BytesIO(body)
    monkeypatch.setattr(bootstrap.urllib.request,"urlopen",lambda *a,**k:source)
    expected="0"*64 if error == "checksum" else hashlib.sha256(body).hexdigest()
    size=len(body)+1 if error == "short" else len(body)-1 if error == "long" else len(body)
    dest=root/"downloads/weights.ckpt"
    with pytest.raises((ValueError,OSError)):
        bootstrap.download("https://example.invalid/model",dest,expected,size)
    assert not dest.exists() and not dest.with_suffix(".ckpt.part").exists()


def archive_fixture(path,entries):
    with tarfile.open(path,"w:gz") as out:
        for name,kind in entries:
            member=tarfile.TarInfo(name)
            if kind == "file":
                member.size=4
                out.addfile(member,io.BytesIO(b"test"))
            else:
                member.type={"symlink":tarfile.SYMTYPE,"hardlink":tarfile.LNKTYPE,"fifo":tarfile.FIFOTYPE}[kind]
                member.linkname="../outside"
                out.addfile(member)


@pytest.mark.parametrize("suffix,kind",[("../outside","file"),("sub/../../outside","file"),
    ("sub\\outside","file"),("linked","symlink"),("linked","hardlink"),("pipe","fifo"),
    ("ABSOLUTE","file"),("WRONG_ROOT","file")])
def test_source_archive_rejects_unsafe_paths_and_links(launcher,suffix,kind):
    _,bootstrap,root=launcher
    prefix="DecoderTCR-"+bootstrap.DECODER_REV
    name="/outside" if suffix == "ABSOLUTE" else "wrong/file" if suffix == "WRONG_ROOT" else prefix+"/"+suffix
    archive=root/"archive.tar.gz"
    archive_fixture(archive,[(name,kind)])
    with pytest.raises(ValueError,match="Unsafe"):
        bootstrap.unpack_source(archive,root/"managed")
    assert not (root/"managed").exists() and not (root/"outside").exists()
    assert not list(root.glob("source-*"))


def test_source_archive_extracts_pinned_tree_without_touching_outside_files(launcher):
    _,bootstrap,root=launcher
    prefix="DecoderTCR-"+bootstrap.DECODER_REV
    archive=root/"archive.tar.gz"
    archive_fixture(archive,[(prefix+"/src/DecoderTCR/utils/predict_from_genes.py","file")])
    target=root/"managed"
    bootstrap.unpack_source(archive,target)
    assert (target/".workbench-revision").read_text().strip() == bootstrap.DECODER_REV
    assert (target/"src/DecoderTCR/utils/predict_from_genes.py").read_text() == "test"
    assert not list(root.glob("source-*"))


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


def test_cpu_reuse_resolves_registry_checkpoint_when_not_explicit(launcher):
    _,bootstrap,root=launcher
    path,values,_=settings_fixture(root)
    values["checkpoint"]=None
    default=path.parent/"decoder/checkpoints/DecoderTCR-ESMC-V0.3/300M.ckpt"
    default.parent.mkdir(parents=True)
    default.write_bytes(b"synthetic default checkpoint")
    path.write_text(json.dumps(values))
    checked=bootstrap.read_settings(path,Path(sys.executable),validation_environment(bootstrap,root))
    assert checked["checkpoint"] == str(default)


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
    monkeypatch.setitem(sys.modules,"DecoderTCR",ModuleType("DecoderTCR"))
    monkeypatch.setattr(bootstrap,"sha256",lambda path:bootstrap.MODEL_SHA)
    def run(command,**kwargs):
        if command[2] != bootstrap.GERMLINE_PROBE:
            monkeypatch.setattr(sys,"argv",["-c",*command[3:]])
            exec(command[2],{})  # Execute the real small probe against a fake Torch runtime.
    monkeypatch.setattr(bootstrap,"run",run)
    settings=dict(device=device,python_executable="unused",model="DecoderTCR-ESMC_300M",checkpoint="unused")
    if passes:
        bootstrap.probe(settings,{},deep=True)
    else:
        with pytest.raises(AssertionError,match="CUDA.*unavailable"):
            bootstrap.probe(settings,{},deep=True)


@pytest.mark.parametrize("system,device",[("Linux","cpu"),("Linux","gpu"),("Darwin","cpu")])
def test_decoder_install_uses_cpu_torch_on_linux_and_frozen_pins(launcher,monkeypatch,system,device):
    module,bootstrap,root=launcher
    decoder,state=root/"decoder",root/".tcr"
    calls=[]
    monkeypatch.setattr(bootstrap.platform,"system",lambda:system)
    monkeypatch.setattr(bootstrap,"run",lambda command,**kwargs:calls.append([str(x) for x in command]))
    bootstrap.install_decoder(["uv","--no-config"],decoder,state,{},device=device)
    if system == "Linux" and device == "cpu":
        exported=next(command for command in calls if "export" in command)
        installed=next(command for command in calls if "install" in command)
        assert "--frozen" in exported and "--no-dev" in exported
        assert exported[exported.index("--prune")+1] == "torch"
        requirements=exported[exported.index("--output-file")+1]
        assert installed[installed.index("-r")+1] == requirements
        assert installed[installed.index("--torch-backend")+1] == "cpu"
        assert "torch==2.10.0" in installed
        assert installed[installed.index("--python")+1] == str(module.python_in(decoder/".venv"))
        assert not any("sync" in command for command in calls)
        assert not any("nvidia" in value.lower() or "cuda" in value.lower() for command in calls for value in command)
    else:
        assert len(calls) == 1 and "sync" in calls[0] and "--frozen" in calls[0]
        assert "--prune" not in calls[0] and "--torch-backend" not in calls[0]


@pytest.mark.parametrize("passes",[True,False])
def test_linux_gpu_setup_publishes_only_after_successful_cuda_probe(launcher,monkeypatch,passes):
    _,bootstrap,root=launcher
    state=config_files(root)
    original={path:path.read_bytes() for path in state.glob("*.json")}
    decoder=state/"DecoderTCR"
    decoder.mkdir()
    (decoder/".workbench-revision").write_text(bootstrap.DECODER_REV)
    devices=[]
    probes=[]
    monkeypatch.setattr(bootstrap.platform,"system",lambda:"Linux")
    monkeypatch.setattr(bootstrap,"ensure_uv",lambda *a:["uv"])
    monkeypatch.setattr(bootstrap,"run",lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,"install_decoder",lambda *args:devices.append(args[-1]))
    monkeypatch.setattr(bootstrap,"download",lambda *a:None)
    def probe(settings,env,*,deep):
        assert {path:path.read_bytes() for path in state.glob("*.json")} == original
        assert deep and settings["device"] == "cuda" and settings["precision"] == "float32"
        probes.append(settings)
        if not passes:
            raise subprocess.CalledProcessError(1,["cuda-probe"],stderr="CUDA device unavailable")
    monkeypatch.setattr(bootstrap,"probe",probe)
    assert bootstrap.main(["setup","--device","gpu"],root=root) == (0 if passes else 1)
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
        with pytest.raises((AssertionError,ValueError),match="Metal|architecture/tokenizer"):
            bootstrap.probe(settings,{},deep=True)


@pytest.mark.parametrize("width,passes",[(960,True),(1152,False)])
def test_custom_checkpoint_probe_checks_inventory_and_labels_unvalidated(launcher,monkeypatch,capsys,width,passes):
    from types import ModuleType, SimpleNamespace

    _,bootstrap,_=launcher
    torch=ModuleType("torch")
    torch.__version__="synthetic"
    state={"model.model.embed.weight":SimpleNamespace(shape=(64,width))}
    state.update({f"model.model.transformer.blocks.{i}.weight":None for i in range(30)})
    def load(path,**kwargs):
        assert kwargs == {"map_location":"cpu","weights_only":True,"mmap":True}
        return {"state_dict":state}
    torch.load=load
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setitem(sys.modules,"DecoderTCR",ModuleType("DecoderTCR"))
    monkeypatch.setattr(bootstrap,"sha256",lambda path:"a"*64)
    monkeypatch.setattr(sys,"path",list(sys.path))
    def run(command,**kwargs):
        if command[2] != bootstrap.GERMLINE_PROBE:
            monkeypatch.setattr(sys,"argv",["-c",*command[3:]])
            exec(command[2],{})
    monkeypatch.setattr(bootstrap,"run",run)
    settings=dict(device="cpu",python_executable="unused",checkpoint="custom",model="DecoderTCR-ESMC_300M")
    if passes:
        bootstrap.probe(settings,{},deep=True)
        output=capsys.readouterr().out
        assert "User-supplied checkpoint" in output and "has not been numerically validated" in output
        assert "a"*64 in output
    else:
        with pytest.raises(ValueError,match="architecture does not match"):
            bootstrap.probe(settings,{},deep=True)
