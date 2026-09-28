"""Explicit, local, resumable setup and read-only diagnostics (standard library only)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import venv

from tcr import python_in

UV_VERSION = "0.12.17"
# The internal package (Python 3.12) bundles the model code, stitchr germlines and
# reconstruction references; model weights are resolved from a shared registry, not
# downloaded here. Override the install source for a local checkout or private index.
DECODER_PACKAGE = "decodertcr-internal==0.5.0"
SETUP_ALIASES = {"esmc-300m": "DecoderTCR-ESMC_300M", "esmc-600m": "DecoderTCR-ESMC_600M",
                 "esmc-6b": "DecoderTCR-ESMC_6B"}
# Registry model IDs for the deep probe, which runs in the model environment where
# only decodertcr_internal (not tcr_workbench) is importable. Mirrors model_registry.
MODEL_IDS = {"DecoderTCR-ESMC_300M": "decodertcr@1.5.0:300M",
             "DecoderTCR-ESMC_600M": "decodertcr@1.5.0:600M",
             "DecoderTCR-ESMC_6B": "decodertcr@1.5.0:6B"}
# Parameter counts come from the exact source tensor inventories; the fp32 registry
# artifact bytes are estimated as parameters * 4 for the pre-install memory check.
SETUP_PARAMETER_COUNTS = {"DecoderTCR-ESMC_300M": 332997184,
                          "DecoderTCR-ESMC_600M": 575036992,
                          "DecoderTCR-ESMC_6B": 6352005184}
CORE_PINS = ["polars==1.36.1", "pyarrow==21.0.0", "pydantic==2.13.5",
             "rapidfuzz==3.13.0", "numpy==2.0.2", "h5py==3.14.0"]
# Reconstruction germlines/IMGT gene sets ship inside decodertcr_internal; a tiny
# normalization exercises the bundled resources without touching the registry.
GERMLINE_PROBE = (
    "import decodertcr_internal as dt; "
    "assert dt.normalize_gene('TRBV19'), 'decodertcr_internal germline resources missing'"
)


def germline_probe(species):
    # Mouse reconstruction uses the local mouse_stitch worker with Stitchr mouse
    # germlines; human resources ship inside decodertcr_internal (see GERMLINE_PROBE).
    if species not in ("human", "mouse"):
        raise ValueError("Germline species must be human or mouse")
    if species == "human":
        return GERMLINE_PROBE
    return (
        "from pathlib import Path; from Stitchr import stitchrfunctions as f; "
        "p=Path(f.data_dir)/'MOUSE'; "
        "assert all((p/n).is_file() and (p/n).stat().st_size > 0 for n in "
        "('TRA.fasta','TRB.fasta','J-region-motifs.tsv','C-region-motifs.tsv')), "
        "'Mouse Stitchr germlines missing'"
    )


def ensure_germlines(python, species, state, env):
    code = germline_probe(species)
    try:
        run([python, "-c", code], env=env, capture=True)
    except subprocess.CalledProcessError:
        with tempfile.TemporaryDirectory(dir=state, prefix="germlines-") as temporary:
            germ_env = dict(env, PATH=str(python.parent) + os.pathsep + env.get("PATH", ""))
            run([python, "-c", "from Stitchr.stitchrdl import main; main()", "-s", species],
                env=germ_env, cwd=temporary)
        run([python, "-c", code], env=env, capture=True)


def clean_env(state: Path) -> dict[str, str]:
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               UV_CACHE_DIR=str(state / "cache/uv"),
               UV_PYTHON_INSTALL_DIR=str(state / "python"),
               UV_NO_MODIFY_PATH="1", UV_NO_PROGRESS="1")
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    return env


def run(command, *, env, cwd=None, capture=False):
    return subprocess.run([str(x) for x in command], env=env, cwd=cwd, check=True,
                          text=True, capture_output=capture)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def setup_lock(state: Path):
    state.mkdir(parents=True, exist_ok=True)
    lock = state / "setup.lock"
    try:
        handle = lock.open("x")
    except FileExistsError:
        raise ValueError(f"Another setup may be running. If it stopped, remove {lock} and retry.")
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def read_settings(path: Path, core: Path, env):
    # Validation uses the very same strict Pydantic contract as normal inference.
    code = ("import json,sys; from pathlib import Path; "
            "from tcr_workbench.runtime_settings import RuntimeSettings,_unique_keys; "
            "from tcr_workbench.model_registry import normalize_device,resolve_model,validate_backend; "
            "v=RuntimeSettings.model_validate(json.loads(open(sys.argv[1]).read(), "
            "object_pairs_hook=_unique_keys)); v.device=normalize_device(v.device); "
            "v.model=resolve_model(v.model).name; "
            "validate_backend(v.model,v.device,v.checkpoint,v.mlx_python,v.precision); "
            "print(v.model_dump_json())")
    value = json.loads(run([core, "-c", code, path], env=env, capture=True).stdout)
    for key in ("decoder_dir", "python_executable", "checkpoint", "mlx_python"):
        if value.get(key):
            # absolute(), not resolve(): preserve venv interpreter symlinks.
            value[key] = str((path.resolve().parent / value[key]).absolute())
            if not Path(value[key]).exists():
                raise ValueError(f"Configured {key} does not exist: {value[key]}")
    if value["device"] == "apple":
        if not value.get("checkpoint") or not Path(value["checkpoint"]).is_dir():
            raise ValueError("Apple checkpoint must be a converted MLX bundle directory")
    # Torch weights are registry-resolved by model ID; no local checkpoint file.
    return value


def probe(settings, env, *, deep=False):
    python = settings["python_executable"]
    run([python, "-c", GERMLINE_PROBE], env=env, capture=True)
    # The registry must be reachable without downloading weights.
    run([python, "-c", "import decodertcr_internal as dt; "
         "assert dt.models(), 'no registry releases are available'; "
         "print('decodertcr_internal', dt.__version__, 'registry OK')"], env=env)
    if deep:
        run([python, "-c", "import torch; import decodertcr_internal; print('PyTorch', torch.__version__)"],
            env=env)
        if settings["device"].startswith("cuda"):
            run([python, "-c", "import torch,sys; d=torch.device(sys.argv[1]); "
                 "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                 "assert (d.index or 0)<torch.cuda.device_count(), 'CUDA index is unavailable'",
                 settings["device"]], env=env)
        if settings["device"] == "apple":
            code = ("import sys; import mlx.core as mx; "
                    "from esmc_mlx.weights import verify_bundle; "
                    "from esmc_mlx.config import validate_decoder_config; "
                    "assert mx.metal.is_available(), 'Metal is unavailable'; "
                    "c,m,_=verify_bundle(sys.argv[1]); "
                    "validate_decoder_config(c,sys.argv[2]); "
                    "print('MLX Metal and bundle integrity OK:',m['checkpoint_identity'])")
            run([settings["mlx_python"], "-c", code, settings["checkpoint"], settings["model"]], env=env)
        else:
            # Confirm the exact registry release resolves and reports its identity.
            code = ("import sys; import decodertcr_internal as dt; "
                    "info=dt.info(sys.argv[1]); "
                    "print('Registry release OK:', info['model_id'], info['member']['sequence_convention'], "
                    "info['member']['weights']['sha256'][:16])")
            run([python, "-c", code, MODEL_IDS[settings["model"]]], env=env)


def install_decoder(uv, env_dir: Path, env, source: str, device="cpu"):
    """Install decodertcr_internal into an isolated Python 3.12 model environment."""
    python = python_in(env_dir)
    if not python.is_file():
        run([*uv, "venv", "--python", "3.12", env_dir], env=env)
    if platform.system() == "Linux" and device == "cpu":
        # Avoid multi-GB NVIDIA wheels for a CPU install of the pinned Torch release.
        run([*uv, "pip", "install", "--python", python, "--torch-backend", "cpu",
             "torch==2.10.0", source], env=env)
    else:
        run([*uv, "pip", "install", "--python", python, source], env=env)
    return python


def configure_registry(python: Path, registry: str, env):
    """Point the model environment at the shared decodertcr registry (no downloads)."""
    run([python, "-m", "decodertcr_internal.cli", "configure", "--registry", registry], env=env)


def ensure_uv(state, env):
    python = python_in(state / "bootstrap")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(state / "bootstrap")
    run([python, "-m", "pip", "install", "--disable-pip-version-check", f"uv=={UV_VERSION}"], env=env)
    return [python.parent / ("uv.exe" if os.name == "nt" else "uv"), "--no-config"]


def check_setup_resources(core: Path, env, model: str, device: str, *, allow_memory_risk=False,
                          checkpoint_storage_bytes=None):
    """Estimate separate setup phases before model installation or allocation.

    The core environment supplies only Pydantic/standard-library resource code;
    this subprocess imports neither Torch nor MLX and never reads model weights.
    """
    code = """
import json,sys
from tcr_workbench.resources import plan_resources
parameter_bytes=int(sys.argv[1])
device=sys.argv[2]
allow=sys.argv[3]=='1'
checkpoint_storage_bytes=int(sys.argv[4])
phases=[('CPU reference/conversion','cpu','torch'),('Apple inference','apple','mlx')] if device=='apple' else [('model runtime',device,'torch')]
blocked=False
for label,selected,backend in phases:
    storage={'checkpoint_storage_bytes':checkpoint_storage_bytes} if backend=='torch' else {}
    plan=plan_resources(parameter_bytes,selected,backend,batch_size=2,sequence_length=64,allow_memory_risk=allow,**storage)
    print('Setup resource check — '+label+': '+json.dumps(plan.model_dump()),flush=True)
    blocked=blocked or plan.blocked
if blocked:
    raise SystemExit('Setup stopped before model installation/conversion: estimated memory exceeds the available budget. Choose sufficient hardware or explicitly pass --allow-memory-risk; this can exhaust memory. No device or model was changed.')
"""
    # The fp32 registry artifact bytes are estimated as parameters * 4; the real
    # on-disk artifact size is re-checked from the registry during preparation.
    checkpoint_bytes = (checkpoint_storage_bytes if checkpoint_storage_bytes is not None else
                        SETUP_PARAMETER_COUNTS[model] * 4)
    run([core, "-c", code, str(SETUP_PARAMETER_COUNTS[model] * 4),
         "cuda" if device == "gpu" else device, "1" if allow_memory_risk else "0",
         str(checkpoint_bytes)], env=env)


def prepare_setup_model(settings, state, core, env, allow_memory_risk):
    """Convert and parity-check the Apple bundle from the registry artifact."""
    code = ("import json,sys; from pathlib import Path; "
            "from tcr_workbench.model_preparation import prepare_model; "
            "path=Path(sys.argv[1]); "
            "prepared,_=prepare_model(json.loads(path.read_text()),sys.argv[2],"
            "allow_memory_risk=sys.argv[3]=='1',sequence_length=64); "
            "path.write_text(json.dumps(prepared))")
    with tempfile.TemporaryDirectory(dir=state, prefix="model-setup-") as temporary:
        path = Path(temporary) / "settings.json"
        atomic_json(path, settings)
        run([core, "-c", code, path, state / "prepared", "1" if allow_memory_risk else "0"], env=env)
        return json.loads(path.read_text())


def setup(args, root: Path):
    state = root / ".tcr"
    if args.device == "apple" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        raise ValueError("Apple setup requires macOS on Apple Silicon (arm64); use --device cpu.")
    if args.device == "gpu" and platform.system() != "Linux":
        raise ValueError("Automatic NVIDIA GPU setup targets Linux. Use --device apple on Apple Silicon.")
    full_setup = not args.core_only and not args.reuse_config
    registry = args.registry or os.environ.get("DECODERTCR_REGISTRY")
    if full_setup and not registry:
        raise ValueError("Model setup requires --registry <root> (shared decodertcr registry) "
                         "or the DECODERTCR_REGISTRY environment variable")
    env = clean_env(state)
    model = SETUP_ALIASES[args.model]
    with setup_lock(state):
        uv = ensure_uv(state, env)
        core = python_in(state / "envs/core")
        if not core.is_file():
            run([*uv, "venv", "--python", "3.12", state / "envs/core"], env=env)
        run([*uv, "pip", "install", "--python", core, *CORE_PINS, "-e", f"{root}[dataset]"], env=env)
        if args.core_only:
            print("Core setup complete. Try: python3 tcr.py screen --help")
            return
        if args.reuse_config:
            values = [read_settings(Path(p).absolute(), core, env) for p in args.reuse_config]
            for value in values:
                probe(value, env, deep=True)
                if getattr(args, "species", "human") == "mouse":
                    ensure_germlines(Path(value["python_executable"]), "mouse", state, env)
            for value in values:
                device = "gpu" if value["device"].startswith("cuda") else value["device"]
                atomic_json(state / f"runtime-{device}.json", value)
            atomic_json(state / "runtime.json", values[-1])
            print("Setup complete; existing environments and registry reused.")
            return
        check_setup_resources(core, env, model, args.device, allow_memory_risk=args.allow_memory_risk)
        print(f"Installing {DECODER_PACKAGE} and connecting to the registry at {registry}. "
              "Model weights are resolved from the shared registry, not downloaded here.", flush=True)
        model_env = state / "envs/model"
        model_python = install_decoder(uv, model_env, env, args.decoder_source,
                                       "cpu" if args.device == "apple" else args.device)
        configure_registry(model_python, registry, env)
        # Human reconstruction resources ship in the package; mouse still uses Stitchr.
        ensure_germlines(model_python, "human", state, env)
        if getattr(args, "species", "human") == "mouse":
            ensure_germlines(model_python, "mouse", state, env)
        settings = dict(decoder_dir=str(model_env), python_executable=str(model_python),
                        model=model, device="cpu", precision="float32",
                        checkpoint=None, mlx_python=None, batch_size=1,
                        token_budget=4096, cache_bytes=67108864, timeout=None)
        if args.device == "apple":
            mlx_python = python_in(state / "envs/mlx")
            if not mlx_python.is_file():
                run([*uv, "venv", "--python", "3.12", state / "envs/mlx"], env=env)
            run([*uv, "pip", "install", "--python", mlx_python, "-e", f"{root / 'esmc-mlx'}[convert]",
                 "torch==2.10.0", "numpy==2.5.3", "pydantic==2.13.5", "safetensors==0.8.0"], env=env)
            # Convert and parity-check the bundle from the resolved registry artifact.
            apple = prepare_setup_model(dict(settings, device="apple", mlx_python=str(mlx_python)),
                                        state, core, env, args.allow_memory_risk)
            probe(apple, env, deep=True)
            atomic_json(state / "runtime-apple.json", apple)
        else:
            selected = dict(settings, device="cuda") if args.device == "gpu" else settings
            probe(selected, env, deep=True)
            if args.device == "gpu":
                atomic_json(state / "runtime-gpu.json", selected)
        atomic_json(state / "runtime-cpu.json", settings)
        atomic_json(state / "runtime.json", apple if args.device == "apple" else selected)
        print("Setup complete. Run: python3 tcr.py doctor\nThen follow docs/usage.md.")


def doctor(args, root: Path) -> int:
    state = root / ".tcr"
    env = clean_env(state)
    core = python_in(state / "envs/core")
    if not core.is_file():
        raise ValueError("Core environment missing. Run: python3 tcr.py setup --core-only (or --device apple/cpu)")
    run([core, "-c", "import tcr_workbench,polars,pyarrow,pydantic; print('Core environment OK')"], env=env)
    path = Path(args.config).absolute() if args.config else state / "runtime.json"
    if not path.exists() and not args.config:
        print("Core-only installation. The model runtime is not configured; use setup --device cpu or apple.")
        return 0
    settings = read_settings(path, core, env)
    probe(settings, env, deep=args.deep)
    print(f"decodertcr_internal, registry and human germlines OK: {settings['model']}, {settings['device']}, "
          f"{settings['precision']}." + ("" if args.deep else " Use doctor --deep to verify weights/Metal."))
    print("This checks installation, not biological accuracy. Examples: docs/usage.md")
    return 0


def main(argv, *, root: Path) -> int:
    parser = argparse.ArgumentParser(prog="python3 tcr.py")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup", help="Install locally and connect to the shared model registry",
                       description="Install into .tcr/ without modifying system Python. Model setup installs "
                       "decodertcr_internal and its dependencies, then connects to the shared registry "
                       "given by --registry or DECODERTCR_REGISTRY. Model weights are resolved from the "
                       "registry on first use, not downloaded here. Apple additionally converts and "
                       "parity-checks an MLX bundle. Linux GPU setup also needs a working NVIDIA driver. "
                       "See docs/model-notices.md.")
    s.add_argument("--device", choices=("cpu", "gpu", "apple"), default="cpu")
    s.add_argument("--species", choices=("human", "mouse"), default="human",
                   help="Also install mouse V/J germlines for mouse TCR reconstruction; human remains available")
    s.add_argument("--model", choices=tuple(SETUP_ALIASES), default="esmc-300m",
                   help="DecoderTCR architecture to configure (default: esmc-300m)")
    s.add_argument("--registry", help="Shared decodertcr registry root; or set DECODERTCR_REGISTRY")
    s.add_argument("--decoder-source", default=DECODER_PACKAGE,
                   help="pip install source for decodertcr_internal (default: the pinned release)")
    group = s.add_mutually_exclusive_group()
    group.add_argument("--core-only", action="store_true", help="No model/framework/germline installation")
    group.add_argument("--reuse-config", action="append", metavar="PATH",
                       help="Reuse existing runtime settings; repeat for CPU and Apple; last is default")
    s.add_argument("--allow-memory-risk", action="store_true",
                   help="Explicitly continue despite a memory estimate exceeding available capacity; may exhaust memory")
    s = sub.add_parser("doctor", help="Read-only installation checks; never installs or downloads")
    s.add_argument("--deep", action="store_true", help="Also check checkpoint hashes and framework/Metal imports")
    s.add_argument("--config", help="Check a specific runtime config")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            setup(args, root)
            return 0
        return doctor(args, root)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        print(f"{args.command} failed: {detail.strip()}\n"
              "Existing results are unchanged. Fix the reported issue and rerun the same command.", file=sys.stderr)
        return 1
