"""Explicit, local, resumable setup and read-only diagnostics (standard library only)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import venv

from tcr import python_in

UV_VERSION = "0.12.17"
DECODER_REV = "3e3d9889d26d79635f674940ddab039c4dc6f9f9"
SOURCE_SHA = "cecc2e1801776f57b73b6b5351f7e82e0a578696295d3f97b3fb6d417e7ee70c"
MODEL_REV = "803fed3bbf3dcc40ed481f7d50a20f668db0ef01"
MODEL_SHA = "18d47c169d0ce992152838b8229e1c682b0a681f55057b471dcdcbf07d2fcad9"
MODEL_SIZE = 3996365253
MODEL_FILE = "DecoderTCR-ESMC-V0.3/300M.ckpt"
SETUP_MODELS = {
    "esmc-300m": ("DecoderTCR-ESMC_300M", MODEL_FILE, MODEL_SHA, MODEL_SIZE),
    "esmc-600m": ("DecoderTCR-ESMC_600M", "DecoderTCR-ESMC-V0.3/600M.ckpt",
                  "4d3c84f30e3781c023e412eb3f3c098ef9fe64fd1423fd34df4b40b9dcbac0aa", 2300241260),
    "esmc-6b": ("DecoderTCR-ESMC_6B", "DecoderTCR-ESMC-V0.3/6B.ckpt",
                "b6bd3170b55a9a06d9c1472e248bd083e092afc7924439b444bf7379b74ab692", 25408220584),
}
# Parameter counts come from the exact source tensor inventories, independent of
# checkpoint containers (the 300M release also stores optimizer state).
SETUP_PARAMETER_COUNTS = {"DecoderTCR-ESMC_300M": 332997184,
                          "DecoderTCR-ESMC_600M": 575036992,
                          "DecoderTCR-ESMC_6B": 6352005184}
CORE_PINS = ["polars==1.36.1", "pyarrow==21.0.0", "pydantic==2.13.5",
             "rapidfuzz==3.13.0", "numpy==2.0.2", "h5py==3.14.0"]
GERMLINE_PROBE = (
    "from pathlib import Path; from Stitchr import stitchrfunctions as f; "
    "p=Path(f.data_dir)/'HUMAN'; "
    "assert all((p/n).is_file() and (p/n).stat().st_size > 0 for n in "
    "('TRA.fasta','TRB.fasta','J-region-motifs.tsv','C-region-motifs.tsv')), "
    "'Human Stitchr germlines missing'"
)


def germline_probe(species):
    if species not in ("human", "mouse"):
        raise ValueError("Germline species must be human or mouse")
    return GERMLINE_PROBE.replace("HUMAN", species.upper()).replace("Human", species.title())


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


def download(url: str, dest: Path, expected_sha=None, expected_size=None):
    """Bounded-memory download; failed transfers never replace an existing artifact."""
    if dest.is_file():
        if expected_sha and sha256(dest) != expected_sha:
            raise ValueError(f"Checksum mismatch: {dest}. Move the damaged file aside and retry.")
        if expected_size and dest.stat().st_size != expected_size:
            raise ValueError(f"Size mismatch: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".part")
    print(f"Downloading {dest.name}…", flush=True)
    digest = hashlib.sha256()
    size = 0
    reported = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TCR-Workbench/0.1"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as out:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if expected_size and size - reported >= 256 * 1024 * 1024:
                    print(f"  {size / 1e9:.1f} / {expected_size / 1e9:.1f} GB", flush=True)
                    reported = size
                if expected_size and size > expected_size:
                    raise ValueError("Download exceeds the pinned artifact size")
        if expected_size and size != expected_size:
            raise ValueError(f"Incomplete download: {size} bytes, expected {expected_size}")
        if expected_sha and digest.hexdigest() != expected_sha:
            raise ValueError("Downloaded file failed SHA-256 verification")
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)


def unpack_source(archive: Path, target: Path):
    """Extract only ordinary files/directories inside the one pinned archive root."""
    prefix = f"DecoderTCR-{DECODER_REV}"
    with tempfile.TemporaryDirectory(dir=target.parent, prefix="source-") as work:
        staging = Path(work)
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            total = 0
            for member in members:
                path = Path(member.name)
                if (not path.parts or path.parts[0] != prefix or path.is_absolute()
                        or ".." in path.parts or "\\" in member.name
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f"Unsafe source archive entry: {member.name}")
                total += member.size
                if total > 2 * 1024**3:
                    raise ValueError("Source archive exceeds 2 GiB")
            for member in members:
                dest = staging / member.name
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as inp, dest.open("wb") as out:
                        shutil.copyfileobj(inp, out, 1024 * 1024)
        extracted = staging / prefix
        if not (extracted / "src/DecoderTCR/utils/predict_from_genes.py").is_file():
            raise ValueError("Downloaded source is missing DecoderTCR")
        (extracted / ".workbench-revision").write_text(DECODER_REV + "\n")
        extracted.rename(target)


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
            "v.checkpoint=v.checkpoint or str(Path(v.decoder_dir)/resolve_model(v.model).checkpoint); "
            "print(v.model_dump_json())")
    value = json.loads(run([core, "-c", code, path], env=env, capture=True).stdout)
    for key in ("decoder_dir", "python_executable", "checkpoint", "mlx_python"):
        if value.get(key):
            # absolute(), not resolve(): preserve venv interpreter symlinks.
            value[key] = str((path.resolve().parent / value[key]).absolute())
            if not Path(value[key]).exists():
                raise ValueError(f"Configured {key} does not exist: {value[key]}")
    if value["device"] == "apple":
        if not Path(value["checkpoint"]).is_dir():
            raise ValueError("Apple checkpoint must be a converted MLX bundle directory")
    elif value.get("checkpoint") and not Path(value["checkpoint"]).is_file():
        raise ValueError("CPU/CUDA checkpoint must be a PyTorch file")
    return value


def probe(settings, env, *, deep=False):
    python = settings["python_executable"]
    run([python, "-c", GERMLINE_PROBE], env=env, capture=True)
    if deep:
        run([python, "-c", "import torch; import DecoderTCR; print('PyTorch', torch.__version__)"],
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
            digest = sha256(Path(settings["checkpoint"]))
            if any(settings["model"] == m[0] and digest == m[2] for m in SETUP_MODELS.values()):
                print("Pinned DecoderTCR checkpoint integrity OK")
            else:
                code = ("import sys,torch; sys.path.insert(0,sys.argv[1]); "
                        "from tcr_workbench.model_registry import resolve_model; "
                        "from tcr_workbench.backends.torch_checkpoint_worker import validate_inventory; "
                        "validate_inventory(torch.load(sys.argv[2],map_location='cpu',"
                        "weights_only=True,mmap=True),resolve_model(sys.argv[3]).arch)")
                run([python, "-c", code, Path(__file__).parent / "src", settings["checkpoint"],
                     settings["model"]], env=env)
                print(f"User-supplied checkpoint; architecture inventory checked, SHA-256 {digest}. "
                      "This release has not been numerically validated by Workbench.")


def install_decoder(uv, decoder: Path, state: Path, env, device="cpu"):
    if platform.system() != "Linux" or device == "gpu":
        run([*uv, "sync", "--project", decoder, "--python", "3.12", "--no-dev", "--frozen"], env=env)
        return
    # Upstream's Linux lock selects CUDA wheels. Retain its non-Torch pins while
    # selecting the same Torch release's CPU build; avoid multi-GB NVIDIA downloads.
    python = python_in(decoder / ".venv")
    if not python.is_file():
        run([*uv, "venv", "--python", "3.12", decoder / ".venv"], env=env)
    requirements = state / "decoder-cpu-requirements.txt"
    run([*uv, "export", "--project", decoder, "--frozen", "--no-dev", "--no-emit-project",
         "--prune", "torch", "--output-file", requirements, "--quiet"], env=env)
    run([*uv, "pip", "install", "--python", python, "--torch-backend", "cpu", "torch==2.10.0",
         "-r", requirements, "-e", decoder], env=env)


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
    # These pinned PyTorch releases use uncompressed tensor archives. Their full
    # file size conservatively includes tensor storage plus archive headers and
    # metadata; the 300M release also contains optimizer state. It is deliberately
    # separate from resident parameters and is not charged to MLX bundle inference.
    checkpoint_bytes = (checkpoint_storage_bytes if checkpoint_storage_bytes is not None else
                        next(entry[3] for entry in SETUP_MODELS.values() if entry[0] == model))
    run([core, "-c", code, str(SETUP_PARAMETER_COUNTS[model] * 4),
         "cuda" if device == "gpu" else device, "1" if allow_memory_risk else "0",
         str(checkpoint_bytes)], env=env)


def prepare_setup_model(settings, state, core, env, expected_sha256, allow_memory_risk):
    """Use the same compatibility and parity checks as local checkpoint preparation."""
    code = ("import json,sys; from pathlib import Path; "
            "from tcr_workbench.model_preparation import prepare_model; "
            "path=Path(sys.argv[1]); "
            "prepared,_=prepare_model(json.loads(path.read_text()),sys.argv[2],"
            "expected_sha256=sys.argv[3],allow_memory_risk=sys.argv[4]=='1',sequence_length=64); "
            "path.write_text(json.dumps(prepared))")
    with tempfile.TemporaryDirectory(dir=state, prefix="model-setup-") as temporary:
        path = Path(temporary) / "settings.json"
        atomic_json(path, settings)
        run([core, "-c", code, path, state / "prepared", expected_sha256,
             "1" if allow_memory_risk else "0"], env=env)
        return json.loads(path.read_text())


def setup(args, root: Path):
    state = root / ".tcr"
    if args.device == "apple" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        raise ValueError("Apple setup requires macOS on Apple Silicon (arm64); use --device cpu.")
    if args.device == "gpu" and platform.system() != "Linux":
        raise ValueError("Automatic NVIDIA GPU setup targets Linux. Use --device apple on Apple Silicon.")
    full_setup = not args.core_only and not args.reuse_config
    custom = args.checkpoint_url is not None or args.expected_sha256 is not None
    if custom:
        if not full_setup:
            raise ValueError("Checkpoint download/checksum options require full model setup")
        if not args.checkpoint and not args.checkpoint_url:
            raise ValueError("--expected-sha256 requires --checkpoint or --checkpoint-url")
        if not args.expected_sha256 or not re.fullmatch(r"[a-fA-F0-9]{64}", args.expected_sha256):
            raise ValueError("Supply the checkpoint publisher's 64-digit --expected-sha256")
        args.expected_sha256 = args.expected_sha256.lower()
        if args.checkpoint_url:
            url = urllib.parse.urlsplit(args.checkpoint_url)
            if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
                raise ValueError("--checkpoint-url must be an HTTPS download URL without credentials or a fragment")
    if full_setup and args.checkpoint is not None and not Path(args.checkpoint).is_file():
        raise ValueError(f"Local checkpoint is missing or not a file: {args.checkpoint}")
    env = clean_env(state)
    model, model_file, model_sha, model_size = SETUP_MODELS[args.model]
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
            print("Setup complete; existing model files and environments reused.")
            return
        resource_options = {}
        if custom:
            # Remote container storage is unknown until inspection; estimate its
            # parameter payload here, then check the actual inventory before loading.
            resource_options["checkpoint_storage_bytes"] = (Path(args.checkpoint).stat().st_size
                if args.checkpoint else SETUP_PARAMETER_COUNTS[model] * 4)
        check_setup_resources(core, env, model, args.device, allow_memory_risk=args.allow_memory_risk,
                              **resource_options)
        if custom:
            print(f"Installing {model} with the supplied checkpoint; compatibility checks run before "
                  "saving defaults.", flush=True)
        elif args.model == "esmc-6b":
            print(f"Installing pinned {model}; weights are {model_size / 1e9:.1f} GB. "
                  "Apple conversion needs about another "
                  "25.4 GB disk space plus runtime dependencies. The 6B Apple adapter is experimental "
                  "until local reference parity succeeds.", flush=True)
        else:
            print(f"Installing pinned {model}. Allow about 12 GB disk space "
                  f"(more for Apple conversion); weights are {model_size / 1e9:.1f} GB.", flush=True)
        decoder = state / "DecoderTCR"
        if not decoder.exists():
            archive = state / "downloads" / f"DecoderTCR-{DECODER_REV}.tar.gz"
            download(f"https://codeload.github.com/Biohub/DecoderTCR/tar.gz/{DECODER_REV}", archive,
                     SOURCE_SHA)
            unpack_source(archive, decoder)
        elif (decoder / ".workbench-revision").read_text().strip() != DECODER_REV:
            raise ValueError("Managed DecoderTCR source revision differs from this release")
        install_decoder(uv, decoder, state, env, args.device)
        decoder_python = python_in(decoder / ".venv")
        ensure_germlines(decoder_python, "human", state, env)
        if getattr(args, "species", "human") == "mouse":
            ensure_germlines(decoder_python, "mouse", state, env)
        checkpoint = decoder / "checkpoints" / model_file
        if args.checkpoint_url:
            checkpoint = state / "checkpoints" / args.model / args.expected_sha256 / "weights.ckpt"
            download(args.checkpoint_url, checkpoint, args.expected_sha256)
        elif args.checkpoint:
            checkpoint = Path(args.checkpoint).absolute()
            if custom:
                if sha256(checkpoint) != args.expected_sha256:
                    raise ValueError("Checkpoint SHA-256 does not match --expected-sha256")
            elif checkpoint.stat().st_size != model_size or sha256(checkpoint) != model_sha:
                raise ValueError(f"Setup --checkpoint must be the pinned {model} release. "
                                 "Add --expected-sha256 for a new release, or use prepare-model.")
        else:
            download(f"https://huggingface.co/biohub/DecoderTCR/resolve/{MODEL_REV}/{model_file}",
                     checkpoint, model_sha, model_size)
        settings = dict(decoder_dir=str(decoder), python_executable=str(decoder_python),
                        model=model, device="cpu", precision="float32",
                        checkpoint=str(checkpoint), mlx_python=None, batch_size=1,
                        token_budget=4096, cache_bytes=67108864, timeout=None)
        if args.device == "apple":
            mlx_python = python_in(state / "envs/mlx")
            if not mlx_python.is_file():
                run([*uv, "venv", "--python", "3.12", state / "envs/mlx"], env=env)
            run([*uv, "pip", "install", "--python", mlx_python, "-e", f"{root / 'esmc-mlx'}[convert]",
                 "torch==2.10.0", "numpy==2.5.3", "pydantic==2.13.5", "safetensors==0.8.0"], env=env)
            if custom:
                apple = prepare_setup_model(dict(settings, device="apple", mlx_python=str(mlx_python)),
                                            state, core, env, args.expected_sha256, args.allow_memory_risk)
            else:
                bundle = state / f"models/decoder-{args.model.removeprefix('esmc-')}-fp32"
                if not bundle.exists():
                    run([mlx_python, root / "esmc-mlx/scripts/convert_weights.py", "--source", checkpoint,
                         "--source-variant", "decodertcr-lightning-v03", "--expected-sha256", model_sha,
                         "--model-size", args.model.removeprefix("esmc-"),
                         "--output", bundle], env=env)
                apple = dict(settings, device="apple", checkpoint=str(bundle), mlx_python=str(mlx_python))
            probe(apple, env, deep=True)
            atomic_json(state / "runtime-apple.json", apple)
        else:
            selected = dict(settings, device="cuda") if args.device == "gpu" else settings
            if custom:
                selected = prepare_setup_model(selected, state, core, env, args.expected_sha256,
                                               args.allow_memory_risk)
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
        print("Core-only installation. DecoderTCR is not configured; use setup --device cpu or apple.")
        return 0
    settings = read_settings(path, core, env)
    if not (Path(settings["decoder_dir"]) / "src/DecoderTCR/utils/predict_from_genes.py").is_file():
        raise ValueError("DecoderTCR source tree is incomplete")
    probe(settings, env, deep=args.deep)
    print(f"DecoderTCR paths and human germlines OK: {settings['model']}, {settings['device']}, "
          f"{settings['precision']}." + ("" if args.deep else " Use doctor --deep to verify weights/Metal."))
    print("This checks installation, not biological accuracy. Examples: docs/usage.md")
    return 0


def main(argv, *, root: Path) -> int:
    parser = argparse.ArgumentParser(prog="python3 tcr.py")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup", help="Install locally; downloads only happen during explicit setup",
                       description="Install into .tcr/ without modifying system Python. Model setup downloads "
                       "pinned source, dependencies, human germlines and weights "
                       "(300M: 4 GB; 600M: 2.3 GB; 6B: 25 GB). "
                       "Use --checkpoint-url and --expected-sha256 for a compatible newer release. "
                       "Linux GPU setup also needs a working NVIDIA driver and additional disk space. "
                       "See docs/model-notices.md.")
    s.add_argument("--device", choices=("cpu", "gpu", "apple"), default="cpu")
    s.add_argument("--species", choices=("human", "mouse"), default="human",
                   help="Also install mouse V/J germlines for mouse TCR reconstruction; human remains available")
    s.add_argument("--model", choices=tuple(SETUP_MODELS), default="esmc-300m",
                   help="DecoderTCR architecture to install (default: esmc-300m)")
    group = s.add_mutually_exclusive_group()
    group.add_argument("--core-only", action="store_true", help="No model/framework/germline downloads")
    group.add_argument("--reuse-config", action="append", metavar="PATH",
                       help="Reuse existing runtime settings; repeat for CPU and Apple; last is default")
    source = s.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="Reuse a local checkpoint; add --expected-sha256 for a newer release")
    source.add_argument("--checkpoint-url", help="Exact HTTPS URL for a compatible checkpoint; requires --expected-sha256")
    s.add_argument("--expected-sha256", help="Publisher's SHA-256 for the supplied checkpoint or download URL")
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
