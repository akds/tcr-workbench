"""Host-side MLX worker orchestration; this module never imports MLX or PyTorch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .precision_contract import Precision, precision_metadata


class AppleRuntime(BaseModel):
    """Strict worker handshake, including the model and numerical execution mode."""
    model_config = ConfigDict(extra="forbid", strict=True)
    backend: Literal["mlx"]
    device: Literal["apple"]
    mode: Literal["scores", "profile", "profiles"]
    context_type: Literal["tcr-pmhc", "pmhc"]
    precision: Precision
    dtype: Precision
    approximate: bool
    math_precision: Literal["float32-no-tf32", "float16-approximate-fp32-reductions"]
    bundle_dtype: Literal["float32"]
    score_reduction_dtype: Literal["float32"]
    rope_frequency_dtype: Literal["float32"]
    tokenizer_variant: Literal["decodertcr-esm1b"]
    source_checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mlx_peak_memory_bytes: int = Field(ge=0)
    model_load_seconds: float = Field(ge=0, allow_inf_nan=False)
    inference_seconds: float = Field(ge=0, allow_inf_nan=False)
    rows: int = Field(ge=1)
    scored: int = Field(ge=0)
    unique_context_forwards: int = Field(ge=0)
    unresolved: Optional[int] = Field(default=None, ge=0)
    cache_hits: Optional[int] = Field(default=None, ge=0)
    batches: Optional[int] = Field(default=None, ge=0)
    peak_cache_bytes: Optional[int] = Field(default=None, ge=0)
    reconstruction: Optional[Dict[str, str]] = None
    status: Optional[Literal["ModelHypothesis", "Unresolved"]] = None
    reason: Optional[str] = None


    @model_validator(mode="after")
    def numerical_mode_matches(self):
        if any(getattr(self, key) != value for key, value in precision_metadata(self.precision).items()):
            raise ValueError("MLX precision fields describe inconsistent numerical modes")
        return self


def validate_options(batch_size, token_budget, cache_bytes):
    for name, value, minimum, maximum in (
        ("batch_size", batch_size, 1, 128),
        ("token_budget", token_budget, 3, 262144),
        ("cache_bytes", cache_bytes, 0, 1024 ** 3),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def bundle_fingerprint(bundle):
    from ..prediction import file_sha256
    root = Path(bundle).resolve()
    if not root.is_dir():
        raise ValueError("Apple --checkpoint must name a converted MLX bundle directory")
    for name in ("manifest.json", "config.json", "tokenizer.json", "model.safetensors"):
        if not (root / name).is_file():
            raise ValueError(f"converted MLX bundle is missing {name}")
    files = {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob("*"))
             if path.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = json.loads((root / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("MLX manifest must be a JSON object")
    source_hash = manifest.get("source_sha256")
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(char not in "0123456789abcdef" for char in source_hash)):
        raise ValueError("MLX manifest requires source_sha256")
    return {"bundle_path": str(root), "bundle_sha256": digest, "bundle_files": files,
            "checkpoint_sha256": source_hash,
            "bundle_source_revision": manifest.get("source_revision"),
            "bundle_model_revision": manifest.get("model_revision")}


def runtime_fingerprint(python, decoder_root):
    from ..prediction import _decoder_environment, file_sha256
    executable = Path(python).absolute()
    if not executable.is_file():
        raise FileNotFoundError(f"MLX Python executable is missing: {executable}")
    script = """
import importlib.util
import importlib.metadata as md
import json
import platform
import sys
spec=importlib.util.find_spec('esmc_mlx')
print(json.dumps({'origin':spec.origin if spec else None,
 'packages':sorted((d.metadata.get('Name',''),d.version) for d in md.distributions()),
 'python':sys.version,'platform':platform.platform()}))
"""
    try:
        process = subprocess.run([str(executable), "-I", "-c", script], cwd=decoder_root,
                                 env=_decoder_environment(executable), capture_output=True,
                                 text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("could not inspect MLX Python environment") from exc
    if process.returncode:
        raise RuntimeError("could not inspect MLX Python environment: " + process.stderr[-2000:])
    info = json.loads(process.stdout)
    if not isinstance(info, dict) or not isinstance(info.get("origin"), str) or not info["origin"]:
        raise ValueError("--mlx-python must have the esmc-mlx package installed")
    code = hashlib.sha256()
    package = Path(info["origin"]).resolve().parent
    for path in sorted(package.rglob("*.py")):
        code.update(str(path.relative_to(package)).encode())
        code.update(file_sha256(path).encode())
    backend = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        backend.update(path.name.encode())
        backend.update(file_sha256(path).encode())
    for path in (Path(__file__).parents[1] / "prediction.py",
                 Path(__file__).parents[1] / "model_registry.py"):
        backend.update(path.name.encode())
        backend.update(file_sha256(path).encode())
    return {"mlx_python_executable": str(executable),
            "mlx_python_sha256": file_sha256(executable),
            "mlx_environment_sha256": hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest(),
            "mlx_source_sha256": code.hexdigest(), "backend_source_sha256": backend.hexdigest()}


def fingerprint(decoder_dir, python, model, bundle, mlx_python, *, batch_size, token_budget,
                cache_bytes, precision="float32"):
    from ..prediction import _decoder_fingerprint
    validate_options(batch_size, token_budget, cache_bytes)
    numerical = precision_metadata(precision)
    converted = bundle_fingerprint(bundle)
    # Reuse the authoritative reconstruction/source/environment audit without
    # requiring the 4GB PyTorch checkpoint beside a converted inference bundle.
    upstream = _decoder_fingerprint(decoder_dir, python)
    return {**upstream, **converted, **runtime_fingerprint(mlx_python, Path(decoder_dir)),
            "model": model, "backend": "mlx", "device": "apple", "batch_size": batch_size,
            "token_budget": token_budget, "cache_bytes": cache_bytes,
            **numerical, "context_type": "tcr-pmhc"}


def execute(input_path, output_path, temporary, provenance, *, timeout, profile=False):
    from ..prediction import _execute, _decoder_fingerprint
    root = Path(provenance["decoder_dir"])
    workers = Path(__file__).resolve().parent
    reconstructed = Path(temporary) / "reconstructed.csv"
    runtime_path = Path(temporary) / "runtime.json"
    from ..species import reconstruction_arguments, verify_biological_fingerprint
    verify_biological_fingerprint(provenance)
    _execute([provenance["python_executable"], str(workers / "reconstruct_worker.py"),
              "--input", str(input_path), "--output", str(reconstructed),
              "--context-type", provenance["context_type"],
              *reconstruction_arguments(provenance)], root, timeout)
    command = [provenance["mlx_python_executable"], "-I", str(workers / "mlx_worker.py"),
               "--input", str(reconstructed), "--output", str(output_path),
               "--bundle", provenance["bundle_path"], "--runtime", str(runtime_path),
               "--model", provenance["model"], "--batch-size", str(provenance["batch_size"]),
               "--token-budget", str(provenance["token_budget"]),
               "--cache-bytes", str(provenance["cache_bytes"]),
               "--context-type", provenance["context_type"],
               "--precision", provenance["precision"]]
    if profile:
        command.append("--profile-batch" if profile == "batch" else "--profile")
    _execute(command, root, timeout)
    runtime = AppleRuntime.model_validate_json(runtime_path.read_text()).model_dump(exclude_none=True)
    if runtime["mode"] != ("profiles" if profile == "batch" else "profile" if profile else "scores"):
        raise ValueError("MLX worker returned the wrong output mode")
    if runtime["context_type"] != provenance["context_type"]:
        raise ValueError("MLX worker returned the wrong biological context mode")
    if (not profile or profile == "batch") and (runtime.get("unresolved") is None
                        or runtime["scored"] + runtime["unresolved"] != runtime["rows"]):
        raise ValueError("MLX worker returned inconsistent row coverage")
    if (runtime.get("source_checkpoint_sha256") != provenance["checkpoint_sha256"]
            or runtime.get("backend") != "mlx" or runtime.get("device") != "apple"
            or any(runtime.get(key) != value for key, value in precision_metadata(provenance["precision"]).items())):
        raise ValueError("MLX worker returned inconsistent model/device provenance")
    # Check complete bundle bytes again before publishing results. This also
    # protects the pre-load metadata/cache inspection from concurrent changes.
    if bundle_fingerprint(provenance["bundle_path"])["bundle_sha256"] != provenance["bundle_sha256"]:
        raise ValueError("MLX bundle changed during inference")
    current = runtime_fingerprint(provenance["mlx_python_executable"], root)
    if any(value != provenance.get(key) for key, value in current.items()):
        raise ValueError("MLX source or Python environment changed during inference")
    upstream = _decoder_fingerprint(root, provenance["python_executable"])
    verify_biological_fingerprint(provenance)
    if any(value != provenance.get(key) for key, value in upstream.items()):
        raise ValueError("DecoderTCR reconstruction source, environment or germlines changed during inference")
    return runtime
