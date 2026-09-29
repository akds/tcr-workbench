"""Build pinned upstream commands without importing PyTorch into Workbench."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..model_registry import resolve_model


class TorchRuntime(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    backend: Literal["torch"]
    device: str = Field(pattern=r"^(cpu|cuda(?::[0-9]+)?)$")
    context_type: Literal["pmhc", "tcr-pmhc"]
    mode: Literal["scores", "profile", "profiles"]
    dtype: Literal["float32"]
    model_load_seconds: float = Field(ge=0, allow_inf_nan=False)
    inference_seconds: float = Field(ge=0, allow_inf_nan=False)
    real_model_forwards: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    peak_cache_bytes: int = Field(ge=0)
    rows: int = Field(ge=1)
    scored: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    status: Optional[Literal["ModelHypothesis", "Unresolved"]] = None
    reason: Optional[str] = None
    reconstruction: Optional[Dict[str, str]] = None



def execute_torch(input_path, candidate, temporary, fingerprint, *, timeout, profile=False):
    from ..prediction import _execute, _decoder_fingerprint
    root = Path(fingerprint["decoder_dir"])
    workers = Path(__file__).resolve().parent
    reconstructed = Path(temporary) / "reconstructed.csv"
    runtime_path = Path(temporary) / "runtime.json"
    from ..species import reconstruction_arguments, verify_biological_fingerprint
    verify_biological_fingerprint(fingerprint)
    spec = resolve_model(fingerprint["model"])
    _execute([fingerprint["python_executable"], str(workers / "reconstruct_worker.py"),
              "--input", str(input_path), "--output", str(reconstructed),
              "--context-type", fingerprint["context_type"], "--convention", spec.convention,
              *reconstruction_arguments(fingerprint)], root, timeout)
    command = [fingerprint["python_executable"], str(workers / "torch_sequence_worker.py"),
               "--input", str(reconstructed), "--output", str(candidate), "--runtime", str(runtime_path),
               "--context-type", fingerprint["context_type"], "--model", fingerprint["model"],
               "--model-id", spec.model_id, "--device", fingerprint["device"]]
    weight_source = fingerprint.get("weight_source", "registry")
    if weight_source == "huggingface":
        command += ["--weight-source", "huggingface", "--hf-repo", fingerprint["hf_repo"],
                    "--hf-weights-sha256", fingerprint["hf_weights_sha256"],
                    "--hf-revision", fingerprint["hf_revision"]]
        source = {"kind": "huggingface", "repo": fingerprint["hf_repo"],
                  "revision": fingerprint["hf_revision"]}
    else:
        source = {"kind": "registry", "model_id": spec.model_id}
    if profile:
        command.append("--profile-batch" if profile == "batch" else "--profile")
    _execute(command, root, timeout)
    runtime = TorchRuntime.model_validate_json(runtime_path.read_text()).model_dump(exclude_none=True)
    if (runtime.get("backend") != "torch"
            or runtime.get("device") != fingerprint["device"]
            or runtime.get("context_type") != fingerprint["context_type"]
            or runtime["mode"] != ("profiles" if profile == "batch" else "profile" if profile else "scores")
            or runtime["scored"] + runtime["unresolved"] != runtime["rows"]):
        raise ValueError("PyTorch worker returned inconsistent model/device provenance")
    current = _decoder_fingerprint(root, fingerprint["python_executable"], source=source)
    verify_biological_fingerprint(fingerprint)
    if any(fingerprint.get(key) != value for key, value in current.items()):
        raise ValueError("DecoderTCR reconstruction environment changed during inference")
    return runtime
