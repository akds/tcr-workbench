"""Explicit model onboarding and cached first-use checks; no network or framework imports."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .model_registry import normalize_device, resolve_model, validate_backend
from .prediction import _decoder_fingerprint, _execute, file_sha256, _json_write
from .backends import mlx_decoder
from .resources import HardwareMetrics, detect_hardware, plan_resources, GIB


class TimingProof(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    load_seconds: float = Field(ge=0, allow_inf_nan=False)
    forward_seconds: float = Field(gt=0, allow_inf_nan=False)
    batch_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)


class InventoryProof(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    tensor_count: int = Field(gt=0)
    parameter_bytes: int = Field(gt=0)
    checkpoint_storage_bytes: int = Field(gt=0)
    backbone: str
    architecture: str
    finite_values_checked: Literal[False]
    forward_checked: Literal[False]


class PreparationChecks(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    strict_tensor_inventory: Literal[True]
    finite_weights: Literal[True]
    finite_forward: Literal[True]
    tensor_count: int = Field(gt=0)
    parameter_bytes: int = Field(gt=0)
    checkpoint_storage_bytes: int = Field(gt=0)
    synthetic_token_rows: Literal[2]
    parity_passed: Optional[Literal[True]] = None
    max_absolute_logit_error: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    relative_l2: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    max_token_relative_l2: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    min_token_cosine: Optional[float] = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    comparison: Optional[str] = None
    gate: Optional[str] = None


class PreparedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: int = Field(default=1, ge=1, le=1)
    fingerprint: dict
    checkpoint: str
    model_id: str
    checks: PreparationChecks
    artifacts: dict[str, str]
    timing: TimingProof
    scope: str = "Technical compatibility on synthetic tokens; not biological or universal numerical validation"


class TorchProof(TimingProof):
    model_config = ConfigDict(extra="forbid", strict=True)
    backend: Literal["torch"]
    device: str
    model: str
    tensor_count: int = Field(gt=0)
    parameter_bytes: int = Field(gt=0)
    checkpoint_storage_bytes: int = Field(gt=0)


class AppleProof(TimingProof):
    model_config = ConfigDict(extra="forbid", strict=True)
    backend: Literal["mlx"]
    device: Literal["apple"]
    model: str
    precision: Literal["float32", "float16"]
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _identity(options):
    spec = resolve_model(options["model"])
    identity = {**_decoder_fingerprint(options["decoder_dir"], options["python_executable"],
                                       model_id=spec.model_id),
                "model": options["model"], "device": options["device"], "precision": options["precision"]}
    if options["device"] == "apple":
        # The converted bundle is a preparation OUTPUT; the registry artifact hash
        # (already in the decoder fingerprint) is the source identity for reuse.
        identity.update(mlx_decoder.runtime_fingerprint(options["mlx_python"], Path(options["decoder_dir"])))
    return identity


@contextmanager
def _lock(path):
    try:
        handle = path.open("x")
    except FileExistsError:
        raise ValueError(f"Preparation is already running, or was interrupted: {path}. "
                         "Remove this lock only after checking that no preparation is running.")
    try:
        with handle:
            handle.write("Preparing checkpoint\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def _registry_artifact_path(options, model_id):
    """Resolve the registry safetensors path in the model environment for conversion."""
    with tempfile.TemporaryDirectory(prefix="tcr-artifact-") as folder:
        output = Path(folder) / "artifact.txt"
        code = ("import sys; from decodertcr_internal.registry import Registry; "
                "r=Registry(None).resolve(sys.argv[1]); "
                "open(sys.argv[2],'w').write(str(r.bundle / r.member['weights']['path']))")
        _execute([options["python_executable"], "-c", code, model_id, str(output)],
                 Path(options["decoder_dir"]), options.get("timeout"))
        return Path(output.read_text().strip())


def _fixture(options, output, *, backend, checkpoint=None, reference=None):
    worker = Path(__file__).parent / "backends/preparation_worker.py"
    python = options["mlx_python"] if backend == "mlx" else options["python_executable"]
    device = "cpu" if options["device"] == "apple" else options["device"]
    command = [python, str(worker), "--backend", backend,
               "--model", options["model"], "--device", device, "--output", str(output),
               "--precision", options["precision"]]
    if backend == "torch":
        command += ["--model-id", resolve_model(options["model"]).model_id]
    else:
        command += ["--checkpoint", str(checkpoint)]
    if reference is not None:
        command += ["--reference", str(reference)]
    _execute(command, Path(options["decoder_dir"]), options.get("timeout"))


def _inspect(options, output):
    """Read the registry artifact header for inventory; do not build or run a model."""
    command = [options["python_executable"], str(Path(__file__).parent / "backends/preparation_worker.py"),
               "--backend", "torch", "--inspect-only", "--model-id", resolve_model(options["model"]).model_id,
               "--model", options["model"], "--output", str(output)]
    _execute(command, Path(options["decoder_dir"]), options.get("timeout"))
    info = InventoryProof.model_validate_json(output.read_text())
    spec = resolve_model(options["model"])
    if (info.model, info.backbone, info.architecture) != (spec.name, spec.backbone, spec.arch):
        raise ValueError("Checkpoint inventory returned a different architecture")
    return info


def _hardware(options):
    metrics = detect_hardware(options["device"])
    if options["device"].startswith("cuda") and metrics.gpu_available_bytes is None:
        # Ask the selected runtime, which respects CUDA_VISIBLE_DEVICES/MIG.
        # Physical nvidia-smi indices cannot safely stand in for logical indices.
        with tempfile.TemporaryDirectory(prefix="tcr-device-") as folder:
            output = Path(folder) / "memory.json"
            code = ("import json,sys,torch; d=torch.device(sys.argv[1]); "
                    "free,total=torch.cuda.mem_get_info(d); "
                    "open(sys.argv[2],'w').write(json.dumps({'free':free,'total':total}))")
            try:
                _execute([options["python_executable"], "-c", code, options["device"], str(output)],
                         Path(options["decoder_dir"]), min(options.get("timeout") or 30, 30))
                values = json.loads(output.read_text())
                metrics = HardwareMetrics.model_validate({**metrics.model_dump(),
                    "gpu_total_bytes": values["total"], "gpu_available_bytes": values["free"],
                    "gpu_device": options["device"] if ":" in options["device"] else "cuda:0",
                    "sources": [*metrics.sources, "selected Torch runtime cuda.mem_get_info"],
                    "warnings": [*metrics.warnings, "VRAM query resolved by the selected Torch runtime."]})
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                metrics = HardwareMetrics.model_validate({**metrics.model_dump(), "warnings": [*metrics.warnings,
                    f"Selected Torch VRAM query failed ({type(exc).__name__}); memory remains unknown."]})
    return metrics


def _resource_report(options, parameter_bytes, *, checkpoint_storage_bytes,
                     needs_preparation, allow_memory_risk, sequence_length):
    common = dict(parameter_bytes=parameter_bytes, allow_memory_risk=allow_memory_risk)
    hardware = _hardware(options)
    plans = {"inference": plan_resources(**common, device=options["device"],
             backend="mlx" if options["device"] == "apple" else "torch",
             batch_size=options.get("batch_size") or 1, sequence_length=sequence_length, hardware=hardware,
             checkpoint_storage_bytes=None if options["device"] == "apple" else checkpoint_storage_bytes)}
    if needs_preparation:
        # Even Apple conversion requires the full FP32 CPU reference. These phases
        # run sequentially, so their memory estimates must not be added together.
        plans["preparation"] = plan_resources(**common,
            device="cpu" if options["device"] == "apple" else options["device"],
            batch_size=2, sequence_length=42, hardware=hardware, checkpoint_storage_bytes=checkpoint_storage_bytes)
    return {name: plan.model_dump(mode="json") for name, plan in plans.items()}


def _enforce_resources(plans):
    for phase, plan in plans.items():
        for warning in plan["warnings"]:
            print(f"Memory check ({phase}): {warning}", file=sys.stderr, flush=True)
        if plan["blocked"]:
            available = plan["available_bytes"]
            detail = f"{available / GIB:.1f} GiB available" if available is not None else "available memory unknown"
            raise ValueError(f"Memory check stopped {phase}: estimated peak {plan['estimated_peak_bytes'] / GIB:.1f} GiB "
                             f"on {plan['memory_pool']} ({detail}); host staging estimate "
                             f"{plan['host_estimated_peak_bytes'] / GIB:.1f} GiB. "
                             + " ".join(plan["suggestions"]) +
                             " Use prepare-model --plan to inspect; --allow-memory-risk explicitly overrides this estimate.")


def _time_estimate(timing, forwards, length, batch_size):
    if forwards is None:
        return None
    if type(forwards) is not int or forwards < 1:
        raise ValueError("--estimate-forwards must be positive")
    if timing is None:
        return {"status": "unavailable", "reason": "No local forward timing yet; prepare this checkpoint first."}
    # Linear projections and quadratic attention bound a rough planning interval.
    # A short synthetic calibration is explicitly not a workload benchmark.
    ratio = length / timing.sequence_length
    rows = forwards / timing.batch_size
    linear = timing.forward_seconds * rows * ratio
    quadratic = linear * ratio
    return {"status": "rough_extrapolation", "context_rows": forwards, "sequence_length": length,
            "requested_batch_size": batch_size, "calibration": timing.model_dump(),
            "lower_seconds": timing.load_seconds + min(linear, quadratic) * 0.25,
            "upper_seconds": timing.load_seconds + max(linear, quadratic) * 4,
            "limitations": "Not a confidence interval or guaranteed workflow ETA. Tiny synthetic warm-up timing; "
            "batch efficiency, compilation, reconstruction, IO and memory pressure vary. "
            "Count distinct model context rows after cache reuse: these workflows mask the whole peptide and reuse "
            "matching receptor/HLA/peptide-length contexts. Benchmark representative inputs for scheduling."}


def _compare(reference, observed, precision):
    with np.load(reference, allow_pickle=False) as first, np.load(observed, allow_pickle=False) as second:
        tokens, candidate_tokens = first["tokens"], second["tokens"]
        expected, actual = first["logits"], second["logits"]
        if (candidate_tokens.dtype.kind not in "iu" or actual.dtype != np.float32
                or not np.array_equal(tokens, candidate_tokens) or expected.shape != actual.shape):
            raise ValueError("Prepared Apple tokenizer/output shape differs from PyTorch")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("Preparation produced non-finite logits")
        expected, actual = expected[tokens != 1].astype(np.float64), actual[tokens != 1].astype(np.float64)
        delta = actual - expected
        error = float(np.max(np.abs(delta)))
        relative = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12))
        norms = np.linalg.norm(expected, axis=1)
        other_norms = np.linalg.norm(actual, axis=1)
        token_error = float(np.max(np.linalg.norm(delta, axis=1) / np.maximum(norms, 1)))
        cosine = np.sum(expected * actual, axis=1) / np.maximum(norms * other_norms, 1e-12)
        cosine[(norms < 1e-12) & (other_norms < 1e-12)] = 1
        min_cosine = float(np.clip(np.min(cosine), -1, 1))
        passed = (np.allclose(actual, expected, atol=1e-4, rtol=1e-4)
                  if precision == "float32" else relative <= 0.01 and token_error <= 0.01 and min_cosine >= 0.999)
        if not passed:
            raise ValueError(f"Apple preparation parity failed: max absolute={error:.6g}, relative L2={relative:.6g}")
        return {"parity_passed": True, "max_absolute_logit_error": error, "relative_l2": relative,
                "max_token_relative_l2": token_error, "min_token_cosine": min_cosine,
                "comparison": "valid token logits; PAD queries excluded",
                "gate": "atol=rtol=1e-4" if precision == "float32" else
                        "global/per-token relative L2 <= 0.01 (token norm floor 1); min token cosine >= 0.999"}


def _cached(path, fingerprint):
    if not path.is_file():
        return None
    try:
        card = PreparedModel.model_validate_json(path.read_text())
        if card.fingerprint != fingerprint:
            return None
        # Torch weights are resolved from the registry and have no local bundle;
        # only Apple keeps a converted bundle directory beside the record.
        expected_checkpoint = str(path.parent / "bundle") if fingerprint["device"] == "apple" else ""
        if card.checkpoint != expected_checkpoint:
            return None
        for name, digest in card.artifacts.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                return None
            artifact = path.parent / relative
            if not artifact.is_file() or file_sha256(artifact) != digest:
                return None
        if not card.artifacts:
            return None
        if fingerprint["device"] == "apple" and not Path(card.checkpoint).exists():
            return None
        if "reference.npz" not in card.artifacts or "reference.npz.json" not in card.artifacts:
            return None
        if fingerprint["device"] == "apple" and (card.checks.parity_passed is not True
                or "apple.npz" not in card.artifacts or "apple.npz.json" not in card.artifacts):
            return None
        proof = TorchProof.model_validate_json((path.parent / "reference.npz.json").read_text())
        if (proof.parameter_bytes != card.checks.parameter_bytes or proof.tensor_count != card.checks.tensor_count
                or proof.checkpoint_storage_bytes != card.checks.checkpoint_storage_bytes
                or proof.model != fingerprint["model"]):
            return None
        if fingerprint["device"] == "apple":
            proof = AppleProof.model_validate_json((path.parent / "apple.npz.json").read_text())
        timing = TimingProof.model_validate({name: getattr(proof, name) for name in TimingProof.model_fields})
        if timing != card.timing:
            return None
        return card
    except (ValueError, OSError):
        return None


def prepare_model(options, state_dir, *, model_id=None, expected_sha256=None, reference_checkpoint=None,
                  plan_only=False, allow_memory_risk=False, sequence_length=2048, estimate_forwards=None):
    """Return effective runtime options after fail-closed local compatibility checks."""
    options = dict(options)
    if type(sequence_length) is not int or not 1 <= sequence_length <= 2048:
        raise ValueError("--sequence-length must be between 1 and 2048 model tokens")
    _time_estimate(None, estimate_forwards, sequence_length, options.get("batch_size") or 1)
    options["device"] = normalize_device(options["device"])
    spec = resolve_model(options["model"])
    options["model"] = spec.name
    # Apple converts its bundle from the registry artifact, so no bundle is required yet.
    validate_backend(spec.name, options["device"], options.get("checkpoint"),
                     options.get("mlx_python"), options["precision"], require_checkpoint=False)
    state = Path(state_dir).absolute()
    fingerprint = _identity(options)
    if expected_sha256 is not None and fingerprint["artifact_sha256"] != expected_sha256:
        raise ValueError("Registry artifact SHA-256 does not match --expected-sha256")
    # The caller's supplied label is part of the record, defaulting to the exact model ID.
    label = model_id or spec.model_id
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    destination = state / key
    card_path = destination / "prepared.json"
    cached = _cached(card_path, fingerprint)
    def cached_result(card):
        plans = _resource_report(options, card.checks.parameter_bytes, needs_preparation=False,
                                 checkpoint_storage_bytes=card.checks.checkpoint_storage_bytes,
                                 allow_memory_risk=allow_memory_risk, sequence_length=sequence_length)
        if not plan_only:
            _enforce_resources(plans)
        effective_checkpoint = card.checkpoint or options.get("checkpoint")
        return dict(options, checkpoint=effective_checkpoint), {"cache_hit": True, "record": str(card_path),
            "plan_only": plan_only, "resources": plans,
            "time_estimate": _time_estimate(card.timing, estimate_forwards, sequence_length, options.get("batch_size") or 1)}
    if cached is not None:
        return cached_result(cached)
    state.mkdir(parents=True, exist_ok=True)
    with _lock(state / (key + ".lock")):
        cached = _cached(card_path, fingerprint)
        if cached is not None:
            return cached_result(cached)
        if destination.exists():
            # Preserve a corrupt/old diagnostic record; do not overwrite it silently.
            raise ValueError(f"Preparation cache failed integrity checks: {destination}. "
                             "Move this directory aside and retry.")
        print(f"Preparing {spec.name} for {options['device']} (one-time checks for this model/runtime)…", file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory(dir=state, prefix="preparing-") as temporary:
            staging = Path(temporary)
            inventory = _inspect(options, staging / "inventory.json")
            plans = _resource_report(options, inventory.parameter_bytes, needs_preparation=True,
                                     checkpoint_storage_bytes=inventory.checkpoint_storage_bytes,
                                     allow_memory_risk=allow_memory_risk, sequence_length=sequence_length)
            if plan_only:
                if _identity(options) != fingerprint:
                    raise ValueError("Registry artifact, code or environment changed during inspection")
                return options, {"cache_hit": False, "plan_only": True, "prepared": False,
                    "artifact_sha256": fingerprint["artifact_sha256"],
                    "inventory": inventory.model_dump(), "resources": plans,
                    "time_estimate": _time_estimate(None, estimate_forwards, sequence_length, options.get("batch_size") or 1)}
            _enforce_resources(plans)
            reference = staging / "reference.npz"
            _fixture(options, reference, backend="torch")
            with np.load(reference, allow_pickle=False) as data:
                data_shape = data["tokens"].shape
                if (data["logits"].ndim != 3 or data["tokens"].shape != data["logits"].shape[:2]
                        or data["tokens"].ndim != 2 or data["tokens"].shape[0] != 2
                        or not 1 <= data["tokens"].shape[1] <= 2048
                        or data["logits"].shape[-1] != (64 if spec.backbone == "esmc" else 33)
                        or data["tokens"].dtype.kind not in "iu"
                        or np.any(data["tokens"] < 0) or np.any(data["tokens"] >= 33)
                        or data["logits"].dtype != np.float32 or not np.isfinite(data["logits"]).all()):
                    raise ValueError("Checkpoint smoke test returned invalid/non-finite logits")
            info = TorchProof.model_validate_json(reference.with_suffix(".npz.json").read_text())
            if (info.model != spec.name or info.device != ("cpu" if options["device"] == "apple" else options["device"])
                    or info.parameter_bytes != inventory.parameter_bytes or info.tensor_count != inventory.tensor_count
                    or info.checkpoint_storage_bytes != inventory.checkpoint_storage_bytes
                    or (info.batch_size, info.sequence_length) != data_shape):
                raise ValueError("Checkpoint worker returned a different model/device")
            checks = {"strict_tensor_inventory": True, "finite_weights": True, "finite_forward": True,
                      "tensor_count": info.tensor_count, "parameter_bytes": info.parameter_bytes,
                      "checkpoint_storage_bytes": info.checkpoint_storage_bytes, "synthetic_token_rows": 2}
            timing = TimingProof.model_validate({name: getattr(info, name) for name in TimingProof.model_fields})
            final_checkpoint = ""
            if options["device"] == "apple":
                size = {"DecoderTCR-ESMC_300M": "300m", "DecoderTCR-ESMC_600M": "600m",
                        "DecoderTCR-ESMC_6B": "6b"}[spec.name]
                artifact = _registry_artifact_path(options, spec.model_id)
                prepared = staging / "bundle"
                code = ("from esmc_mlx.weights import convert_checkpoint; import sys; "
                        "convert_checkpoint(sys.argv[1],sys.argv[2],'decodertcr-safetensors-v1',"
                        "model_id=sys.argv[3],model_size=sys.argv[4])")
                _execute([options["mlx_python"], "-c", code, str(artifact), str(prepared),
                          label, size], Path(options["decoder_dir"]), options.get("timeout"))
                observed = staging / "apple.npz"
                _fixture(options, observed, backend="mlx", checkpoint=prepared, reference=reference)
                checks.update(_compare(reference, observed, options["precision"]))
                info = AppleProof.model_validate_json(observed.with_suffix(".npz.json").read_text())
                if (info.source_sha256 != fingerprint["artifact_sha256"]
                        or info.model != spec.name or info.precision != options["precision"]
                        or (info.batch_size, info.sequence_length) != data_shape):
                    raise ValueError("Apple worker returned a different artifact/model/precision")
                timing = TimingProof.model_validate({name: getattr(info, name) for name in TimingProof.model_fields})
                final_checkpoint = str(destination / "bundle")
            if _identity(options) != fingerprint:
                raise ValueError("Registry artifact, code or environment changed during preparation")
            artifacts = {str(p.relative_to(staging)): file_sha256(p) for p in staging.rglob("*") if p.is_file()}
            card = PreparedModel(fingerprint=fingerprint, checkpoint=final_checkpoint, model_id=label,
                                 checks=checks, artifacts=artifacts, timing=timing)
            _json_write(staging / "prepared.json", card.model_dump())
            staging.rename(destination)
        print(f"Model ready. Compatibility record: {card_path}", file=sys.stderr, flush=True)
        effective_checkpoint = final_checkpoint or options.get("checkpoint")
        return dict(options, checkpoint=effective_checkpoint), {"cache_hit": False, "record": str(card_path),
            "checks": checks, "resources": plans,
            "time_estimate": _time_estimate(timing, estimate_forwards, sequence_length, options.get("batch_size") or 1)}
