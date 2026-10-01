"""Model identity is independent of inference hardware; unknown choices fail closed."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from .backends.precision_contract import precision_metadata


@dataclass(frozen=True)
class DecoderModel:
    name: str
    backbone: str
    arch: str
    model_id: str
    convention: str = "v2"
    apple_supported: bool = False
    # Optional HuggingFace source for the released fp32 safetensors. When set,
    # `--weight-source huggingface` loads these weights via decodertcr_internal
    # from_pretrained instead of the shared internal registry. The file name is
    # the flat safetensors published beside a self-describing config.json.
    hf_repo: Optional[str] = None
    hf_file: str = "model.safetensors"


MODELS = {
    name: DecoderModel(name, backbone, arch, model_id, convention, apple, hf_repo)
    for name, backbone, arch, model_id, convention, apple, hf_repo in [
        ("DecoderTCR-ESMC_300M", "esmc", "DecoderTCRC_300M",
         "decodertcr@1.5.0:300M", "v2", True, "denilau17/DecoderTCR-ESMC-300M-test"),
        ("DecoderTCR-ESMC_600M", "esmc", "DecoderTCRC_600M",
         "decodertcr@1.5.0:600M", "v2", True, None),
        ("DecoderTCR-ESMC_6B", "esmc", "DecoderTCRC_6B",
         "decodertcr@1.5.0:6B", "v2", True, None),
    ]
}
WEIGHT_SOURCES = ("registry", "huggingface")
ALIASES = {"esmc-300m": "DecoderTCR-ESMC_300M", "esmc-600m": "DecoderTCR-ESMC_600M",
           "esmc-6b": "DecoderTCR-ESMC_6B"}


def resolve_model(name: str) -> DecoderModel:
    canonical = ALIASES.get(name.lower(), name) if isinstance(name, str) else name
    if canonical not in MODELS:
        raise ValueError(f"unsupported model {name}; choose one of {sorted(MODELS)}")
    return MODELS[canonical]


def registry_model_id(name: str) -> str:
    """Return the decodertcr_internal registry model ID for a Workbench model name."""
    return resolve_model(name).model_id


def normalize_weight_source(source) -> str:
    """Weights come from the internal registry by default, or HuggingFace on request."""
    value = "registry" if source is None else str(source).strip().lower()
    if value not in WEIGHT_SOURCES:
        raise ValueError(f"unsupported weight source {source!r}; choose one of {WEIGHT_SOURCES}")
    return value


def huggingface_repo(name: str, override: Optional[str] = None) -> str:
    """Resolve the HuggingFace repo id for a model; an explicit override wins.

    Fails closed when a model has no published HuggingFace source and none is
    supplied, rather than silently falling back to the internal registry.
    """
    repo = (override or "").strip() or resolve_model(name).hf_repo
    if not repo:
        raise ValueError(
            f"model {name} has no HuggingFace source configured; use --weight-source registry "
            "or pass --hf-repo with an explicit repository id"
        )
    return repo


def normalize_device(device: str) -> str:
    if not isinstance(device, str):
        raise ValueError("device must be cpu, gpu, cuda, cuda:N, or apple")
    value = device.strip().lower()
    if value in ("apple", "mlx"):
        return "apple"
    if value == "gpu":
        return "cuda"
    if value == "cpu" or re.fullmatch(r"cuda(?::(?:0|[1-9][0-9]*))?", value):
        return value
    raise ValueError("unsupported device; use cpu, gpu, cuda, cuda:N, or apple (MLX Metal)")


def validate_precision(device: str, precision: str) -> str:
    precision_metadata(precision)
    if precision == "float16" and device != "apple":
        raise ValueError("float16 precision is supported only on Apple ESM-C 300M; use --precision float32 for CPU/CUDA")
    return precision


def validate_backend(model: str, device: str, checkpoint, mlx_python, precision: str = "float32",
                     *, require_checkpoint: bool = True) -> DecoderModel:
    validate_precision(device, precision)
    spec = resolve_model(model)
    if device == "apple":
        if not spec.apple_supported:
            raise ValueError(f"Apple MLX supports ESM-C 300M, 600M and 6B architectures only, not {spec.name}")
        if spec.name != "DecoderTCR-ESMC_300M" and precision != "float32":
            raise ValueError(f"Apple {spec.name} currently requires float32 precision")
        if mlx_python is None:
            raise ValueError("Apple MLX requires --mlx-python PYTHON")
        # Preparation converts the bundle from the registry artifact, so the bundle
        # need not exist yet; running inference still requires the converted bundle.
        if require_checkpoint and checkpoint is None:
            raise ValueError("Apple MLX requires --checkpoint CONVERTED_BUNDLE and --mlx-python PYTHON")
    else:
        # CPU/CUDA always load the registry (or --weight-source huggingface) weights
        # selected by --model. A checkpoint is silently ignored there, so reject it
        # rather than score with weights the configuration did not actually choose.
        if checkpoint is not None:
            raise ValueError("--checkpoint applies only to --device apple; CPU/CUDA load the registry "
                             "(or --weight-source huggingface) weights selected by --model. Remove "
                             "--checkpoint and clear it from any saved settings.")
        if mlx_python is not None:
            raise ValueError("--mlx-python applies only to --device apple")
    return spec
