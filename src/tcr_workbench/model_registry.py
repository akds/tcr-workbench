"""Model identity is independent of inference hardware; unknown choices fail closed."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .backends.precision_contract import precision_metadata


@dataclass(frozen=True)
class DecoderModel:
    name: str
    backbone: str
    arch: str
    checkpoint: str
    apple_supported: bool = False


MODELS = {
    name: DecoderModel(name, backbone, arch, path, apple)
    for name, backbone, arch, path, apple in [
        ("DecoderTCR-ESMC_300M", "esmc", "DecoderTCRC_300M",
         "checkpoints/DecoderTCR-ESMC-V0.3/300M.ckpt", True),
        ("DecoderTCR-ESMC_600M", "esmc", "DecoderTCRC_600M",
         "checkpoints/DecoderTCR-ESMC-V0.3/600M.ckpt", True),
        ("DecoderTCR-ESMC_6B", "esmc", "DecoderTCRC_6B",
         "checkpoints/DecoderTCR-ESMC-V0.3/6B.ckpt", True),
        ("DecoderTCR_650M", "esm2", "ESM2_650M",
         "checkpoints/DecoderTCR-ESM2-V0.1/650M_DecoderTCR.ckpt", False),
        ("DecoderTCR_3B", "esm2", "ESM2_3B",
         "checkpoints/DecoderTCR-ESM2-V0.1/3B_DecoderTCR.ckpt", False),
    ]
}
ALIASES = {"esmc-300m": "DecoderTCR-ESMC_300M", "esmc-600m": "DecoderTCR-ESMC_600M",
           "esmc-6b": "DecoderTCR-ESMC_6B", "esm2-650m": "DecoderTCR_650M",
           "esm2-3b": "DecoderTCR_3B"}


def resolve_model(name: str) -> DecoderModel:
    canonical = ALIASES.get(name.lower(), name) if isinstance(name, str) else name
    if canonical not in MODELS:
        raise ValueError(f"unsupported model {name}; choose one of {sorted(MODELS)}")
    return MODELS[canonical]


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


def validate_backend(model: str, device: str, checkpoint, mlx_python, precision: str = "float32") -> DecoderModel:
    validate_precision(device, precision)
    spec = resolve_model(model)
    if device == "apple":
        if not spec.apple_supported:
            raise ValueError(f"Apple MLX supports ESM-C 300M, 600M and 6B architectures only, not {spec.name}")
        if spec.name != "DecoderTCR-ESMC_300M" and precision != "float32":
            raise ValueError(f"Apple {spec.name} currently requires float32 precision")
        if checkpoint is None or mlx_python is None:
            raise ValueError("Apple MLX requires --checkpoint CONVERTED_BUNDLE and --mlx-python PYTHON")
    elif mlx_python is not None:
        raise ValueError("--mlx-python applies only to --device apple")
    return spec
