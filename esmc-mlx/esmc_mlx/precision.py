"""Explicit matrix precision; M5's default TF32 is not full FP32 parity."""
from __future__ import annotations

import os
import math


def configure_precision():
    # MLX reads this when initializing its matmul subsystem, so set it before
    # importing/using MLX. A numeric check also catches an already initialized
    # embedding application's incompatible subsystem state.
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    if os.environ["MLX_ENABLE_TF32"] != "0":
        raise RuntimeError("ESM-C FP32 inference requires MLX_ENABLE_TF32=0 before starting Python")


def verify_precision():
    import mlx.core as mx
    import numpy as np
    configure_precision()
    value = np.float32(1.0001)
    a = mx.full((128, 960), float(value), dtype=mx.float32)
    b = mx.full((960, 128), float(value), dtype=mx.float32)
    error = mx.max(mx.abs(a @ b - float(960 * float(value) ** 2))).item()
    if not math.isfinite(error) or error > 0.002:
        raise RuntimeError("MLX matmul failed full-FP32 precision check; restart Python with MLX_ENABLE_TF32=0")


configure_precision()
