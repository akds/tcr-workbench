"""Native MLX RoPE with explicit FP32 frequencies matching the source oracle.

The reference computes 1/(base ** (arange(0,head_dim,2)/head_dim)) in
FP32, then multiplies positions by those inverse frequencies. MLX's native
base-only path uses a different frequency calculation: on the validated M5
this caused growing angle error at longer contexts. Passing explicit frequency
denominators preserves the fused native kernel and removes that discrepancy.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class NativeRoPE(nn.Module):
    def __init__(self, dims: int, base: float = 10000.0):
        super().__init__()
        if not isinstance(dims, int) or isinstance(dims, bool) or dims <= 0 or dims % 2:
            raise ValueError("RoPE dimensions must be a positive even integer")
        if not np.isfinite(base) or base <= 1:
            raise ValueError("RoPE base must be finite and greater than one")
        self.dims = dims
        self.base = base
        exponents = np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims)
        # For the validated 64-wide head / base=10000 this matches all 32
        # PyTorch CPU frequency values bit-for-bit. Leading underscore keeps
        # this derived FP32 constant out of checkpoint parameters/dtype casts.
        self._freqs = mx.array(np.power(np.float32(base), exponents), dtype=mx.float32)

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(x, self.dims, traditional=False, base=None,
                            freqs=self._freqs, scale=1.0, offset=offset)
