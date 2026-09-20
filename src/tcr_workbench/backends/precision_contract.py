"""Shared numerical-mode identities; safe in both host and isolated workers."""
from __future__ import annotations

from typing import Literal

Precision = Literal["float32", "float16"]


def precision_metadata(precision):
    if not isinstance(precision, str) or precision not in ("float32", "float16"):
        raise ValueError("precision must be float32 or float16")
    return {"precision": precision, "dtype": precision, "approximate": precision == "float16",
            "math_precision": ("float32-no-tf32" if precision == "float32"
                               else "float16-approximate-fp32-reductions"),
            "bundle_dtype": "float32", "score_reduction_dtype": "float32",
            "rope_frequency_dtype": "float32"}
