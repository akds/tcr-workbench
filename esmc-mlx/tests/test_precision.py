"""Precision guard tests use a fake MLX module: no GPU or model initialization."""
import sys
from types import ModuleType

import numpy as np
import pytest

from esmc_mlx.precision import configure_precision, verify_precision


def test_missing_precision_setting_defaults_to_full_fp32(monkeypatch):
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    configure_precision()
    import os
    assert os.environ["MLX_ENABLE_TF32"] == "0"


@pytest.mark.parametrize("setting", ["1", "true", "", "false"])
def test_incompatible_precision_setting_fails(monkeypatch, setting):
    monkeypatch.setenv("MLX_ENABLE_TF32", setting)
    with pytest.raises(RuntimeError, match="MLX_ENABLE_TF32=0"):
        configure_precision()


def fake_probe(monkeypatch, error):
    value = np.float32(1.0001)
    expected = float(960 * float(value) ** 2)
    class ProbeMatrix:
        def __matmul__(self, other):
            return np.array([expected + error], dtype=np.float64)
    core = ModuleType("mlx.core")
    core.full = lambda *args, **kwargs: ProbeMatrix()
    core.max, core.abs, core.float32 = np.max, np.abs, np.float32
    module = ModuleType("mlx")
    module.core = core
    monkeypatch.setitem(sys.modules, "mlx", module)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


def test_precision_probe_accepts_full_precision(monkeypatch):
    fake_probe(monkeypatch, 0.0005)
    verify_precision()


@pytest.mark.parametrize("error", [0.01, 0.2, float("inf"), float("nan")])
def test_precision_probe_rejects_stale_or_nonfinite_matmul(monkeypatch, error):
    # Environment already says 0, yet an initialized/stale backend can still
    # execute reduced-precision products; the numeric result must decide.
    fake_probe(monkeypatch, error)
    with pytest.raises(RuntimeError, match="precision check"):
        verify_precision()
