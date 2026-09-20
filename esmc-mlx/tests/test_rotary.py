import mlx.core as mx
import numpy as np
import pytest

from esmc_mlx.rotary import NativeRoPE


def test_frequencies_exactly_match_pytorch_cpu():
    torch = pytest.importorskip("torch")
    rope = NativeRoPE(64)
    expected = (10000.0 ** (torch.arange(0,64,2,dtype=torch.float32)/64)).numpy()
    np.testing.assert_array_equal(np.asarray(rope._freqs), expected)
    assert not rope.parameters(), "derived frequencies must not become checkpoint weights"


def test_native_explicit_frequencies_match_long_reference():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(123)
    values = rng.normal(size=(2,3,2048,64)).astype(np.float32)
    x = torch.from_numpy(values)
    inverse = 1.0/(10000.0**(torch.arange(0,64,2,dtype=torch.float32)/64))
    angles = torch.outer(torch.arange(2048,dtype=torch.float32),inverse)
    cos, sin = torch.cos(angles), torch.sin(angles)
    a,b = x[...,:32],x[...,32:]
    expected = torch.cat((a*cos-b*sin,a*sin+b*cos),dim=-1).numpy()
    actual = np.asarray(NativeRoPE(64)(mx.array(values)))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dims,base", [(0,10000.0),(3,10000.0),(True,10000.0),(64,float("nan")),(64,1.0)])
def test_invalid_rope_config(dims,base):
    with pytest.raises(ValueError):
        NativeRoPE(dims,base)
