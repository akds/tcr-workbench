import mlx.core as mx
import numpy as np
import pytest

from esmc_mlx.config import ModelConfig
from esmc_mlx.model import ESMC


def tiny(variant="decodertcr-lightning-v03"):
    return ESMC(ModelConfig(model_id="tiny-test", source_variant=variant,
                           tokenizer_variant="decodertcr-esm1b" if variant.startswith("decoder") else "biohub-esmc",
                           hidden_size=32, num_attention_heads=4, num_hidden_layers=3,
                           intermediate_size=64))


@pytest.mark.parametrize("variant", ["decodertcr-lightning-v03", "biohub-esmc-published-v1"])
def test_position_head_and_padding(variant):
    mx.random.seed(19)
    model = tiny(variant)
    ids = mx.array([[0, 5, 32, 9, 2], [0, 7, 2, 1, 1]])
    full = model(ids, capture=True)
    positions = mx.array([[1, 2], [1, 2]])
    selected = model(ids, positions=positions, output_embeddings=False)
    np.testing.assert_allclose(np.array(selected["logits"]),
                               np.array(full["logits"][:, 1:3]), atol=1e-6)
    single = model(ids[1:2, :3])
    np.testing.assert_allclose(np.array(single["logits"]),
                               np.array(full["logits"][1:2, :3]), atol=2e-5, rtol=1e-5)
    assert set(selected) == {"logits"}
    assert "block0.q_rope" in full["stages"]
    assert full["stages"]["block0.q_rope"].shape == (2, 4, 5, 8)
    assert set(model(ids, compute_logits=False)) == {"prenorm", "postnorm"}


@pytest.mark.parametrize("ids", [mx.array([0, 2]), mx.array([[0., 2.]]),
                                  mx.array([[0, -1]]), mx.array([[0, 64]])])
def test_bad_tokens_fail(ids):
    with pytest.raises(ValueError):
        tiny()(ids)


@pytest.mark.parametrize("positions", [mx.array([[2]]), mx.array([[-1]]),
                                        mx.array([[0.]]), mx.array([[0], [0]])])
def test_bad_positions_fail(positions):
    with pytest.raises(ValueError):
        tiny()(mx.array([[0, 2]]), positions=positions)


def test_all_padding_rows_and_bad_capture_fail():
    with pytest.raises(ValueError, match="non-padding"):
        tiny()(mx.array([[0, 2], [1, 1]]))
    with pytest.raises(ValueError, match="capture"):
        tiny()(mx.array([[0, 2]]), capture={3})


def test_boolean_sdpa_mask_excludes_false_keys():
    # A reviewer incorrectly assumed Boolean masks become additive 0/1 biases.
    # Uniform attention over [1,9] would return 5 without the exclusion.
    q = k = mx.zeros((1, 1, 2, 8), dtype=mx.float32)
    v = mx.concatenate([mx.ones((1, 1, 1, 8)), mx.full((1, 1, 1, 8), 9.)], axis=2)
    mask = mx.array([[[[True, False]]]])
    boolean = mx.fast.scaled_dot_product_attention(q, k, v, scale=8 ** -0.5, mask=mask)
    additive = mx.fast.scaled_dot_product_attention(q, k, v, scale=8 ** -0.5,
                                                   mask=mx.where(mask, 0., -float("inf")))
    np.testing.assert_array_equal(np.array(boolean), np.ones((1, 1, 2, 8)))
    np.testing.assert_array_equal(np.array(boolean), np.array(additive))



def test_runtime_fp16_cast_preserves_storage_config_and_rope_frequencies():
    from mlx.utils import tree_flatten

    mx.random.seed(29)
    model = tiny()
    expected = [np.array(block.attn.rope._freqs) for block in model.transformer.blocks]
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    assert model.config.dtype == "float32"  # Bundle metadata is not runtime precision.
    assert all(value.dtype == mx.float16 for _, value in tree_flatten(model.parameters()))
    for block, frequencies in zip(model.transformer.blocks, expected):
        assert block.attn.rope._freqs.dtype == mx.float32
        np.testing.assert_array_equal(np.array(block.attn.rope._freqs), frequencies)
    ids = mx.array([[0, 5, 32, 9, 2], [0, 7, 32, 2, 1]])
    full = model(ids, output_embeddings=False)["logits"]
    selected = model(ids, positions=mx.array([[1, 2], [1, 2]]), output_embeddings=False)["logits"]
    assert selected.dtype == mx.float16
    np.testing.assert_allclose(np.array(selected), np.array(full[:, 1:3]), atol=0.002, rtol=0.002)
