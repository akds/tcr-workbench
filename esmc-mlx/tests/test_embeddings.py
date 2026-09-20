import numpy as np
import pytest

from esmc_mlx.inference import iter_embeddings
from esmc_mlx.tokenizer import Tokenizer
from test_model import tiny


def test_buckets_restore_order_and_pool_only_residues():
    model = tiny()
    tokenizer = Tokenizer("decodertcr-esm1b")
    sequences = ["ACDEFG", "A", "", "W<mask>Q", "CKLM"]
    results = list(iter_embeddings(model, tokenizer, iter(sequences), batch_size=2,
                                   max_tokens=20, bucket_size=3))
    assert [r.index for r in results] == list(range(len(sequences)))
    for sequence, result in zip(sequences, results):
        ids = tokenizer.encode(sequence)
        expected_positions = [p for p, token in enumerate(ids) if token not in {0, 1, 2}]
        assert result.positions.tolist() == expected_positions
        assert result.token_ids.tolist() == [ids[p] for p in expected_positions]
        assert result.prenorm.shape == result.postnorm.shape == (len(expected_positions), 32)
        if expected_positions:
            np.testing.assert_allclose(result.pooled, result.postnorm.mean(axis=0), atol=1e-6)
        else:
            assert result.pooled is None


def test_bounded_window_does_not_consume_all_input():
    consumed = []
    def sequences():
        for i in range(20):
            consumed.append(i)
            yield "AC"
    iterator = iter_embeddings(tiny(), Tokenizer("decodertcr-esm1b"), sequences(), bucket_size=3)
    assert next(iterator).index == 0
    assert len(consumed) == 3
    iterator.close()


@pytest.mark.parametrize("kwargs", [{"max_tokens": 2}, {"batch_size": 0},
                                     {"bucket_size": True}])
def test_invalid_budgets_fail(kwargs):
    with pytest.raises(ValueError):
        list(iter_embeddings(tiny(), Tokenizer(), ["A"], **kwargs))


def test_tokenizer_mismatch_fails_before_forward():
    with pytest.raises(ValueError, match="tokenizer identity"):
        list(iter_embeddings(tiny(), Tokenizer("biohub-esmc"), ["A|C"]))
    with pytest.raises(ValueError, match="tokenizer identity"):
        list(iter_embeddings(tiny(), Tokenizer("decodertcr-esm1b", max_length=16), ["AC"]))
