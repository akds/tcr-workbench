import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from esmc_mlx.config import ModelConfig, config_300m
from esmc_mlx.tokenizer import AMINO_ACIDS, Tokenizer, TokenizerConfig

FIXTURE = json.loads((Path(__file__).parents[1] / "references/tokenizer_oracles.json").read_text())


@pytest.mark.parametrize("variant", ["biohub-esmc", "decodertcr-esm1b"])
def test_exact_oracle_ids(variant):
    tokenizer = Tokenizer(variant)
    for case in FIXTURE["cases"]:
        expected = case[variant]
        if expected is None:
            with pytest.raises(ValueError):
                tokenizer.encode(case["sequence"])
        else:
            assert tokenizer.encode(case["sequence"]) == expected, repr(case["sequence"])


@pytest.mark.parametrize("variant", ["biohub-esmc", "decodertcr-esm1b"])
def test_alphabet_channels_and_special_injection(variant):
    tokenizer = Tokenizer(variant)
    assert tokenizer.encode(AMINO_ACIDS, False) == [5,23,13,9,18,6,21,12,15,4,20,17,14,16,10,8,11,7,22,19]
    assert len(tokenizer.vocabulary) == 33
    assert config_300m("biohub-esmc-published-v1").vocab_size == 64
    assert tokenizer.encode("<mask>", False) == [32]
    assert tokenizer.encode("") == [0, 2]


def test_padding_and_literal_pad_mask():
    tokens, mask = Tokenizer().batch_encode(["A", "AC<pad>D"])
    assert tokens.dtype == np.int32
    assert tokens.tolist() == [[0,5,2,1,1,1], [0,5,23,1,13,2]]
    np.testing.assert_array_equal(mask, tokens != 1)
    with pytest.raises(ValueError):
        Tokenizer().batch_encode([])
    with pytest.raises(TypeError):
        Tokenizer().batch_encode("ACD")


def test_limits_and_rejection_no_implicit_truncation():
    tokenizer = Tokenizer()
    assert len(tokenizer.encode("A" * 2046)) == 2048
    with pytest.raises(ValueError, match="truncation"):
        tokenizer.encode("A" * 2047)
    with pytest.raises(TypeError):
        tokenizer.encode(None)
    with pytest.raises(TypeError):
        tokenizer.encode("A", 1)
    with pytest.raises(ValidationError):
        Tokenizer("unknown")
    with pytest.raises(ValidationError):
        Tokenizer(max_length=4096)
    with pytest.raises(ValidationError):
        TokenizerConfig(variant="biohub-esmc", vocabulary=("bad",))


def test_strict_config_boundary():
    c = config_300m("decodertcr-lightning-v03")
    assert c.head_dim == 64 and c.ffn_hidden == 2560
    for updates in ({"unknown": True}, {"hidden_size": "960"}, {"hidden_size": 961},
                    {"tokenizer_variant": "biohub-esmc"}, {"rope_base": float("nan")},
                    {"dtype": "float16"}, {"vocab_size": 33}):
        with pytest.raises(ValidationError):
            ModelConfig.model_validate(c.model_dump() | updates)
