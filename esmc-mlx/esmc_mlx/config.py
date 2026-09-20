"""Versioned architecture contracts; no framework imports at configuration time."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SourceVariant = Literal["biohub-esmc-published-v1", "decodertcr-lightning-v03"]
TokenizerVariant = Literal["biohub-esmc", "decodertcr-esm1b"]
ModelSize = Literal["300m", "600m", "6b"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format_version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    source_variant: SourceVariant
    tokenizer_variant: TokenizerVariant
    hidden_size: int = Field(default=960, ge=2, le=8192)
    num_attention_heads: int = Field(default=15, ge=1, le=128)
    num_hidden_layers: int = Field(default=30, ge=1, le=128)
    intermediate_size: int = Field(default=2560, ge=1, le=32768)
    vocab_size: Literal[64] = 64
    pre_norm_bias: bool = True
    final_norm_bias: bool = False
    qk_norm_bias: bool = False
    layer_norm_eps: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    rope_base: float = Field(default=10000.0, gt=1, allow_inf_nan=False)
    residue_scaling_base: float = Field(default=36.0, gt=0, allow_inf_nan=False)
    max_position_embeddings: int = Field(default=2048, ge=2, le=2048)
    dtype: Literal["float32"] = "float32"
    pad_token_id: Literal[1] = 1
    mask_token_id: Literal[32] = 32

    @model_validator(mode="after")
    def consistent_architecture(self):
        if self.hidden_size % self.num_attention_heads or self.head_dim % 2:
            raise ValueError("hidden_size must divide into even-width attention heads")
        required = {"biohub-esmc-published-v1": "biohub-esmc", "decodertcr-lightning-v03": "decodertcr-esm1b"}
        if required[self.source_variant] != self.tokenizer_variant:
            raise ValueError("tokenizer_variant does not match source_variant")
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def ffn_hidden(self) -> int:
        return self.intermediate_size

    @property
    def residue_scaling_factor(self) -> float:
        return math.sqrt(self.num_hidden_layers / self.residue_scaling_base)


def config_300m(source_variant: SourceVariant) -> ModelConfig:
    """Only two explicitly inventoried 300M source formats are convertible."""
    decoder = source_variant == "decodertcr-lightning-v03"
    return ModelConfig(
        model_id="decodertcr-esmc-300m" if decoder else "esmc-300m",
        source_variant=source_variant,
        tokenizer_variant="decodertcr-esm1b" if decoder else "biohub-esmc",
    )


def config_600m(source_variant: SourceVariant) -> ModelConfig:
    """Pinned DecoderTCR 600M uses the same encoder/tokenizer with larger dimensions."""
    if source_variant != "decodertcr-lightning-v03":
        raise ValueError("600M conversion currently supports DecoderTCR Lightning checkpoints only")
    return ModelConfig(model_id="decodertcr-esmc-600m", source_variant=source_variant,
                       tokenizer_variant="decodertcr-esm1b", hidden_size=1152,
                       num_attention_heads=18, num_hidden_layers=36, intermediate_size=3072)


def config_6b(source_variant: SourceVariant) -> ModelConfig:
    """Source-verified 6B architecture; checkpoint inference needs local preparation.

    This definition has no claim of real-checkpoint parity on the development
    machine. The upstream SwiGLU expansion rounds 8/3 * 2560 up to 6912.
    """
    if source_variant != "decodertcr-lightning-v03":
        raise ValueError("6B conversion currently supports DecoderTCR Lightning checkpoints only")
    return ModelConfig(model_id="decodertcr-esmc-6b", source_variant=source_variant,
                       tokenizer_variant="decodertcr-esm1b", hidden_size=2560,
                       num_attention_heads=40, num_hidden_layers=80, intermediate_size=6912)


def decoder_model_size(model: str) -> ModelSize:
    """Bind a workflow's declared model to its architecture, never to its filename."""
    sizes: dict[str, ModelSize] = {"DecoderTCR-ESMC_300M": "300m", "DecoderTCR-ESMC_600M": "600m",
                                  "DecoderTCR-ESMC_6B": "6b"}
    if model not in sizes:
        raise ValueError("Apple supports DecoderTCR ESM-C 300M, 600M and 6B architectures only")
    return sizes[model]


def validate_decoder_config(config: ModelConfig, model: str) -> None:
    size = decoder_model_size(model)
    factory = {"300m": config_300m, "600m": config_600m, "6b": config_6b}[size]
    expected = factory("decodertcr-lightning-v03")
    if config.model_dump(exclude={"model_id"}) != expected.model_dump(exclude={"model_id"}):
        raise ValueError(f"Apple bundle architecture/tokenizer does not match {model}")
