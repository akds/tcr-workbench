"""Small source-specific ESM tokenizers with explicitly different unknown policies.

Vocabulary/data conventions follow Biohub ESM (MIT) and DecoderTCR's bundled
Meta ESM alphabet (MIT). See THIRD_PARTY_NOTICES.md. No upstream tokenizer code
is imported; literal special-token and whitespace behavior is oracle-tested.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import TokenizerVariant

BIOHUB_VOCAB = tuple("<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - | <mask>".split())
DECODER_VOCAB = (*BIOHUB_VOCAB[:31], "<null_1>", "<mask>")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    format_version: int = Field(default=1, ge=1, le=1)
    variant: TokenizerVariant
    vocabulary: tuple[str, ...]
    max_length: int = Field(default=2048, ge=2, le=2048)

    @model_validator(mode="after")
    def check_vocabulary(self):
        expected = DECODER_VOCAB if self.variant == "decodertcr-esm1b" else BIOHUB_VOCAB
        if self.vocabulary != expected:
            raise ValueError("tokenizer vocabulary must exactly match the selected source")
        return self


class Tokenizer:
    cls_token_id = 0
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    mask_token_id = 32

    def __init__(self, variant: TokenizerVariant = "biohub-esmc", max_length: int = 2048):
        vocab = DECODER_VOCAB if variant == "decodertcr-esm1b" else BIOHUB_VOCAB
        self.config = TokenizerConfig(variant=variant, vocabulary=vocab, max_length=max_length)
        self.variant = variant
        self.max_length = max_length
        self.vocabulary = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        special = sorted((t for t in vocab if len(t) > 1), key=len, reverse=True)
        self._pattern = re.compile("|".join(re.escape(t) for t in special) + r"|[\s\S]")
        self.amino_acid_ids = tuple(self.token_to_id[aa] for aa in AMINO_ACIDS)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        if not isinstance(sequence, str):
            raise TypeError("sequence must be a string")
        if not isinstance(add_special_tokens, bool):
            raise TypeError("add_special_tokens must be bool")
        ids = []
        residue_limit = self.max_length - (2 if add_special_tokens else 0)
        for match in self._pattern.finditer(sequence):
            token = match.group()
            if self.variant == "decodertcr-esm1b":
                if token.isspace():
                    continue
                if token not in self.token_to_id:
                    raise ValueError(f"Unsupported DecoderTCR alphabet token {token!r}; case is preserved")
                ids.append(self.token_to_id[token])
            else:
                # The official BPE has no normalization/pre-tokenizer and one UNK
                # per unknown Unicode code point (including whitespace/lowercase).
                ids.append(self.token_to_id.get(token, self.unk_token_id))
            if len(ids) > residue_limit:
                raise ValueError(f"Sequence exceeds {self.max_length} total tokens; truncation is never implicit")
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.eos_token_id]
        if len(ids) > self.max_length:
            raise ValueError(f"Sequence has {len(ids)} tokens; maximum is {self.max_length}; truncation is never implicit")
        return ids

    def batch_encode(self, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(sequences, (str, bytes)):
            raise TypeError("batch_encode expects a sequence of strings")
        encoded = [self.encode(sequence) for sequence in sequences]
        if not encoded:
            raise ValueError("batch_encode requires at least one sequence")
        width = max(map(len, encoded))
        ids = np.full((len(encoded), width), self.pad_token_id, dtype=np.int32)
        for index, values in enumerate(encoded):
            ids[index, :len(values)] = values
        return ids, ids != self.pad_token_id


# Descriptive alias for callers that prefer an explicit type name.
SequenceTokenizer = Tokenizer
