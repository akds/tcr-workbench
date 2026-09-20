"""Verified loading and bounded, length-bucketed protein embeddings.

No PyTorch import is needed to load a converted bundle or run inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from collections.abc import Iterable, Iterator

from .precision import verify_precision
import mlx.core as mx
import numpy as np

from .model import ESMC
from .tokenizer import Tokenizer
from .weights import verify_bundle


def load_bundle(path: str | Path) -> tuple[ESMC, Tokenizer, dict]:
    """Verify all bundle hashes and load every parameter strictly on Metal."""
    if not mx.metal.is_available():
        raise RuntimeError("Apple backend requires an accessible Apple Silicon Metal GPU")
    mx.set_default_device(mx.gpu)
    verify_precision()
    config, manifest, weight_path = verify_bundle(path)
    model = ESMC(config)
    model.load_weights(str(weight_path), strict=True)
    model.eval()
    mx.eval(model.parameters())
    tokenizer = Tokenizer(config.tokenizer_variant, config.max_position_embeddings)
    return model, tokenizer, manifest


@dataclass(frozen=True)
class EmbeddingResult:
    index: int
    token_ids: np.ndarray
    positions: np.ndarray
    prenorm: np.ndarray
    postnorm: np.ndarray
    pooled: np.ndarray | None


def iter_embeddings(model: ESMC, tokenizer: Tokenizer, sequences: Iterable[str], *,
                    batch_size: int = 4, max_tokens: int = 4096,
                    bucket_size: int = 8, pool: bool = True) -> Iterator[EmbeddingResult]:
    """Yield input-order embeddings without retaining the complete input/output.

    Batches are length-sorted within bounded windows. Residue positions exclude
    CLS/EOS/PAD; supported mask/unknown/ambiguous residues are retained. ``pooled``
    is an FP32 mean of the postnorm residues (None for zero residues or pool=False).
    Original zero-based token positions and indices make the exclusions explicit.
    This API is for proteins; it does not implicitly pool TCR/HLA complexes.
    ``bucket_size`` bounds retained host outputs independently of ``max_tokens``
    (which bounds each forward). Eight 2046-residue 300M outputs use about 120MiB.
    """
    for name, value in (("batch_size", batch_size), ("max_tokens", max_tokens),
                        ("bucket_size", bucket_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(sequences, (str, bytes)):
        raise TypeError("sequences must be an iterable of protein strings")
    if (tokenizer.variant != model.config.tokenizer_variant
            or tokenizer.max_length != model.config.max_position_embeddings):
        raise ValueError("tokenizer identity/context limit must match the loaded model")
    source = enumerate(sequences)
    while window := list(islice(source, bucket_size)):
        encoded = [(index, tokenizer.encode(sequence)) for index, sequence in window]
        if any(len(ids) > max_tokens for _, ids in encoded):
            raise ValueError("one sequence exceeds max_tokens; increase the explicit batch token budget")
        encoded.sort(key=lambda item: len(item[1]))
        outputs = {}
        offset = 0
        while offset < len(encoded):
            stop = offset + 1
            while (stop < len(encoded) and stop - offset < batch_size
                   and (stop + 1 - offset) * len(encoded[stop][1]) <= max_tokens):
                stop += 1
            batch = encoded[offset:stop]
            ids = np.full((len(batch), len(batch[-1][1])), tokenizer.pad_token_id, np.int32)
            for row, (_, tokens) in enumerate(batch):
                ids[row, :len(tokens)] = tokens
            result = model(mx.array(ids), compute_logits=False)
            mx.eval(result)
            pre, post = np.array(result["prenorm"]), np.array(result["postnorm"])
            for row, (index, _) in enumerate(batch):
                positions = np.flatnonzero(~np.isin(ids[row], [0, 1, 2]))
                selected_pre, selected_post = pre[row, positions], post[row, positions]
                pooled = (selected_post.astype(np.float32).mean(axis=0)
                          if pool and len(positions) else None)
                outputs[index] = EmbeddingResult(index, ids[row, positions].copy(), positions,
                                                 selected_pre, selected_post, pooled)
            offset = stop
        for index, _ in window:
            yield outputs.pop(index)
