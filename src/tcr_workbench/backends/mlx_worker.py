"""Standalone, torch-free MLX DecoderTCR scoring and profile worker.

Only peptide-position logits leave the device. A byte-bounded LRU shares fully
masked contexts across panel peptides; caches exist for one model invocation.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from itertools import islice
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDS = np.array([5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19])
SEQUENCE_FIELDS = ("HLA_a", "HLA_b", "TCR_a", "TCR_b")


class LogitCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.nbytes = 0
        self.data = OrderedDict()

    def get(self, key):
        value = self.data.get(key)
        if value is not None:
            self.data.move_to_end(key)
        return value

    def put(self, key, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("cached logits must be a NumPy array")
        if not value.flags.owndata:
            value = value.copy()
        # Account for retained sequence keys as well as numeric logits. Python
        # overhead is bounded separately by the maximum number of cache entries.
        size = (sys.getsizeof(value) + sys.getsizeof(key)
                + sum(sys.getsizeof(part) for part in key) + 256)
        if size > self.max_bytes:
            return
        if key in self.data:
            self.nbytes -= self.data.pop(key)[1]
        while self.data and (self.nbytes + size > self.max_bytes or len(self.data) >= 8192):
            self.nbytes -= self.data.popitem(last=False)[1][1]
        self.data[key] = (value, size)
        self.nbytes += size


def context_key(row, context_type="tcr-pmhc"):
    if context_type == "pmhc":
        if row.get("TCR_a") or row.get("TCR_b"):
            raise ValueError("HLA-only context must not contain TCR chains")
        return (row["HLA_a"], row["HLA_b"], "", "", len(row["peptide"]))
    return tuple(row[field] for field in SEQUENCE_FIELDS) + (len(row["peptide"]),)


def masked_tokens(key, tokenizer, max_length, context_type="tcr-pmhc"):
    ha, hb, ta, tb, length = key
    if not all((ha, hb)) or (context_type == "tcr-pmhc" and not all((ta, tb))):
        raise ValueError("reconstruction marked successful but a full chain is empty")
    sequence = ha + hb + "A" * length + ta + tb
    if len(sequence) + 2 > max_length:
        raise ValueError(f"context has {len(sequence) + 2} tokens; maximum is {max_length}")
    tokens = tokenizer.encode(sequence, add_special_tokens=True)
    if len(tokens) != len(sequence) + 2:
        raise ValueError("reconstructed context contains unsupported token syntax")
    positions = np.arange(1 + len(ha) + len(hb), 1 + len(ha) + len(hb) + length)
    tokens = np.asarray(tokens, dtype=np.int32)
    tokens[positions] = 32
    return tokens, positions.astype(np.int32)


def log_softmax(logits):
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 64 or not np.isfinite(values).all():
        raise ValueError("MLX returned invalid or non-finite 64-channel peptide logits")
    maximum = values.max(axis=1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def score_peptides(peptides, logits):
    """Vectorized PLL, with all 64 channels retained in the normalizer."""
    log_probs = log_softmax(logits)
    if not peptides or any(len(peptide) != len(log_probs) for peptide in peptides):
        raise ValueError("peptide lengths must match the masked logit positions")
    lookup = np.full(256, -1, dtype=np.int32)
    lookup[np.frombuffer(AA.encode(), dtype=np.uint8)] = AA_IDS
    encoded = np.frombuffer("".join(peptides).encode("ascii"), dtype=np.uint8)
    ids = lookup[encoded].reshape(len(peptides), len(log_probs))
    if (ids < 0).any():
        raise ValueError("peptides must contain only standard amino acids")
    return log_probs[np.arange(len(log_probs))[None, :], ids].mean(axis=1, dtype=np.float32)


def profile_probabilities(logits):
    values = np.asarray(logits, dtype=np.float32)
    log_softmax(values)  # Validate all channels, including the unreported tail.
    values = values[:, AA_IDS]
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def process_rows(reader, writer, *, infer, tokenizer, max_length, batch_size,
                 token_budget, cache_bytes, chunk_size=1024, context_type="tcr-pmhc"):
    cache = LogitCache(cache_bytes)
    counts = {"rows": 0, "scored": 0, "unresolved": 0, "unique_context_forwards": 0,
              "cache_hits": 0, "batches": 0, "peak_cache_bytes": 0}
    for rows in iter(lambda: list(islice(reader, chunk_size)), []):
        groups = {}
        pending = []
        for index, row in enumerate(rows):
            row["_score"] = ""
            row["inference_reason"] = ""
            if row.get("ok", "").lower() not in ("true", "1"):
                continue
            key = context_key(row, context_type)
            groups.setdefault(key, []).append(index)

        def assign_scores(key, logits):
            indices = groups[key]
            scores = score_peptides([rows[index]["peptide"] for index in indices], logits)
            for index, score in zip(indices, scores):
                rows[index]["_score"] = float(score)
                counts["scored"] += 1

        for key, indices in groups.items():
            cached = cache.get(key)
            if cached is not None:
                assign_scores(key, cached[0])
                counts["cache_hits"] += len(indices)
                continue
            try:
                tokens, positions = masked_tokens(key, tokenizer, max_length, context_type)
            except ValueError as exc:
                for index in indices:
                    rows[index]["ok"] = "False"
                    rows[index]["inference_reason"] = str(exc)
                continue
            if len(tokens) > token_budget:
                raise ValueError(f"--token-budget {token_budget} is below context length {len(tokens)}")
            pending.append((key, tokens, positions))
        # Bucket by peptide length (position dimensions) then context length.
        pending.sort(key=lambda entry: (entry[0][-1], len(entry[1])))
        offset = 0
        while offset < len(pending):
            batch = [pending[offset]]
            offset += 1
            while offset < len(pending) and len(batch) < batch_size:
                next_entry = pending[offset]
                if (next_entry[0][-1] != batch[0][0][-1]
                        or (len(batch) + 1) * len(next_entry[1]) > token_budget):
                    break
                batch.append(next_entry)
                offset += 1
            length = max(len(entry[1]) for entry in batch)
            tokens = np.full((len(batch), length), 1, dtype=np.int32)
            positions = np.stack([entry[2] for entry in batch])
            for index, (_, sequence, _) in enumerate(batch):
                tokens[index, :len(sequence)] = sequence
            logits = np.asarray(infer(tokens, positions), dtype=np.float32)
            if logits.shape != (len(batch), positions.shape[1], 64):
                raise ValueError("MLX returned incorrect peptide logit dimensions")
            counts["batches"] += 1
            counts["unique_context_forwards"] += len(batch)
            for index, (key, _, _) in enumerate(batch):
                # Copy out each small logit slice; never retain a large parent batch.
                value = logits[index].copy()
                # Consume this batch immediately. Keeping all chunk logits in a
                # second dictionary defeats cache eviction for long peptides.
                assign_scores(key, value)
                cache.put(key, value)
                counts["peak_cache_bytes"] = max(counts["peak_cache_bytes"], cache.nbytes)
        for row in rows:
            writer(row)
        counts["rows"] += len(rows)
    counts["unresolved"] = counts["rows"] - counts["scored"]
    return counts


def prepare_model_precision(model, precision, mx, flatten):
    """Cast verified runtime parameters only, keeping derived RoPE constants FP32."""
    if __package__:
        from .precision_contract import precision_metadata
    else:
        # Isolated Python (-I) excludes the script directory from sys.path.
        # Load only this audited sibling; never add arbitrary import roots.
        import runpy
        precision_metadata = runpy.run_path(str(Path(__file__).with_name("precision_contract.py")))["precision_metadata"]
    numerical = precision_metadata(precision)
    if precision == "float16":
        model.set_dtype(mx.float16)
        mx.eval(model.parameters())
        # Release inactive FP32 initialization/cast buffers once. The process
        # high-water mark still includes startup; never clear per forward.
        mx.clear_cache()
    parameters = flatten(model.parameters())
    if not parameters or any(value.dtype != getattr(mx, precision) for _, value in parameters):
        raise RuntimeError("MLX model parameter dtype differs from requested precision")
    if any(block.attn.rope._freqs.dtype != mx.float32 for block in model.transformer.blocks):
        raise RuntimeError("RoPE frequency constants must remain float32")
    return numerical


def main():
    parser = argparse.ArgumentParser()
    for name in ("input", "output", "bundle", "runtime", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--cache-bytes", type=int, default=64 * 1024 * 1024)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--profile", action="store_true")
    modes.add_argument("--profile-batch", action="store_true")
    parser.add_argument("--precision", choices=("float32", "float16"), default="float32")
    parser.add_argument("--context-type", choices=("tcr-pmhc", "pmhc"), default="tcr-pmhc")
    args = parser.parse_args()
    if args.profile_batch and args.context_type != "pmhc":
        raise ValueError("Batched profiles require MHC-only context")
    if args.batch_size < 1 or args.token_budget < 3 or args.cache_bytes < 0:
        raise ValueError("invalid batch, token budget or cache size")
    os.environ["MLX_ENABLE_TF32"] = "0"
    import mlx.core as mx
    from esmc_mlx.inference import load_bundle
    from esmc_mlx.config import decoder_model_size, validate_decoder_config
    from mlx.utils import tree_flatten
    if not mx.metal.is_available():
        raise RuntimeError("Apple backend requires MLX Metal; CPU fallback is disabled")
    decoder_model_size(args.model)
    if args.model != "DecoderTCR-ESMC_300M" and args.precision != "float32":
        raise ValueError(f"Apple {args.model} currently requires float32 precision")
    mx.set_default_device(mx.gpu)
    load_started = perf_counter()
    model, tokenizer, manifest = load_bundle(args.bundle)
    config = model.config
    validate_decoder_config(config, args.model)
    if tokenizer.encode(AA, add_special_tokens=False) != AA_IDS.tolist():
        raise ValueError("DecoderTCR tokenizer amino-acid channel mapping differs")

    numerical = prepare_model_precision(model, args.precision, mx, tree_flatten)
    load_seconds = perf_counter() - load_started
    inference_seconds = 0.0

    def infer(tokens, positions):
        nonlocal inference_seconds
        started = perf_counter()
        result = model(mx.array(tokens), positions=mx.array(positions), output_embeddings=False)
        logits = result["logits"].astype(mx.float32)
        mx.eval(logits)
        values = np.asarray(logits)
        inference_seconds += perf_counter() - started
        return values

    with open(args.input, newline="", encoding="utf-8") as source, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        if args.profile_batch:
            import runpy
            write_profiles = runpy.run_path(str(Path(__file__).with_name("profile_batch_worker.py")))["write_profiles"]

            def infer_profile(row):
                tokens, positions = masked_tokens(context_key(row, "pmhc"), tokenizer,
                                                  config.max_position_embeddings, "pmhc")
                if len(tokens) > args.token_budget:
                    raise ValueError("--token-budget is below profile context length")
                return profile_probabilities(infer(tokens[None], positions[None])[0])

            counts = write_profiles(reader, target, infer_profile)
            counts["unique_context_forwards"] = counts["scored"]
        elif args.profile:
            rows = list(reader)
            if len(rows) != 1:
                raise ValueError("profile requires exactly one HLA context")
            if rows[0].get("ok", "").lower() not in ("true", "1"):
                reasons = [str(row.get(key, "")) for row in rows
                           for key in ("tcr_reason", "hla_reason")]
                csv.writer(target).writerow(["position"] + list(AA))
                counts = {"rows": 1, "scored": 0, "unresolved": 1, "unique_context_forwards": 0,
                          "reconstruction": rows[0], "status": "Unresolved",
                          "reason": "; ".join(reason for reason in reasons if reason) or "reconstruction failed"}
            else:
                tokens, positions = masked_tokens(context_key(rows[0], args.context_type), tokenizer,
                                                  config.max_position_embeddings, args.context_type)
                if len(tokens) > args.token_budget:
                    raise ValueError("--token-budget is below profile context length")
                logits = infer(tokens[None], positions[None])[0]
                values = profile_probabilities(logits)
                writer = csv.writer(target)
                writer.writerow(["position"] + list(AA))
                for position, probabilities in enumerate(values, start=1):
                    writer.writerow([position] + probabilities.tolist())
                counts = {"rows": 1, "scored": 1, "unique_context_forwards": 1,
                          "reconstruction": rows[0], "status": "ModelHypothesis"}
        else:
            metric = "pll_" + args.model
            fields = list(reader.fieldnames) + ["inference_reason", metric]
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()

            def write_row(row):
                row[metric] = row.pop("_score")
                writer.writerow(row)

            counts = process_rows(reader, write_row, infer=infer, tokenizer=tokenizer,
                                  max_length=config.max_position_embeddings,
                                  batch_size=args.batch_size, token_budget=args.token_budget,
                                  cache_bytes=args.cache_bytes, context_type=args.context_type)
    runtime = {"backend": "mlx", "device": "apple", **numerical,
               "mode": "profiles" if args.profile_batch else "profile" if args.profile else "scores",
               "context_type": args.context_type,
               "tokenizer_variant": config.tokenizer_variant,
               "source_checkpoint_sha256": manifest["source_sha256"],
               "model_load_seconds": load_seconds, "inference_seconds": inference_seconds,
               "mlx_peak_memory_bytes": mx.get_peak_memory(), **counts}
    Path(args.runtime).write_text(json.dumps(runtime, indent=2) + "\n")


if __name__ == "__main__":
    main()
