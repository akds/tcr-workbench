"""Bounded upstream PyTorch scoring of explicitly reconstructed HLA/TCR sequences.

Weights are resolved from the decodertcr_internal registry by exact model ID; an
already loaded LoadedModel is reused for every row. Context type is explicit and
never inferred from missing TCR columns: absent chains are passed as empty
sequences. Model weights are loaded once, only if at least one row can score.
"""
from __future__ import annotations

import argparse
import csv
from itertools import islice
import json
import math
from time import perf_counter
from pathlib import Path


AA = "ACDEFGHIKLMNPQRSTVWY"


def sequence_entry(row, context_type):
    """Return a prepared five-chain entry; absent chains are explicit empties."""
    entry = {"HLA_a": row["HLA_a"], "HLA_b": row["HLA_b"], "Peptide": row["peptide"],
             "TCR_a": "", "TCR_b": ""}
    if context_type == "tcr-pmhc":
        entry.update(TCR_a=row["TCR_a"], TCR_b=row["TCR_b"])
    return entry


def main():
    parser = argparse.ArgumentParser()
    for name in ("input", "output", "runtime", "model", "device", "model-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--registry")
    parser.add_argument("--context-type", choices=("tcr-pmhc", "pmhc"), required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--profile", action="store_true")
    modes.add_argument("--profile-batch", action="store_true")
    parser.add_argument("--cache-size", type=int, default=256)
    args = parser.parse_args()
    if args.profile_batch and args.context_type != "pmhc":
        raise ValueError("Batched profiles require MHC-only context")
    import torch
    import decodertcr_internal as dt
    from decodertcr_internal import load
    if not 0 <= args.cache_size <= 65536:
        raise ValueError("cache-size must be between zero and 65536 entries")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()
            or (device.index is not None and device.index >= torch.cuda.device_count())):
        raise RuntimeError("requested CUDA device is unavailable; CPU fallback is disabled")
    if device.type not in ("cuda", "cpu"):
        raise ValueError("PyTorch workflow requires cpu or cuda")
    loaded = None
    load_seconds = 0.0
    inference_seconds = 0.0

    def get_model():
        nonlocal loaded, load_seconds
        if loaded is None:
            started = perf_counter()
            loaded = load(args.model_id, device=str(device), registry=args.registry or None)
            loaded.module.eval()
            if any(parameter.is_floating_point() and parameter.dtype != torch.float32
                   for parameter in loaded.module.parameters()):
                raise RuntimeError("PyTorch workflow requires actual float32 model parameters")
            actual_device = next(loaded.module.parameters()).device
            if (actual_device.type != device.type
                    or (device.index is not None and actual_device.index != device.index)):
                raise RuntimeError("upstream model is on a different device; fallback is disabled")
            load_seconds = perf_counter() - started
        return loaded

    counts = {"rows": 0, "scored": 0, "unresolved": 0}
    with open(args.input, newline="", encoding="utf-8") as source, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        if args.profile_batch:
            from profile_batch_worker import write_profiles

            def infer_profile(row):
                nonlocal inference_seconds
                model = get_model()
                started = perf_counter()
                frame = dt.peptide_profile(sequence_entry(row, "pmhc"),
                    len(row["peptide"]), model=model, input_format="prepared")
                inference_seconds += perf_counter() - started
                return frame.reset_index().sort_values("position")[list(AA)].to_numpy()

            counts = write_profiles(reader, target, infer_profile)
        elif args.profile:
            rows = list(reader)
            if len(rows) != 1:
                raise ValueError("profile requires exactly one reconstructed context")
            row = rows[0]
            counts.update(rows=1, reconstruction=row)
            if row["ok"].lower() not in ("true", "1"):
                csv.writer(target).writerow(["position"] + list(AA))
                counts.update(unresolved=1, status="Unresolved",
                              reason="; ".join(row.get(key, "") for key in ("hla_reason", "tcr_reason")))
            else:
                model = get_model()
                started = perf_counter()
                frame = dt.peptide_profile(sequence_entry(row, args.context_type),
                    len(row["peptide"]), model=model, input_format="prepared")
                frame.reset_index()[["position"] + list(AA)].to_csv(target, index=False)
                inference_seconds += perf_counter() - started
                counts.update(scored=1, status="ModelHypothesis")
        else:
            metric = "pll_" + args.model
            writer = csv.DictWriter(target, fieldnames=list(reader.fieldnames) + ["inference_reason", metric])
            writer.writeheader()
            for rows in iter(lambda: list(islice(reader, 128)), []):
                valid = []
                for row in rows:
                    row[metric], row["inference_reason"] = "", ""
                    if row["ok"].lower() in ("true", "1"):
                        entry = sequence_entry(row, args.context_type)
                        length = 2 + sum(len(value) for value in entry.values())
                        if length > 2048:
                            row["ok"] = "False"
                            row["inference_reason"] = f"context has {length} tokens; maximum is 2048"
                        else:
                            valid.append((row, entry))
                if valid:
                    model = get_model()
                    started = perf_counter()
                    scored = model.score([entry for _, entry in valid], mask_region="peptide",
                                         input_format="prepared", cache_size=args.cache_size)
                    scores = scored["pll"].tolist()
                    inference_seconds += perf_counter() - started
                    if len(scores) != len(valid):
                        raise ValueError("upstream returned incorrect score count")
                    for (row, _), score in zip(valid, scores):
                        if math.isfinite(float(score)):
                            row[metric] = float(score)
                            counts["scored"] += 1
                        else:
                            row["ok"] = "False"
                            row["inference_reason"] = "upstream returned a non-finite model score"
                writer.writerows(rows)
                counts["rows"] += len(rows)
            counts["unresolved"] = counts["rows"] - counts["scored"]
    Path(args.runtime).write_text(json.dumps({"backend": "torch", "device": str(device),
        "mode": "profiles" if args.profile_batch else "profile" if args.profile else "scores",
        "context_type": args.context_type, "dtype": "float32", "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "real_model_forwards": counts["scored"],
        "cache_hits": 0,
        "peak_cache_bytes": 0,
        **counts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
