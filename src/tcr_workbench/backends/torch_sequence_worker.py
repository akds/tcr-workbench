"""Bounded upstream PyTorch scoring of explicitly reconstructed HLA/TCR sequences.

An already loaded module is passed to DecoderTCR.score; this avoids its implicit
CUDA-to-CPU fallback. Context type is explicit and never inferred from missing
TCR columns. Model weights are loaded once, only if at least one row can score.
"""
from __future__ import annotations

import argparse
import csv
from itertools import islice
import json
import math
from pathlib import Path
from time import perf_counter


AA = "ACDEFGHIKLMNPQRSTVWY"


def sequence_entry(row, context_type):
    entry = {"HLA_a": row["HLA_a"], "HLA_b": row["HLA_b"], "Peptide": row["peptide"]}
    if context_type == "tcr-pmhc":
        entry.update(TCR_a=row["TCR_a"], TCR_b=row["TCR_b"])
    return entry


def main():
    parser = argparse.ArgumentParser()
    for name in ("input", "output", "runtime", "model", "device"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--context-type", choices=("tcr-pmhc", "pmhc"), required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--profile", action="store_true")
    modes.add_argument("--profile-batch", action="store_true")
    parser.add_argument("--cache-bytes", type=int, default=67108864)
    args = parser.parse_args()
    if args.profile_batch and args.context_type != "pmhc":
        raise ValueError("Batched profiles require MHC-only context")
    import torch
    import DecoderTCR as dt
    from DecoderTCR.utils.model_zoo import load, resolve
    from torch_checkpoint_worker import validate_inventory
    from torch_cache import wrap_scoring_model
    if not 0 <= args.cache_bytes <= 1024 ** 3:
        raise ValueError("cache-bytes must be between zero and 1GiB")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()
            or (device.index is not None and device.index >= torch.cuda.device_count())):
        raise RuntimeError("requested CUDA device is unavailable; CPU fallback is disabled")
    if device.type not in ("cuda", "cpu"):
        raise ValueError("PyTorch workflow requires cpu or cuda")
    spec = resolve(args.model)
    checkpoint = Path(args.checkpoint) if args.checkpoint else spec.ckpt_path
    model = None
    layers = None
    load_seconds = 0.0
    inference_seconds = 0.0
    cache = None

    def get_model():
        nonlocal model, layers, load_seconds, cache
        if model is None:
            started = perf_counter()
            inventory = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            validate_inventory(inventory, spec.arch)
            del inventory
            model, layers = load(args.model, device=device, checkpoint=checkpoint,
                                 backbone=spec.backbone, arch=spec.arch)
            model.eval()
            if any(parameter.is_floating_point() and parameter.dtype != torch.float32
                   for parameter in model.parameters()):
                raise RuntimeError("PyTorch workflow requires actual float32 model parameters")
            actual_device = next(model.parameters()).device
            if (actual_device.type != device.type
                    or (device.index is not None and actual_device.index != device.index)):
                raise RuntimeError("upstream model is on a different device; fallback is disabled")
            if not (args.profile or args.profile_batch):
                model, cache = wrap_scoring_model(model, args.cache_bytes, torch)
            load_seconds = perf_counter() - started
        return model, layers

    counts = {"rows": 0, "scored": 0, "unresolved": 0}
    with open(args.input, newline="", encoding="utf-8") as source, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        if args.profile_batch:
            from profile_batch_worker import write_profiles

            def infer_profile(row):
                nonlocal inference_seconds
                mdl, n = get_model()
                started = perf_counter()
                with torch.inference_mode():
                    frame = dt.peptide_profile(sequence_entry(row, "pmhc"),
                        length=len(row["peptide"]), model=mdl, num_layers=n,
                        device=device, from_genes=False)
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
                mdl, n = get_model()
                started = perf_counter()
                with torch.inference_mode():
                    frame = dt.peptide_profile(sequence_entry(row, args.context_type),
                        length=len(row["peptide"]), model=mdl, num_layers=n,
                        device=device, from_genes=False)
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
                    mdl, n = get_model()
                    started = perf_counter()
                    with torch.inference_mode():
                        scores = dt.score([entry for _, entry in valid], model=mdl, num_layers=n,
                            device=device, with_tcr=args.context_type == "tcr-pmhc", return_dataframe=False)
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
        "real_model_forwards": cache.forwards if cache is not None else counts["scored"],
        "cache_hits": cache.hits if cache is not None else 0,
        "peak_cache_bytes": cache.peak_bytes if cache is not None else 0,
        **counts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
