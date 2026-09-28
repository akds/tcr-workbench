"""One-time strict registry-artifact inspection and tiny synthetic forward checks.

Weights are resolved from the decodertcr_internal registry by exact model ID. The
torch backend loads the released artifact and runs a tiny synthetic forward as the
Apple parity reference; the mlx backend runs the converted bundle.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


def _artifact_inventory(model_id):
    """Return tensor/byte inventory and identity from the registry safetensors header."""
    from decodertcr_internal.registry import Registry
    from safetensors import safe_open

    resolved = Registry(None).resolve(model_id)
    weights = resolved.bundle / resolved.member["weights"]["path"]
    tensor_count = 0
    parameter_bytes = 0
    with safe_open(str(weights), framework="np") as reader:
        for key in reader.keys():
            record = reader.get_slice(key)
            shape = record.get_shape()
            dtype = np.dtype(record.get_dtype())
            tensor_count += 1
            parameter_bytes += int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    # The on-disk artifact bytes replace the former optimizer-inclusive ckpt storage.
    storage_bytes = int(Path(weights).stat().st_size)
    return tensor_count, parameter_bytes, storage_bytes, resolved.member["backbone"], resolved.member["arch"]


def torch_fixture(args):
    import torch
    from decodertcr_internal import load
    from decodertcr_internal.constants import ALPHABET

    tensor_count, parameter_bytes, storage_bytes, backbone, arch = _artifact_inventory(args.model_id)
    if args.inspect_only:
        return {"model": args.model, "tensor_count": tensor_count, "parameter_bytes": parameter_bytes,
                "checkpoint_storage_bytes": storage_bytes,
                "backbone": backbone, "architecture": arch,
                "finite_values_checked": False, "forward_checked": False}
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()
            or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError("Requested CUDA device is unavailable; no CPU fallback")
    started = perf_counter()
    loaded = load(args.model_id, device=str(device), registry=None)
    model = loaded.module
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Reference model must use float32 parameters")
    first = next(model.parameters()).device
    if first.type != device.type or (device.index is not None and first.index != device.index):
        raise ValueError("Model loaded on a different device")
    # Scan parameters for finite values in bounded chunks.
    for tensor in model.parameters():
        if tensor.is_floating_point():
            for chunk in tensor.detach().view(-1).split(1024 * 1024):
                if not torch.isfinite(chunk).all():
                    raise ValueError("Non-finite model parameters")
    load_seconds = perf_counter() - started
    # Real alphabet, mask, EOS and a padded mixed-length batch. No donor data.
    _, _, tokens = ALPHABET.get_batch_converter()([
        ("short", "ACDEFGHIK<mask>LMNPQRSTVWY"),
        ("long", "MNPQRSTVWYACDEFGHIKLMNPQRSTVWY<mask>ACDEFGHIK"),
    ])
    started = perf_counter()
    with torch.inference_mode():
        output = model(tokens.to(device))
    logits = output["logits"].detach().float().cpu().numpy()
    forward_seconds = perf_counter() - started
    if not np.isfinite(logits).all():
        raise ValueError("Reference forward produced non-finite logits")
    return tokens.numpy(), logits, {"tensor_count": tensor_count, "parameter_bytes": parameter_bytes,
        "checkpoint_storage_bytes": storage_bytes,
        "backend": "torch", "device": str(device), "load_seconds": load_seconds,
        "forward_seconds": forward_seconds, "batch_size": len(tokens), "sequence_length": tokens.shape[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "mlx"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-id")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--precision", default="float32", choices=("float32", "float16"))
    args = parser.parse_args()
    if args.backend == "torch":
        if not args.model_id:
            raise ValueError("Torch preparation requires --model-id")
        if args.inspect_only:
            Path(args.output).write_text(json.dumps(torch_fixture(args), indent=2) + "\n")
            return
        tokens, logits, info = torch_fixture(args)
    else:
        if not args.checkpoint:
            raise ValueError("MLX preparation requires the converted --checkpoint bundle")
        os.environ["MLX_ENABLE_TF32"] = "0"
        import mlx.core as mx
        from esmc_mlx.inference import load_bundle
        from mlx.utils import tree_flatten
        from mlx_worker import prepare_model_precision
        if not mx.metal.is_available():
            raise ValueError("MLX Metal is unavailable; no CPU fallback")
        mx.set_default_device(mx.gpu)
        started = perf_counter()
        model, tokenizer, manifest = load_bundle(args.checkpoint)
        from esmc_mlx.config import validate_decoder_config
        validate_decoder_config(model.config, args.model)
        with np.load(args.reference, allow_pickle=False) as data:
            tokens = data["tokens"]
        rows = ["ACDEFGHIK<mask>LMNPQRSTVWY", "MNPQRSTVWYACDEFGHIKLMNPQRSTVWY<mask>ACDEFGHIK"]
        for row, sequence in zip(tokens, rows):
            if tokenizer.encode(sequence) != row[row != 1].tolist():
                raise ValueError("Apple and reference tokenizer outputs differ")
        prepare_model_precision(model, args.precision, mx, tree_flatten)
        load_seconds = perf_counter() - started
        started = perf_counter()
        result = model(mx.array(tokens), output_embeddings=False)
        mx.eval(result["logits"])
        logits = np.asarray(result["logits"].astype(mx.float32))
        forward_seconds = perf_counter() - started
        if not np.isfinite(logits).all():
            raise ValueError("Apple forward produced non-finite logits")
        info = {"backend": "mlx", "device": "apple", "precision": args.precision,
                "source_sha256": manifest["source_sha256"], "load_seconds": load_seconds,
                "forward_seconds": forward_seconds, "batch_size": len(tokens), "sequence_length": tokens.shape[1]}
    info["model"] = args.model
    np.savez(args.output, tokens=tokens, logits=logits)
    Path(args.output + ".json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
