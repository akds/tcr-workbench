"""One-time strict checkpoint inspection and tiny synthetic forward checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


def checkpoint_storage_bytes(value, torch):
    """Count distinct mapped storages, including optimizer state, without reading values."""
    containers, storages = set(), set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages.add((storage.data_ptr(), storage.nbytes()))
        elif isinstance(item, (dict, list, tuple)) and id(item) not in containers:
            containers.add(id(item))
            pending.extend(item.values() if isinstance(item, dict) else item)
    return sum(size for _, size in storages)


def torch_fixture(args):
    import torch
    from DecoderTCR.utils.model_zoo import resolve, load
    from DecoderTCR.constants import ALPHABET
    from torch_checkpoint_worker import validate_inventory

    spec = resolve(args.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    validate_inventory(checkpoint, spec.arch)
    # Build only tensor metadata, not a second set of real model parameters.
    with torch.device("meta"):
        if spec.backbone == "esmc":
            from DecoderTCR.model.DecoderTCRC import DecoderTCRC
            empty = DecoderTCRC(spec.arch, init_weights=None, use_packed_flash=False)
        else:
            from DecoderTCR.model.DecoderTCR import DecoderTCR
            empty = DecoderTCR(spec.arch, init_weights=None)
    expected = {"model." + name: tensor for name, tensor in empty.state_dict().items()}
    actual = checkpoint["state_dict"]
    if set(expected) != set(actual):
        raise ValueError(f"Checkpoint tensor keys differ: missing={sorted(set(expected)-set(actual))[:10]}, "
                         f"unexpected={sorted(set(actual)-set(expected))[:10]}")
    for name, reference in expected.items():
        tensor = actual[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tuple(tensor.shape) != tuple(reference.shape) or tensor.dtype != reference.dtype):
            raise ValueError(f"Checkpoint tensor shape/dtype/layout differs: {name}")
        if tensor.is_floating_point() and not args.inspect_only:
            # Scan in bounded chunks so 6B inventories do not allocate a giant boolean tensor.
            if not tensor.is_contiguous():
                raise ValueError(f"Checkpoint tensor must be contiguous: {name}")
            for chunk in tensor.view(-1).split(1024 * 1024):
                if not torch.isfinite(chunk).all():
                    raise ValueError(f"Non-finite checkpoint values: {name}")
    tensor_count = len(actual)
    parameter_bytes = sum(t.numel() * t.element_size() for t in actual.values())
    storage_bytes = checkpoint_storage_bytes(checkpoint, torch)
    if args.inspect_only:
        return {"model": args.model, "tensor_count": tensor_count, "parameter_bytes": parameter_bytes,
                "checkpoint_storage_bytes": storage_bytes,
                "backbone": spec.backbone, "architecture": spec.arch,
                "finite_values_checked": False, "forward_checked": False}
    del expected, actual, empty, checkpoint
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()
            or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError("Requested CUDA device is unavailable; no CPU fallback")
    started = perf_counter()
    model, layers = load(args.model, device=device, checkpoint=args.checkpoint,
                         backbone=spec.backbone, arch=spec.arch)
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Reference model must use float32 parameters")
    first = next(model.parameters()).device
    if first.type != device.type or (device.index is not None and first.index != device.index):
        raise ValueError("Model loaded on a different device")
    load_seconds = perf_counter() - started
    # Real alphabet, mask, EOS and a padded mixed-length batch. No donor data.
    _, _, tokens = ALPHABET.get_batch_converter()([
        ("short", "ACDEFGHIK<mask>LMNPQRSTVWY"),
        ("long", "MNPQRSTVWYACDEFGHIKLMNPQRSTVWY<mask>ACDEFGHIK"),
    ])
    started = perf_counter()
    with torch.inference_mode():
        output = model(tokens.to(device), repr_layers=[layers], return_contacts=False)
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--precision", default="float32", choices=("float32", "float16"))
    args = parser.parse_args()
    if args.backend == "torch":
        if args.inspect_only:
            Path(args.output).write_text(json.dumps(torch_fixture(args), indent=2) + "\n")
            return
        tokens, logits, info = torch_fixture(args)
    else:
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
