"""Validate a custom checkpoint's architecture before invoking the upstream CLI.

The upstream ESM-C loader trusts saved hyperparameters instead of its ``arch``
argument. A wrong-size checkpoint must not silently acquire the selected model's
label. This lightweight inventory reads only memory-mapped tensor headers.
"""
from __future__ import annotations

import argparse
import gc
import re
import runpy
import sys

ARCHITECTURES = {
    "DecoderTCRC_300M": ("embed", "transformer.blocks", 64, 960, 30),
    "DecoderTCRC_600M": ("embed", "transformer.blocks", 64, 1152, 36),
    "DecoderTCRC_6B": ("embed", "transformer.blocks", 64, 2560, 80),
    "ESM2_650M": ("embed_tokens", "layers", 33, 1280, 33),
    "ESM2_3B": ("embed_tokens", "layers", 33, 2560, 36),
}


def validate_inventory(checkpoint, arch):
    if arch not in ARCHITECTURES:
        raise ValueError("unsupported DecoderTCR checkpoint architecture")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("custom DecoderTCR checkpoint requires a Lightning state_dict")
    embedding, blocks, vocab, width, depth = ARCHITECTURES[arch]
    state = checkpoint["state_dict"]
    weight = state.get(f"model.model.{embedding}.weight")
    prefix = "model.model." + blocks + "."
    indices = {int(match.group(1)) for key in state
               if (match := re.match(re.escape(prefix) + r"([0-9]+)\.", key))}
    if weight is None or tuple(weight.shape) != (vocab, width) or indices != set(range(depth)):
        raise ValueError(f"checkpoint architecture does not match requested {arch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--module", choices=("DecoderTCR.utils.predict_from_genes",
                                           "DecoderTCR.utils.peptide_profile"), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    validate_inventory(checkpoint, args.expected_arch)
    del checkpoint
    gc.collect()
    sys.argv = [args.module] + (args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    runpy.run_module(args.module, run_name="__main__")


if __name__ == "__main__":
    main()
