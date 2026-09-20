#!/usr/bin/env python3
"""Convert an explicitly selected source format to an atomic FP32 MLX bundle."""
from __future__ import annotations
import argparse
import json
from esmc_mlx.weights import convert_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-variant", choices=("biohub-esmc-published-v1", "decodertcr-lightning-v03"), required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--model-id", help="Required for unrecognized checkpoint bytes; records user-supplied identity")
    parser.add_argument("--model-size", choices=("300m", "600m", "6b"), default="300m",
                        help="Explicit architecture; 600m/6b support DecoderTCR source only; 6b requires local parity validation")
    args = parser.parse_args()
    manifest = convert_checkpoint(args.source, args.output, args.source_variant, args.expected_sha256,
                                  args.model_id, model_size=args.model_size)
    print(json.dumps({key: manifest[key] for key in ("model_id", "source_sha256", "source_tensor_count", "parameter_count")}, indent=2))


if __name__ == "__main__":
    main()
