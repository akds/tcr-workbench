"""Isolated reconstruction and FP32 independent-chain receptor embeddings.

Only pooled vectors leave the accelerator. Exact chain sequences are deduplicated
in a disk-backed index; no peptide or MHC is accepted by this worker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
import sqlite3
from time import perf_counter


GENES = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b")
SELECTED = ("TRAV", "TRAJ", "TRBV", "TRBJ")
INPUT_FIELDS = ("embedding_id", "species", *GENES, "status", "reason")
CHAIN_FIELDS = ("TCR_a", "TCR_b", *SELECTED, "alpha_sha256", "beta_sha256",
                "reconstructed_receptor_sha256")
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
WIDTHS = {"DecoderTCR-ESMC_300M": 960, "DecoderTCR-ESMC_600M": 1152,
          "DecoderTCR-ESMC_6B": 2560}
FEATURE_VERSION = "independent-chain-postnorm-mean-l2-v1"


def chain_hashes(alpha, beta):
    return {"alpha_sha256": hashlib.sha256(alpha.encode()).hexdigest(),
            "beta_sha256": hashlib.sha256(beta.encode()).hexdigest(),
            "reconstructed_receptor_sha256": hashlib.sha256(
                json.dumps([alpha, beta], separators=(",", ":")).encode()).hexdigest()}


def reconstruct(input_path, output_path, species, stitch=None):
    if stitch is None:
        if species == "mouse":
            from mouse_stitch import stitch_mouse_tcrs as stitch
        else:
            from DecoderTCR.reconstruct.tcr import stitch_tcrs as stitch
    with open(input_path, newline="", encoding="utf-8") as source, open(
            output_path, "w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != INPUT_FIELDS:
            raise ValueError("Receptor embedding reconstruction input schema changed")
        writer = csv.DictWriter(target, fieldnames=(*INPUT_FIELDS, *CHAIN_FIELDS))
        writer.writeheader()
        for rows in iter(lambda: list(islice(reader, 512)), []):
            unique = {}
            for row in rows:
                if row["species"] != species or row["status"] not in ("Pending", "Unresolved"):
                    raise ValueError("Reconstruction species/status contract changed")
                if row["status"] == "Pending":
                    key = tuple(row[field] for field in GENES)
                    unique.setdefault(key, {"name": row["embedding_id"], **dict(zip(GENES, key))})
            outputs = stitch(list(unique.values())) if unique else {}
            if set(outputs) != {row["name"] for row in unique.values()}:
                raise ValueError("Reconstructor changed receptor identities")
            for row in rows:
                row.update(dict.fromkeys(CHAIN_FIELDS, ""))
                if row["status"] == "Pending":
                    record = outputs[unique[tuple(row[field] for field in GENES)]["name"]]
                    alpha, beta = record.get("TCR_a", ""), record.get("TCR_b", "")
                    reason = str(record.get("reason") or "")
                    ok = record.get("ok") is True
                    if ok:
                        if any(not isinstance(chain, str) or not chain or set(chain) - AA
                               for chain in (alpha, beta)):
                            ok = False
                            reason = "; ".join(filter(None, [reason, "Reconstructed chain must contain only AA20 residues"]))
                        elif any(len(chain) > 2046 for chain in (alpha, beta)):
                            ok = False
                            reason = "; ".join(filter(None, [reason, "Reconstructed chain exceeds 2048 tokens; no truncation"]))
                    row.update({key: str(record.get(key) or "") for key in SELECTED})
                    row.update(status="Embedded" if ok else "Unresolved",
                               reason=reason or ("" if ok else "Full-chain reconstruction failed"))
                    if ok:
                        row.update(TCR_a=alpha, TCR_b=beta, **chain_hashes(alpha, beta))
                writer.writerow(row)


def length_batches(cursor, batch_size, token_budget):
    batch = []
    for identifier, sequence in cursor:
        length = len(sequence) + 2
        if length > token_budget:
            raise ValueError(f"One reconstructed chain needs {length} tokens; increase --token-budget")
        if batch and (len(batch) >= batch_size or (len(batch) + 1) * length > token_budget):
            yield batch
            batch = []
        batch.append((identifier, sequence))
    if batch:
        yield batch


def token_array(sequences, encode, pad_token_id):
    import numpy as np
    if pad_token_id != 1:
        raise ValueError("Tokenizer must use the verified DecoderTCR PAD token ID 1")
    encoded = [encode(sequence) for sequence in sequences]
    if any(len(ids) != len(sequence) + 2 or ids[0] != 0 or ids[-1] != 2
           or any(token < 4 or token > 23 for token in ids[1:-1])
           for sequence, ids in zip(sequences, encoded)):
        raise ValueError("Tokenizer must preserve AA20 residues with exactly one CLS/EOS pair")
    tokens = np.full((len(encoded), max(map(len, encoded))), pad_token_id, np.int32)
    for index, ids in enumerate(encoded):
        tokens[index, :len(ids)] = ids
    return tokens


def torch_pool(raw_model, tokens, torch):
    """Use the reference blocks/norm without hidden-state stacking or the head."""
    valid = tokens != raw_model.tokenizer.pad_token_id
    hidden = raw_model.embed(tokens)
    for block in raw_model.transformer.blocks:
        hidden = block(hidden, valid)
    hidden = raw_model.transformer.norm(hidden)
    residues = (tokens != 0) & (tokens != 1) & (tokens != 2)
    pooled = (hidden.float() * residues.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / residues.sum(dim=1, keepdim=True)
    return pooled / torch.linalg.vector_norm(pooled, dim=1, keepdim=True)


def mlx_pool(model, tokens, mx):
    result = model(tokens, compute_logits=False)
    hidden = result["postnorm"].astype(mx.float32)
    residues = (tokens != 0) & (tokens != 1) & (tokens != 2)
    pooled = (hidden * residues[..., None]).sum(axis=1) / residues.sum(axis=1, keepdims=True)
    return pooled / mx.linalg.norm(pooled, axis=1, keepdims=True)


def load_encoder(args):
    import numpy as np
    started = perf_counter()
    if args.device == "apple":
        import mlx.core as mx
        from esmc_mlx.inference import load_bundle
        from esmc_mlx.config import validate_decoder_config
        model, tokenizer, manifest = load_bundle(args.checkpoint)
        validate_decoder_config(model.config, args.model)
        if model.config.dtype != "float32" or tokenizer.variant != "decodertcr-esm1b":
            raise ValueError("Receptor embeddings require a DecoderTCR FP32 bundle/tokenizer")
        if manifest["source_sha256"] != args.checkpoint_sha256:
            raise ValueError("Embedding bundle source-checkpoint identity changed")
        def encode(sequences):
            tokens = token_array(sequences, tokenizer.encode, tokenizer.pad_token_id)
            pooled = mlx_pool(model, mx.array(tokens), mx)
            mx.eval(pooled)
            return np.array(pooled, dtype=np.float32)
        def memory():
            return int(mx.get_peak_memory())
        runtime = {"backend": "mlx", "device": "apple", "tokenizer_variant": tokenizer.variant}
    else:
        import torch
        from DecoderTCR.utils.model_zoo import load, resolve
        # -I removes the script directory from sys.path. Import this sibling by
        # its explicit trusted file path rather than weakening isolation.
        import runpy
        validate_inventory = runpy.run_path(str(Path(__file__).with_name("torch_checkpoint_worker.py")))["validate_inventory"]
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda") or (device.type == "cuda" and (
                not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()))):
            raise RuntimeError("Requested embedding device is unavailable; fallback is disabled")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        spec = resolve(args.model)
        inventory = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        validate_inventory(inventory, spec.arch)
        del inventory
        wrapper, _ = load(args.model, device=device, checkpoint=Path(args.checkpoint),
                          backbone=spec.backbone, arch=spec.arch)
        wrapper.eval()
        if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in wrapper.parameters()):
            raise ValueError("Receptor embeddings require actual FP32 parameters")
        actual = next(wrapper.parameters()).device
        if actual.type != device.type or (device.index is not None and actual.index != device.index):
            raise RuntimeError("Embedding model device changed; fallback is disabled")
        raw_model = wrapper.model
        tokenizer = raw_model.tokenizer
        def encode(sequences):
            tokens = token_array(sequences, tokenizer.encode, tokenizer.pad_token_id)
            with torch.inference_mode():
                pooled = torch_pool(raw_model, torch.as_tensor(tokens, dtype=torch.long, device=device), torch)
                return pooled.cpu().numpy()
        def memory():
            return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        runtime = {"backend": "torch", "device": str(device), "tokenizer_variant": "decodertcr-esm1b"}
    return encode, memory, {**runtime, "model_load_seconds": perf_counter() - started}


def infer(args, encoder_loader=load_encoder):
    import numpy as np
    width = WIDTHS[args.model]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    database = root / ".chain-vectors.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-16384")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("CREATE TABLE chains (id INTEGER PRIMARY KEY, sequence TEXT UNIQUE)")
        connection.execute("CREATE TABLE vectors (chain_id INTEGER PRIMARY KEY REFERENCES chains(id), vector BLOB NOT NULL)")
        rows = eligible = 0
        with open(args.input, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != (*INPUT_FIELDS, *CHAIN_FIELDS):
                raise ValueError("Reconstructed embedding schema changed")
            for row in reader:
                rows += 1
                if row["status"] == "Embedded":
                    eligible += 1
                    if chain_hashes(row["TCR_a"], row["TCR_b"]) != {key: row[key] for key in chain_hashes("", "")}:
                        raise ValueError("Reconstructed chain hash changed")
                    connection.executemany("INSERT OR IGNORE INTO chains(sequence) VALUES (?)", [(row["TCR_a"],), (row["TCR_b"],)])
                elif row["status"] != "Unresolved" or not row["reason"]:
                    raise ValueError("Invalid reconstruction status or missing abstention reason")
        connection.commit()
        unique_chains = connection.execute("SELECT COUNT(*) FROM chains").fetchone()[0]
        runtime = {"backend": "mlx" if args.device == "apple" else "torch", "device": args.device,
                   "tokenizer_variant": "decodertcr-esm1b", "model_load_seconds": 0.0}
        elapsed = 0.0
        batches = 0
        peak = None
        if unique_chains:
            longest = connection.execute("SELECT MAX(length(sequence)) + 2 FROM chains").fetchone()[0]
            if longest > args.token_budget:
                raise ValueError(f"One reconstructed chain needs {longest} tokens; increase --token-budget")
            encode, memory, runtime = encoder_loader(args)
            cursor = connection.execute("SELECT id,sequence FROM chains ORDER BY length(sequence),id")
            for batch in length_batches(cursor, args.batch_size, args.token_budget):
                started = perf_counter()
                vectors = np.asarray(encode([sequence for _, sequence in batch]))
                elapsed += perf_counter() - started
                if vectors.shape != (len(batch), width) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                    raise ValueError("Embedding worker produced invalid pooled vectors")
                if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5, rtol=0):
                    raise ValueError("Embedding worker produced non-unit chain vectors")
                connection.executemany("INSERT INTO vectors(chain_id,vector) VALUES (?,?)",
                    [(item[0], vector.tobytes()) for item, vector in zip(batch, vectors)])
                batches += 1
            connection.commit()
            peak = memory()
        missing = connection.execute(
            "SELECT COUNT(*) FROM chains LEFT JOIN vectors ON vectors.chain_id=chains.id "
            "WHERE vectors.vector IS NULL OR length(vectors.vector) != ?", (width * 4,)).fetchone()[0]
        if missing or connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] != unique_chains:
            raise ValueError("Embedding chain vector coverage is incomplete")
        output = np.lib.format.open_memmap(root / "receptor_embeddings.npy", mode="w+", dtype=np.float32,
                                            shape=(eligible, 2 * width))
        identifiers = []
        with open(args.input, newline="", encoding="utf-8") as source, (
                root / "embedding_audit.csv").open("w", newline="", encoding="utf-8") as target:
            reader = csv.DictReader(source)
            fields = [field for field in reader.fieldnames if field not in ("TCR_a", "TCR_b")]
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                if row["status"] == "Embedded":
                    combined = np.empty(2 * width, dtype=np.float32)
                    for offset, key in ((0, "TCR_a"), (width, "TCR_b")):
                        blob = connection.execute(
                            "SELECT vectors.vector FROM chains JOIN vectors ON vectors.chain_id=chains.id "
                            "WHERE chains.sequence=?", (row[key],)).fetchone()[0]
                        vector = np.frombuffer(blob, dtype=np.float32)
                        combined[offset:offset + width] = vector
                    combined /= np.linalg.norm(combined)
                    output[len(identifiers)] = combined
                    identifiers.append(row["embedding_id"])
                writer.writerow({field: row[field] for field in fields})
        output.flush()
        del output
        (root / "embedding_ids.json").write_text(json.dumps(identifiers) + "\n", encoding="utf-8")
        runtime.update(model=args.model, source_checkpoint_sha256=args.checkpoint_sha256,
                       feature_version=FEATURE_VERSION, precision="float32", math_precision="float32-no-tf32",
                       rows=rows, embedded=eligible, unresolved=rows-eligible, unique_chains=unique_chains,
                       chain_dimension=width, vector_dimension=2 * width, batches=batches,
                       inference_seconds=elapsed, accelerator_peak_memory_bytes=peak)
        (root / "embedding_runtime.json").write_text(json.dumps(runtime, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    finally:
        connection.close()
        database.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("reconstruct", "infer"))
    for name in ("input", "output", "species"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model", choices=tuple(WIDTHS))
    parser.add_argument("--device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=4096)
    args = parser.parse_args()
    if args.species not in ("human", "mouse"):
        raise ValueError("Unsupported reconstruction species")
    if args.phase == "reconstruct":
        reconstruct(args.input, args.output, args.species)
    else:
        if not 1 <= args.batch_size <= 128 or not 3 <= args.token_budget <= 262144:
            raise ValueError("Invalid embedding batch size/token budget")
        if not all((args.model, args.device, args.checkpoint, args.checkpoint_sha256)):
            raise ValueError("Inference requires explicit model, device and checkpoint identity")
        infer(args)


if __name__ == "__main__":
    main()
