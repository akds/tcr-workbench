"""Standalone reconstruction worker, executed only in the pinned upstream env.

Uses the upstream reconstruction functions without loading model weights. TCR
reconstruction is amortized across peptide/HLA panels, retaining upstream gene
choices and failure reasons. The bounded cache and chunking limit Python rows.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import hashlib
from itertools import islice
from pathlib import Path
import re

GENES = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b")
RECONSTRUCTION = ("TRAV", "TRAJ", "TRBV", "TRBJ", "HLA_a", "HLA_b", "TCR_a", "TCR_b",
                  "ok", "tcr_ok", "tcr_reason", "hla_ok", "hla_reason")


def exact_hla_key(allele):
    """Map an explicit modern molecule to the pinned compact training key.

    That legacy table records two fields with a two-digit first field. Refuse
    higher resolution and variable-width first fields: removing all delimiters
    would otherwise alias distinct modern names such as A*10:256/A*102:56.
    """
    parts = allele.split("/")
    molecules = []
    for part in parts:
        match = re.fullmatch(r"HLA-(A|B|C|DRA|DRB1|DRB3|DRB4|DRB5|DQA1|DQB1|DPA1|DPB1)"
                             r"\*([0-9]{2}):([0-9]{2,})", part)
        if match is None:
            raise KeyError("training HLA reference requires explicit two-field names with a two-digit first field")
        locus, first, second = match.groups()
        molecules.append((locus, locus + first + second))
    if len(molecules) == 1 and molecules[0][0] in ("A", "B", "C"):
        return molecules[0][1] + "__B2M"
    allowed = {("DRA", "DRB1"), ("DRA", "DRB3"), ("DRA", "DRB4"), ("DRA", "DRB5"),
               ("DQA1", "DQB1"), ("DPA1", "DPB1")}
    molecules.sort()
    if len(molecules) == 2 and tuple(locus for locus, _ in molecules) in allowed:
        return "__".join(key for _, key in molecules)
    raise KeyError("HLA-only scoring requires class I A/B/C or an explicit homologous class II alpha/beta pair")


def lookup_exact_hla(allele, reference):
    key = exact_hla_key(allele)
    if key not in reference:
        raise KeyError(f"explicit HLA molecule {allele!r} (key {key!r}) is absent from the training reference")
    sequences = reference[key]
    if not sequences.get("HLA_a") or not sequences.get("HLA_b"):
        raise ValueError("training HLA reference contains an empty chain")
    return sequences["HLA_a"], sequences["HLA_b"]


def reconstruct_pmhc(input_path, output_path, reference, *, lookup=None):
    cache = {}
    with open(input_path, newline="", encoding="utf-8") as source, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=list(reader.fieldnames)
                                + ["HLA_a", "HLA_b", "ok", "hla_ok", "hla_reason"])
        writer.writeheader()
        for row in reader:
            allele = row["hla"]
            if allele not in cache:
                try:
                    ha, hb = lookup(allele) if lookup else lookup_exact_hla(allele, reference)
                    cache[allele] = dict(HLA_a=ha, HLA_b=hb,
                                         ok=True, hla_ok=True, hla_reason="")
                except KeyError as exc:
                    cache[allele] = dict(HLA_a="", HLA_b="", ok=False, hla_ok=False, hla_reason=str(exc))
                if len(cache) > 32768:
                    cache.pop(next(iter(cache)))
            row.update(cache[allele])
            writer.writerow(row)


def reconstruct(input_path, output_path, *, stitch, lookup, chunk_size=1024, cache_entries=32768):
    cache = OrderedDict()
    hla_cache = OrderedDict()
    with open(input_path, newline="", encoding="utf-8") as source, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=list(reader.fieldnames) + list(RECONSTRUCTION))
        writer.writeheader()
        for chunk in iter(lambda: list(islice(reader, chunk_size)), []):
            keys = [tuple(row[key] for key in GENES) for row in chunk]
            missing = list(dict.fromkeys(key for key in keys if key not in cache))
            if missing:
                stitched = stitch([dict(zip(GENES, key), name=f"tcr_{i}")
                                   for i, key in enumerate(missing)])
                fresh = {key: stitched[f"tcr_{i}"] for i, key in enumerate(missing)}
            else:
                fresh = {}
            # Snapshot cached members too: inserting a fresh receptor can evict
            # an existing member needed later in this same chunk.
            current = {key: fresh[key] if key in fresh else cache[key] for key in keys}
            for row, key in zip(chunk, keys):
                st = current[key]
                cache[key] = st
                cache.move_to_end(key)
                if len(cache) > cache_entries:
                    cache.popitem(last=False)
                allele = row["hla"]
                if allele not in hla_cache:
                    try:
                        ha, hb = lookup(allele)
                        hla_cache[allele] = (ha, hb, True, "")
                    except KeyError as exc:
                        hla_cache[allele] = ("", "", False, str(exc))
                    if len(hla_cache) > cache_entries:
                        hla_cache.popitem(last=False)
                ha, hb, hla_ok, hla_reason = hla_cache[allele]
                hla_cache.move_to_end(allele)
                row.update({key: st.get(key, "") for key in ("TRAV", "TRAJ", "TRBV", "TRBJ")})
                row.update(TCR_a=st["TCR_a"], TCR_b=st["TCR_b"], tcr_ok=st["ok"],
                           tcr_reason=st["reason"], HLA_a=ha, HLA_b=hb, hla_ok=hla_ok,
                           hla_reason=hla_reason, ok=bool(st["ok"] and hla_ok))
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-type", choices=("tcr-pmhc", "pmhc"), default="tcr-pmhc")
    parser.add_argument("--species", choices=("human", "mouse"), default="human")
    parser.add_argument("--mhc-reference")
    parser.add_argument("--mhc-reference-sha256")
    args = parser.parse_args()
    if args.species == "mouse":
        from mhc_context import load_reference
        if not args.mhc_reference or not args.mhc_reference_sha256:
            raise ValueError("Mouse reconstruction requires an explicitly hashed MHC sequence reference")
        path = Path(args.mhc_reference)
        if hashlib.sha256(path.read_bytes()).hexdigest() != args.mhc_reference_sha256:
            raise ValueError("Mouse MHC sequence reference hash mismatch")
        reference = load_reference(path)["molecules"]

        def lookup_mouse(allele):
            if allele not in reference:
                raise KeyError(f"Explicit mouse MHC molecule {allele!r} is absent from the sequence reference; supply --mhc-reference with both exact mature chains")
            return reference[allele]["HLA_a"], reference[allele]["HLA_b"]

        if args.context_type == "pmhc":
            reconstruct_pmhc(args.input, args.output, reference, lookup=lookup_mouse)
        else:
            from mouse_stitch import stitch_mouse_tcrs
            reconstruct(args.input, args.output, stitch=stitch_mouse_tcrs, lookup=lookup_mouse)
        return
    if args.mhc_reference or args.mhc_reference_sha256:
        raise ValueError("Human reconstruction uses the pinned human reference")
    if args.context_type == "pmhc":
        from DecoderTCR.reconstruct.hla import _reference
        reconstruct_pmhc(args.input, args.output, _reference())
        return
    from DecoderTCR.reconstruct.tcr import stitch_tcrs
    from DecoderTCR.reconstruct.hla import lookup_hla, _reference

    def lookup_molecule(allele):
        # Preserve the original class-I oracle's exact audit messages. Its
        # normalizer cannot represent class-II pairs, so those use strict keys.
        return lookup_exact_hla(allele, _reference()) if "/" in allele else lookup_hla(allele)

    reconstruct(args.input, args.output, stitch=stitch_tcrs, lookup=lookup_molecule)


if __name__ == "__main__":
    main()
