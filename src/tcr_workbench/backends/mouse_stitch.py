"""Mouse V/J/CDR3 reconstruction with Stitchr's explicit MOUSE reference.

Unlike the upstream human convenience normalizer, no family-level gene or
unknown allele is replaced by an arbitrary gene. Temporary files are bounded
to one chunk and cleaned up even after a failed subprocess.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile

GENES = ("trav", "traj", "trbv", "trbj")


@lru_cache(maxsize=1)
def _gene_set(germlines):
    if not all((germlines / name).is_file() for name in ("TRA.fasta", "TRB.fasta")):
        raise ValueError("Mouse Stitchr germlines are missing. Run python3 tcr.py setup --species mouse with your installed device/config.")
    available = set()
    for filename in ("TRA.fasta", "TRB.fasta"):
        with (germlines / filename).open() as source:
            for line in source:
                if line.startswith(">") and "|" in line:
                    gene = line.split("|")[1]
                    available.add(gene)
                    available.add(gene.split("*")[0])
    return frozenset(available)


def stitch_mouse_tcrs(rows):
    from Stitchr import stitchrfunctions as sf
    available = _gene_set(Path(sf.data_dir) / "MOUSE")
    out = {}
    eligible = []
    for row in rows:
        missing = [row[key] for key in GENES if row[key] not in available]
        if missing:
            out[row["name"]] = {"TCR_a": "", "TCR_b": "", "ok": False,
                "reason": "Mouse V/J gene or allele is absent from the installed reference; no substitution: " + ", ".join(missing),
                **{key.upper(): row[key] for key in GENES}}
        else:
            eligible.append(row)
    if not eligible:
        return out
    headers = ["TCR_name", "TRAV", "TRAJ", "TRA_CDR3", "TRBV", "TRBJ", "TRB_CDR3",
               "TRAC", "TRBC", "TRA_leader", "TRB_leader", "Linker", "Link_order",
               "TRA_5_prime_seq", "TRA_3_prime_seq", "TRB_5_prime_seq", "TRB_3_prime_seq"]
    with tempfile.TemporaryDirectory(prefix="tcr-mouse-stitch-") as temporary:
        source = Path(temporary) / "input.tsv"
        target = Path(temporary) / "output.tsv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(headers)
            for row in eligible:
                writer.writerow([row["name"], row["trav"], row["traj"], row["cdr3a"],
                                 row["trbv"], row["trbj"], row["cdr3b"], *([""] * 10)])
        process = subprocess.run(["thimble", "-in", str(source), "-o", str(target),
                                  "-s", "MOUSE", "-sc"], capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f"Mouse thimble reconstruction failed: {process.stderr[-2000:]}")
        by_name = {row["name"]: row for row in eligible}
        with target.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle, delimiter="\t"):
                name = record["TCR_name"]
                # Thimble can expand ambiguous inputs. Never choose an output
                # or silently collapse duplicate names to a convenient receptor.
                if name not in by_name or name in out:
                    raise ValueError("Mouse thimble returned unexpected or duplicated receptor names")
                row = by_name[name]
                alpha, beta = record.get("TRA_aa", ""), record.get("TRB_aa", "")
                reason = ""
                if not alpha or not beta:
                    reason = "empty mouse chain (gene/CDR3 not stitchable)"
                elif "*" in alpha or "*" in beta:
                    reason = "premature stop in reconstructed mouse chain"
                ok = not reason
                warning = record.get("Warnings/Errors", "").strip()
                if warning and warning != "[None]":
                    reason = "; ".join(filter(None, [reason, warning]))
                selected = {key.upper(): record.get(key.upper()) or row[key] for key in GENES}
                out[name] = {"TCR_a": alpha, "TCR_b": beta, "ok": ok, "reason": reason, **selected}
        for row in eligible:
            out.setdefault(row["name"], {"TCR_a": "", "TCR_b": "", "ok": False,
                "reason": "Mouse receptor dropped by thimble", **{key.upper(): row[key] for key in GENES}})
    return out
