"""Shared streaming profile output for framework workers; no model imports."""
from __future__ import annotations

import csv
import math

AA = "ACDEFGHIKLMNPQRSTVWY"
FIELDS = ["name", "hla", "peptide_length", "position", "status", "reason", *AA]


def write_profiles(reader, target, infer_profile):
    """Keep one model lifetime while writing one profile per MHC-only context."""
    writer = csv.DictWriter(target, fieldnames=FIELDS)
    writer.writeheader()
    counts = {"rows": 0, "scored": 0, "unresolved": 0}
    for row in reader:
        if row.get("TCR_a") or row.get("TCR_b"):
            raise ValueError("MHC-only profile generation must not contain TCR chains")
        length = len(row["peptide"])
        if not 1 <= length <= 50:
            raise ValueError("Profile peptide length must be 1–50")
        base = {"name": row["name"], "hla": row["hla"], "peptide_length": length}
        counts["rows"] += 1
        if row.get("ok", "").lower() not in ("true", "1"):
            reason = "; ".join(row.get(key, "") for key in ("hla_reason", "tcr_reason") if row.get(key))
            writer.writerow({**base, "status": "Unresolved", "reason": reason or "MHC sequence unavailable"})
            counts["unresolved"] += 1
            continue
        values = list(infer_profile(row))
        if len(values) != length:
            raise ValueError("Profile output does not match requested length")
        for position, probabilities in enumerate(values, start=1):
            probabilities = [float(value) for value in probabilities]
            if (len(probabilities) != len(AA) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                    or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-5)):
                raise ValueError("Profile output must contain normalized, finite AA20 probabilities")
            writer.writerow({**base, "position": position, "status": "Profiled", "reason": "",
                             **dict(zip(AA, probabilities))})
        counts["scored"] += 1
    return counts
