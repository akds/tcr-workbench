"""Small standard-library-only reference reader shared by isolated workers."""
from __future__ import annotations

import json
from pathlib import Path
import re


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate MHC reference JSON key: {key}")
        result[key] = value
    return result


def load_reference(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("MHC reference exceeds 8 MiB; supply a focused molecule reference")
    reference = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if (not isinstance(reference, dict)
            or set(reference) != {"schema_version", "species", "source", "molecules"}
            or type(reference["schema_version"]) is not int or reference["schema_version"] != 1
            or reference["species"] != "mouse"
            or not isinstance(reference["source"], str) or not reference["source"].strip()
            or not isinstance(reference["molecules"], dict) or not reference["molecules"]):
        raise ValueError("MHC reference requires schema_version=1, species=mouse, source and nonempty molecules")
    for name, molecule in reference["molecules"].items():
        if not re.fullmatch(r"H-2-(K|D|L|IA|IE)[a-z][a-z0-9]*", name):
            raise ValueError(f"MHC reference needs canonical mouse identifiers such as H-2-Kb or H-2-IAb: {name!r}")
        if not isinstance(molecule, dict) or set(molecule) != {"class", "HLA_a", "HLA_b", "source"}:
            raise ValueError(f"{name}: specify class, HLA_a, HLA_b, source (both exact mature chains)")
        expected_class = "II" if name.startswith(("H-2-IA", "H-2-IE")) else "I"
        if molecule["class"] != expected_class:
            raise ValueError(f"{name}: MHC class disagrees with molecule name")
        if not isinstance(molecule["source"], str) or not molecule["source"].strip():
            raise ValueError(f"{name}: cite sequence accessions, version and residue ranges")
        for key in ("HLA_a", "HLA_b"):
            value = molecule[key]
            if not isinstance(value, str) or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{50,600}", value):
                raise ValueError(f"{name} {key}: supply an exact mature standard-amino-acid chain (50–600 residues)")
    return reference
