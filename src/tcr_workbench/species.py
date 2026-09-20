"""Explicit biological species and versioned sequence reference boundaries."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from .hla import compare_hla, normalize_hla, parse_hla
from .backends.mhc_context import load_reference

Species = Literal["human", "mouse"]


def validate_species(species: str) -> str:
    if species not in ("human", "mouse"):
        raise ValueError("species must be human or mouse")
    return species


def normalize_mhc(value: str, species: str = "human") -> str:
    validate_species(species)
    name = normalize_hla(value)
    if name.startswith("H-2-") != (species == "mouse"):
        raise ValueError(f"MHC {name!r} does not match --species {species}; human HLA and mouse H-2 are never substituted")
    return name


def model_mhc(value: str, species: str = "human") -> str:
    """Require an explicit molecule, without promising sequence availability."""
    value = normalize_mhc(value, species)
    if species == "mouse":
        return value
    molecule = parse_hla(value)
    if (not value.startswith(("HLA-A*", "HLA-B*", "HLA-C*"))
            and molecule.kind != "heterodimer"):
        raise ValueError("DecoderTCR requires class I A/B/C or an explicit complete class II heterodimer")
    if compare_hla(value, value).status != "compatible":
        raise ValueError("DecoderTCR requires an unambiguous allele-level HLA restriction")
    if any(len(a.fields) != 2 or len(a.fields[0]) != 2 for a in molecule.alleles):
        raise ValueError("DecoderTCR training-reference lookup requires exact two-field HLA names with a two-digit first field; no compact-key alias or resolution truncation is used")
    return value


def biological_fingerprint(species: str = "human", mhc_reference=None) -> dict:
    """Bind the exact biological context/reference independently of checkpoint prep."""
    validate_species(species)
    result = {"species": species}
    if mhc_reference is None and species == "human":
        return result
    if species != "mouse":
        raise ValueError("--mhc-reference currently supports mouse contexts only; human HLA uses the pinned training reference")
    path = Path(mhc_reference).resolve() if mhc_reference else Path(__file__).parent / "data/mouse_mhc.json"
    reference = load_reference(path)
    result.update(mhc_reference_path=str(path),
                  mhc_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  mhc_reference_source=reference["source"],
                  mhc_reference_kind="user-supplied" if mhc_reference else "bundled-uniprot",
                  species_interpretation="Mouse molecular context. Predictive performance depends on the selected checkpoint and requires independent validation.")
    return result


def verify_biological_fingerprint(fingerprint: dict) -> None:
    path = fingerprint.get("mhc_reference_path")
    if path and hashlib.sha256(Path(path).read_bytes()).hexdigest() != fingerprint["mhc_reference_sha256"]:
        raise ValueError("MHC sequence reference changed during inference")


def reconstruction_arguments(fingerprint: dict) -> list[str]:
    args = ["--species", fingerprint.get("species", "human")]
    if fingerprint.get("mhc_reference_path"):
        args += ["--mhc-reference", fingerprint["mhc_reference_path"],
                 "--mhc-reference-sha256", fingerprint["mhc_reference_sha256"]]
    return args
