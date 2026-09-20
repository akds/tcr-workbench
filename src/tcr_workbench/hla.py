"""Conservative, offline HLA compatibility; syntax is not allele-database validation.

Two-field protein identity is sufficient for ordinary expressed alleles. G/P
groups need an explicit, versioned membership database, which this MVP does not
bundle. Their names must never be reduced to their apparent two-field prefix.
DQ/DP restrictions require an explicitly reported alpha/beta heterodimer.
DRB-only names encode a DRB-level restriction, but cannot confirm a reference
that explicitly specifies a DRA/DRB heterodimer without alpha-chain typing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

_ALLELE = re.compile(
    r"^(A|B|C|DRA|DRB1|DRB3|DRB4|DRB5|DQA1|DQB1|DPA1|DPB1)"
    r"\*([0-9]{2,}(?::[0-9]{2,}){0,3})([NLSCAQGP]?)$"
)
_PAIRS = {
    frozenset(("DQA1", "DQB1")),
    frozenset(("DPA1", "DPB1")),
    frozenset(("DRA", "DRB1")),
    frozenset(("DRA", "DRB3")),
    frozenset(("DRA", "DRB4")),
    frozenset(("DRA", "DRB5")),
}
_NEEDS_PARTNER = {"DRA", "DQA1", "DQB1", "DPA1", "DPB1"}


@dataclass(frozen=True)
class Allele:
    locus: str
    fields: tuple[str, ...]
    suffix: str = ""

    @property
    def name(self) -> str:
        if self.locus.startswith("H-2-"):
            return self.locus + self.fields[0]
        return "HLA-" + self.locus + "*" + ":".join(self.fields) + self.suffix


@dataclass(frozen=True)
class HLA:
    alleles: tuple[Allele, ...]
    kind: str

    @property
    def name(self) -> str:
        delimiter = "|" if self.kind == "ambiguous" else "/"
        return delimiter.join(a.name for a in self.alleles)


@dataclass(frozen=True)
class HLACompatibility:
    status: str
    reason: str


@lru_cache(maxsize=16384)
def parse_hla(value: str) -> HLA:
    """Parse modern names, optional HLA prefix, and explicit class-II pairs.

    A single-field name and alternatives at one locus (``A*02:01|A*02:02``)
    are retained as ambiguous typing. A slash between two different class-II
    loci denotes a heterodimer, never an allele ambiguity or inferred phase.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("HLA must be a non-empty allele or class-II heterodimer")
    mouse = re.fullmatch(r"H-?2-?(K|D|L|IA|IE)([A-Z][A-Z0-9]*)", value.strip(), re.I)
    if mouse:
        locus, haplotype = mouse.groups()
        return HLA((Allele("H-2-" + locus.upper(), (haplotype.lower(),)),), "allele")
    text = re.sub(r"\s+", "", value).upper().replace("HLA-", "")
    parts = re.split(r"[/|,;]", text)
    alleles = []
    for part in parts:
        match = _ALLELE.fullmatch(part)
        if match is None:
            raise ValueError(
                f"Unsupported HLA name {value!r}; use modern names such as HLA-A*02:01 "
                "or HLA-DQA1*05:01/HLA-DQB1*02:01"
            )
        locus, fields, suffix = match.groups()
        allele = Allele(locus, tuple(fields.split(":")), suffix)
        if suffix == "G" and len(allele.fields) != 3:
            raise ValueError(f"G-group names require three fields: {value!r}")
        if suffix == "P" and len(allele.fields) != 2:
            raise ValueError(f"P-group names require two fields: {value!r}")
        alleles.append(allele)
    if len(alleles) == 1:
        return HLA(tuple(alleles), "allele")
    loci = frozenset(a.locus for a in alleles)
    if len(loci) == 1:
        distinct = tuple(sorted(set(alleles), key=lambda a: a.name))
        return HLA(distinct, "allele" if len(distinct) == 1 else "ambiguous")
    if len(alleles) == 2 and loci in _PAIRS and "/" in text and not re.search(r"[|,;]", text):
        return HLA(tuple(sorted(alleles, key=lambda a: a.locus)), "heterodimer")
    raise ValueError("HLA rows require one locus or an explicit DQ, DP, or DR alpha/beta pair")


def normalize_hla(value: str) -> str:
    """Canonicalize a literal identifier, without establishing compatibility.

    Equal G/P-group names still require versioned membership information before
    they can establish the surface-restriction compatibility used by this tool.
    """
    return parse_hla(value).name


def _allele_compatibility(left: Allele, right: Allele) -> HLACompatibility:
    if left.locus != right.locus:
        return HLACompatibility("incompatible", "Different HLA loci")
    if left.suffix in {"N", "S", "C"} or right.suffix in {"N", "S", "C"}:
        return HLACompatibility(
            "incompatible",
            "Null, secreted-only, or cytoplasmic HLA cannot confirm surface restriction",
        )
    # Group names are representatives, not prefixes shared by every member.
    if left.suffix in {"G", "P"} or right.suffix in {"G", "P"}:
        return HLACompatibility("unresolved", "G/P-group membership requires a versioned mapping")
    # Without a delimiter, four or more digits can be an old concatenated
    # allele rather than one modern field. Never guess its field boundaries.
    # Explicit colon-delimited fields can legitimately exceed three digits
    # (for example DPB1*1000:01), so do not impose a field-width ceiling.
    if any(len(a.fields) == 1 and len(a.fields[0]) > 3 for a in (left, right)):
        return HLACompatibility(
            "unresolved",
            "Delimiter-free HLA name could be a legacy concatenated allele; "
            "supply colon-delimited fields instead of inferring field boundaries",
        )
    common = min(len(left.fields), len(right.fields), 2)
    if left.fields[:common] != right.fields[:common]:
        return HLACompatibility("incompatible", "Different HLA protein alleles")
    if common < 2:
        return HLACompatibility("unresolved", "HLA typing has fewer than two fields")
    if left.suffix or right.suffix:
        return HLACompatibility("unresolved", "HLA expression is low, aberrant, or questionable")
    return HLACompatibility("compatible", "HLA protein alleles agree at two-field resolution")


@lru_cache(maxsize=32768)
def compare_hla(left: str | None, right: str | None) -> HLACompatibility:
    """Compare restrictions without converting typing uncertainty to compatibility."""
    if not left or not right:
        return HLACompatibility("unresolved", "HLA restriction or typing is missing")
    first, second = parse_hla(left), parse_hla(right)
    mouse_first = first.alleles[0].locus.startswith("H-2-")
    mouse_second = second.alleles[0].locus.startswith("H-2-")
    if mouse_first or mouse_second:
        if not (mouse_first and mouse_second):
            return HLACompatibility("incompatible", "Human and mouse MHC contexts differ")
        if first.name == second.name:
            return HLACompatibility("compatible", "Explicit mouse MHC molecule identifiers agree; expression is not measured")
        return HLACompatibility("incompatible", "Different explicit mouse MHC molecules; no haplotype substitution is inferred")
    for molecule in (first, second):
        if molecule.kind != "ambiguous" and any(
            a.suffix in {"N", "S", "C"} for a in molecule.alleles
        ):
            return HLACompatibility(
                "incompatible",
                "An explicitly null, secreted-only, or cytoplasmic chain cannot confirm surface restriction",
            )
    if first.kind == "ambiguous" or second.kind == "ambiguous":
        possibilities = [_allele_compatibility(a, b) for a in first.alleles for b in second.alleles]
        if all(item.status == "incompatible" for item in possibilities):
            return HLACompatibility("incompatible", "No allele alternative is compatible")
        return HLACompatibility("unresolved", "Explicit HLA allele ambiguity is retained")
    a_map = {a.locus: a for a in first.alleles}
    b_map = {a.locus: a for a in second.alleles}
    shared = set(a_map) & set(b_map)
    if not shared:
        return HLACompatibility("incompatible", "Different HLA loci")
    checks = [_allele_compatibility(a_map[locus], b_map[locus]) for locus in sorted(shared)]
    if any(check.status == "incompatible" for check in checks):
        return next(check for check in checks if check.status == "incompatible")
    if any(check.status == "unresolved" for check in checks):
        return next(check for check in checks if check.status == "unresolved")
    if set(a_map) != set(b_map):
        if first.kind == second.kind == "heterodimer":
            differing_a = ", ".join(sorted(set(a_map) - set(b_map)))
            differing_b = ", ".join(sorted(set(b_map) - set(a_map)))
            return HLACompatibility(
                "unresolved",
                f"Class II heterodimers specify different chain genes ({differing_a} versus {differing_b}); "
                "restriction equivalence is not inferred",
            )
        return HLACompatibility(
            "unresolved",
            "HLA heterodimer is only partially specified; missing alpha/beta typing is not inferred",
        )
    if len(a_map) == 1 and shared & _NEEDS_PARTNER:
        family = "DRA-only DR" if "DRA" in shared else "DQ/DP"
        return HLACompatibility(
            "unresolved", f"{family} restriction requires both alpha and beta chains"
        )
    return HLACompatibility(
        "compatible", "HLA protein restriction is compatible; expression is not measured"
    )


def donor_compatibility(restriction: str | None, donor_hla: Iterable[str]) -> HLACompatibility:
    """Find a reported compatible molecule without assuming typing completeness.

    The donor schema records positive typing observations, not an exhaustive
    genotype. Even two reported alleles do not assert locus/copy completeness.
    Nonmatches therefore remain unresolved; explicit non-surface restrictions
    are still incompatible. DQ/DP pairs are never inferred from separate chains.
    """
    values = tuple(donor_hla)
    if restriction and compare_hla(restriction, restriction).status == "incompatible":
        return compare_hla(restriction, restriction)
    if not values:
        return HLACompatibility("unresolved", "Donor HLA typing is missing")
    checks = [compare_hla(restriction, value) for value in values]
    # A DRB-level restriction names no particular alpha allele. An explicitly
    # typed expressed donor DR molecule can support that beta-level claim.
    # This is directional evidence, not a change to symmetric molecule identity.
    requested = parse_hla(restriction) if restriction else None
    if requested and requested.kind == "allele" and requested.alleles[0].locus.startswith("DRB"):
        locus = requested.alleles[0].locus
        for value in values:
            typed = parse_hla(value)
            if typed.kind != "heterodimer" or compare_hla(value, value).status != "compatible":
                continue
            beta = next((a for a in typed.alleles if a.locus == locus), None)
            if beta and compare_hla(restriction, beta.name).status == "compatible":
                return HLACompatibility(
                    "compatible", "Reported donor DR molecule supports the DRB-level restriction"
                )
    for status in ("compatible", "unresolved"):
        for check in checks:
            if check.status == status:
                return check
    if restriction:
        requested_loci = {allele.locus for allele in parse_hla(restriction).alleles}
        reported_loci = {allele.locus for value in values for allele in parse_hla(value).alleles}
        if not requested_loci & reported_loci:
            return HLACompatibility(
                "unresolved", "Donor typing does not cover the reference HLA locus"
            )
    return HLACompatibility(
        "unresolved",
        "No reported donor HLA is compatible; supplied typing does not assert genotype completeness",
    )
