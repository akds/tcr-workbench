"""Auditable CDR3 reference screening with bounded-memory native edit search.

Exact means CDR3 identity, not an identical full receptor or antigen proof.
Every reference provenance row is retained. Results are antigen hypotheses,
and missing chains, V/J uncertainty, and HLA uncertainty are explicit.
"""

# Python 3.9 Pydantic evaluates field annotations; keep Optional rather than PEP 604.
# ruff: noqa: UP045
from __future__ import annotations

import csv
import re
from array import array
from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .hla import compare_hla, donor_compatibility, normalize_hla

PathLike = Union[str, Path]
_AMINO = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
_PRESERVED_AMINO = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZJUO*?.-]+$")
_CHAIN_FIELDS = ("cdr3a", "cdr3b", "trav", "traj", "trbv", "trbj")


def _optional_text(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        # Pydantic v2 deliberately does not wrap TypeError from validators.
        raise ValueError("Expected text, not a coerced number or boolean")  # noqa: TRY004
    return value.strip() or None


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, str_strip_whitespace=True)


class _SequenceRow(_BoundaryModel):
    cdr3a: Optional[str] = None
    cdr3b: Optional[str] = None
    trav: Optional[str] = None
    traj: Optional[str] = None
    trbv: Optional[str] = None
    trbj: Optional[str] = None

    @field_validator(*_CHAIN_FIELDS, mode="before")
    @classmethod
    def normalize_chain_field(cls, value: Any) -> Optional[str]:
        text = _optional_text(value)
        return text.upper() if text else None

    @field_validator("cdr3a", "cdr3b")
    @classmethod
    def sequence_syntax(cls, value: Optional[str]) -> Optional[str]:
        if value and not _PRESERVED_AMINO.fullmatch(value):
            raise ValueError(
                "CDR3 must contain amino-acid symbols; uncertainty is retained explicitly"
            )
        return value


class ReferenceRow(_SequenceRow):
    reference_id: str = Field(min_length=1)
    peptide: str = Field(min_length=1)
    hla: Optional[str] = None
    source: str = Field(min_length=1)
    evidence: str = Field(min_length=1)

    @field_validator("peptide", mode="before")
    @classmethod
    def peptide_sequence(cls, value: Any) -> str:
        text = _optional_text(value)
        if not text or not _AMINO.fullmatch(text.upper()):
            raise ValueError(
                "Peptide must be an unmodified sequence of the 20 standard amino acids"
            )
        return text.upper()

    @field_validator("hla", mode="before")
    @classmethod
    def hla_name(cls, value: Any) -> Optional[str]:
        text = _optional_text(value)
        return normalize_hla(text) if text else None

    @model_validator(mode="after")
    def needs_chain(self) -> ReferenceRow:
        if self.cdr3a is None and self.cdr3b is None:
            raise ValueError(
                "Reference needs at least one CDR3; incomplete or nonproductive sequences may be retained"
            )
        return self


class PanelRow(_BoundaryModel):
    peptide: str = Field(min_length=1)
    hla: Optional[str] = None
    description: Optional[str] = None

    _peptide = field_validator("peptide", mode="before")(ReferenceRow.peptide_sequence.__func__)
    _hla = field_validator("hla", mode="before")(ReferenceRow.hla_name.__func__)


class DonorRow(_BoundaryModel):
    donor_id: str = Field(min_length=1)
    hla: str = Field(min_length=1)

    @field_validator("hla", mode="before")
    @classmethod
    def hla_name(cls, value: Any) -> str:
        text = _optional_text(value)
        if text is None:
            raise ValueError("Donor HLA row requires an allele; omit the row for missing typing")
        return normalize_hla(text)


class _ReceptorRow(_SequenceRow):
    receptor_id: str = Field(min_length=1)
    donor_id: Optional[str] = None
    pairing_status: str = "unknown"
    unusable_chain_context: str = ""
    # AIRR observations without a cell_id cannot assert any counted cells.
    cell_count: int = Field(default=1, ge=0)

    _donor = field_validator("donor_id", mode="before")(_optional_text)


def _validate_frame(frame: pl.DataFrame, model: type[BaseModel], label: str) -> pl.DataFrame:
    """Validate in bounded batches, keeping extra columns and original row order."""
    missing = [
        name
        for name, field in model.model_fields.items()
        if field.is_required() and name not in frame.columns
    ]
    if missing:
        raise ValueError("{} missing required columns: {}".format(label, ", ".join(missing)))
    if not frame.height:
        return frame
    batches = []
    offset = 0
    for batch in frame.iter_slices(8192):
        normalized = []
        for index, row in enumerate(batch.iter_rows(named=True)):
            try:
                normalized.append(model.model_validate(row).model_dump())
            except ValidationError as exc:
                raise ValueError(f"{label} row {offset + index + 2}: {exc}") from exc
        batches.append(pl.DataFrame(normalized, infer_schema_length=None))
        offset += batch.height
    return pl.concat(batches, how="diagonal_relaxed")


def _read_table(path: PathLike, model: type[BaseModel], label: str) -> pl.DataFrame:
    path = Path(path)
    # Detection uses the header only; quoted data never controls the delimiter.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    separator = "\t" if "\t" in header else ","
    columns = next(csv.reader([header], delimiter=separator), [])
    if not columns or any(not name or name != name.strip() for name in columns):
        raise ValueError(f"{label} headers must be nonempty and have no surrounding whitespace")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{label} contains duplicate column names")
    try:
        frame = pl.read_csv(
            path,
            separator=separator,
            infer_schema=False,
            encoding="utf8",
            null_values=[""],
            truncate_ragged_lines=False,
        )
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"Cannot read {label} {path}: {exc}") from exc
    return _validate_frame(frame, model, label)


def _unique_references(frame: pl.DataFrame) -> pl.DataFrame:
    identities = frame.get_column("reference_id")
    duplicates = identities.is_duplicated()
    if duplicates.any():
        raise ValueError(
            "reference_id must be unique; duplicate: {!r}".format(identities[duplicates.arg_max()])
        )
    return frame


def read_references(path: PathLike) -> pl.DataFrame:
    """Load reference CSV/TSV; preserve all evidence rows and sequence uncertainty."""
    return _unique_references(_read_table(path, ReferenceRow, "references"))


def read_panel(path: PathLike) -> pl.DataFrame:
    """Load peptides and restrictions; blank HLA remains unresolved."""
    return _read_table(path, PanelRow, "panel")


def read_donors(path: PathLike) -> pl.DataFrame:
    """Load one allele/heterodimer per donor row; do not infer phased DQ/DP pairs."""
    return _read_table(path, DonorRow, "donors")


def _usable_cdr3(value: Optional[str]) -> bool:
    return bool(
        value
        and len(value) >= 3
        and value.startswith("C")
        and value[-1] in "FW"
        and _AMINO.fullmatch(value)
    )


_CacheInfo = namedtuple("SequenceCacheInfo", "hits misses maxsize currsize cached_hits max_hits")


class _SequenceQueryCache:
    """Bound both query count and retained match count, including duplicate evidence.

    A very common sequence can have millions of provenance rows. Such a query
    must still return every row, but must not occupy millions of cached tuples.
    """

    def __init__(
        self, search, maxsize: int = 4096, max_hits: int = 32768, max_entry_hits: int = 2048
    ):
        self.search = search
        self.maxsize = maxsize
        self.max_hits = max_hits
        self.max_entry_hits = max_entry_hits
        self._entries = OrderedDict()
        self._cached_hits = 0
        self._hits = 0
        self._misses = 0

    def __call__(self, query: str) -> tuple[tuple[int, int], ...]:
        if query in self._entries:
            self._hits += 1
            self._entries.move_to_end(query)
            return self._entries[query]
        self._misses += 1
        result = self.search(query)
        size = len(result)
        if size <= min(self.max_entry_hits, self.max_hits):
            while self._entries and (
                len(self._entries) >= self.maxsize or self._cached_hits + size > self.max_hits
            ):
                _, evicted = self._entries.popitem(last=False)
                self._cached_hits -= len(evicted)
            self._entries[query] = result
            self._cached_hits += size
        return result

    def cache_info(self):
        return _CacheInfo(
            self._hits,
            self._misses,
            self.maxsize,
            len(self._entries),
            self._cached_hits,
            self.max_hits,
        )


class _SequenceIndex:
    """Exact lookup plus recall-preserving 4-mer/native Levenshtein search.

    No all-pairs Python loop or dense distance matrix. Each distinct query is
    cached within a fixed limit; repeated clonotypes across donors reuse work.
    """

    # Seeded build-inclusive benchmarks show small scans are cheaper; the
    # auxiliary index amortizes at roughly 650–900 distinct eligible queries.
    _PREFILTER_MIN_SEQUENCES = 4096
    _PREFILTER_MIN_QUERIES = 1024
    _GRAM_LENGTH = 4

    def __init__(
        self,
        rows: pl.DataFrame,
        chain: str,
        max_distance: int,
        *,
        query_count: Optional[int] = None,
    ):
        by_sequence = defaultdict(list)
        values = rows.get_column(chain) if chain in rows.columns else ()
        for index, value in enumerate(values):
            if _usable_cdr3(value):
                by_sequence[value].append(index)
        # Retain the built mapping instead of briefly duplicating its hash table.
        by_sequence.default_factory = None
        self.by_sequence = by_sequence
        lengths = defaultdict(list)
        for sequence in self.by_sequence:
            lengths[len(sequence)].append(sequence)
        self.lengths = dict(lengths)
        self.max_distance = max_distance
        self._sequences = ()
        self._qgrams = None
        if (
            max_distance > 0
            and len(self.by_sequence) >= self._PREFILTER_MIN_SEQUENCES
            and (query_count is None or query_count >= self._PREFILTER_MIN_QUERIES)
        ):
            self._sequences = tuple(self.by_sequence)
            # Compact native integer postings retain every unique sequence. IDs
            # expand back to all provenance rows only after exact verification.
            typecode = "I" if len(self._sequences) < 2**32 else "Q"
            qgrams = defaultdict(lambda: array(typecode))
            for index, sequence in enumerate(self._sequences):
                for gram in {
                    sequence[pos : pos + self._GRAM_LENGTH]
                    for pos in range(len(sequence) - self._GRAM_LENGTH + 1)
                }:
                    qgrams[gram].append(index)
            qgrams.default_factory = None
            self._qgrams = qgrams
        self.find = _SequenceQueryCache(self._find)

    def _prefilter_choices(self, sequence: str) -> Optional[list[str]]:
        """Return a verified-candidate superset, or None for a native full scan.

        Partition the query into k+1 disjoint segments and choose one 4-mer
        entirely within each segment. Each edit can disrupt at most one selected
        substring, so <=k edits leave at least one selected substring unchanged
        somewhere in the reference. Taking the UNION of its postings is therefore
        recall-preserving, including insertions/deletions at segment boundaries.
        Rarest-within-segment selection only affects speed, never recall.
        """
        parts = self.max_distance + 1
        width = self._GRAM_LENGTH
        if self._qgrams is None or len(sequence) < parts * width:
            return None
        selected = set()
        for part in range(parts):
            start, stop = len(sequence) * part // parts, len(sequence) * (part + 1) // parts
            grams = (sequence[pos : pos + width] for pos in range(start, stop - width + 1))
            selected.add(min(grams, key=lambda gram: (len(self._qgrams.get(gram, ())), gram)))
        postings = [self._qgrams.get(gram, ()) for gram in selected]
        low, high = len(sequence) - self.max_distance, len(sequence) + self.max_distance
        compatible_count = sum(len(self.lengths.get(length, ())) for length in range(low, high + 1))
        # Broad/low-complexity postings are faster to scan natively. This is a
        # complete-search fallback, never a cap on returned candidates/evidence.
        if sum(map(len, postings)) > compatible_count // 4:
            return None
        candidates = set()
        for posting in postings:
            candidates.update(posting)
        return [
            self._sequences[index]
            for index in sorted(candidates)
            if low <= len(self._sequences[index]) <= high
        ]

    def _find(self, sequence: str) -> tuple[tuple[int, int], ...]:
        if self.max_distance == 0:
            return tuple((index, 0) for index in self.by_sequence.get(sequence, ()))
        found = []
        choices = self._prefilter_choices(sequence)
        buckets = (
            [choices]
            if choices is not None
            else (
                self.lengths.get(length, ())
                for length in range(
                    len(sequence) - self.max_distance, len(sequence) + self.max_distance + 1
                )
            )
        )
        for choices in buckets:
            for choice, distance, _ in process.extract_iter(
                sequence, choices, scorer=Levenshtein.distance, score_cutoff=self.max_distance
            ):
                found.extend((index, int(distance)) for index in self.by_sequence[choice])
        return tuple(found)


_SHARED_ALPHA_NAMES = {
    "TRAV14/DV4": "TRAV14",
    "TRAV23/DV6": "TRAV23",
    "TRAV29/DV5": "TRAV29",
    "TRAV36/DV7": "TRAV36",
    "TRAV38-2/DV8": "TRAV38-2",
}
_GENE_TOKEN = re.compile(
    r"^(TR[AB][VJ])[0-9]+(?:-[0-9]+)*P?(?:/(?:DV[0-9]+|OR[0-9]+(?:-[0-9]+)*))?"
    r"(?:\*[0-9]{2,})?$"
)


@lru_cache(maxsize=16384)
def _gene_alternatives(call: str, prefix: str) -> Optional[frozenset[str]]:
    """Recognize gene-call syntax without converting annotations into conflicts.

    Missing-like strings and unrecognized text remain in the input frame for
    audit. They are not comparable gene identities. This boundary intentionally
    does not apply format-specific Cell Ranger sentinel cleanup to references.
    """
    # A slash preceding another full TRA/TRB gene is an alternative. The slash
    # within TRAV14/DV4 or the orphan TRBV20/OR9-2 is part of one gene name.
    alternatives = frozenset(token.strip() for token in re.split(r"[,|;]|/(?=TR[AB][VJ])", call))
    for token in alternatives:
        match = _GENE_TOKEN.fullmatch(token)
        if match is None or match[1] != prefix:
            return None
    return alternatives


def _gene_alternatives_may_agree(first: str, second: str) -> bool:
    a, _, allele_a = first.partition("*")
    b, _, allele_b = second.partition("*")
    if _SHARED_ALPHA_NAMES.get(a, a) != _SHARED_ALPHA_NAMES.get(b, b):
        return False
    # Alias spelling alone does not establish allele-level equivalence. Preserve
    # that uncertainty, while ordinary fully named disjoint alleles conflict.
    return a != b or not allele_a or not allele_b or allele_a == allele_b


def _gene_status(
    receptor: dict[str, Any], reference: dict[str, Any], shared: tuple[str, ...]
) -> str:
    unknown = False
    for chain in shared:
        for field in ("trav", "traj") if chain == "cdr3a" else ("trbv", "trbj"):
            left, right = receptor.get(field), reference.get(field)
            if not left or not right:
                unknown = True
                continue
            a = _gene_alternatives(left, field.upper())
            b = _gene_alternatives(right, field.upper())
            if a is None or b is None:
                unknown = True
                continue
            if not any(_gene_alternatives_may_agree(x, y) for x in a for y in b):
                return "conflict"
            if len(a) > 1 or len(b) > 1 or a != b:
                unknown = True
    return "unresolved" if unknown else "compatible"


_OUTPUT_SCHEMA = {
    "receptor_id": pl.String,
    "donor_id": pl.String,
    "status": pl.String,
    "peptide": pl.String,
    "hla": pl.String,
    "evidence_type": pl.String,
    "reference_id": pl.String,
    "source": pl.String,
    "reference_evidence": pl.String,
    "distance": pl.Int64,
    "alpha_distance": pl.Int64,
    "beta_distance": pl.Int64,
    "hla_status": pl.String,
    "donor_hla_status": pl.String,
    "panel_hla": pl.String,
    "panel_hla_status": pl.String,
    "gene_status": pl.String,
    "pairing_status": pl.String,
    "unusable_chain_context": pl.String,
    "reason": pl.String,
    "evidence_rank": pl.Int64,
}


class _EvidenceBuffer:
    """Keep Python-object overhead bounded while retaining every evidence row."""

    def __init__(self, batch_size: int = 8192):
        self.batch_size = batch_size
        self.rows = []
        self.chunks = []
        self.count = 0

    def __len__(self) -> int:
        return self.count

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if self.rows:
            self.chunks.append(pl.DataFrame(self.rows, schema=_OUTPUT_SCHEMA))
            self.rows.clear()

    def finish(self) -> pl.DataFrame:
        self._flush()
        return (
            pl.concat(self.chunks, rechunk=False)
            if self.chunks
            else pl.DataFrame(schema=_OUTPUT_SCHEMA)
        )


def screen(
    receptors: pl.DataFrame,
    references: pl.DataFrame,
    panel: pl.DataFrame,
    donors: pl.DataFrame,
    max_distance: int = 1,
) -> pl.DataFrame:
    """Screen a user panel using CDR3 evidence, never probabilities or proof.

    Distance is the sum of Levenshtein edits across all jointly observed usable
    chains, limited to 0..3. A conflicting observed second chain cannot become
    a single-chain match. Every reference/panel restriction retained is emitted;
    a receptor without retained support gets one explicit Unresolved row.
    """
    if (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, int)
        or not 0 <= max_distance <= 3
    ):
        raise ValueError("max_distance must be an integer from 0 to 3")
    receptors = _validate_frame(receptors, _ReceptorRow, "receptors")
    references = _unique_references(_validate_frame(references, ReferenceRow, "references"))
    panel = _validate_frame(panel, PanelRow, "panel")
    donors = _validate_frame(donors, DonorRow, "donors")
    return _screen_validated(receptors, references, panel, donors, max_distance=max_distance)


def _screen_validated(
    receptors: pl.DataFrame,
    references: pl.DataFrame,
    panel: pl.DataFrame,
    donors: pl.DataFrame,
    max_distance: int = 1,
) -> pl.DataFrame:
    """Materialize the validated iterator for callers requiring a DataFrame."""
    return pl.concat(
        list(
            _iter_screen_validated(receptors, references, panel, donors, max_distance=max_distance)
        ),
        rechunk=False,
    )


def _iter_screen_validated(
    receptors: pl.DataFrame,
    references: pl.DataFrame,
    panel: pl.DataFrame,
    donors: pl.DataFrame,
    max_distance: int = 1,
) -> Iterator[pl.DataFrame]:
    """Internal CLI path after all four tables pass their ingestion/loaders.

    These callers already own normalized, validated frames. The public ``screen``
    boundary always validates arbitrary caller-provided frames before entry here.
    Avoid serializing every reference into Python/Pydantic twice in the CLI.
    Complete receptor groups are sorted/ranked in batches of at most 8192 rows;
    a single larger receptor group is yielded whole. Python row dictionaries
    remain chunked even within that group. Peak retained output is therefore
    proportional to the largest receptor group, rather than all evidence.
    Empty input yields one empty frame with the stable output schema.
    """
    if (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, int)
        or not 0 <= max_distance <= 3
    ):
        raise ValueError("max_distance must be an integer from 0 to 3")
    if receptors.height and receptors["receptor_id"].is_duplicated().any():
        raise ValueError("receptor_id must be unique")

    # Dict keys deduplicate restrictions in input order without quadratic list
    # membership checks when a peptide is paired with many panel HLA molecules.
    panel_by_peptide = defaultdict(dict)
    for row in panel.iter_rows(named=True):
        panel_by_peptide[row["peptide"]][row.get("hla")] = None
    donor_by_id = defaultdict(list)
    for row in donors.iter_rows(named=True):
        donor_by_id[row["donor_id"]].append(row["hla"])
    # Canonical immutable typing keys share compatibility work across donors
    # with identical reported alleles, without assuming genotype completeness.
    donor_by_id = {
        identity: tuple(sorted(set(alleles))) for identity, alleles in donor_by_id.items()
    }

    @lru_cache(maxsize=32768)
    def cached_donor_compatibility(restriction, alleles):
        return donor_compatibility(restriction, alleles)

    # Filter before building an index: irrelevant reference antigens cost no search time.
    allowed = pl.Series("peptide", list(panel_by_peptide), dtype=pl.String)
    reference_frame = references.filter(pl.col("peptide").is_in(allowed.implode()))
    indexes = {}
    for chain in ("cdr3a", "cdr3b"):
        has_queries = (
            chain in receptors.columns and receptors[chain].null_count() < receptors.height
        )
        # Build an auxiliary index only when enough distinct, long-enough queries
        # can amortize it; one shared alpha sequence should not pay this cost.
        query_count = (
            receptors.select(
                pl.col(chain)
                .filter(
                    pl.col(chain).cast(pl.String).str.len_bytes()
                    >= _SequenceIndex._GRAM_LENGTH * (max_distance + 1)
                )
                .drop_nulls()
                .n_unique()
            ).item()
            if has_queries
            else 0
        )
        indexes[chain] = _SequenceIndex(
            reference_frame if has_queries else reference_frame.head(0),
            chain,
            max_distance,
            query_count=query_count,
        )

    @lru_cache(maxsize=4096)
    def reference_row(index: int) -> dict[str, Any]:
        return reference_frame.row(index, named=True)

    pending = []
    emitted = False
    for receptor in receptors.sort("receptor_id").iter_rows(named=True):
        output = _EvidenceBuffer()
        donor_alleles = donor_by_id.get(receptor.get("donor_id"), ())
        usable = tuple(chain for chain in ("cdr3a", "cdr3b") if _usable_cdr3(receptor.get(chain)))
        hits = {chain: dict(indexes[chain].find(receptor[chain])) for chain in usable}
        candidate_ids = (
            set().union(*(mapping.keys() for mapping in hits.values())) if hits else set()
        )
        incompatible_count = 0
        for index in sorted(candidate_ids):
            reference = reference_row(index)
            shared = tuple(chain for chain in usable if _usable_cdr3(reference.get(chain)))
            # A missing/unsafe chain stays missing; a known discordant chain rejects the match.
            if not shared or any(index not in hits[chain] for chain in shared):
                continue
            distance = sum(hits[chain][index] for chain in shared)
            if distance > max_distance:
                continue
            gene_status = _gene_status(receptor, reference, shared)
            compatibility = cached_donor_compatibility(reference.get("hla"), donor_alleles)
            if compatibility.status == "incompatible":
                incompatible_count += 1
                continue
            evidence_type = ("exact" if distance == 0 else "similar") + (
                "_paired" if len(shared) == 2 else "_single_chain"
            )
            for panel_hla in panel_by_peptide[reference["peptide"]]:
                panel_match = compare_hla(reference.get("hla"), panel_hla)
                if panel_match.status == "incompatible":
                    incompatible_count += 1
                    continue
                hla_status = (
                    "compatible"
                    if compatibility.status == panel_match.status == "compatible"
                    else "unresolved"
                )
                reasons = [
                    f"CDR3 {evidence_type} evidence; antigen hypothesis requires experimental validation"
                ]
                if len(shared) == 1:
                    reasons.append(
                        "Only {} chain is jointly observed and usable".format(
                            "alpha" if shared[0] == "cdr3a" else "beta"
                        )
                    )
                reasons.append("V/J gene calls: " + gene_status)
                if receptor.get("unusable_chain_context"):
                    reasons.append(
                        "Excluded observed chains: " + receptor["unusable_chain_context"]
                    )
                if receptor["pairing_status"] not in {"paired", "single_alpha", "single_beta"}:
                    reasons.append("Receptor pairing status: " + receptor["pairing_status"])
                if any(
                    reference.get(chain) and not _usable_cdr3(reference[chain])
                    for chain in ("cdr3a", "cdr3b")
                ):
                    reasons.append(
                        "Reference contains an additional unsafe or nonproductive CDR3; preserved without matching it"
                    )
                reasons.append(compatibility.reason)
                if panel_match.status != "compatible":
                    reasons.append("Panel: " + panel_match.reason)
                output.append(
                    {
                        "receptor_id": receptor["receptor_id"],
                        "donor_id": receptor.get("donor_id"),
                        "status": "Candidate"
                        if hla_status == "compatible" and gene_status != "conflict"
                        else "Unresolved",
                        "peptide": reference["peptide"],
                        "hla": reference.get("hla"),
                        "evidence_type": evidence_type,
                        "reference_id": reference["reference_id"],
                        "source": reference["source"],
                        "reference_evidence": reference["evidence"],
                        "distance": distance,
                        "alpha_distance": hits.get("cdr3a", {}).get(index),
                        "beta_distance": hits.get("cdr3b", {}).get(index),
                        "hla_status": hla_status,
                        "donor_hla_status": compatibility.status,
                        "panel_hla": panel_hla,
                        "panel_hla_status": panel_match.status,
                        "gene_status": gene_status,
                        "pairing_status": receptor["pairing_status"],
                        "unusable_chain_context": receptor.get("unusable_chain_context", ""),
                        "reason": "; ".join(reasons),
                    }
                )
        if not output:
            reason = (
                "No usable, unambiguous CDR3 with conserved C and F/W anchors"
                if not usable
                else (
                    "No reference evidence within the total edit-distance limit and user peptide panel"
                )
            )
            if incompatible_count:
                reason += f"; {incompatible_count} sequence-supported reference/panel comparisons excluded by HLA incompatibility"
            if receptor.get("unusable_chain_context"):
                reason += "; Excluded observed chains: " + receptor["unusable_chain_context"]
            output.append(
                {
                    "receptor_id": receptor["receptor_id"],
                    "donor_id": receptor.get("donor_id"),
                    "status": "Unresolved",
                    "evidence_type": "none",
                    "hla_status": "unresolved",
                    "donor_hla_status": "unresolved",
                    "gene_status": "unresolved",
                    "pairing_status": receptor["pairing_status"],
                    "unusable_chain_context": receptor.get("unusable_chain_context", ""),
                    "reason": reason,
                }
            )
        # Flushed chunks must be consumed even if the buffer's chunk size is
        # tuned below the normal emitted-batch target in a future optimization.
        if len(output) >= 8192 or output.chunks:
            if pending:
                batch = _rank_evidence(pl.DataFrame(pending, schema=_OUTPUT_SCHEMA))
                pending.clear()
                emitted = True
                yield batch
            batch = _rank_evidence(output.finish())
            output = None
            emitted = True
            yield batch
        else:
            if pending and len(pending) + len(output) > 8192:
                batch = _rank_evidence(pl.DataFrame(pending, schema=_OUTPUT_SCHEMA))
                pending.clear()
                emitted = True
                yield batch
            pending.extend(output.rows)
    if pending:
        batch = _rank_evidence(pl.DataFrame(pending, schema=_OUTPUT_SCHEMA))
        pending.clear()
        yield batch
    elif not emitted:
        yield pl.DataFrame(schema=_OUTPUT_SCHEMA)


def _rank_evidence(result: pl.DataFrame) -> pl.DataFrame:
    """Sort one or more COMPLETE receptor groups using the report ranking key."""
    return (
        result.with_columns(
            (pl.col("status") != "Candidate").cast(pl.Int8).alias("_status_order"),
            pl.col("evidence_type")
            .replace_strict(
                {
                    "exact_paired": 0,
                    "exact_single_chain": 1,
                    "similar_paired": 2,
                    "similar_single_chain": 3,
                },
                default=4,
            )
            .alias("_sequence_order"),
            pl.col("reference_evidence")
            .str.to_lowercase()
            .replace_strict({"functional": 0, "multimer": 1, "curated": 2}, default=3)
            .alias("_assay_order"),
        )
        .sort(
            [
                "receptor_id",
                "_status_order",
                "_sequence_order",
                "distance",
                "_assay_order",
                "reference_id",
                "panel_hla",
            ],
            nulls_last=True,
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("receptor_id").alias("evidence_rank"))
        .drop("_status_order", "_sequence_order", "_assay_order")
    )
