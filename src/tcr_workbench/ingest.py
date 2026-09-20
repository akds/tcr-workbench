"""Columnar, lossless ingestion of processed human/mouse alpha/beta TCR tables.

``chains`` retains every source record (paired rows become one row per chain),
including all original columns under ``input_`` names. Only productive or
productivity-unknown, complete, unambiguous TRA/TRB junctions without explicit
negative cell/confidence flags enter matching.
A cell with no usable junction still receives an unresolved receptor record.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Optional, Union

import polars as pl
from pydantic import ValidationError

from .models import CELL_SCHEMA, RECEPTOR_SCHEMA, IngestOptions, IngestResult, InputError

GENES = ["trav", "traj", "trbv", "trbj"]
SEQUENCES = ["cdr3a", "cdr3b"]
RECEPTOR_FIELDS = ["donor_id", *SEQUENCES, *GENES, "pairing_status", "unusable_chain_context"]
KEY = ["donor_id", "_cell_key"]
AA = "ACDEFGHIKLMNPQRSTVWY"


def _null() -> pl.Expr:
    return pl.lit(None, dtype=pl.String)


def _clean(name: str) -> pl.Expr:
    return pl.col(name).str.strip_chars().replace("", None)


def _optional(frame: pl.DataFrame, name: str) -> pl.Expr:
    return _clean(name) if name in frame.columns else _null()


def _qc(qc: list, code: str, count: int, message: str) -> None:
    if count:
        qc.append({"code": code, "count": count, "message": message})


def _required(frame: pl.DataFrame, names: list, context: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise InputError(f"{context}: missing required columns: {', '.join(missing)}")


def _check_values(frame: pl.DataFrame, condition: pl.Expr, message: str) -> None:
    bad = frame.filter(condition.fill_null(False))
    if bad.height:
        rows = bad["source_row"].head(5).to_list() if "source_row" in bad.columns else []
        raise InputError(f"{message}; {bad.height} record(s), source rows {rows}")


def _read_table(path: Path) -> pl.DataFrame:
    if not path.is_file():
        raise InputError(f"Input file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(65536)
        if not sample.strip():
            raise InputError(f"Input table is empty: {path}")
        first = sample.splitlines()[0]
        delimiter = "\t" if first.count("\t") > first.count(",") else ","
        headers = next(csv.reader([first], delimiter=delimiter))
        if len(headers) != len(set(headers)):
            raise InputError("Duplicate column names in input header")
        if any(not header or header != header.strip() for header in headers):
            raise InputError("Input column names must be nonempty without surrounding whitespace")
        # Reading all columns as strings preserves barcodes, alleles and raw records.
        frame = pl.read_csv(
            path,
            separator=delimiter,
            infer_schema=False,
            null_values="",
            encoding="utf8",
            try_parse_dates=False,
        )
    except (OSError, UnicodeError, csv.Error, pl.exceptions.PolarsError) as exc:
        raise InputError(f"Cannot read input table {path}: {exc}") from exc
    if not frame.height:
        raise InputError("Input table has a header but no records")
    return frame


def _boolean(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(None, dtype=pl.Boolean)
    cleaned = _clean(column).str.to_lowercase()
    _check_values(
        frame,
        cleaned.is_not_null() & ~cleaned.is_in(["true", "false", "t", "f", "1", "0"]),
        f"Invalid boolean in {column}; use true/false, T/F, or 1/0",
    )
    return pl.when(cleaned.is_null()).then(None).otherwise(cleaned.is_in(["true", "t", "1"]))


def _normalize_10x_sentinels(frame: pl.DataFrame, qc: list) -> pl.DataFrame:
    """Normalize Cell Ranger's explicit missing-call sentinel, retaining raw input."""
    columns = [
        name for name in
        ("v_gene", "d_gene", "j_gene", "c_gene", "productive", "high_confidence", "is_cell")
        if name in frame.columns
    ]
    if not columns:
        return frame
    sentinels = [(_clean(name).str.to_lowercase() == "none").fill_null(False) for name in columns]
    count = sum(frame.select(flag.sum().alias(name) for name, flag in zip(columns, sentinels)).row(0))
    _qc(
        qc, "cellranger_none_sentinel", count,
        "Cell Ranger literal None gene/boolean values were normalized to null; raw values are retained.",
    )
    return frame.with_columns(
        pl.when(flag).then(None).otherwise(pl.col(name)).alias(name)
        for name, flag in zip(columns, sentinels)
    )


def _donors(frame: pl.DataFrame, donor_id: Optional[str]) -> pl.DataFrame:
    if "donor_id" not in frame.columns:
        if donor_id is None:
            raise InputError("Missing donor_id: supply a donor_id column or --donor-id")
        return frame.with_columns(pl.lit(donor_id).alias("donor_id"))
    frame = frame.with_columns(_clean("donor_id"))
    if donor_id is not None:
        _check_values(
            frame,
            pl.col("donor_id").is_not_null() & (pl.col("donor_id") != donor_id),
            "The supplied donor_id conflicts with the donor_id column",
        )
        frame = frame.with_columns(pl.col("donor_id").fill_null(donor_id))
    _check_values(frame, pl.col("donor_id").is_null(), "Missing donor_id")
    _check_values(
        frame,
        pl.col("donor_id").str.contains(r"[\x00-\x1f]"),
        "donor_id contains control characters",
    )
    return frame


def _junction_flags(sequence: pl.Expr) -> tuple:
    ambiguous = (sequence.is_not_null() & ~sequence.str.contains(f"^[{AA}]+$")).fill_null(False)
    incomplete = (sequence.is_null() | ~sequence.str.contains(r"^C[A-Z*]*[FW]$")).fill_null(True)
    return ambiguous, incomplete


def _unusable_flags(
    ambiguous: pl.Expr,
    incomplete: pl.Expr,
    productive: pl.Expr,
    confidence: pl.Expr,
    is_cell: pl.Expr,
    observed: pl.Expr,
) -> dict:
    return {
        "ambiguous_sequence": observed & ambiguous,
        "incomplete_junction": observed & incomplete,
        "nonproductive": observed & (productive == False).fill_null(False),  # noqa: E712
        "low_confidence": observed & (confidence == False).fill_null(False),  # noqa: E712
        "noncell": observed & (is_cell == False).fill_null(False),  # noqa: E712
    }


def _context_text(terms: list) -> pl.Expr:
    return pl.concat_str(
        [pl.when(flag).then(pl.lit(label)).otherwise(None) for label, flag in terms],
        separator="; ",
        ignore_nulls=True,
    ).alias("unusable_chain_context")


def _validate_chains(chains: pl.DataFrame, qc: list) -> pl.DataFrame:
    _check_values(
        chains,
        pl.col("cdr3").is_not_null()
        & ~pl.col("cdr3").str.contains(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO*]+$"),
        "Invalid amino acid symbols in a CDR3 junction",
    )
    _check_values(
        chains,
        pl.col("cell_id").str.contains(r"[\x00-\x1f]"),
        "cell_id contains control characters",
    )
    _check_values(chains, pl.col("locus").is_null(), "Missing chain/locus")
    for gene in ("v_call", "j_call"):
        _check_values(
            chains, pl.col(gene).str.contains(r"[\x00-\x1f]"), f"{gene} contains control characters"
        )
        for locus in ("TRA", "TRB"):
            prefix = locus + ("V" if gene == "v_call" else "J")
            token = prefix + r"[0-9A-Z/*.-]+"
            _check_values(
                chains,
                (pl.col("locus") == locus)
                & pl.col(gene).is_not_null()
                & ~pl.col(gene).str.contains(r"^" + token + r"([,;|] *" + token + r")*$"),
                f"Malformed or locus-inconsistent {gene}; expected {prefix} gene calls",
            )
    ambiguous, incomplete = _junction_flags(pl.col("cdr3"))
    chains = chains.with_columns(
        ambiguous.alias("sequence_ambiguous"),
        incomplete.alias("junction_incomplete"),
    ).with_columns(
        (
            pl.col("locus").is_in(["TRA", "TRB"])
            & pl.col("productive").fill_null(True)
            & pl.col("high_confidence").fill_null(True)
            & pl.col("is_cell").fill_null(True)
            & ~pl.col("sequence_ambiguous")
            & ~pl.col("junction_incomplete")
        ).alias("eligible")
    )
    stats = chains.select(
        (pl.col("productive") == False).sum().alias("nonproductive"),  # noqa: E712
        pl.col("productive").is_null().sum().alias("productivity_unknown"),
        (~pl.col("locus").is_in(["TRA", "TRB"])).sum().alias("unsupported_locus"),
        pl.col("sequence_ambiguous").sum(),
        pl.col("junction_incomplete").sum(),
        (pl.col("high_confidence") == False).sum().alias("low_confidence"),  # noqa: E712
        (pl.col("is_cell") == False).sum().alias("noncell"),  # noqa: E712
        (pl.col("v_call").str.contains(r"[,;|]") | pl.col("j_call").str.contains(r"[,;|]"))
        .sum()
        .alias("gene_ambiguity"),
    ).row(0, named=True)
    messages = {
        "nonproductive": "Nonproductive chains are retained and excluded from candidate pairing.",
        "productivity_unknown": "Unknown productivity is retained; usable junctions remain candidates.",
        "unsupported_locus": "Non-TRA/TRB chains are retained and excluded from candidate pairing.",
        "sequence_ambiguous": "Ambiguous/stop-containing junctions are retained; no sequence is guessed.",
        "junction_incomplete": "Missing or non-C-to-F/W junctions are retained but not matched.",
        "low_confidence": "Low-confidence chains are retained and excluded from candidate pairing.",
        "noncell": "Contigs marked is_cell=false are retained and excluded from candidate pairing.",
        "gene_ambiguity": "Multiple gene/allele calls are retained verbatim; no allele is selected.",
    }
    for code, count in stats.items():
        _qc(qc, code, count, messages[code])
    return chains


def _check_cell_collisions(chains: pl.DataFrame) -> None:
    """Reject recycled donor barcodes when library provenance proves a collision."""
    for name in ("input_sample_id", "input_library_id"):
        if name in chains.columns:
            collisions = (
                chains.filter(pl.col("cell_id").is_not_null())
                .group_by("donor_id", "cell_id")
                .agg(pl.col(name).drop_nulls().n_unique().alias("libraries"))
                .filter(pl.col("libraries") > 1)
            )
            if collisions.height:
                raise InputError(
                    f"Barcode collision across {name.removeprefix('input_')} values in "
                    f"{collisions.height} donor/cell combinations; prefix cell IDs with the "
                    "sample/library identifier before combining libraries"
                )


def _long_chains(frame: pl.DataFrame, raw: pl.DataFrame, fmt: str, qc: list) -> pl.DataFrame:
    if fmt == "10x":
        _required(frame, ["barcode", "chain", "cdr3"], "10x input")
        mapping = {
            "cell_id": "barcode",
            "locus": "chain",
            "cdr3": "cdr3",
            "v_call": "v_gene",
            "j_call": "j_gene",
            "chain_id": "contig_id",
        }
    else:
        _required(
            frame,
            ["locus", "junction_aa"],
            "AIRR input (junction_aa includes conserved C/F/W anchors)",
        )
        mapping = {
            "cell_id": "cell_id",
            "locus": "locus",
            "cdr3": "junction_aa",
            "v_call": "v_call",
            "j_call": "j_call",
            "chain_id": "sequence_id",
        }
    expressions = [
        _optional(frame, original).alias(normalized) for normalized, original in mapping.items()
    ]
    chains = frame.select(
        "source_row",
        "donor_id",
        *expressions,
        _boolean(frame, "productive").alias("productive"),
        _boolean(frame, "high_confidence").alias("high_confidence"),
        _boolean(frame, "is_cell").alias("is_cell"),
        pl.lit(True).alias("chain_observed"),
    )
    chains = chains.with_columns(
        pl.col("chain_id")
        .fill_null(pl.concat_str([pl.lit("row:"), pl.col("source_row")]))
        .alias("chain_id"),
        pl.col("locus", "cdr3", "v_call", "j_call").str.to_uppercase(),
    )
    if fmt == "10x":
        _check_values(chains, pl.col("cell_id").is_null(), "10x input has a missing barcode")
    duplicates = chains.group_by("donor_id", "chain_id").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise InputError(
            "Duplicate chain identifiers within a donor; contig_id/sequence_id must be unique"
        )
    return _validate_chains(pl.concat([chains, raw], how="horizontal"), qc)


def _paired_chains(frame: pl.DataFrame, raw: pl.DataFrame, qc: list) -> pl.DataFrame:
    _required(frame, ["cell_id", "cdr3a", "cdr3b"], "Paired input")
    frame = frame.with_columns(
        _clean("cell_id"),
        *[_optional(frame, name).str.to_uppercase().alias(name) for name in [*SEQUENCES, *GENES]],
    )
    _check_values(frame, pl.col("cell_id").is_null(), "Paired input has a missing cell_id")
    sides = []
    for suffix, locus, v, j in [("a", "TRA", "trav", "traj"), ("b", "TRB", "trbv", "trbj")]:
        side = frame.select(
            "source_row",
            "donor_id",
            "cell_id",
            pl.lit(locus).alias("locus"),
            pl.col(f"cdr3{suffix}").alias("cdr3"),
            pl.col(v).alias("v_call"),
            pl.col(j).alias("j_call"),
            pl.concat_str([pl.lit(f"row:{suffix}:"), pl.col("source_row")]).alias("chain_id"),
            _boolean(
                frame,
                f"productive_{suffix}" if f"productive_{suffix}" in frame.columns else "productive",
            ).alias("productive"),
            _boolean(frame, "high_confidence").alias("high_confidence"),
            _boolean(frame, "is_cell").alias("is_cell"),
            (
                pl.col(f"cdr3{suffix}").is_not_null()
                | pl.col(v).is_not_null()
                | pl.col(j).is_not_null()
            ).alias("chain_observed"),
            (pl.col("cdr3a").is_null() & pl.col("cdr3b").is_null()).alias("_both_cdr3_missing"),
        )
        # Keep an empty alpha placeholder when the complete source row lacks junctions.
        keep = (
            pl.col("cdr3").is_not_null()
            | pl.col("v_call").is_not_null()
            | pl.col("j_call").is_not_null()
        )
        if suffix == "a":
            keep = keep | pl.col("_both_cdr3_missing")
        sides.append(
            pl.concat([side, raw], how="horizontal").filter(keep).drop("_both_cdr3_missing")
        )
    return _validate_chains(pl.concat(sides, how="vertical"), qc)


def _cell_keys(chains: pl.DataFrame) -> pl.DataFrame:
    # A missing AIRR cell_id never forms a pair, even with adjacent identical sequences.
    return chains.with_columns(
        pl.when(pl.col("cell_id").is_null())
        .then(pl.concat_str([pl.lit("unpaired:"), pl.col("chain_id")]))
        .otherwise(pl.concat_str([pl.lit("cell:"), pl.col("cell_id")]))
        .alias("_cell_key")
    )


def _candidate_pairs(chains: pl.DataFrame, limit: int, qc: list) -> pl.DataFrame:
    cells = chains.select(*KEY, "cell_id").unique()
    usable = chains.filter(pl.col("eligible"))
    a = (
        usable.filter(pl.col("locus") == "TRA")
        .select(
            *KEY,
            pl.col("cdr3").alias("cdr3a"),
            pl.col("v_call").alias("trav"),
            pl.col("j_call").alias("traj"),
        )
        .unique()
    )
    b = (
        usable.filter(pl.col("locus") == "TRB")
        .select(
            *KEY,
            pl.col("cdr3").alias("cdr3b"),
            pl.col("v_call").alias("trbv"),
            pl.col("j_call").alias("trbj"),
        )
        .unique()
    )
    counts = (
        cells.join(a.group_by(KEY).len().rename({"len": "na"}), on=KEY, how="left")
        .join(b.group_by(KEY).len().rename({"len": "nb"}), on=KEY, how="left")
        .with_columns(pl.col("na", "nb").fill_null(0))
    )
    counts = counts.with_columns(
        (
            pl.col("na").clip(lower_bound=1).cast(pl.UInt64)
            * pl.col("nb").clip(lower_bound=1).cast(pl.UInt64)
            > limit
        ).alias("excess")
    )
    _qc(
        qc,
        "excess_chains",
        counts["excess"].sum(),
        f"Cells exceeding {limit} candidate pairs are unresolved; all original chains remain available.",
    )
    allowed = counts.filter(~pl.col("excess")).select(KEY)
    pairs = counts.join(a.join(allowed, on=KEY, how="semi"), on=KEY, how="left").join(
        b.join(allowed, on=KEY, how="semi"), on=KEY, how="left"
    )
    flags = _unusable_flags(
        pl.col("sequence_ambiguous"),
        pl.col("junction_incomplete"),
        pl.col("productive"),
        pl.col("high_confidence"),
        pl.col("is_cell"),
        pl.col("chain_observed"),
    )
    context_columns = []
    context_terms = []
    for locus, label in (("TRA", "alpha"), ("TRB", "beta")):
        for reason, flag in flags.items():
            name = f"_context_{label}_{reason}"
            context_columns.append(((pl.col("locus") == locus) & flag).any().alias(name))
            context_terms.append((f"{label}:{reason}", pl.col(name)))
    quality = chains.group_by(KEY).agg(
        (pl.col("sequence_ambiguous") | pl.col("junction_incomplete")).any().alias("unsafe"),
        (pl.col("high_confidence") == False).any().alias("low_confidence"),  # noqa: E712
        (pl.col("is_cell") == False).any().alias("noncell"),  # noqa: E712
        *context_columns,
    )
    pairs = pairs.join(quality, on=KEY, how="left").with_columns(
        _context_text(context_terms),
        pl.when(pl.col("excess"))
        .then(pl.lit("excess_chains"))
        .when((pl.col("na") + pl.col("nb") == 0) & (pl.col("low_confidence") | pl.col("noncell")))
        .then(pl.lit("excluded_quality"))
        .when((pl.col("na") + pl.col("nb") == 0) & pl.col("unsafe"))
        .then(pl.lit("unresolved_sequence"))
        .when(pl.col("na") + pl.col("nb") == 0)
        .then(pl.lit("no_productive_chains"))
        .when(pl.col("cell_id").is_null())
        .then(pl.lit("unpaired_airr"))
        .when((pl.col("na") > 1) & (pl.col("nb") > 1))
        .then(pl.lit("ambiguous_pairing"))
        .when(pl.col("na") > 1)
        .then(pl.lit("dual_alpha"))
        .when(pl.col("nb") > 1)
        .then(pl.lit("dual_beta"))
        .when(pl.col("na") == 0)
        .then(pl.lit("single_beta"))
        .when(pl.col("nb") == 0)
        .then(pl.lit("single_alpha"))
        .otherwise(pl.lit("paired"))
        .alias("pairing_status"),
    )
    for status in ("dual_alpha", "dual_beta", "ambiguous_pairing"):
        count = pairs.filter(pl.col("pairing_status") == status).select(KEY).unique().height
        _qc(
            qc,
            status,
            count,
            "Within-cell alternatives are retained; physical pairing is not asserted.",
        )
    return pairs


def _paired_candidates(frame: pl.DataFrame, qc: list) -> pl.DataFrame:
    # Keep explicit wide pairs wide: reconstructing them from the long audit
    # table would need two million-key joins on a million-cell repertoire.
    frame = frame.with_columns(
        _clean("cell_id"),
        *[_optional(frame, name).str.to_uppercase().alias(name) for name in [*SEQUENCES, *GENES]],
    )
    confidence = _boolean(frame, "high_confidence").fill_null(True)
    is_cell = _boolean(frame, "is_cell").fill_null(True)
    expressions = []
    context_terms = []
    observed_terms = []
    unsafe_terms = []
    for side, genes in [("a", ["trav", "traj"]), ("b", ["trbv", "trbj"])]:
        sequence = pl.col(f"cdr3{side}")
        productivity = _boolean(
            frame, f"productive_{side}" if f"productive_{side}" in frame.columns else "productive"
        ).fill_null(True)
        ambiguous, incomplete = _junction_flags(sequence)
        safe = (~ambiguous & ~incomplete & productivity & confidence & is_cell).fill_null(False)
        observed = pl.any_horizontal(pl.col(name).is_not_null() for name in [f"cdr3{side}", *genes])
        observed_terms.append(observed)
        unsafe_terms.append(observed & (ambiguous | incomplete))
        flags = _unusable_flags(ambiguous, incomplete, productivity, confidence, is_cell, observed)
        label = "alpha" if side == "a" else "beta"
        context_terms.extend((f"{label}:{reason}", flag) for reason, flag in flags.items())
        expressions.extend(
            pl.when(safe).then(pl.col(name)).otherwise(None).alias(name)
            for name in [f"cdr3{side}", *genes]
        )
    pairs = frame.select(
        "donor_id", "cell_id", *expressions, _context_text(context_terms),
        (pl.any_horizontal(unsafe_terms) | ~pl.any_horizontal(observed_terms)).alias("_unsafe"),
        (~confidence | ~is_cell).alias("_excluded_quality"),
    ).with_columns(
        pl.concat_str([pl.lit("cell:"), pl.col("cell_id")]).alias("_cell_key"),
        pl.when(pl.col("cdr3a").is_null() & pl.col("cdr3b").is_null() & pl.col("_excluded_quality"))
        .then(pl.lit("excluded_quality"))
        .when(pl.col("cdr3a").is_null() & pl.col("cdr3b").is_null() & pl.col("_unsafe"))
        .then(pl.lit("unresolved_sequence"))
        .when(pl.col("cdr3a").is_null() & pl.col("cdr3b").is_null())
        .then(pl.lit("no_productive_chains"))
        .when(pl.col("cdr3a").is_null())
        .then(pl.lit("single_beta"))
        .when(pl.col("cdr3b").is_null())
        .then(pl.lit("single_alpha"))
        .otherwise(pl.lit("paired"))
        .alias("pairing_status"),
    )
    repeated = pairs.filter(pl.struct(KEY).is_duplicated())
    alternatives = (
        repeated.select(*KEY, *SEQUENCES, *GENES)
        .unique()
        .group_by(KEY)
        .len()
        .filter(pl.col("len") > 1)
    )
    _qc(
        qc,
        "multiple_paired_rows",
        alternatives.height,
        "Cells with multiple supplied pairs retain those pairs without creating new combinations.",
    )
    if alternatives.height:
        pairs = pairs.join(
            alternatives.select(*KEY, pl.lit(True).alias("multiple")), on=KEY, how="left"
        ).with_columns(
            pl.when(pl.col("multiple").fill_null(False))
            .then(pl.lit("ambiguous_input_pairs"))
            .otherwise(pl.col("pairing_status"))
            .alias("pairing_status")
        )
    return pairs


def _stable_id(value: str) -> str:
    return "tcr_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _collapse(pairs: pl.DataFrame) -> tuple:
    # Stable, length-prefixed identity distinguishes missing values and arbitrary strings.
    # SHA is applied only after vectorized deduplication, not to every input chain.
    # Two unknown receptors cannot establish a shared clonotype. Scope null
    # placeholders to their original cell or unpaired observation.
    pairs = pairs.with_columns(
        pl.when(pl.col("cdr3a").is_null() & pl.col("cdr3b").is_null())
        .then(pl.col("_cell_key"))
        .otherwise(None)
        .alias("_unresolved_key")
    )
    identity_fields = [*RECEPTOR_FIELDS, "_unresolved_key"]
    unique = pairs.select(identity_fields).unique().sort(identity_fields, nulls_last=True)
    identity = pl.concat_str(
        [
            pl.when(pl.col(name).is_null())
            .then(pl.lit("-1:"))
            .otherwise(pl.concat_str([pl.col(name).str.len_bytes(), pl.lit(":"), pl.col(name)]))
            for name in identity_fields
        ],
        separator="",
    )
    unique = unique.with_columns(
        identity.map_elements(_stable_id, return_dtype=pl.String).alias("receptor_id")
    )
    pairs = pairs.join(unique, on=identity_fields, how="left", nulls_equal=True)
    # Missing AIRR cell IDs cannot establish cell abundance. Observation count
    # separately records independent unpaired observations or known cell units.
    counts = pairs.group_by("receptor_id").agg(
        pl.col("cell_id").drop_nulls().n_unique().cast(pl.UInt64).alias("cell_count"),
        pl.col("_cell_key").n_unique().cast(pl.UInt64).alias("observation_count"),
    )
    receptors = (
        unique.join(counts, on="receptor_id")
        .select(list(RECEPTOR_SCHEMA))
        .cast(RECEPTOR_SCHEMA)
        .sort("receptor_id")
    )
    cells = (
        pairs.select(list(CELL_SCHEMA))
        .unique()
        .sort(["donor_id", "cell_id", "receptor_id"], nulls_last=True)
    )
    return receptors, cells


def read_receptors(
    path: Union[str, Path],
    format: str = "auto",
    donor_id: Optional[str] = None,
    max_pairings_per_cell: int = 16,
    species: str = "human",
) -> IngestResult:
    """Read CSV/TSV input without discarding biological or schema uncertainty.

    AIRR requires ``junction_aa``; AIRR ``cdr3_aa`` alone lacks conserved anchors.
    ``max_pairings_per_cell`` bounds multi-chain expansion; excessive cells get
    unresolved placeholders. ``cell_count`` is per candidate and must not be
    summed across receptor alternatives to estimate unique cell abundance.
    Unpaired AIRR rows have cell_count=0; observation_count records their count.
    """
    try:
        options = IngestOptions(
            path=Path(path),
            format=format,
            donor_id=donor_id,
            max_pairings_per_cell=max_pairings_per_cell,
            species=species,
        )
    except (ValidationError, TypeError) as exc:
        raise InputError(str(exc)) from exc
    original = _read_table(options.path)
    if "species" in original.columns:
        supplied = original["species"].drop_nulls().str.to_lowercase().str.strip_chars()
        if supplied.filter(supplied != options.species).len():
            raise InputError("Input species column disagrees with --species; use separate runs for human and mouse")
    if "source_row" in original.columns:
        # Its raw value remains in input_source_row; normalized source_row is ours.
        frame = original.drop("source_row")
    else:
        frame = original
    frame = frame.with_row_index("source_row", offset=2)
    frame = _donors(frame, options.donor_id)
    raw = original.rename({name: f"input_{name}" for name in original.columns})
    fmt = options.format
    if fmt == "auto":
        if {"barcode", "chain", "cdr3"}.issubset(frame.columns):
            fmt = "10x"
        elif {"cdr3a", "cdr3b"}.issubset(frame.columns):
            fmt = "paired"
        elif "locus" in frame.columns and (
            "junction_aa" in frame.columns or "cdr3_aa" in frame.columns
        ):
            fmt = "airr"
        else:
            raise InputError("Cannot detect receptor format; use --format 10x, airr, or paired")
    qc = []
    if fmt == "10x":
        frame = _normalize_10x_sentinels(frame, qc)
    chains = (
        _paired_chains(frame, raw, qc) if fmt == "paired" else _long_chains(frame, raw, fmt, qc)
    )
    _check_cell_collisions(chains)
    pairs = (
        _paired_candidates(frame, qc)
        if fmt == "paired"
        else _candidate_pairs(_cell_keys(chains), options.max_pairings_per_cell, qc)
    )
    receptors, cells = _collapse(pairs)
    if options.species == "mouse":
        # Existing human identifiers remain stable. Mouse identities cannot
        # collide with otherwise identical human donor/gene/junction records.
        identity = {rid: "mouse_" + rid for rid in receptors["receptor_id"]}
        receptors = receptors.with_columns(pl.col("receptor_id").replace_strict(identity))
        cells = cells.with_columns(pl.col("receptor_id").replace_strict(identity))
    receptors = receptors.with_columns(pl.lit(options.species).alias("species"))
    cells = cells.with_columns(pl.lit(options.species).alias("species"))
    chains = chains.with_columns(pl.lit(options.species).alias("species"))
    missing_cells = chains["cell_id"].null_count()
    _qc(
        qc,
        "unpaired_airr",
        missing_cells,
        "AIRR rows without cell_id remain separate chain observations; pairing is never inferred.",
    )
    qc.insert(
        0,
        {
            "code": "input_records",
            "count": original.height,
            "message": f"Read {original.height} source records; retained {chains.height} chain records.",
        },
    )
    return IngestResult(
        chains=chains,
        receptors=receptors,
        cells=cells,
        qc=qc,
        source=str(options.path.resolve()),
        format=fmt,
        species=options.species,
    )
