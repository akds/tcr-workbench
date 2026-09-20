"""Optional model execution and explicitly labelled, reproducible peptide evidence.

DecoderTCR runs in its own Python 3.12 environment. This module never downloads
weights, imports torch, or treats pseudo-log-likelihood as binding probability.
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .model_registry import MODELS, normalize_device, resolve_model, validate_backend

PathLike = Union[str, Path]
AA = "ACDEFGHIKLMNPQRSTVWY"
GENES = ["trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b"]
COMPONENTS = GENES + ["hla", "peptide"]
DEFAULT_MODEL = "DecoderTCR-ESMC_300M"
CHECKPOINTS = {name: spec.checkpoint for name, spec in MODELS.items()}


def normalize_decoder_device(device: str) -> str:
    """Normalize hardware names independently of the selected checkpoint size."""
    return normalize_device(device)


def file_sha256(path: PathLike) -> str:
    """Hash with bounded memory, including checkpoints larger than RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".manifest-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _sidecar(path: PathLike) -> Path:
    return Path(str(path) + ".manifest.json")


def _read_manifest(path: PathLike, required: List[str]) -> Dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid manifest JSON: {path}") from exc
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
    ):
        raise ValueError(f"unsupported or missing manifest schema_version: {path}")
    if any(not isinstance(manifest.get(key), str) or not manifest[key] for key in required):
        raise ValueError(f"manifest requires nonempty fields {required}: {path}")
    return manifest


def _read_csv(path: PathLike) -> pl.DataFrame:
    delimiter = "\t" if str(path).endswith(".tsv") else ","
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    if not header or any(not column.strip() for column in header):
        raise ValueError("CSV must have nonempty column names")
    if len(header) != len(set(header)):
        raise ValueError("CSV contains duplicate column names")
    return pl.read_csv(
        path,
        separator=delimiter,
        infer_schema=False,
        null_values=[""],
    )


def _require(frame: pl.DataFrame, fields: List[str], where: str) -> None:
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"{where}: missing columns {sorted(missing)}")


def _sequence(value: str) -> str:
    value = value.strip().upper()
    if not value or any(letter not in AA for letter in value):
        raise ValueError("sequence must contain only the 20 standard amino acids")
    return value


def _hla(value: str) -> str:
    # Lazy import keeps the independent numerical profile usable during startup.
    from .hla import normalize_hla

    return normalize_hla(value)


class _ReceptorComponents(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    trav: str = Field(min_length=1)
    traj: str = Field(min_length=1)
    cdr3a: str
    trbv: str = Field(min_length=1)
    trbj: str = Field(min_length=1)
    cdr3b: str

    @field_validator("cdr3a", "cdr3b")
    @classmethod
    def cdr3(cls, value: str) -> str:
        value = _sequence(value)
        if not value.startswith("C") or value[-1] not in "FW":
            raise ValueError("CDR3 must retain IMGT conserved C and F/W anchors")
        return value

    @field_validator("trav", "traj", "trbv", "trbj")
    @classmethod
    def gene(cls, value: str, info: ValidationInfo) -> str:
        if not value.startswith(info.field_name.upper()):
            raise ValueError(f"{info.field_name} must be a {info.field_name.upper()} gene")
        # IMGT dual-use names identify one gene, not a slash-separated choice.
        # Mouse paralogs include D suffixes and numbered DV subgenes (for
        # example TRAV15D-1/DV6D-1). Exact species-reference membership is
        # checked during reconstruction; arbitrary slash alternatives fail.
        dual_use = re.fullmatch(
            r"TRAV[0-9]+D?(?:-[0-9]+)?/DV[0-9]+D?(?:-[0-9]+)?(?:\*[0-9]+)?", value
        )
        if any(c in value for c in ";,| ") or (
            "/" in value and not (info.field_name == "trav" and dual_use)
        ):
            raise ValueError("one unambiguous gene call is required")
        return value


class _Pair(_ReceptorComponents):
    species: Literal["human", "mouse"] = "human"
    name: str = Field(min_length=1)
    hla: str = Field(min_length=1)
    peptide: str

    @field_validator("peptide")
    @classmethod
    def sequence(cls, value: str) -> str:
        return _sequence(value)

    @field_validator("hla")
    @classmethod
    def validate_hla_species(cls, value: str, info: ValidationInfo) -> str:
        from .species import model_mhc
        return model_mhc(value, info.data.get("species", "human"))

    @classmethod
    def supported_hla(cls, value: str) -> str:
        from .species import model_mhc
        return model_mhc(value, "human")



class LibraryMetadata(BaseModel):
    """Required companion JSON for a precomputed peptide–MHC score table."""

    model_config = ConfigDict(extra="forbid", strict=True)
    predictor: str = Field(min_length=1)
    version: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    units: str = Field(min_length=1)
    higher_is_better: bool
    source: str = Field(min_length=1)
    library_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration: str = "Uncalibrated; these scores are not binding probabilities."


def _validated_panel(panel: pl.DataFrame) -> pl.DataFrame:
    _require(panel, ["peptide", "hla"], "peptide panel")
    if not panel.height:
        return pl.DataFrame(schema={"peptide": pl.String, "hla": pl.String})
    if panel["peptide"].dtype != pl.String or panel["hla"].dtype not in (pl.String, pl.Null):
        raise ValueError("peptide panel sequences and HLA must be strings")
    normalized = panel.select(
        pl.col("peptide").str.strip_chars().str.to_uppercase(), pl.col("hla").cast(pl.String)
    )
    if normalized.filter(
        pl.col("peptide").is_null() | ~pl.col("peptide").str.contains("^[" + AA + "]+$")
    ).height:
        raise ValueError("peptide panel contains missing or invalid peptide sequences")
    try:
        alleles = {
            allele: _hla(allele) if allele and allele.strip() else None
            for allele in normalized["hla"].unique().to_list()
        }
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"peptide panel contains invalid HLA: {exc}") from exc
    return normalized.with_columns(
        pl.col("hla").replace_strict(alleles, return_dtype=pl.String)
    ).unique(maintain_order=True)


def export_decoder_input(
    receptors: pl.DataFrame,
    panel: pl.DataFrame,
    output_path: PathLike,
    *,
    donors: Optional[pl.DataFrame] = None,
    max_pairs: int = 1_000_000,
    species: str = "human",
) -> Dict[str, Any]:
    """Export label-free pairs, mapping, skipped-pair audit and content manifest.

    Only single paired alpha-beta receptors with full V/J calls are eligible.
    Supplied donor typing restricts export to confirmed compatible restrictions.
    Class II requires an explicitly supplied complete alpha/beta heterodimer.
    """
    _require(receptors, ["receptor_id"] + GENES, "receptors")
    if (
        receptors["receptor_id"].null_count()
        or receptors["receptor_id"].n_unique() != receptors.height
    ):
        raise ValueError("receptor_id must be non-null and unique")
    from .species import model_mhc, validate_species
    validate_species(species)
    if "species" in receptors.columns and receptors.filter(pl.col("species") != species).height:
        raise ValueError("Receptor table species disagrees with requested species")
    panel = _validated_panel(panel)
    if max_pairs < 1 or receptors.height * panel.height > max_pairs:
        raise ValueError(f"export exceeds max_pairs={max_pairs}; restrict the peptide panel")
    donor_hla = {}
    if donors is not None:
        _require(donors, ["donor_id", "hla"], "donor typing")
        _require(receptors, ["donor_id"], "receptors with donor typing")
        for row in donors.iter_rows(named=True):
            try:
                donor_id = row["donor_id"]
                if not isinstance(donor_id, str) or not donor_id.strip():
                    raise ValueError("donor_id must be a nonempty string")
                donor_hla.setdefault(donor_id, []).append(_hla(row["hla"]))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"donor typing row {row.get('donor_id')!r}: {exc}") from exc
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping_path = Path(str(output) + ".mapping.csv")
    skipped_path = Path(str(output) + ".skipped.csv")
    n_pairs, n_skipped = 0, 0
    # Bound peak memory independently of the Cartesian output size. Only panel
    # rows are retained, never millions of pair dictionaries.
    from .hla import donor_compatibility

    target_reasons = {}
    for hla in panel["hla"].unique().to_list():
        if not hla:
            target_reasons[hla] = "panel HLA restriction is missing; unresolved"
        else:
            try:
                model_mhc(hla, species)
                target_reasons[hla] = None
            except ValueError as exc:
                target_reasons[hla] = str(exc)
    targets = panel.to_dicts()
    donor_cache = {}
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output.parent) as temporary:
        paths = [Path(temporary) / p.name for p in [output, mapping_path, skipped_path]]
        from contextlib import ExitStack

        with ExitStack() as stack:
            handles = [
                stack.enter_context(p.open("w", newline="", encoding="utf-8")) for p in paths
            ]
            pair_writer = csv.DictWriter(handles[0], fieldnames=["name"] + COMPONENTS)
            map_writer = csv.DictWriter(
                handles[1], fieldnames=["name", "receptor_id", "donor_id", "peptide", "hla"]
            )
            skip_writer = csv.DictWriter(
                handles[2],
                fieldnames=["receptor_id", "donor_id", "peptide", "hla", "status", "reason"],
            )
            for writer in (pair_writer, map_writer, skip_writer):
                writer.writeheader()
            for receptor in receptors.iter_rows(named=True):
                receptor_reason = None
                if receptor.get("pairing_status") not in ("paired", "single_pair"):
                    receptor_reason = "a single resolved alpha-beta pairing_status is required"
                try:
                    values = _ReceptorComponents.model_validate(receptor).model_dump()
                except (ValueError, TypeError) as exc:
                    values = None
                    receptor_reason = (
                        receptor_reason or f"invalid or ambiguous model components: {exc}"
                    )
                for target in targets:
                    hla = target["hla"]
                    reason = target_reasons[hla] or receptor_reason
                    donor_id = receptor.get("donor_id")
                    if reason is None and donors is not None:
                        key = donor_id, hla
                        if key not in donor_cache:
                            if donor_id not in donor_hla:
                                donor_cache[key] = "donor HLA typing not supplied"
                            else:
                                match = donor_compatibility(hla, donor_hla[donor_id])
                                donor_cache[key] = (
                                    None
                                    if match.status == "compatible"
                                    else (f"donor HLA {match.status}: {match.reason}")
                                )
                        reason = donor_cache[key]
                    base = {"receptor_id": receptor["receptor_id"], "donor_id": donor_id, **target}
                    if reason is None:
                        stable = json.dumps(
                            [receptor["receptor_id"], hla, target["peptide"]], separators=(",", ":")
                        )
                        name = "pair_" + hashlib.sha256(stable.encode()).hexdigest()[:24]
                        pair = {"name": name, **values, **target}
                        pair_writer.writerow(pair)
                        map_writer.writerow({"name": pair["name"], **base})
                        n_pairs += 1
                    else:
                        skip_writer.writerow({**base, "status": "Unresolved", "reason": reason})
                        n_skipped += 1
        for temporary_path, destination in zip(paths, [output, mapping_path, skipped_path]):
            temporary_path.replace(destination)
    result = {
        "schema_version": 1,
        "input_sha256": file_sha256(output),
        "mapping_sha256": file_sha256(mapping_path),
        "mapping_path": mapping_path.name,
        "skipped_path": skipped_path.name,
        "exported_pairs": n_pairs,
        "skipped_pairs": n_skipped,
        "labels_in_model_input": False,
        "species": species,
    }
    _json_write(_sidecar(output), result)
    return result


def _input_frame(path: PathLike, *, species: str = "human") -> pl.DataFrame:
    frame = _read_csv(path)
    _require(frame, ["name"] + COMPONENTS, "DecoderTCR input")
    if set(frame.columns) != set(["name"] + COMPONENTS):
        raise ValueError(
            "model input may contain only name and receptor/pMHC components; keep labels separate"
        )
    if frame["name"].null_count() or frame["name"].n_unique() != frame.height:
        raise ValueError("DecoderTCR input name must be non-null and unique per pair")
    if frame.filter(pl.col("name").str.strip_chars() == "").height:
        raise ValueError("DecoderTCR input name cannot be blank")
    for row in frame.select(GENES).unique().iter_rows(named=True):
        validated = _ReceptorComponents.model_validate(row).model_dump()
        if any(validated[k] != row[k] for k in GENES):
            raise ValueError("model input must already contain normalized components")
    from .species import model_mhc
    for hla in frame["hla"].unique().to_list():
        if model_mhc(hla, species) != hla:
            raise ValueError("model input must already contain normalized HLA")
    if frame.filter(
        pl.col("peptide").is_null() | ~pl.col("peptide").str.contains("^[" + AA + "]+$")
    ).height:
        raise ValueError("model input must contain normalized peptide sequences")
    return frame


def import_decoder_scores(
    input_path: PathLike,
    scored_path: PathLike,
    *,
    model: str = DEFAULT_MODEL,
    require_manifest: bool = False,
    species: str = "human",
) -> pl.DataFrame:
    """Validate pair identities and retain missing/failed rows as Unresolved.

    External scores without a run manifest are marked external_unverified.
    PLL is an uncalibrated model ranking, never an identified antigen.
    """
    model = resolve_model(model).name
    expected = _input_frame(input_path, species=species)
    scored = _read_csv(scored_path)
    metric = "pll_" + model
    _require(scored, ["name"] + COMPONENTS + [metric], "DecoderTCR scores")
    if scored["name"].null_count() or scored["name"].n_unique() != scored.height:
        raise ValueError("scored names must be non-null and unique")
    if scored.join(expected.select("name"), on="name", how="anti").height:
        raise ValueError("scored output contains unknown pair IDs")
    identity = scored.select(["name"] + COMPONENTS).join(expected, on="name", suffix="_expected")
    for field in COMPONENTS:
        if identity.filter(~pl.col(field).eq_missing(pl.col(field + "_expected"))).height:
            raise ValueError(f"scored output changes input components: {field}")
    provenance = "external_unverified"
    manifest_path = _sidecar(scored_path)
    if manifest_path.exists():
        manifest = _read_manifest(manifest_path, ["input_sha256", "output_sha256", "model"])
        if (
            manifest.get("input_sha256") != file_sha256(input_path)
            or manifest.get("output_sha256") != file_sha256(scored_path)
            or manifest.get("model") != model
            or manifest.get("species", "human") != species
        ):
            raise ValueError("score manifest does not match input, output, or model; stale scores")
        provenance = "hash_verified"
    elif require_manifest:
        raise ValueError("verified DecoderTCR run manifest is required")
    reasons = [c for c in scored.columns if c.endswith("_reason")]
    reconstruction = [
        c
        for c in [
            "TRAV",
            "TRAJ",
            "TRBV",
            "TRBJ",
            "HLA_a",
            "HLA_b",
            "TCR_a",
            "TCR_b",
            "tcr_ok",
            "hla_ok",
        ]
        if c in scored.columns
    ]
    result = expected.join(
        scored.select(
            ["name", metric] + reasons + reconstruction + (["ok"] if "ok" in scored.columns else [])
        ),
        on="name",
        how="left",
        maintain_order="left",
    )
    # Keep million-pair imports in columnar buffers rather than Python records.
    raw_score = (
        pl.when(pl.col(metric).is_in(["NA", "NaN", "nan"])).then(None).otherwise(pl.col(metric))
    )
    result = result.with_columns(raw_score.alias("_raw_score"))
    result = result.with_columns(pl.col("_raw_score").cast(pl.Float64, strict=False).alias("score"))
    if result.filter(pl.col("_raw_score").is_not_null() & pl.col("score").is_null()).height:
        raise ValueError("invalid numeric DecoderTCR score")
    if "ok" in result.columns:
        result = result.with_columns(pl.col("ok").str.strip_chars().str.to_lowercase())
        if result.filter(
            pl.col("ok").is_not_null() & ~pl.col("ok").is_in(["true", "false", "1", "0"])
        ).height:
            raise ValueError("invalid DecoderTCR ok flag")
        backend_failed = pl.col("ok").is_in(["false", "0"]).fill_null(False)
    else:
        backend_failed = pl.lit(False)
    reason = (
        pl.concat_str(
            [
                pl.when(pl.col(c).str.strip_chars() != "").then(pl.col(c)).otherwise(None)
                for c in reasons
            ],
            separator="; ",
            ignore_nulls=True,
        )
        if reasons
        else pl.lit("")
    )
    result = result.with_columns(reason.alias("reason"))
    failed = (
        pl.col("score").is_null()
        | ~pl.col("score").is_finite()
        | backend_failed
        | (pl.col("reason") != "")
    )
    result = result.with_columns(failed.alias("_failed"))
    result = result.with_columns(
        pl.when(pl.col("_failed")).then(None).otherwise(pl.col("score")).alias("score"),
        pl.when(pl.col("_failed"))
        .then(pl.lit("Unresolved"))
        .otherwise(pl.lit("ModelHypothesis"))
        .alias("status"),
        pl.when(pl.col("reason") != "")
        .then(pl.col("reason"))
        .when(backend_failed)
        .then(pl.lit("backend reported reconstruction failure"))
        .when(pl.col("_failed"))
        .then(pl.lit("missing or non-finite model score"))
        .otherwise(pl.lit("uncalibrated PLL; experimental validation required"))
        .alias("reason"),
        pl.lit(metric).alias("metric"),
        pl.lit(provenance).alias("provenance"),
    ).select(
        ["name"]
        + COMPONENTS
        + ["score", "status", "reason", "metric", "provenance"]
        + reconstruction
        + (["ok"] if "ok" in result.columns else [])
        + reasons
    )
    result = _annotate_gene_resolution(result)
    input_manifest = _sidecar(input_path)
    if input_manifest.exists():
        manifest = _read_manifest(
            input_manifest, ["input_sha256", "mapping_path", "mapping_sha256"]
        )
        if manifest.get("input_sha256") != file_sha256(input_path):
            raise ValueError("input manifest is stale")
        mapping_path = Path(manifest["mapping_path"])
        if not mapping_path.is_absolute():
            mapping_path = Path(input_path).parent / mapping_path
        if file_sha256(mapping_path) != manifest.get("mapping_sha256"):
            raise ValueError("input receptor mapping has changed")
        mapping = _read_csv(mapping_path)
        _require(mapping, ["name", "receptor_id", "donor_id"], "input receptor mapping")
        mapping = mapping.select("name", "receptor_id", "donor_id")
        if (
            mapping["name"].null_count()
            or mapping["receptor_id"].null_count()
            or mapping["name"].n_unique() != mapping.height
            or mapping.height != expected.height
            or mapping.join(expected.select("name"), on="name", how="anti").height
        ):
            raise ValueError("input receptor mapping must cover each pair ID exactly once")
        result = result.join(mapping, on="name", how="left", validate="1:1", maintain_order="left")
        result = result.with_columns(pl.lit("linked").alias("mapping_status"))
    else:
        result = result.with_columns(
            pl.lit(None, dtype=pl.String).alias("receptor_id"),
            pl.lit(None, dtype=pl.String).alias("donor_id"),
            pl.lit("mapping_unavailable").alias("mapping_status"),
            pl.concat_str(
                "reason",
                pl.lit("receptor mapping unavailable; export sidecar not supplied"),
                separator="; ",
            ).alias("reason"),
        )
    return result


def _annotate_gene_resolution(frame: pl.DataFrame) -> pl.DataFrame:
    """Expose upstream family/allele assumptions without silently rewriting input.

    DecoderTCR can replace unknown/family calls with a functional family member.
    Harmless numeric zero-padding is distinguished from changed gene identity.
    Missing allele suffixes remain explicit assumptions even when genes agree.
    """
    gene_fields = ["trav", "traj", "trbv", "trbj"]
    missing_reports, inferred, spelling, messages = [], [], [], []
    for field in gene_fields:
        provided = (
            pl.col(field)
            .str.replace(r"^(TR[AB][VJ])0+([0-9])", "${1}${2}")
            .str.replace(r"-0+([0-9])", "-${1}")
        )
        selected_field = field.upper()
        if selected_field not in frame.columns:
            missing_reports.append(pl.lit(True))
            continue
        selected = (
            pl.col(selected_field)
            .str.replace(r"^(TR[AB][VJ])0+([0-9])", "${1}${2}")
            .str.replace(r"-0+([0-9])", "-${1}")
        )
        missing = pl.col(selected_field).is_null() | (
            pl.col(selected_field).str.strip_chars() == ""
        )
        missing_reports.append(missing)
        supplied_allele = pl.col(field).str.contains("*", literal=True)
        differs = (
            ~missing
            & pl.when(supplied_allele)
            .then(provided != selected)
            .otherwise(provided != selected.str.split("*").list.first())
        ).fill_null(False)
        inferred.append(differs)
        spelling.append(~missing & ~differs & (pl.col(field) != pl.col(selected_field)))
        messages.append(
            pl.when(differs)
            .then(
                pl.concat_str(
                    [
                        pl.lit(f"upstream selected {selected_field}="),
                        pl.col(selected_field),
                        pl.lit(f" for input {field}="),
                        pl.col(field),
                    ]
                )
            )
            .otherwise(None)
        )
        messages.append(
            pl.when(spelling[-1])
            .then(
                pl.concat_str(
                    [
                        pl.lit(f"normalized {field} nomenclature from "),
                        pl.col(field),
                        pl.lit(" to "),
                        pl.col(selected_field),
                    ]
                )
            )
            .otherwise(None)
        )
    lacks_allele = pl.any_horizontal(
        [~pl.col(c).str.contains("*", literal=True) for c in gene_fields]
    )
    unreported = pl.any_horizontal(missing_reports)
    changed = pl.any_horizontal(inferred) if inferred else pl.lit(False)
    reformatted = pl.any_horizontal(spelling) if spelling else pl.lit(False)
    messages.extend(
        [
            pl.when(unreported)
            .then(pl.lit("some reconstruction gene choices were not reported"))
            .otherwise(None),
            pl.when(lacks_allele)
            .then(
                pl.lit(
                    "input gene alleles were not fully specified; upstream may use default IMGT alleles"
                )
            )
            .otherwise(None),
        ]
    )
    frame = frame.with_columns(
        pl.when(changed)
        .then(pl.lit("inferred_gene_choice"))
        .when(unreported)
        .then(pl.lit("unreported"))
        .when(lacks_allele)
        .then(pl.lit("alleles_unspecified"))
        .when(reformatted)
        .then(pl.lit("nomenclature_normalized"))
        .otherwise(pl.lit("reported_match"))
        .alias("gene_resolution_status"),
        pl.concat_str(messages, separator="; ", ignore_nulls=True).alias("gene_resolution_reason"),
    )
    return frame.with_columns(
        pl.when((pl.col("status") == "ModelHypothesis") & (pl.col("gene_resolution_reason") != ""))
        .then(pl.concat_str("reason", "gene_resolution_reason", separator="; "))
        .otherwise(pl.col("reason"))
        .alias("reason")
    )


def _model_fingerprint(
    decoder_dir: PathLike, python_executable: PathLike, model: str, *, checkpoint=None
) -> Dict[str, Any]:
    root = Path(decoder_dir).resolve()
    if model not in CHECKPOINTS:
        raise ValueError(f"unsupported model {model}; choose one of {sorted(CHECKPOINTS)}")
    checkpoint = Path(checkpoint).resolve() if checkpoint is not None else root / CHECKPOINTS[model]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DecoderTCR checkpoint is not installed: {checkpoint}")
    return {"model": model, "checkpoint_sha256": file_sha256(checkpoint),
            **_decoder_fingerprint(root, python_executable)}


def _decoder_fingerprint(decoder_dir: PathLike, python_executable: PathLike) -> Dict[str, Any]:
    """Fingerprint reconstruction code and environment without scanning model weights."""
    root = Path(decoder_dir).resolve()
    source = root / "src" / "DecoderTCR"
    if not (source / "utils" / "predict_from_genes.py").is_file():
        raise ValueError(f"not a supported DecoderTCR checkout: {root}")
    code = hashlib.sha256()
    # Include packaged reconstruction references and lockfiles, not only Python.
    paths = [p for p in (root / "src").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths.extend(p for p in [root / "uv.lock", root / "pyproject.toml"] if p.is_file())
    for path in sorted(paths):
        code.update(str(path.relative_to(root)).encode())
        code.update(file_sha256(path).encode())
    python = Path(python_executable).absolute()
    if not python.is_file():
        raise FileNotFoundError(f"DecoderTCR Python executable is missing: {python}")
    environment = _environment_fingerprint(python, root)
    # HLA parsing, schema, and reporting contracts are transitive dependencies
    # of this adapter. Any package source change invalidates cached model output.
    from .report import source_digest
    adapter_sha256 = source_digest()
    return {
        "decoder_source_sha256": code.hexdigest(),
        "workbench_adapter_sha256": adapter_sha256,
        "python_executable": str(python),
        "python_sha256": file_sha256(python),
        "decoder_dir": str(root),
        **environment,
    }


def _decoder_environment(python: PathLike) -> Dict[str, str]:
    environment = os.environ.copy()
    # The chosen model environment owns its imports; parent notebook/shell path
    # overrides must not redirect Python or inject packages from another env.
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = (
        str(Path(python).absolute().parent) + os.pathsep + environment.get("PATH", "")
    )
    # Pinned upstream otherwise obeys ambient USE_FLASH_ATTN even when its model
    # loader disables the separate packed-flash training implementation.
    environment["USE_FLASH_ATTN"] = "0"
    # MLX on M5 otherwise permits TF32 matmul for FP32 arrays. The validated
    # Apple path requires genuine FP32 accumulation, set before importing MLX.
    environment["MLX_ENABLE_TF32"] = "0"
    return environment


def _environment_fingerprint(python: Path, root: Path) -> Dict[str, str]:
    """Identify installed dependencies and external stitchr germline resources.

    Query package metadata and stitchr's resource location in the model environment;
    do not import torch or instantiate a model to decide whether to reuse scores.
    """
    script = """
import importlib.metadata as metadata
import importlib.util
import importlib
import json
from pathlib import Path
import platform
import sys
spec = importlib.util.find_spec('DecoderTCR')
germline_roots = []
for name in ('Stitchr', 'stitchr'):
    package = importlib.util.find_spec(name)
    if package and package.origin:
        germline_roots.append(str(Path(package.origin).resolve().parent))
if importlib.util.find_spec('Stitchr') is not None:
    sf = importlib.import_module('Stitchr.stitchrfunctions')
    germline_roots.append(str(Path(sf.data_dir).resolve()))
packages = sorted((d.metadata.get('Name', ''), d.version) for d in metadata.distributions())
print(json.dumps({'decoder_origin': str(Path(spec.origin).resolve()) if spec and spec.origin else None,
                  'germline_roots': sorted(set(germline_roots)), 'packages': packages,
                  'python_version': sys.version, 'platform': platform.platform()}))
"""
    try:
        process = subprocess.run(
            [str(python), "-c", script],
            cwd=root,
            env=_decoder_environment(python),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("could not inspect the DecoderTCR Python environment") from exc
    if process.returncode:
        raise RuntimeError(f"could not inspect DecoderTCR environment: {process.stderr[-2000:]}")
    try:
        info = json.loads(process.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError("DecoderTCR environment returned invalid metadata") from exc
    expected = (root / "src" / "DecoderTCR" / "__init__.py").resolve()
    if info.get("decoder_origin") != str(expected):
        raise ValueError(
            "Python environment must import DecoderTCR from the supplied checkout; "
            f"expected {expected}, found {info.get('decoder_origin')}"
        )
    germlines = hashlib.sha256()
    for package_dir in info.pop("germline_roots", []):
        package = Path(package_dir)
        for path in sorted(package.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                germlines.update(str(path.relative_to(package)).encode())
                germlines.update(file_sha256(path).encode())
    environment = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return {"environment_sha256": environment, "germline_sha256": germlines.hexdigest()}


_WORKER_PROGRESS_INTERVAL = 30.0


def _worker_progress(stop: threading.Event, started: float) -> None:
    """An elapsed-time heartbeat, never an invented completion estimate."""
    while not stop.wait(_WORKER_PROGRESS_INTERVAL):
        if stop.is_set() or sys.stderr is None:
            return
        try:
            print(f"DecoderTCR worker: {time.monotonic() - started:.0f}s elapsed; still running.",
                  file=sys.stderr, flush=True)
        except (OSError, ValueError):
            # A closed diagnostics stream must not alter model execution.
            return


def _execute(command: List[str], cwd: Path, timeout: Optional[float]) -> None:
    # Stream diagnostics to disk, not a growing capture_output buffer.
    with tempfile.TemporaryFile(mode="w+b") as log:
        stop = threading.Event()
        progress = threading.Thread(target=_worker_progress, args=(stop, time.monotonic()),
                                    name="DecoderTCR-progress", daemon=True)
        try:
            progress.start()
            process = subprocess.run(
                command,
                cwd=cwd,
                env=_decoder_environment(command[0]),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"DecoderTCR exceeded timeout={timeout}s") from exc
        finally:
            stop.set()
            if progress.ident is not None:
                progress.join()
        if process.returncode:
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - 12000))
            detail = log.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DecoderTCR failed (exit {process.returncode}):\n{detail}")


def run_decoder(
    input_path: PathLike,
    output_path: PathLike,
    *,
    decoder_dir: PathLike,
    python_executable: PathLike,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    precision: str = "float32",
    timeout: Optional[float] = None,
    force: bool = False,
    checkpoint: Optional[PathLike] = None,
    mlx_python: Optional[PathLike] = None,
    batch_size: int = 1,
    token_budget: int = 4096,
    cache_bytes: int = 64 * 1024 * 1024,
    species: str = "human",
    mhc_reference: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Run the selected backend atomically; reuse only fully matching hashes.

    A checkpoint is hashed once per call with bounded memory. Any mismatch in
    model, input, source, device or existing output requires force=True.
    """
    device = normalize_decoder_device(device)
    model = validate_backend(model, device, checkpoint, mlx_python, precision).name
    from .backends.mlx_decoder import validate_options
    validate_options(batch_size, token_budget, cache_bytes)
    from .species import biological_fingerprint, verify_biological_fingerprint
    biology = biological_fingerprint(species, mhc_reference)
    frame = _input_frame(input_path, species=species)
    if not frame.height:
        raise ValueError("no eligible pairs to score; inspect the skipped-pair audit")
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output:
        raise ValueError("input and output paths must differ")
    if device == "apple":
        from .backends import mlx_decoder
        model_fingerprint = mlx_decoder.fingerprint(
            decoder_dir, python_executable, model, checkpoint, mlx_python,
            batch_size=batch_size, token_budget=token_budget, cache_bytes=cache_bytes, precision=precision)
    else:
        model_fingerprint = (_model_fingerprint(decoder_dir, python_executable, model)
            if checkpoint is None else _model_fingerprint(
                decoder_dir, python_executable, model, checkpoint=checkpoint))
    fingerprint = {
        **model_fingerprint,
        **biology,
        "input_sha256": file_sha256(source),
        "device": device,
        "precision": precision,
        "approximate": precision == "float16",
        "schema_version": 1,
    }
    if device != "apple":
        fingerprint.update(context_type="tcr-pmhc", backend="torch", dtype="float32", cache_bytes=cache_bytes)
        if checkpoint is not None:
            fingerprint["checkpoint_path"] = str(Path(checkpoint).resolve())
    if output.exists() and not force:
        sidecar = _sidecar(output)
        previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if all(previous.get(k) == v for k, v in fingerprint.items()) and previous.get(
            "output_sha256"
        ) == file_sha256(output):
            import_decoder_scores(source, output, model=model, require_manifest=True, species=species)
            return {**previous, "cache_hit": True}
        raise ValueError(
            "existing scores have missing or different provenance; use force to replace"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="decoder-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "scores.csv"
        if device == "apple":
            runtime = mlx_decoder.execute(source, candidate, temporary, fingerprint, timeout=timeout)
            fingerprint.update(runtime)
        else:
            from .backends.torch_decoder import execute_torch
            fingerprint.update(execute_torch(source, candidate, temporary, fingerprint, timeout=timeout))
        if file_sha256(source) != fingerprint["input_sha256"]:
            raise ValueError("model input changed during inference")
        import_decoder_scores(source, candidate, model=model, species=species)
        verify_biological_fingerprint(fingerprint)
        fingerprint["output_sha256"] = file_sha256(candidate)
        candidate.replace(output)
    fingerprint["metric"] = "pll_" + model
    fingerprint["interpretation"] = "Uncalibrated model hypothesis, not binding probability."
    _json_write(_sidecar(output), fingerprint)
    return {**fingerprint, "cache_hit": False}


def peptide_mhc_lookup(
    panel: pl.DataFrame, library_path: PathLike, metadata_path: PathLike
) -> pl.DataFrame:
    """Exact peptide/HLA lookup in a hashed score library; missing stays unresolved."""
    metadata = LibraryMetadata.model_validate_json(Path(metadata_path).read_text())
    if file_sha256(library_path) != metadata.library_sha256:
        raise ValueError("peptide-MHC library hash does not match metadata")
    targets = _validated_panel(panel)
    library = _read_csv(library_path)
    _require(library, ["peptide", "hla", "score"], "peptide-MHC library")
    # Native expressions validate every row without copying the complete library
    # into Python dicts. HLA parsing is amortized over distinct allele strings.
    scores = library.select(
        pl.col("peptide").str.strip_chars().str.to_uppercase(),
        pl.col("hla"),
        pl.when(pl.col("score").is_in(["NA", "NaN", "nan"]))
        .then(None)
        .otherwise(pl.col("score"))
        .alias("raw_score"),
    )
    invalid_peptide = scores.filter(
        pl.col("peptide").is_null() | ~pl.col("peptide").str.contains("^[" + AA + "]+$")
    )
    if invalid_peptide.height:
        raise ValueError("peptide-MHC library contains missing or invalid peptide sequences")
    alleles = scores["hla"].unique().to_list()
    try:
        normalized = {allele: _hla(allele) for allele in alleles}
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"peptide-MHC library contains invalid HLA: {exc}") from exc
    scores = scores.with_columns(
        pl.col("hla").replace_strict(normalized),
        pl.col("raw_score").cast(pl.Float64, strict=False).alias("score"),
    )
    if scores.filter(
        (pl.col("raw_score").is_not_null() & pl.col("score").is_null())
        | (pl.col("score").is_not_null() & ~pl.col("score").is_finite())
    ).height:
        raise ValueError("peptide-MHC library contains invalid or non-finite numeric scores")
    scores = scores.drop("raw_score")
    if scores.select(pl.struct("peptide", "hla").is_duplicated().any()).item():
        raise ValueError("library contains duplicate peptide/HLA keys; select a single metric/run")
    scores = scores.with_columns(pl.lit(True).alias("_library_key_present"))
    result = targets.join(scores, on=["peptide", "hla"], how="left", maintain_order="left")
    result = result.with_columns(
        pl.when(pl.col("score").is_not_null())
        .then(pl.lit("PrecomputedScore"))
        .otherwise(pl.lit("Unresolved"))
        .alias("status"),
        pl.when(pl.col("score").is_not_null())
        .then(pl.lit("peptide-MHC score; not TCR recognition"))
        .when(pl.col("hla").is_null())
        .then(pl.lit("panel HLA restriction is missing; unresolved"))
        .when(pl.col("_library_key_present").fill_null(False))
        .then(pl.lit("library contains this peptide/HLA pair but its score is missing"))
        .otherwise(pl.lit("no score for this exact peptide/HLA pair"))
        .alias("reason"),
    )
    return result.drop("_library_key_present").with_columns(
        [pl.lit(value).alias(key) for key, value in metadata.model_dump().items()]
    )


def empirical_profile(peptides: pl.DataFrame, *, pseudocount: float = 0.5) -> pl.DataFrame:
    """Length/HLA-stratified descriptive PSSM; never a fitted binding predictor.

    Each distinct peptide is counted once per HLA. Uniform background is 1/20.
    Frequencies use a pseudocount per amino acid; positions are 1-based.
    Missing HLA remains a separate null-HLA composition group.
    """
    if not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be positive and finite")
    panel = _validated_panel(peptides).with_columns(
        pl.col("peptide").str.len_chars().alias("length")
    )
    rows = []
    lookup = np.full(256, -1, dtype=np.int16)
    lookup[np.frombuffer(AA.encode(), dtype=np.uint8)] = np.arange(20)
    for (hla, length), group in panel.group_by("hla", "length", maintain_order=True):
        n = group.height
        encoded = np.frombuffer(
            "".join(group["peptide"].to_list()).encode(), dtype=np.uint8
        ).reshape(n, length)
        tokens = lookup[encoded]
        offsets = np.arange(length, dtype=np.int64) * 20
        counts = np.bincount((tokens + offsets).ravel(), minlength=length * 20).reshape(length, 20)
        frequencies = (counts + pseudocount) / (n + 20 * pseudocount)
        odds = np.log2(frequencies * 20)
        for pos in range(length):
            for aa_index, aa in enumerate(AA):
                rows.append(
                    {
                        "hla": hla,
                        "length": length,
                        "position": pos + 1,
                        "amino_acid": aa,
                        "count": int(counts[pos, aa_index]),
                        "frequency": float(frequencies[pos, aa_index]),
                        "log2_odds": float(odds[pos, aa_index]),
                        "n_peptides": n,
                        "interpretation": "descriptive library composition; not a binding predictor",
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "hla": pl.String,
            "length": pl.Int64,
            "position": pl.Int64,
            "amino_acid": pl.String,
            "count": pl.Int64,
            "frequency": pl.Float64,
            "log2_odds": pl.Float64,
            "n_peptides": pl.Int64,
            "interpretation": pl.String,
        },
    )


def run_decoder_profile(
    components: Dict[str, str],
    output_path: PathLike,
    *,
    length: int,
    decoder_dir: PathLike,
    python_executable: PathLike,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    precision: str = "float32",
    timeout: Optional[float] = None,
    force: bool = False,
    checkpoint: Optional[PathLike] = None,
    mlx_python: Optional[PathLike] = None,
    batch_size: int = 1,
    token_budget: int = 4096,
    cache_bytes: int = 64 * 1024 * 1024,
    species: str = "human",
    mhc_reference: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Get conditional amino-acid marginals using the selected backend.

    Marginals are token distributions, not peptide-binding probabilities.
    Existing output requires force=True. Export supports one complete receptor/HLA context.
    """
    device = normalize_decoder_device(device)
    model = validate_backend(model, device, checkpoint, mlx_python, precision).name
    from .backends.mlx_decoder import validate_options
    validate_options(batch_size, token_budget, cache_bytes)
    if not 1 <= length <= 50:
        raise ValueError("peptide length must be between 1 and 50")
    from .species import biological_fingerprint, verify_biological_fingerprint
    biology = biological_fingerprint(species, mhc_reference)
    pair = _Pair.model_validate({**components, "species": species, "name": "profile", "peptide": "A" * length})
    hla = _hla(pair.hla)
    from .hla import compare_hla

    if compare_hla(hla, hla).status != "compatible":
        raise ValueError("DecoderTCR profile requires an unambiguous allele-level HLA restriction")
    output = Path(output_path).resolve()
    if output.exists() and not force:
        raise ValueError("profile output already exists; use force to replace")
    output.parent.mkdir(parents=True, exist_ok=True)
    if device == "apple":
        from .backends import mlx_decoder
        fingerprint = mlx_decoder.fingerprint(
            decoder_dir, python_executable, model, checkpoint, mlx_python,
            batch_size=batch_size, token_budget=token_budget, cache_bytes=cache_bytes, precision=precision)
    else:
        fingerprint = (_model_fingerprint(decoder_dir, python_executable, model)
            if checkpoint is None else _model_fingerprint(
                decoder_dir, python_executable, model, checkpoint=checkpoint))
    fingerprint.update(precision=precision, approximate=precision == "float16", **biology)
    with tempfile.TemporaryDirectory(prefix="profile-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "profile.csv"
        source = Path(temporary) / "profile_input.csv"
        pl.DataFrame([pair.model_dump()]).select(["name"] + COMPONENTS).write_csv(source)
        if device == "apple":
            executor = mlx_decoder.execute
        else:
            from .backends.torch_decoder import execute_torch
            executor = execute_torch
            fingerprint.update(context_type="tcr-pmhc", backend="torch", device=device,
                               dtype="float32", cache_bytes=cache_bytes)
            if checkpoint is not None:
                fingerprint["checkpoint_path"] = str(Path(checkpoint).resolve())
        runtime = executor(source, candidate, temporary, fingerprint, timeout=timeout, profile=True)
        fingerprint.update(runtime)
        verify_biological_fingerprint(fingerprint)
        profile = pl.read_csv(candidate)
        _require(profile, ["position"] + list(AA), "DecoderTCR profile")
        values = profile.select(list(AA)).to_numpy()
        if runtime.get("status") == "Unresolved":
            if (profile.height or profile.columns != ["position"] + list(AA)
                    or not str(runtime.get("reason") or "").strip()
                    or runtime.get("rows") != 1 or runtime.get("scored") != 0
                    or runtime.get("unresolved") != 1):
                raise ValueError("DecoderTCR returned an invalid unresolved profile audit")
            profile = pl.DataFrame(schema={"position": pl.Int64, **{aa: pl.Float64 for aa in AA}})
        elif (
            profile.height != length
            or profile["position"].null_count()
            or profile["position"].n_unique() != length
            or profile["position"].sort().to_list() != list(range(1, length + 1))
            or not np.isfinite(values).all()
            or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-4)
        ):
            raise ValueError("DecoderTCR returned an invalid marginal profile")
        profile.sort("position").write_csv(candidate)
        candidate.replace(output)
    fingerprint.update(
        {
            "components": {k: getattr(pair, k) for k in GENES + ["hla"]},
            "length": length,
            "device": device,
            "output_sha256": file_sha256(output),
            "interpretation": "conditional amino-acid marginals; not binding probabilities",
        }
    )
    _json_write(_sidecar(output), fingerprint)
    return fingerprint
