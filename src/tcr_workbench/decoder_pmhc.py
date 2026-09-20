"""Auditable DecoderTCR peptide–MHC scoring and HLA-only peptide profiles.

This is the same DecoderTCR masked language model used for receptor screening,
with both TCR chains absent. PLL and profile-derived PSSMs are model hypotheses,
not calibrated affinity, presentation or binding probabilities. Exact class-II
training-reference heterodimers are supported; missing partners are not inferred.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from .backends.precision_contract import Precision
from .hla import normalize_hla, parse_hla
from .models import InputError
from .species import Species, normalize_mhc, biological_fingerprint
from .prediction import AA, DEFAULT_MODEL, _validated_panel, _read_csv, file_sha256
from .report import (output_bundle, source_digest, write_frame, write_json, snapshot_inputs,
                     write_workflow_report, model_report_metadata)

PathLike = Union[str, Path]
INTERPRETATION = "Uncalibrated DecoderTCR peptide log probability conditioned on HLA alone; not binding affinity or probability"
PROFILE_INTERPRETATION = "Conditional amino-acid marginals with all peptide positions masked; not binding probabilities"
RESULT_SCHEMA = {
    "row_id": pl.String, "input_row": pl.UInt64, "input_peptide": pl.String,
    "input_hla": pl.String, "peptide": pl.String, "hla": pl.String,
    "status": pl.String, "score": pl.Float64, "metric": pl.String,
    "units": pl.String, "higher_is_better": pl.Boolean, "reason": pl.String,
    "interpretation": pl.String,
    "background_percentile": pl.Float64, "background_n": pl.UInt32,
}
PROFILE_SCHEMA = {
    "hla": pl.String, "length": pl.UInt32, "position": pl.UInt32,
    "amino_acid": pl.String, "probability": pl.Float64, "background": pl.Float64,
    "log2_odds": pl.Float64, "floor_applied": pl.Boolean,
    "interpretation": pl.String,
}


class PMHCOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    decoder_dir: PathLike
    python_executable: PathLike
    species: Species = "human"
    mhc_reference: Optional[PathLike] = None
    model: str = DEFAULT_MODEL
    device: str = "cpu"
    precision: Precision = "float32"
    checkpoint: Optional[PathLike] = None
    mlx_python: Optional[PathLike] = None
    batch_size: int = Field(default=1, ge=1, le=128)
    token_budget: int = Field(default=4096, ge=3, le=262144)
    cache_bytes: int = Field(default=64 * 1024 * 1024, ge=0, le=1024 * 1024 * 1024)
    timeout: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


class PanelOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    max_pairs: int = Field(default=100000, ge=1, le=10000000)


class ProfileOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    length: int = Field(ge=1, le=50)
    probability_floor: float = Field(default=1e-12, gt=0, le=0.05, allow_inf_nan=False)


def _context_reason(hla: Optional[str], species: str = "human") -> str:
    """Return an abstention reason, or empty string for a lookup-eligible context."""
    if hla is None:
        return "HLA context is missing"
    normalize_mhc(hla, species)
    if species == "mouse":
        return ""
    molecule = parse_hla(hla)
    if molecule.kind == "ambiguous":
        return "HLA allele ambiguity is unresolved; no alternative is selected"
    if any(a.suffix in {"N", "S", "C"} for a in molecule.alleles):
        return "An explicitly null, secreted-only, or cytoplasmic HLA chain cannot confirm surface restriction"
    if any(a.suffix for a in molecule.alleles):
        return "HLA group or expression suffix is unresolved; no representative allele is selected"
    if any(len(a.fields) != 2 for a in molecule.alleles):
        return "Exact two-field HLA protein identifiers are required for the training-reference lookup"
    if any(len(a.fields[0]) != 2 for a in molecule.alleles):
        return "Delimiter-free training-reference keys cannot distinguish this allele safely; a two-digit first field is required"
    if molecule.kind == "heterodimer":
        # parse_hla already enforces DRA/DRB, DQA1/DQB1 or DPA1/DPB1.
        return ""
    if len(molecule.alleles) == 1 and molecule.alleles[0].locus in {"A", "B", "C"}:
        return ""
    return "Class-II context requires an explicit alpha/beta heterodimer; missing partners are not inferred"


def _normalize_rows(panel: pl.DataFrame, species: str = "human") -> tuple[pl.DataFrame, pl.DataFrame]:
    # Reuse the existing panel's strict peptide and HLA syntax contract, while
    # preserving every source row rather than dropping duplicate observations.
    normalized_unique = _validated_panel(panel)
    if not panel.height:
        panel = panel.with_columns(pl.col("peptide").cast(pl.String), pl.col("hla").cast(pl.String))
    hla_map = {value: normalize_mhc(value, species) if value and value.strip() else None
               for value in panel["hla"].cast(pl.String).unique().to_list()}
    normalized = panel.select(
        pl.col("peptide").alias("input_peptide"), pl.col("hla").cast(pl.String).alias("input_hla"),
        pl.col("peptide").str.strip_chars().str.to_uppercase().alias("peptide"),
        pl.col("hla").cast(pl.String).replace_strict(hla_map, return_dtype=pl.String).alias("hla"),
    ).with_row_index("_input_index").with_columns(
        (pl.col("_input_index").cast(pl.UInt64) + 1).alias("input_row"),
        pl.concat_str(pl.lit("pmhc_"), (pl.col("_input_index") + 1).cast(pl.String)).alias("row_id"),
    ).drop("_input_index")
    unique = normalized_unique.with_row_index("_unique_index").with_columns(
        pl.concat_str(pl.lit("context_"), (pl.col("_unique_index") + 1).cast(pl.String)).alias("name")
    ).drop("_unique_index")
    return normalized, unique


def _options(kwargs: dict[str, Any]) -> PMHCOptions:
    try:
        from .model_registry import normalize_device, validate_backend
        options = PMHCOptions.model_validate(kwargs)
        biological_fingerprint(options.species, options.mhc_reference)
        device = normalize_device(options.device)
        model = validate_backend(options.model, device, options.checkpoint, options.mlx_python, options.precision)
        return PMHCOptions.model_validate(options.model_dump() | {"model": model.name, "device": device})
    except ValueError as exc:
        raise InputError(f"Invalid DecoderTCR pMHC options: {exc}") from exc


def _backend(input_path: Path, output_path: Path, options: PMHCOptions, *, profile: bool = False) -> dict:
    from .backends.pmhc_decoder import run_pmhc_backend
    return run_pmhc_backend(input_path, output_path, **options.model_dump(), profile=profile)


def _manifest(staging: Path, *, task: str, options: PMHCOptions, backend: dict, summary: dict, source_sha256: str) -> dict:
    if source_digest() != source_sha256:
        raise InputError("Workbench source changed during pMHC analysis; rerun from stable code")
    return {
        "schema_version": 1, "task": task, "predictor": "DecoderTCR", "with_tcr": False,
        "source_sha256": source_sha256,
        "parameters": json.loads(options.model_dump_json()), "backend": backend, "summary": summary,
        "outputs": {path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                    for path in sorted(staging.iterdir()) if path.is_file()},
    }


def _validated_scores(raw: pl.DataFrame, expected: pl.DataFrame, metric: str) -> pl.DataFrame:
    required = {"name", "peptide", "hla", "ok", "hla_reason", "inference_reason", metric}
    if required - set(raw.columns):
        raise InputError(f"DecoderTCR pMHC output is missing columns: {sorted(required-set(raw.columns))}")
    if raw.height != expected.height or raw["name"].n_unique() != raw.height:
        raise InputError("DecoderTCR pMHC output changed row count or contains duplicate identities")
    for column in ("name", "peptide", "hla"):
        if raw[column].null_count() or raw[column].dtype != pl.String:
            raise InputError(f"DecoderTCR pMHC output has invalid {column}")
    if not raw.select("name", "peptide", "hla").sort("name").equals(expected.select("name", "peptide", "hla").sort("name")):
        raise InputError("DecoderTCR pMHC output identities or peptide/HLA contexts changed")
    flags = raw["ok"].cast(pl.String).str.to_lowercase()
    if flags.is_null().any() or (~flags.is_in(["true", "false", "1", "0"])).any():
        raise InputError("DecoderTCR pMHC output has invalid success flags")
    values = raw[metric].cast(pl.Float64, strict=False)
    if (raw[metric].is_not_null() & values.is_null()).any():
        raise InputError("DecoderTCR pMHC output contains a nonnumeric score")
    good = flags.is_in(["true", "1"])
    if (good & (values.is_null() | ~values.is_finite() | (values > 1e-7))).any():
        raise InputError("DecoderTCR marked a missing, non-finite or positive log probability successful")
    if ((~good) & values.is_not_null() & values.is_finite()).any():
        raise InputError("DecoderTCR returned a finite score for a failed reconstruction")
    reasons = ["; ".join(str(value).strip() for value in values if value is not None and str(value).strip())
               for values in raw.select("hla_reason", "inference_reason").iter_rows()]
    return raw.select("name", "peptide", "hla").with_columns(
        pl.Series("score", [float(value) if valid else None for value, valid in zip(values, good)], dtype=pl.Float64),
        pl.Series("status", ["Scored" if valid else "Unresolved" for valid in good]),
        pl.Series("reason", [reason or ("Exact training-reference HLA sequence; requires experimental validation" if valid
                                       else "DecoderTCR could not resolve or score this HLA context")
                             for reason, valid in zip(reasons, good)], dtype=pl.String),
    )


def prepare_background(contexts: pl.DataFrame, staging: Path, *, background_mode: str,
                       background_peptides: int, background_seed: int,
                       options: PMHCOptions) -> tuple[pl.DataFrame, dict]:
    """Generate one MHC-only profile per distinct MHC/length, without TCR inputs."""
    from .background import background_groups, random_panel
    groups, settings = background_groups(contexts, peptides=background_peptides,
        seed=background_seed, mode=background_mode)
    profiles, failures, runs = {}, {}, []
    execution = None
    if settings.mode == "mhc-profile" and settings.peptides:
        requests = []
        for row in groups.select("hla", "peptide_length").unique().sort("hla", "peptide_length").iter_rows(named=True):
            hla, length = row["hla"], row["peptide_length"]
            identity = json.dumps({**row, "species": options.species}, sort_keys=True)
            prefix = "background_profile_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            reason = _context_reason(hla, options.species)
            requests.append({"name": prefix, "hla": hla, "peptide": "A" * length, "reason": reason})
        request_frame = pl.DataFrame(requests, schema={key: pl.String for key in ("name", "hla", "peptide", "reason")})
        input_path = staging / "background_profile_requests.csv"
        request_frame.write_csv(input_path)
        runnable = request_frame.filter(pl.col("reason") == "").drop("reason")
        parts = {}
        if runnable.height:
            batch_input, batch_output = staging / "background_profile_input.csv", staging / "background_profiles.csv"
            runnable.write_csv(batch_input)
            execution = _backend(batch_input, batch_output, options, profile="batch")
            from .backends.pmhc_decoder import validate_output
            validate_output(runnable, batch_output, options.model, execution, profile="batch")
            parts = pl.read_csv(batch_output).partition_by("name", as_dict=True)
        request_hash = file_sha256(input_path)
        for request in requests:
            hla, length, prefix, reason = request["hla"], len(request["peptide"]), request["name"], request["reason"]
            profile_path = staging / (prefix + ".csv")
            profile = pl.DataFrame(schema={"position": pl.Int64, **{aa: pl.Float64 for aa in AA}})
            if not reason:
                part = parts[(prefix,)]
                if part["status"][0] == "Unresolved":
                    reason = part["reason"][0]
                else:
                    profile = part.select("position", *AA).sort("position")
            if reason:
                failures[(hla, length)] = reason
            else:
                profiles[(hla, length)] = profile
            profile.write_csv(profile_path)
            runs.append({"hla": hla, "peptide_length": length, "species": options.species,
                         "status": "Unresolved" if reason else "Profiled", "reason": reason,
                         "input_file": input_path.name, "input_sha256": request_hash,
                         "profile_file": profile_path.name, "profile_sha256": file_sha256(profile_path)})
    sampled, metadata = random_panel(groups, peptides=settings.peptides, seed=settings.seed,
        mode=settings.mode, profiles=profiles, profile_failures=failures)
    metadata.update(profile_runs=runs, profile_execution=execution,
        percentile_definition="100 * (1 + count(background_score >= candidate_score)) / (finite_background_count + 1)",
        percentile_direction="Lower means fewer reference draws score at least as highly; ties are included",
        percentile_interpretation="Descriptive model-reference upper-tail percentage, not a binding probability or calibrated p-value")
    return sampled, metadata


def verify_background_execution(metadata: dict, scoring: dict) -> None:
    """Reject changed model/runtime identities between generation and scoring."""
    keys = ("model", "checkpoint_sha256", "bundle_sha256", "precision", "device", "species",
            "mhc_reference_sha256", "decoder_source_sha256", "environment_sha256",
            "mlx_environment_sha256", "mlx_source_sha256")
    generation = metadata.get("profile_execution") or {}
    for key in keys:
        if key in generation and generation[key] != scoring.get(key):
            raise InputError(f"Background generation and candidate scoring differ in {key}; rerun from stable inputs")


def score_pmhc(panel: Union[pl.DataFrame, str, Path], output_dir: PathLike, *, max_pairs: int = 100000,
               background_peptides: int = 1000, background_seed: int = 0,
               background_mode: str = "uniform",
               **backend_kwargs) -> tuple[pl.DataFrame, dict]:
    """Score every panel observation, sharing computation for duplicate pairs.

    Valid but unsupported/ambiguous HLA and peptides outside the adapter's 1–50
    length range remain explicit Unresolved rows. Corrupt schemas and malformed
    peptide/HLA syntax fail before model execution. Outputs publish atomically.
    """
    options = _options(backend_kwargs)
    source_sha256 = source_digest()
    limit = PanelOptions(max_pairs=max_pairs)
    snapshot = None
    input_path = None
    if isinstance(panel, (str, Path)):
        input_path = Path(panel).resolve()
        snapshot = snapshot_inputs([input_path])
        panel = pl.read_parquet(input_path) if input_path.suffix.lower() == ".parquet" else _read_csv(input_path)
    if not isinstance(panel, pl.DataFrame):
        raise InputError("panel must be a Polars DataFrame or CSV/TSV/Parquet path")
    panel = panel.clone()
    if panel.height > limit.max_pairs:
        raise InputError(f"Panel exceeds {limit.max_pairs} rows; partition it or increase max_pairs")
    try:
        rows, unique = _normalize_rows(panel, options.species)
    except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
        raise InputError(f"Invalid peptide–MHC panel: {exc}") from exc
    reasons = {hla: _context_reason(hla, options.species) for hla in unique["hla"].unique().to_list()}
    unique = unique.with_columns(pl.col("hla").replace_strict(reasons, return_dtype=pl.String).alias("reason"))
    unique = unique.with_columns(pl.when(pl.col("reason") == "").then(
        pl.when(pl.col("peptide").str.len_chars().is_between(1,50)).then(pl.lit(""))
        .otherwise(pl.lit("Peptide length is outside the supported adapter range of 1–50 residues")))
        .otherwise(pl.col("reason")).alias("reason"))
    eligible = unique.filter(pl.col("reason") == "").select("name", "peptide", "hla")
    from .background import background_groups, background_percentiles
    contexts = eligible.with_columns(pl.col("peptide").str.len_chars().alias("peptide_length"))
    background_groups(contexts, peptides=background_peptides, seed=background_seed, mode=background_mode)
    skipped = unique.filter(pl.col("reason") != "").with_columns(
        pl.lit(None, dtype=pl.Float64).alias("score"), pl.lit("Unresolved").alias("status"))
    from .model_registry import resolve_model
    metric = "pll_" + resolve_model(options.model).name
    with output_bundle(Path(output_dir), input_snapshot=snapshot) as staging:
        background_input, background_metadata = prepare_background(contexts, staging,
            background_mode=background_mode, background_peptides=background_peptides,
            background_seed=background_seed, options=options)
        model_input = pl.concat([eligible, background_input.select(eligible.columns)])
        write_frame(panel, staging / "input_panel.csv")
        logical_digest = hashlib.sha256()
        logical_digest.update(json.dumps({key:str(dtype) for key,dtype in panel.schema.items()}, sort_keys=True).encode())
        logical_digest.update(file_sha256(staging / "input_panel.csv").encode())
        input_identity = {"logical_sha256": logical_digest.hexdigest(), "logical_encoding": "SHA256(sorted dtype schema JSON + SHA256 of canonical CSV export)", "rows": panel.height,
                          "file": snapshot[str(input_path)] if snapshot is not None else None}
        write_frame(rows, staging / "normalized_panel.csv")
        model_input.write_csv(staging / "model_input.csv")
        if model_input.height:
            backend = _backend(staging / "model_input.csv", staging / "model_scores.csv", options)
            verify_background_execution(background_metadata, backend)
            raw = pl.read_csv(staging / "model_scores.csv", infer_schema=False, null_values=[""])
            combined_scores = _validated_scores(raw, model_input, metric)
            scored = combined_scores.join(eligible.select("name"), on="name", how="semi")
            background = combined_scores.join(background_input.select("name", "peptide_length"), on="name", how="inner", validate="1:1")
        else:
            backend = {"status": "not_run", "reason": "No supported, unambiguous model inputs"}
            scored = pl.DataFrame(schema={"name": pl.String, "peptide": pl.String, "hla": pl.String,
                                          "score": pl.Float64, "status": pl.String, "reason": pl.String})
            background = scored.with_columns(pl.lit(None, dtype=pl.UInt32).alias("peptide_length"))
        all_scores = pl.concat([scored, skipped.select(scored.columns)], how="vertical")
        result = rows.join(all_scores.drop("name"), on=["peptide", "hla"], how="left", nulls_equal=True,
                           validate="m:1", maintain_order="left").with_columns(
            pl.lit(metric).alias("metric"), pl.lit("natural_log_units").alias("units"),
            pl.lit(True).alias("higher_is_better"), pl.lit(INTERPRETATION + ("; approximate FP16 inference" if options.precision == "float16" else "")).alias("interpretation"),
        )
        result = background_percentiles(result, background).select(list(RESULT_SCHEMA)).cast(RESULT_SCHEMA)
        if result.height != panel.height or result["status"].null_count():
            raise InputError("pMHC result mapping lost input rows")
        write_frame(result, staging / "results.csv")
        write_frame(background, staging / "background_scores.csv")
        background_metadata.update(scored_rows=background.filter(pl.col("status") == "Scored").height,
                                   unresolved_rows=background.filter(pl.col("status") != "Scored").height)
        write_json(staging / "background_metadata.json", background_metadata)
        summary = {"precision": options.precision, "approximate": options.precision == "float16", "input_rows": panel.height, "unique_pairs": unique.height,
                   "eligible_unique_pairs": eligible.height, "scored_rows": result.filter(pl.col("status") == "Scored").height,
                   "unresolved_rows": result.filter(pl.col("status") == "Unresolved").height,
                   "metric": metric, "interpretation": INTERPRETATION,
                   "comparison_scope": "Same checkpoint, explicit HLA molecule and peptide length; comparisons across contexts are uncalibrated"}
        write_json(staging / "summary.json", summary)
        write_workflow_report(staging, "Peptide–MHC scores", table=result,
            workflow="pmhc-score", background=background, background_metadata=background_metadata,
            columns=["input_row", "peptide", "hla", "score", "status", "reason"],
            summary={key: summary[key] for key in ("input_rows", "unique_pairs", "scored_rows", "unresolved_rows")},
            metadata=model_report_metadata(options.model_dump(), backend), manifest_pending=True,
            context={"Species": options.species, "Conditioning": "MHC only; both TCR chains absent", "Score units": "Natural-log PLL",
                     "Peptide reference": f"{background_peptides} {background_mode} draws per MHC/length; seed {background_seed}; background_metadata.json",
                     "Comparison groups": "Same checkpoint, MHC and peptide length"},
            interpretation="DecoderTCR PLL scores measure model compatibility between each peptide and its MHC context. They are uncalibrated scores, not affinity measurements or binding probabilities. Compare scores within the same species, checkpoint, MHC molecule, peptide length and precision. Duplicate observations are retained. Unresolved denotes insufficient supported context, not nonbinding.")
        manifest = _manifest(staging, task="decoder-peptide-mhc", options=options, backend=backend, summary=summary, source_sha256=source_sha256)
        manifest["input"] = input_identity
        manifest["background"] = background_metadata
        write_json(staging / "manifest.json", manifest)
    return result, manifest


def profile_to_pssm(profile: pl.DataFrame, *, hla: Optional[str] = None, length: Optional[int] = None,
                    probability_floor: float = 1e-12) -> pl.DataFrame:
    """Convert validated 20-AA marginals to log2 odds against uniform background.

    A numeric floor affects only log2_odds, never the reported probabilities.
    There is no pseudocount or renormalization, and no class-II binding-register
    inference. Positions refer to the supplied full peptide length.
    """
    length = profile.height if length is None else length
    options = ProfileOptions(length=length, probability_floor=probability_floor)
    canonical = normalize_hla(hla) if hla is not None else None
    if not {"position", *AA} <= set(profile.columns) or profile.height != length:
        raise InputError("Profile must contain one position row and all 20 amino acids per requested residue")
    if profile["position"].dtype not in (pl.Int8,pl.Int16,pl.Int32,pl.Int64,pl.UInt8,pl.UInt16,pl.UInt32,pl.UInt64):
        raise InputError("Profile positions must be integers")
    if profile["position"].null_count() or profile["position"].sort().to_list() != list(range(1,length+1)):
        raise InputError("Profile positions must be unique, complete and one-based")
    if any(not profile.schema[aa].is_numeric() for aa in AA):
        raise InputError("Profile probabilities must have numeric, non-boolean dtypes")
    profile = profile.sort("position")
    try:
        values = profile.select(list(AA)).to_numpy().astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise InputError("Profile probabilities must be numeric") from exc
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-4):
        raise InputError("Profile probabilities must be finite, nonnegative and sum to one per position")
    positions = np.repeat(np.arange(1,length+1,dtype=np.uint32),20)
    probabilities = values.reshape(-1)
    return pl.DataFrame({
        "hla": [canonical] * len(probabilities), "length": np.full(len(probabilities),length,dtype=np.uint32),
        "position": positions, "amino_acid": list(AA) * length, "probability": probabilities,
        "background": np.full(len(probabilities),0.05),
        "log2_odds": np.log2(np.maximum(probabilities,options.probability_floor)/0.05),
        "floor_applied": probabilities < options.probability_floor,
        "interpretation": [PROFILE_INTERPRETATION] * len(probabilities),
    }, schema=PROFILE_SCHEMA)


def profile_pmhc(hla: Optional[str], length: int, output_dir: PathLike, *,
                 probability_floor: float = 1e-12, **backend_kwargs) -> tuple[pl.DataFrame, dict]:
    """Produce an HLA-only peptide profile, PSSM and explicit coverage audit."""
    options = _options(backend_kwargs)
    source_sha256 = source_digest()
    profile_options = ProfileOptions(length=length, probability_floor=probability_floor)
    try:
        canonical = normalize_mhc(hla, options.species) if hla and hla.strip() else None
        reason = _context_reason(canonical, options.species)
    except (ValueError, TypeError, AttributeError) as exc:
        raise InputError(f"Invalid HLA context: {exc}") from exc
    profile = pl.DataFrame(schema={"position": pl.Int64, **{aa:pl.Float64 for aa in AA}})
    pssm = pl.DataFrame(schema=PROFILE_SCHEMA)
    with output_bundle(Path(output_dir)) as staging:
        request = {"input_hla": hla, "hla": canonical, "length": length, "species": options.species,
                   "with_tcr": False, "probability_floor": probability_floor}
        write_json(staging / "request.json", request)
        model_input = pl.DataFrame({"name": ["hla_profile"], "hla": [canonical], "peptide": ["A"*length]},
                                  schema={"name":pl.String,"hla":pl.String,"peptide":pl.String})
        model_input.write_csv(staging / "model_input.csv")
        if not reason:
            backend = _backend(staging / "model_input.csv", staging / "model_profile.csv", options, profile=True)
            if backend.get("status") == "Unresolved":
                reason = backend.get("reason") or "The exact HLA training-reference sequence is unavailable"
            else:
                profile = pl.read_csv(staging / "model_profile.csv")
                pssm = profile_to_pssm(profile, hla=canonical, length=length,
                                       probability_floor=profile_options.probability_floor)
                profile = profile.sort("position")
        else:
            backend = {"status": "not_run", "reason": reason}
        if options.precision == "float16" and pssm.height:
            pssm = pssm.with_columns((pl.col("interpretation") + pl.lit("; approximate FP16 inference")).alias("interpretation"))
        write_frame(profile, staging / "profile.csv")
        write_frame(pssm, staging / "pssm.csv")
        summary = {**request, "precision": options.precision, "approximate": options.precision == "float16", "status": "Unresolved" if reason else "Profiled", "reason": reason,
                   "profile_positions": profile.height, "background": "uniform 0.05 for each of 20 standard amino acids",
                   "pssm_definition": "log2(max(probability, probability_floor) / 0.05)",
                   "numeric_floor_applies_only_to_log2_odds": True,
                   "conditioning_context": "HLA only; both TCR chains absent",
                   "pssm_score_relationship": "The PSSM uses a 20-amino-acid normalizer and cannot reconstruct the absolute full-vocabulary PLL (64 channels for ESM-C; 33 for ESM-2)",
                   "floored_entries": int(pssm["floor_applied"].sum() or 0),
                   "interpretation": PROFILE_INTERPRETATION,
                   "class_ii_register_policy": "No binding-core register is inferred; positions span the requested peptide length"}
        write_json(staging / "summary.json", summary)
        write_workflow_report(staging, "MHC-conditioned peptide profile", table=pssm, profile=profile,
            workflow="pmhc-profile",
            columns=["position", "amino_acid", "probability", "log2_odds", "floor_applied"],
            summary={"status": summary["status"], "requested_positions": length,
                     "profile_positions": profile.height, "amino_acids": 20},
            metadata=model_report_metadata(options.model_dump(), backend), manifest_pending=True, reason=reason,
            context={"MHC": canonical, "Species": options.species, "Conditioning": "MHC only; both TCR chains absent"},
            interpretation="The profile describes conditional amino-acid preferences, not measured binding motifs or peptide-binding probabilities. Positions span the requested full peptide length; no class II binding register is inferred. An unresolved context produces no profile and records an explicit reason.")
        manifest = _manifest(staging, task="decoder-hla-only-profile", options=options, backend=backend, summary=summary, source_sha256=source_sha256)
        write_json(staging / "manifest.json", manifest)
    return pssm, manifest
