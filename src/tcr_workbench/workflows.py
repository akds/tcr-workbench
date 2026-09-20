"""Complete DecoderTCR repertoire and single-receptor workflows with cell audits."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from . import prediction
from .ingest import read_receptors
from .matching import read_donors, read_panel
from .model_registry import resolve_model, normalize_device, validate_backend
from .report import (manifest, output_bundle, sha256_file, snapshot_inputs, source_digest,
                     write_frame, write_json, write_workflow_report, model_report_metadata)


RESULT_SCHEMA = {
    "name": pl.String, "receptor_id": pl.String, "donor_id": pl.String,
    **{name: pl.String for name in prediction.COMPONENTS},
    "score": pl.Float64, "status": pl.String, "reason": pl.String,
    "metric": pl.String, "provenance": pl.String,
    **{name: pl.String for name in ("TRAV", "TRAJ", "TRBV", "TRBJ", "HLA_a", "HLA_b",
                                     "TCR_a", "TCR_b", "ok", "tcr_ok", "hla_ok",
                                     "tcr_reason", "hla_reason", "inference_reason")},
    "gene_resolution_status": pl.String, "gene_resolution_reason": pl.String,
    "mapping_status": pl.String, "peptide_length": pl.UInt32, "rank": pl.UInt32,
}


def _normalize_result_schema(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep a stable typed minimum contract even when no pair reaches a model.

    Missing reconstruction fields remain null rather than suggesting a failed
    or successful reconstruction was performed. Additional backend audit columns
    are preserved after the standard columns.
    """
    extra = [name for name in frame.columns if name not in RESULT_SCHEMA]
    frame = frame.with_columns([
        (pl.col(name).cast(dtype, strict=True) if name in frame.columns
         else pl.lit(None, dtype=dtype)).alias(name)
        for name, dtype in RESULT_SCHEMA.items()
    ])
    return frame.select([*RESULT_SCHEMA, *extra])


def rank_hypotheses(frame: pl.DataFrame) -> pl.DataFrame:
    """Rank only within the same receptor, HLA context and peptide length."""
    return frame.with_columns(pl.col("peptide").str.len_chars().alias("peptide_length")).with_columns(
        pl.col("score").rank(method="min", descending=True)
        .over(["receptor_id", "hla", "peptide_length"]).alias("rank")
    ).sort(["receptor_id", "hla", "peptide_length", "rank", "peptide"], nulls_last=True)


def score_repertoire(input_path, panel_path, output, *, format="auto", donor_id=None,
                     donors_path=None, max_pairs=1_000_000, species="human", mhc_reference=None, **backend):
    """Ingest, prepare, infer, import and rank; publish one atomic audit bundle."""
    source_sha256 = source_digest()
    from .species import biological_fingerprint
    biology = biological_fingerprint(species, mhc_reference)
    backend = {**backend, "species": species, "mhc_reference": mhc_reference}
    input_path, panel_path, output = map(Path, (input_path, panel_path, output))
    paths = [input_path, panel_path] + ([Path(donors_path)] if donors_path else [])
    snapshot = snapshot_inputs(paths)
    model = resolve_model(backend.get("model", prediction.DEFAULT_MODEL)).name
    backend = {**backend, "model": model, "device": normalize_device(backend.get("device", "cpu"))}
    backend["precision"] = backend.get("precision", "float32")
    validate_backend(model, backend["device"], backend.get("checkpoint"),
                     backend.get("mlx_python"), backend["precision"])
    receptors = read_receptors(input_path, format=format, donor_id=donor_id, species=species)
    panel = read_panel(panel_path)
    if not panel.height:
        raise ValueError("repertoire scoring requires at least one peptide/HLA panel row")
    donors = read_donors(donors_path) if donors_path else None
    with output_bundle(output, input_snapshot=snapshot) as out:
        for name in ("chains", "receptors", "cells"):
            write_frame(getattr(receptors, name), out / f"{name}.csv")
        input_csv, scores_csv = out / "decoder_input.csv", out / "scores.csv"
        exported = prediction.export_decoder_input(receptors.receptors, panel, input_csv,
                                                    donors=donors, max_pairs=max_pairs, species=species)
        if exported["exported_pairs"]:
            execution = prediction.run_decoder(input_csv, scores_csv, **backend)
            evidence = prediction.import_decoder_scores(input_csv, scores_csv,
                                                         model=model, require_manifest=True, species=species)
        else:
            execution = {"model": model, "device": backend["device"],
                         "precision": backend["precision"], "approximate": backend["precision"] == "float16",
                         "status": "not_run_no_eligible_pairs"}
            evidence = pl.DataFrame(schema={"receptor_id": pl.String, "donor_id": pl.String,
                "peptide": pl.String, "hla": pl.String, "score": pl.Float64,
                "status": pl.String, "reason": pl.String})
        skipped = pl.read_csv(str(input_csv) + ".skipped.csv", infer_schema=False)
        if skipped.height:
            skipped = skipped.with_columns(pl.lit(None, dtype=pl.Float64).alias("score"),
                                           pl.lit("not_scored").alias("provenance"))
            evidence = pl.concat([evidence, skipped], how="diagonal_relaxed")
        evidence = _normalize_result_schema(rank_hypotheses(evidence))
        write_frame(evidence, out / "results.csv")
        write_json(out / "qc.json", receptors.qc)
        known_cells = receptors.cells.filter(pl.col("cell_id").is_not_null()).select(
            "donor_id", "cell_id").unique().height
        summary = {"precision": backend["precision"], "approximate": backend["precision"] == "float16", "receptors": receptors.receptors.height, "cells": known_cells,
                   "cell_receptor_links": receptors.cells.height,
                   "scored_pairs": evidence.filter(pl.col("score").is_not_null()).height,
                   "unresolved_pairs": evidence.filter(pl.col("status") == "Unresolved").height,
                   "ranking_scope": ["receptor_id", "hla", "peptide_length"],
                   "interpretation": "Uncalibrated PLL hypotheses; experimental validation required"}
        write_json(out / "summary.json", summary)
        write_workflow_report(out, "Repertoire scores", table=evidence,
            workflow="repertoire-score",
            columns=["receptor_id", "hla", "peptide", "peptide_length", "score", "rank", "status", "reason"],
            summary={key: summary[key] for key in ("receptors", "cells", "scored_pairs", "unresolved_pairs")},
            metadata=model_report_metadata(backend, execution), manifest_pending=True,
            context={"Species": species, "Comparison groups": "Same receptor, MHC and peptide length",
                     "Donor MHC typing": "Supplied; see compatibility records" if donors_path else "Not supplied; donor compatibility unconfirmed"},
            interpretation="DecoderTCR PLL scores quantify peptide compatibility within each receptor–MHC context. Higher-ranked peptides are antigen hypotheses, not identified antigens. Scores are not binding probabilities. Ranks compare the same receptor, MHC and peptide length within one species, checkpoint and precision. Unresolved does not mean nonbinding. Complete output tables retain all input cells and chains; alternative pairings are not independent cells.")
        run = manifest({"receptors": input_path, "panel": panel_path,
                        **({"donors": Path(donors_path)} if donors_path else {})},
                       {"command": "repertoire-score", "format": format, "donor_id": donor_id, "species": species,
                        "max_pairs": max_pairs, "model": model, "device": backend["device"],
                        "precision": backend["precision"], "approximate": backend["precision"] == "float16"},
                       input_snapshot=snapshot)
        run.update(export=exported, execution=execution, summary=summary, biological_context=biology)
        run["outputs"] = {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                          for p in sorted(out.iterdir()) if p.is_file()}
        if source_digest() != source_sha256:
            raise ValueError("Workbench source changed during repertoire analysis; rerun from stable code")
        run["software"]["source_sha256"] = source_sha256
        write_json(out / "manifest.json", run)
    return evidence, run
