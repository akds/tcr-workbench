"""One paired receptor against a peptide panel and a matched random comparator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from . import prediction
from .background import background_groups, background_percentiles
from .model_registry import resolve_model
from .report import (model_report_metadata, output_bundle, sha256_file, snapshot_inputs,
                     source_digest, write_frame, write_json, write_workflow_report)
from .workflows import rank_hypotheses


def score_tcr(components: dict, output: Path, *, peptide: str | None = None,
              panel: Path | None = None, background_peptides: int = 1000,
              background_seed: int = 0, max_pairs: int = 100000,
              background_mode: str = "uniform",
              species: str = "human", mhc_reference=None, **backend):
    if (peptide is None) == (panel is None):
        raise ValueError("Supply exactly one --peptide or --panel")
    source_hash = source_digest()
    from .species import biological_fingerprint, normalize_mhc
    biological_fingerprint(species, mhc_reference)
    backend = {**backend, "species": species, "mhc_reference": mhc_reference}
    snapshot = snapshot_inputs([panel]) if panel is not None else None
    if panel is not None:
        panel = Path(panel)
        frame = pl.read_parquet(panel) if panel.suffix.lower() == ".parquet" else prediction._read_csv(panel)
        if "peptide" not in frame.columns or frame["peptide"].dtype != pl.String:
            raise ValueError("TCR peptide panel needs a text peptide column")
        if "hla" in frame.columns:
            expected = normalize_mhc(components["hla"], species)
            if any(value is None or normalize_mhc(value, species) != expected for value in frame["hla"].to_list()):
                raise ValueError("Every panel MHC must match --hla for this receptor; use repertoire-score for multiple MHC contexts")
    else:
        frame = pl.DataFrame({"peptide": [peptide]})
    if not 1 <= frame.height <= max_pairs:
        raise ValueError(f"TCR panel needs between 1 and {max_pairs:,} rows")
    pairs = []
    for i, sequence in enumerate(frame["peptide"]):
        pair = prediction._Pair.model_validate({"name": f"pair_{i + 1}", **components,
                                                "species": species, "peptide": sequence}).model_dump(exclude={"species"})
        if not 1 <= len(pair["peptide"]) <= 50:
            raise ValueError("TCR peptides must have 1–50 residues")
        pairs.append(pair)
    inputs = pl.DataFrame(pairs)
    receptor_id = "receptor_" + hashlib.sha256(json.dumps(
        {"species": species, **{key: pairs[0][key] for key in prediction.GENES}}, sort_keys=True).encode()).hexdigest()[:16]
    contexts = inputs.with_columns(pl.lit(receptor_id).alias("receptor_id"),
                                  pl.col("peptide").str.len_chars().alias("peptide_length"))
    background_groups(contexts, peptides=background_peptides, seed=background_seed, mode=background_mode)
    with output_bundle(Path(output), input_snapshot=snapshot) as out:
        from .decoder_pmhc import _options, prepare_background, verify_background_execution
        random, background_metadata = prepare_background(contexts, out,
            background_mode=background_mode, background_peptides=background_peptides,
            background_seed=background_seed, options=_options(backend))
        random_input = random.with_columns([pl.lit(pairs[0][key]).alias(key)
                                           for key in inputs.columns if key not in random.columns]).select(inputs.columns)
        combined = pl.concat([inputs, random_input])
        write_frame(frame, out / "input_panel.csv")
        input_csv = out / "decoder_input.csv"
        combined.write_csv(input_csv)
        execution = prediction.run_decoder(input_csv, out / "scores.csv", **backend)
        verify_background_execution(background_metadata, execution)
        scored = prediction.import_decoder_scores(input_csv, out / "scores.csv",
                    model=resolve_model(backend.get("model", prediction.DEFAULT_MODEL)).name,
                    require_manifest=True, species=species).with_columns(pl.lit(receptor_id).alias("receptor_id"))
        results = scored.join(inputs.select("name"), on="name", how="semi", maintain_order="left")
        # Replicate observations stay visible, but do not displace another
        # distinct peptide's rank. Match the report's distinct-peptide ranking.
        rank_keys = ["receptor_id", "hla", "peptide"]
        ranks = rank_hypotheses(results.unique(subset=rank_keys, maintain_order=True))
        results = results.join(ranks.select(*rank_keys, "peptide_length", "rank"),
                               on=rank_keys, how="left", validate="m:1", maintain_order="left")
        background = scored.join(random.select("name", "peptide_length"), on="name", how="inner", validate="1:1")
        results = background_percentiles(results, background)
        write_frame(results, out / "results.csv")
        write_frame(background, out / "background_scores.csv")
        successful = pl.col("status").is_in(["Scored", "ModelHypothesis"]) & pl.col("score").is_finite()
        background_metadata.update(scored_rows=background.filter(successful).height,
                                   unresolved_rows=background.filter(~successful.fill_null(False)).height)
        write_json(out / "background_metadata.json", background_metadata)
        summary = {"tested_peptides": results.height,
                   "scored_pairs": results.filter(successful).height,
                   "unresolved_pairs": results.filter(~successful.fill_null(False)).height}
        write_workflow_report(out, "TCR–pMHC scores", workflow="tcr-score",
            table=results, background=background, background_metadata=background_metadata,
            columns=["receptor_id", "hla", "peptide", "peptide_length", "score", "rank", "status", "reason"],
            summary=summary, metadata=model_report_metadata(backend, execution),
            context={**components, "Species": species, "Receptor ID": receptor_id,
                     "Peptide reference": f"{background_peptides} {background_mode} draws per MHC/length; seed {background_seed}; background_metadata.json"},
            manifest_pending=True,
            interpretation="DecoderTCR PLL scores measure peptide compatibility with the specified paired TCR and MHC. They are not binding probabilities or measured affinities. Compare scores within the same species, receptor, MHC, peptide length, checkpoint and precision. Random peptides provide a synthetic reference, not known nonbinders. Unresolved denotes no supported score, not nonbinding.")
        if source_digest() != source_hash:
            raise ValueError("Workbench source changed during TCR scoring; rerun from stable code")
        record = {"schema_version": 1, "command": "tcr-score", "components": components,
                  "receptor_id": receptor_id, "species": species, "execution": execution, "summary": summary,
                  "background": background_metadata, "input_snapshot": snapshot,
                  "source_sha256": source_hash,
                  "outputs": {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                              for p in sorted(out.iterdir()) if p.is_file()}}
        write_json(out / "manifest.json", record)
    return results, record
