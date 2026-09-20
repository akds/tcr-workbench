"""Small, discoverable command line interface for laboratory workflows."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from . import __version__
from .report import (manifest, output_bundle, sha256_file, snapshot_inputs, write_evidence_batches,
                     write_frame, write_json, write_report, write_workflow_report, model_report_metadata,
                     profile_reconstruction_context)


def _decoder_options(command):
    command.add_argument("--decoder-dir", type=Path)
    command.add_argument("--python", dest="python_executable")
    command.add_argument("--model", help="esmc-300m, esmc-600m, esmc-6b, esm2-650m, or esm2-3b")
    command.add_argument("--precision", choices=("float32", "float16"),
                         help="float32 default; float16 is approximate Apple ESM-C 300M inference")
    command.add_argument("--device", help="cpu, gpu/cuda[:N], or apple (MLX Metal)")
    command.add_argument("--checkpoint", type=Path, help="Local PyTorch checkpoint or MLX bundle; launcher prepares Apple weights automatically")
    command.add_argument("--mlx-python", help="MLX environment; --python remains the reconstruction environment")
    command.add_argument("--batch-size", type=int)
    command.add_argument("--token-budget", type=int)
    command.add_argument("--cache-bytes", type=int)
    command.add_argument("--timeout", type=float)
    command.add_argument("--reference-checkpoint", type=Path,
                         help="Original PyTorch checkpoint for first-use parity of an existing Apple bundle")
    command.add_argument("--allow-memory-risk", action="store_true",
                         help="Explicitly override estimated memory limits; may exhaust RAM/VRAM or swap heavily")


def _decoder_kwargs(args):
    return {key: getattr(args, key) for key in (
        "decoder_dir", "python_executable", "model", "device", "precision", "checkpoint", "mlx_python",
        "batch_size", "token_budget", "cache_bytes", "timeout")}


def _background_options(command):
    command.add_argument("--background-mode", choices=("uniform", "mhc-profile"), default="uniform",
                         help="Reference distribution: uniform AA20 (default), or samples from the MHC-only profile")
    command.add_argument("--background-peptides", type=int, default=1000,
                         help="Reference peptides per context/length (default: 1000; 0 disables)")
    command.add_argument("--background-seed", type=int, default=0,
                         help="Reproducible random-peptide seed (default: 0)")


def _biology_options(command):
    command.add_argument("--species", choices=("human", "mouse"), default="human",
                         help="Biological species (default: human); use mouse for H-2 and mouse TCR reconstruction")
    command.add_argument("--mhc-reference", type=Path,
                         help="Versioned mouse MHC sequence JSON; default: bundled exact mouse contexts")


def _biology_kwargs(args):
    return {"species": args.species, "mhc_reference": args.mhc_reference}


def _output_hashes(directory):
    return {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}


def _write_manifest(directory, content):
    write_json(directory / "manifest.json", {**content, "outputs": _output_hashes(directory)})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tcr-workbench",
        description="Prioritize antigen hypotheses for human or mouse alpha-beta TCRs with auditable evidence.",
        epilog="Start with: tcr-workbench example --out demo, then follow the printed command.",
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--config", type=Path, help="Runtime settings JSON (default: local .tcr-workbench.json if present)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("configure", help="Save local DecoderTCR runtime defaults once")
    _decoder_options(s)
    s.add_argument("--settings-out", type=Path, default=Path(".tcr-workbench.json"))
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("prepare-model", help="Check a local checkpoint, smoke-test it, and prepare Apple weights")
    _decoder_options(s)
    s.add_argument("--state-dir", type=Path, default=None, help="Prepared model cache (default: managed .tcr/prepared)")
    s.add_argument("--model-id", help="Explicit release/custom identity; defaults to the checkpoint SHA-256")
    s.add_argument("--expected-sha256", help="Provider's checkpoint checksum, if available")
    s.add_argument("--plan", action="store_true", help="Inspect tensor metadata and memory estimates without loading/running the model")
    s.add_argument("--sequence-length", type=int, default=2048, help="Model tokens for planning (default: conservative 2048)")
    s.add_argument("--estimate-forwards", type=int,
                   help="Rough timing for N distinct model context rows after cache reuse (requires local preparation timing)")
    for command, help_text in (("screen", "Match receptors against a reference and peptide panel"),
                               ("validate", "Validate and normalize receptors; retain chains and QC"),
                               ("decoder-export", "Prepare unique receptor × peptide inputs for DecoderTCR"),
                               ("repertoire-score", "Score a repertoire with DecoderTCR and retain cell/chain audits")):
        s = sub.add_parser(command, help=help_text)
        s.add_argument("--input", required=True, type=Path)
        s.add_argument("--format", choices=("auto", "10x", "airr", "paired"), default="auto")
        s.add_argument("--donor-id", help="Donor for single-donor inputs without a donor_id column")
        s.add_argument("--out", required=True, type=Path, help="New output directory")
        if command in ("screen", "decoder-export", "repertoire-score"):
            s.add_argument("--panel", required=True, type=Path, help="CSV/TSV with peptide,hla")
            s.add_argument("--donors", required=command == "screen", type=Path,
                           help="CSV/TSV with donor_id,hla; one restriction per row")
        if command == "screen":
            s.add_argument("--reference", required=True, type=Path)
            s.add_argument("--max-distance", type=int, choices=range(4), default=1,
                           help="Maximum summed shared-chain CDR3 edit distance (default: 1)")
            s.add_argument("--embedding-matching", action="store_true",
                           help="Also retrieve experimental DecoderTCR receptor-embedding neighbors; requires reference species and paired V/J/CDR3s")
            s.add_argument("--embedding-top-k", type=int, default=10,
                           help="Maximum distinct reference receptors per query in experimental embedding matching (default: 10)")
            _decoder_options(s)
        if command in ("decoder-export", "repertoire-score"):
            s.add_argument("--max-pairs", type=int, default=1_000_000)
        if command == "repertoire-score":
            _decoder_options(s)
            _biology_options(s)
        else:
            s.add_argument("--species", choices=("human", "mouse"), default="human")
    for name, help_text in (("pmhc-score", "Score peptide–MHC pairs with DecoderTCR PLL"),
                            ("pmhc-profile", "Generate an HLA-only DecoderTCR peptide profile and PSSM"),
                            ("tcr-score", "Score peptides with one paired TCR and MHC using DecoderTCR")):
        s = sub.add_parser(name, help=help_text)
        if name == "pmhc-score":
            s.add_argument("--panel", required=True, type=Path)
            s.add_argument("--max-pairs", type=int, default=100000)
        elif name == "pmhc-profile":
            s.add_argument("--hla", required=True)
            s.add_argument("--length", required=True, type=int)
        else:
            for field in ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b", "hla"):
                s.add_argument("--" + field, required=True)
            peptides = s.add_mutually_exclusive_group(required=True)
            peptides.add_argument("--peptide", help="One peptide sequence")
            peptides.add_argument("--panel", type=Path, help="CSV/TSV/Parquet with peptide; optional hla must match --hla")
            s.add_argument("--max-pairs", type=int, default=100000)
        if name in ("pmhc-score", "tcr-score"):
            _background_options(s)
        _decoder_options(s)
        _biology_options(s)
        s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("example", help="Create small synthetic example inputs")
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("dataset", help="Map h5ad observation metadata without loading expression")
    s.add_argument("--input", required=True, type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--inspect", action="store_true")
    s = sub.add_parser("decoder-run", help="Run DecoderTCR in its own Python environment")
    s.add_argument("--input", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    _decoder_options(s)
    _biology_options(s)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("decoder-import", help="Validate scored pairs and retain failed predictions")
    s.add_argument("--input", required=True, type=Path)
    s.add_argument("--scores", required=True, type=Path)
    s.add_argument("--model", default="DecoderTCR-ESMC_300M")
    s.add_argument("--require-manifest", action="store_true")
    s.add_argument("--species", choices=("human", "mouse"), default="human")
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("mhc", help="Look up peptide–MHC scores from a documented precomputed library")
    s.add_argument("--panel", required=True, type=Path)
    s.add_argument("--library", required=True, type=Path)
    s.add_argument("--metadata", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("mhc-predict", help="Predict peptide–MHC affinity with installed class I/II models")
    s.add_argument("--panel", required=True, type=Path)
    s.add_argument("--python", required=True, dest="python_executable",
                   help="Python interpreter in the separate peptide–MHC predictor environment")
    s.add_argument("--mhcflurry-models", type=Path, help="Installed class I affinity model directory")
    s.add_argument("--mhcnuggets-models", type=Path, help="Installed class II binding-affinity weights")
    s.add_argument("--batch-size", type=int, default=1024)
    s.add_argument("--max-pairs", type=int, default=100000)
    s.add_argument("--cpu-threads", type=int, default=4)
    s.add_argument("--timeout", type=float, default=600)
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("profile", help="Compute descriptive length-stratified peptide frequencies/PSSM")
    s.add_argument("--panel", required=True, type=Path)
    s.add_argument("--pseudocount", type=float, default=0.5)
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("decoder-profile", aliases=["tcr-profile"], help="Compute TCR-conditioned peptide marginals and PSSM")
    for gene in ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b", "hla"):
        s.add_argument("--" + gene, required=True)
    s.add_argument("--length", required=True, type=int)
    _decoder_options(s)
    _biology_options(s)
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("evaluate", help="Evaluate clone-level score rankings with missing-score coverage")
    s.add_argument("--scores", required=True, type=Path)
    s.add_argument("--labels", required=True, type=Path)
    s.add_argument("--lower-is-better", action="store_true")
    s.add_argument("--out", required=True, type=Path)
    return p


def make_example(directory: Path) -> None:
    with output_bundle(directory) as out:
        (out / "receptors.csv").write_text(
            "cell_id,donor_id,cdr3a,cdr3b,trav,traj,trbv,trbj\n"
            "cell1,donor1,CAVRPGGAGPFF,CASSLGQAYEQYF,TRAV21,TRAJ6,TRBV7-9,TRBJ2-7\n"
            "cell2,donor1,CAVRPGGAGPFF,CASSLGQAYEQYF,TRAV21,TRAJ6,TRBV7-9,TRBJ2-7\n"
            "cell3,donor1,CAVRPGGAGPFF,CASSLGQAYEQFF,TRAV21,TRAJ6,TRBV7-9,TRBJ2-7\n"
            "cell4,donor1,CAVAAAAAAF,CASSAAAAAAF,TRAV1-2,TRAJ33,TRBV20-1,TRBJ2-1\n"
        )
        (out / "reference.csv").write_text(
            "reference_id,cdr3a,cdr3b,trav,traj,trbv,trbj,peptide,hla,source,evidence\n"
            "synthetic-1,CAVRPGGAGPFF,CASSLGQAYEQYF,TRAV21,TRAJ6,TRBV7-9,TRBJ2-7,"
            "GILGFVFTL,HLA-A*02:01,SYNTHETIC_DEMO_NOT_BIOLOGICAL_EVIDENCE,functional\n"
        )
        (out / "panel.csv").write_text("peptide,hla\nGILGFVFTL,HLA-A*02:01\nELAGIGILTV,HLA-A*02:01\n")
        (out / "donors.csv").write_text("donor_id,hla\ndonor1,HLA-A*02:01\n")
        (out / "README.txt").write_text("Synthetic software demonstration only. These fabricated receptor–antigen associations are not biological evidence.\n")
    import shlex
    d = shlex.quote(str(directory))
    print(f"Synthetic inputs created. Run:\ntcr-workbench screen --input {d}/receptors.csv "
          f"--reference {d}/reference.csv --panel {d}/panel.csv --donors {d}/donors.csv --out {d}/results")


def execute(args: argparse.Namespace) -> None:
    command = "decoder-profile" if args.command == "tcr-profile" else args.command
    if command == "configure":
        from .runtime_settings import save_settings
        save_settings(args)
        print(f"Runtime defaults saved to {args.settings_out}")
        return
    embedding_screen = command == "screen" and args.embedding_matching
    if embedding_screen and not 1 <= args.embedding_top_k <= 1000:
        raise ValueError("--embedding-top-k must be between 1 and 1000")
    if command in {"prepare-model", "decoder-run", "decoder-profile", "pmhc-score", "pmhc-profile", "tcr-score", "repertoire-score"} or embedding_screen:
        from .runtime_settings import resolve_settings
        resolve_settings(args)
        if embedding_screen and args.precision != "float32":
            raise ValueError("Experimental embedding matching requires --precision float32")
        if args.precision == "float16":
            print("Approximate Apple FP16 inference selected; numerical results can differ from FP32.", file=sys.stderr)
    if getattr(args, "out", None) is not None and args.out.exists():
        raise ValueError(f"Output already exists: {args.out}. Choose a new --out directory.")
    model_commands = {"decoder-run", "decoder-profile", "pmhc-score", "pmhc-profile", "tcr-score", "repertoire-score"}
    if command == "prepare-model" or ((command in model_commands or embedding_screen) and os.environ.get("TCR_WORKBENCH_PREPARE_DIR")):
        from .model_preparation import prepare_model
        # Reject simple missing input paths before any checkpoint work.
        for field in ("input", "panel", "donors", "reference", "mhc_reference"):
            path = getattr(args, field, None)
            if path is not None and not path.is_file():
                raise ValueError(f"Input file is missing: {path}")
        state = getattr(args, "state_dir", None) or os.environ.get("TCR_WORKBENCH_PREPARE_DIR", ".tcr/prepared")
        prepared, record = prepare_model(_decoder_kwargs(args), state,
            model_id=getattr(args, "model_id", None), expected_sha256=getattr(args, "expected_sha256", None),
            reference_checkpoint=getattr(args, "reference_checkpoint", None),
            plan_only=getattr(args, "plan", False), allow_memory_risk=args.allow_memory_risk,
            sequence_length=getattr(args, "sequence_length", 2048), estimate_forwards=getattr(args, "estimate_forwards", None))
        args.checkpoint = prepared["checkpoint"]
        if command == "prepare-model":
            print(json.dumps(record, indent=2))
            return
    file_arguments = {"input", "reference", "panel", "donors", "library", "metadata", "scores", "labels"}
    # The h5ad bridge hashes selected obs metadata; scoring workflows own their
    # snapshots. Avoid redundant full-file reads before those guarded calls.
    owns_snapshot = command in {"dataset", "repertoire-score", "pmhc-score", "tcr-score", "decoder-run"}
    input_snapshot = snapshot_inputs([value for name, value in vars(args).items()
                                      if not owns_snapshot and name in file_arguments
                                      and value is not None])
    if command == "example":
        make_example(args.out)
        return
    if command == "repertoire-score":
        from .workflows import score_repertoire
        frame, _ = score_repertoire(args.input, args.panel, args.out, format=args.format,
            donor_id=args.donor_id, donors_path=args.donors, max_pairs=args.max_pairs,
            **_decoder_kwargs(args), **_biology_kwargs(args))
        print(f"DecoderTCR repertoire report written to {args.out} ({frame.height:,} pairs)")
        return
    if command in ("pmhc-score", "pmhc-profile"):
        from .decoder_pmhc import score_pmhc, profile_pmhc
        if command == "pmhc-score":
            frame, run = score_pmhc(args.panel, args.out, max_pairs=args.max_pairs,
                background_peptides=args.background_peptides, background_seed=args.background_seed,
                background_mode=args.background_mode,
                **_decoder_kwargs(args), **_biology_kwargs(args))
        else:
            frame, run = profile_pmhc(args.hla, args.length, args.out,
                **_decoder_kwargs(args), **_biology_kwargs(args))
        if command == "pmhc-profile" and run.get("summary", {}).get("status") == "Unresolved":
            print(f"Unresolved: {run['summary']['reason']}. Audit written to {args.out}")
            return
        print(f"DecoderTCR peptide–MHC results written to {args.out} ({frame.height:,} rows)")
        return
    if command in ("screen", "validate", "decoder-export"):
        from .ingest import read_receptors
        started = time.perf_counter()
        result = read_receptors(args.input, format=args.format, donor_id=args.donor_id, species=args.species)
        inputs = {"receptors": args.input}
        parameters = {"format": args.format, "donor_id": args.donor_id, "species": args.species}
        if command in ("screen", "decoder-export"):
            from .matching import read_donors, read_panel
            panel = read_panel(args.panel)
            donors = read_donors(args.donors) if args.donors else None
            inputs["panel"] = args.panel
            if args.donors:
                inputs["donors"] = args.donors
        if command == "screen":
            from .matching import _iter_screen_validated, read_references
            references = read_references(args.reference)
            inputs["reference"] = args.reference
            parameters["max_distance"] = args.max_distance
            sequence_references = references
            if embedding_screen:
                if "species" not in references.columns:
                    raise ValueError("Embedding matching requires an explicit species column in references")
                # The experimental reference can contain multiple species. Keep
                # the accompanying CDR3 search within the requested species too.
                sequence_references = references.filter(pl.col("species") == args.species)
                parameters.update(embedding_matching=True, embedding_top_k=args.embedding_top_k,
                                  reference_species_filter=args.species)
            evidence_batches = _iter_screen_validated(result.receptors, sequence_references, panel, donors,
                                                      max_distance=args.max_distance)
        with output_bundle(args.out, input_snapshot=input_snapshot) as out:
            for name in ("chains", "receptors", "cells"):
                write_frame(getattr(result, name), out / f"{name}.csv")
            run = manifest(inputs, parameters, input_snapshot=input_snapshot)
            run["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            if command == "screen":
                preview, evidence_summary = write_evidence_batches(evidence_batches, out / "evidence.csv")
                if embedding_screen:
                    from .embedding_matching import run_embedding_matching
                    run["embedding_matching"] = run_embedding_matching(
                        result.receptors, references, panel, donors, out,
                        top_k=args.embedding_top_k, species=args.species, **_decoder_kwargs(args))
                run["elapsed_seconds"] = round(time.perf_counter() - started, 4)
                write_report(out, preview, result.receptors, result.qc, run,
                             evidence_summary=evidence_summary)
            elif command == "decoder-export":
                from .prediction import export_decoder_input
                run["decoder"] = export_decoder_input(result.receptors, panel,
                    out / "decoder_input.csv", donors=donors, max_pairs=args.max_pairs, species=args.species)
                run["parameters"]["species"] = args.species
                write_json(out / "qc.json", result.qc)
                _write_manifest(out, run)
            else:
                write_json(out / "qc.json", result.qc)
                _write_manifest(out, run)
        print(f"Wrote {args.out} ({result.receptors.height:,} receptors; {len(result.qc):,} QC events)")
        return
    from . import prediction
    if command == "tcr-score":
        from .tcr_scoring import score_tcr
        components = {key: getattr(args, key) for key in prediction.GENES + ["hla"]}
        score_tcr(components, args.out, peptide=args.peptide, panel=args.panel,
            background_peptides=args.background_peptides, background_seed=args.background_seed,
            background_mode=args.background_mode,
            max_pairs=args.max_pairs, **_decoder_kwargs(args), **_biology_kwargs(args))
        print(f"DecoderTCR TCR–peptide–MHC results written to {args.out}")
        return
    if command == "dataset":
        from .dataset import build_dataset, inspect_h5ad
        if args.inspect:
            print(json.dumps(inspect_h5ad(args.input), indent=2))
        else:
            if args.out is None:
                raise ValueError("dataset requires --out, or --inspect")
            with output_bundle(args.out, input_snapshot=input_snapshot) as out:
                result = build_dataset(args.input, out)
            print(f"Dataset mapping written to {args.out}")
        return
    if command == "decoder-run":
        result = prediction.run_decoder(args.input, args.output, decoder_dir=args.decoder_dir,
            python_executable=args.python_executable, model=args.model, device=args.device, precision=args.precision,
            timeout=args.timeout, force=args.force, checkpoint=args.checkpoint,
            mlx_python=args.mlx_python, batch_size=args.batch_size,
            token_budget=args.token_budget, cache_bytes=args.cache_bytes, **_biology_kwargs(args))
        print(json.dumps(result, indent=2))
        return
    if command == "decoder-profile":
        from .decoder_pmhc import PROFILE_SCHEMA, profile_to_pssm
        components = {key: getattr(args, key) for key in prediction.GENES + ["hla"]}
        with output_bundle(args.out, input_snapshot=input_snapshot) as out:
            execution = prediction.run_decoder_profile(components, out / "profile.csv", length=args.length,
                decoder_dir=args.decoder_dir, python_executable=args.python_executable,
                model=args.model, device=args.device, precision=args.precision, timeout=args.timeout,
                checkpoint=args.checkpoint, mlx_python=args.mlx_python,
                batch_size=args.batch_size, token_budget=args.token_budget, cache_bytes=args.cache_bytes,
                **_biology_kwargs(args))
            profile = pl.read_csv(out / "profile.csv")
            unresolved = execution.get("status") == "Unresolved"
            pssm = (pl.DataFrame(schema=PROFILE_SCHEMA) if unresolved else
                    profile_to_pssm(profile, hla=components["hla"], length=args.length))
            if args.precision == "float16" and pssm.height:
                pssm = pssm.with_columns((pl.col("interpretation") + pl.lit("; approximate FP16 inference")).alias("interpretation"))
            write_frame(pssm, out / "pssm.csv")
            write_json(out / "pssm_metadata.json", {"units": "log2 odds", "background": "uniform 1/20",
                "probability_floor": 1e-12, "precision": args.precision,
                "approximate": args.precision == "float16",
                "status": "Unresolved" if unresolved else "Profiled",
                "reason": execution.get("reason", ""),
                "interpretation": "Conditional amino-acid preferences; not binding probabilities"})
            write_workflow_report(out, "TCR–MHC-conditioned peptide profile", table=pssm, profile=profile,
                workflow="tcr-profile",
                columns=["position", "amino_acid", "probability", "log2_odds", "floor_applied"],
                summary={"status": "Unresolved" if unresolved else "Profiled",
                         "requested_positions": args.length, "profile_positions": profile.height, "amino_acids": 20},
                metadata=model_report_metadata(_decoder_kwargs(args), execution),
                context={**components, "Species": args.species,
                         **profile_reconstruction_context(components, execution)},
                reason=execution.get("reason", ""), manifest_pending=True,
                interpretation="The profile describes amino-acid preferences conditioned on the paired TCR and MHC. These are model predictions, not binding probabilities or measured binding motifs. No class II binding-core register is inferred. Unresolved contexts retain the failure reason; an unavailable profile does not imply nonbinding.")
            _write_manifest(out, {"schema_version": 1, "command": command,
                "components": components, "species": args.species, "peptide_length": args.length, "execution": execution,
                "interpretation": "AA20 preferences cannot recover full-vocabulary PLL (64 channels for ESM-C; 33 for ESM-2)"})
        if unresolved:
            print(f"Unresolved: {execution.get('reason', 'context reconstruction failed')}. Audit written to {args.out}")
        else:
            print(f"Conditional peptide marginals written to {args.out}")
        return
    if command == "mhc-predict":
        from .matching import read_panel
        from .mhc_predict import predict_peptide_mhc
        panel = read_panel(args.panel)
        frame, provenance = predict_peptide_mhc(panel, python=args.python_executable,
            mhcflurry_models=args.mhcflurry_models, mhcnuggets_models=args.mhcnuggets_models,
            batch_size=args.batch_size, timeout=args.timeout, max_pairs=args.max_pairs,
            cpu_threads=args.cpu_threads)
        with output_bundle(args.out, input_snapshot=input_snapshot) as out:
            write_frame(frame, out / "results.csv")
            write_json(out / "predictor_manifest.json", provenance)
            _write_manifest(out, manifest({"panel": args.panel},
                {"command": command, "batch_size": args.batch_size, "timeout": args.timeout,
                 "max_pairs": args.max_pairs, "cpu_threads": args.cpu_threads},
                input_snapshot=input_snapshot))
        print(f"Peptide–MHC predictions written to {args.out} ({frame.height:,} rows)")
        return
    inputs = {}
    parameters = {"command": command}
    if command == "evaluate":
        from .evaluation import evaluate_scores
        scores = pl.read_csv(args.scores, infer_schema=False, null_values=[""])
        labels = pl.read_csv(args.labels, infer_schema=False, null_values=[""]).with_columns(
            pl.col("label").cast(pl.Int64, strict=True))
        frame = evaluate_scores(scores, labels, higher_is_better=not args.lower_is_better)
        inputs = {"scores": args.scores, "labels": args.labels}
        parameters["higher_is_better"] = not args.lower_is_better
    elif command == "decoder-import":
        frame = prediction.import_decoder_scores(args.input, args.scores, model=args.model,
                                                  require_manifest=args.require_manifest, species=args.species)
        inputs = {"input": args.input, "scores": args.scores}
        parameters["model"] = args.model
    elif command in ("mhc", "profile"):
        from .matching import read_panel
        panel = read_panel(args.panel)
        inputs = {"panel": args.panel}
        if command == "mhc":
            frame = prediction.peptide_mhc_lookup(panel, args.library, args.metadata)
            inputs.update(library=args.library, metadata=args.metadata)
        else:
            frame = prediction.empirical_profile(panel, pseudocount=args.pseudocount)
            parameters["pseudocount"] = args.pseudocount
    else:
        raise ValueError(f"Unknown command: {command}")
    with output_bundle(args.out, input_snapshot=input_snapshot) as out:
        write_frame(frame, out / "results.csv")
        _write_manifest(out, manifest(inputs, parameters, input_snapshot=input_snapshot))
    print(f"Wrote {args.out} ({frame.height:,} rows)")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        execute(args)
    except (ValueError, ValidationError, OSError, ImportError, pl.exceptions.PolarsError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
