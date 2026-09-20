import json
from types import SimpleNamespace

import polars as pl
import pytest

from tcr_workbench import prediction
from tcr_workbench.cli import main
from tcr_workbench.runtime_settings import resolve_settings
from tcr_workbench.workflows import rank_hypotheses


def fake_run(source, output, **kwargs):
    frame = pl.read_csv(source).with_columns(
        pl.lit(-2.0).alias("pll_" + prediction.DEFAULT_MODEL))
    frame.write_csv(output)
    run = {"schema_version": 1, "model": prediction.DEFAULT_MODEL,
           "input_sha256": prediction.file_sha256(source),
           "output_sha256": prediction.file_sha256(output)}
    prediction._json_write(prediction._sidecar(output), run)
    return run


def test_repertoire_cli_preserves_cells_skips_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "run_decoder", fake_run)
    source = tmp_path / "paired.csv"
    source.write_text("cell_id,donor_id,trav,traj,cdr3a,trbv,trbj,cdr3b\n"
        "c1,d1,TRAV21,TRAJ6,CAVRPGGAGPFF,TRBV7-9,TRBJ2-7,CASSLGQAYEQYF\n"
        "c2,d1,TRAV21,TRAJ6,CAVRPGGAGPFF,TRBV7-9,TRBJ2-7,CASSLGQAYEQYF\n"
        "c3,d1,,,,TRBV7-9,TRBJ2-7,CASSLGQAYEQYF\n")
    panel = tmp_path / "panel.csv"
    panel.write_text("peptide,hla\nGILGFVFTL,HLA-A*02:01\nGLCTLVAML,HLA-A*02:01\n")
    output = tmp_path / "run"
    assert main(["repertoire-score", "--input", str(source), "--panel", str(panel),
                 "--out", str(output), "--decoder-dir", str(tmp_path), "--python", "unused"]) == 0
    results = pl.read_parquet(output / "results.parquet")
    assert results.height == 4
    assert results.filter(pl.col("status") == "Unresolved").height == 2
    assert results.filter(pl.col("score").is_not_null())["rank"].to_list() == [1, 1]
    assert pl.read_parquet(output / "cells.parquet").height == 3
    record = json.loads((output / "manifest.json").read_text())
    for name, metadata in record["outputs"].items():
        assert prediction.file_sha256(output / name) == metadata["sha256"]
    assert record["summary"]["ranking_scope"] == ["receptor_id", "hla", "peptide_length"]


def test_ranks_do_not_cross_context_or_length():
    frame = pl.DataFrame({"receptor_id": ["r"] * 4, "hla": ["A", "A", "B", "A"],
                          "peptide": ["AAAA", "AAAC", "AAAA", "AAAAA"],
                          "score": [-1., -2., -10., -20.]})
    ranks = {r["peptide"] + r["hla"]: r["rank"] for r in rank_hypotheses(frame).to_dicts()}
    assert ranks == {"AAAAA": 1, "AAACA": 2, "AAAAB": 1, "AAAAAA": 1}


def test_all_unresolved_repertoire_does_not_start_model(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("no model should run for an ineligible repertoire")
    monkeypatch.setattr(prediction, "run_decoder", unexpected)
    source, panel, out = tmp_path / "input.csv", tmp_path / "panel.csv", tmp_path / "out"
    source.write_text("cell_id,donor_id,cdr3a,cdr3b\nc1,d1,,CASSLGQAYEQYF\n")
    panel.write_text("peptide,hla\nGILGFVFTL,HLA-A*02:01\n")
    assert main(["repertoire-score", "--input", str(source), "--panel", str(panel),
                 "--out", str(out), "--decoder-dir", str(tmp_path), "--python", "unused"]) == 0
    results = pl.read_parquet(out / "results.parquet")
    assert results["status"].to_list() == ["Unresolved"]
    assert results["rank"].to_list() == [None]
    assert json.loads((out / "manifest.json").read_text())["execution"]["status"] == "not_run_no_eligible_pairs"


def test_configured_commands_and_flag_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "decoder").mkdir()
    python = tmp_path / "python"
    python.write_text("placeholder")
    assert main(["configure", "--decoder-dir", "decoder", "--python", "python", "--model", "esmc-300m"]) == 0
    options = resolve_settings(SimpleNamespace(config=None, device="gpu", model=None))
    assert options["device"] == "cuda"
    assert options["python_executable"] == str(python)
    assert options["model"] == prediction.DEFAULT_MODEL
    assert main(["configure", "--decoder-dir", "decoder", "--python", "python"]) == 2


def test_named_config_paths_are_relative_to_config(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"decoder_dir": "decoder", "python_executable": "env/bin/python"}))
    result = resolve_settings(SimpleNamespace(config=path))
    assert result["decoder_dir"] == str(tmp_path / "decoder")
    assert result["python_executable"] == str(tmp_path / "env/bin/python")


def test_bad_config_fails_loudly(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text('{"decoder_dir":".","python_executable":"python","devcie":"apple"}')
    with pytest.raises(ValueError):
        resolve_settings(SimpleNamespace(config=path))


def test_duplicate_config_key_is_rejected(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text('{"decoder_dir":".","python_executable":"python","device":"cpu","device":"apple"}')
    with pytest.raises(ValueError, match="Duplicate runtime setting"):
        resolve_settings(SimpleNamespace(config=path))


def test_single_tcr_cli_publishes_verified_output_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "run_decoder", fake_run)
    output = tmp_path / "run"
    flags = ["tcr-score", "--trav", "TRAV21", "--traj", "TRAJ6", "--cdr3a", "CAVRPGGAGPFFVVF",
             "--trbv", "TRBV7-9", "--trbj", "TRBJ2-7", "--cdr3b", "CASSLGQAYEQYF",
             "--hla", "HLA-B*27:05", "--peptide", "LRVMMLAPF", "--out", str(output),
             "--decoder-dir", str(tmp_path), "--python", "unused"]
    assert main(flags) == 0
    run = json.loads((output / "manifest.json").read_text())
    assert "results.parquet" in run["outputs"]
    for name, entry in run["outputs"].items():
        assert prediction.file_sha256(output / name) == entry["sha256"]


def test_tcr_profile_alias_writes_pssm_and_provenance(tmp_path, monkeypatch):
    from tcr_workbench.prediction import AA

    def profile(components, output, **kwargs):
        pl.DataFrame({"position": [1, 2], **{aa: [0.05, 0.05] for aa in AA}}).write_csv(output)
        return {"model": prediction.DEFAULT_MODEL, "status": "mock_execution"}

    monkeypatch.setattr(prediction, "run_decoder_profile", profile)
    output = tmp_path / "run"
    assert main(["tcr-profile", "--trav", "TRAV21", "--traj", "TRAJ6", "--cdr3a", "CAVRPGGAGPFFVVF",
        "--trbv", "TRBV7-9", "--trbj", "TRBJ2-7", "--cdr3b", "CASSLGQAYEQYF", "--hla", "HLA-B*27:05",
        "--length", "2", "--out", str(output), "--decoder-dir", str(tmp_path), "--python", "unused"]) == 0
    pssm = pl.read_parquet(output / "pssm.parquet")
    assert pssm.height == 40
    assert pssm["hla"].unique().to_list() == ["HLA-B*27:05"]
    run = json.loads((output / "manifest.json").read_text())
    for name, entry in run["outputs"].items():
        assert prediction.file_sha256(output / name) == entry["sha256"]


def test_unresolved_tcr_profile_remains_auditable(tmp_path, monkeypatch, capsys):
    def profile(components, output, **kwargs):
        pl.DataFrame(schema={"position": pl.Int64, **{aa: pl.Float64 for aa in prediction.AA}}).write_csv(output)
        return {"model": prediction.DEFAULT_MODEL, "status": "Unresolved", "reason": "HLA absent from reference"}

    monkeypatch.setattr(prediction, "run_decoder_profile", profile)
    output = tmp_path / "run"
    assert main(["tcr-profile", "--trav", "TRAV21", "--traj", "TRAJ6", "--cdr3a", "CAVRPGGAGPFFVVF",
        "--trbv", "TRBV7-9", "--trbj", "TRBJ2-7", "--cdr3b", "CASSLGQAYEQYF", "--hla", "HLA-A*99:99",
        "--length", "9", "--out", str(output), "--decoder-dir", str(tmp_path), "--python", "unused"]) == 0
    assert "Unresolved: HLA absent" in capsys.readouterr().out
    assert pl.read_parquet(output / "pssm.parquet").height == 0
    assert json.loads((output / "manifest.json").read_text())["execution"]["status"] == "Unresolved"
    metadata = json.loads((output / "pssm_metadata.json").read_text())
    assert metadata["status"] == "Unresolved" and metadata["reason"] == "HLA absent from reference"


def test_repertoire_dual_alpha_counts_one_cell_and_two_links(tmp_path, monkeypatch):
    from tcr_workbench.workflows import score_repertoire

    def unexpected(*args, **kwargs):
        raise AssertionError("ambiguous dual-alpha pairing must not start a model")

    monkeypatch.setattr(prediction, "run_decoder", unexpected)
    source, panel, output = tmp_path / "chains.csv", tmp_path / "panel.csv", tmp_path / "run"
    pl.DataFrame({"barcode": ["c1"] * 3, "chain": ["TRA", "TRA", "TRB"],
                  "cdr3": ["CAVF", "CAGF", "CASSF"], "productive": ["true"] * 3}).write_csv(source)
    panel.write_text("peptide,hla\nGILGFVFTL,HLA-A*02:01\n")
    _, run = score_repertoire(source, panel, output, donor_id="d1", decoder_dir=tmp_path,
                              python_executable="unused")
    assert run["summary"]["cells"] == 1
    assert run["summary"]["cell_receptor_links"] == 2
    assert pl.read_parquet(output / "cells.parquet").height == 2
    assert pl.read_parquet(output / "chains.parquet").height == 3


def test_repertoire_airr_without_cell_ids_counts_zero_cells(tmp_path, monkeypatch):
    from tcr_workbench.workflows import score_repertoire

    def unexpected(*args, **kwargs):
        raise AssertionError("unpaired AIRR observations must not start a model")

    monkeypatch.setattr(prediction, "run_decoder", unexpected)
    source, panel, output = tmp_path / "airr.csv", tmp_path / "panel.csv", tmp_path / "run"
    pl.DataFrame({"sequence_id": ["a", "b"], "locus": ["TRA", "TRB"],
                  "junction_aa": ["CAVF", "CASSF"], "productive": ["T", "T"]}).write_csv(source)
    panel.write_text("peptide,hla\nGILGFVFTL,HLA-A*02:01\n")
    _, run = score_repertoire(source, panel, output, donor_id="d1", decoder_dir=tmp_path,
                              python_executable="unused")
    assert run["summary"]["cells"] == 0
    assert run["summary"]["cell_receptor_links"] == 2
    assert pl.read_parquet(output / "cells.parquet")["cell_id"].null_count() == 2


def test_repertoire_source_change_rejects_publication(tmp_path, monkeypatch):
    from tcr_workbench import workflows

    source, panel, output = tmp_path / "paired.csv", tmp_path / "panel.csv", tmp_path / "run"
    source.write_text("cell_id,donor_id,cdr3a,cdr3b\nc1,d1,,CASSLGQAYEQYF\n")
    panel.write_text("peptide,hla\nGILGFVFTL,HLA-A*02:01\n")
    digest = {"value": "before-ingestion"}
    original_read = workflows.read_receptors

    def read_then_change(*args, **kwargs):
        result = original_read(*args, **kwargs)
        digest["value"] = "changed-during-analysis"
        return result

    monkeypatch.setattr(workflows, "read_receptors", read_then_change)
    monkeypatch.setattr(workflows, "source_digest", lambda: digest["value"], raising=False)
    with pytest.raises(ValueError, match="source changed"):
        workflows.score_repertoire(source, panel, output, decoder_dir=tmp_path,
                                   python_executable="unused")
    assert not output.exists()


def test_repertoire_result_schema_stable_when_every_pair_is_skipped(tmp_path, monkeypatch):
    from tcr_workbench.workflows import RESULT_SCHEMA, score_repertoire

    monkeypatch.setattr(prediction, "run_decoder", fake_run)
    panel = tmp_path / "panel.csv"
    panel.write_text("peptide,hla\nGILGFVFTL,HLA-A*02:01\n")
    header = "cell_id,donor_id,trav,traj,cdr3a,trbv,trbj,cdr3b\n"
    rows = {
        "eligible": "c1,d1,TRAV21,TRAJ6,CAVRPGGAGPFF,TRBV7-9,TRBJ2-7,CASSLGQAYEQYF\n",
        "skipped": "c1,d1,,,,TRBV7-9,TRBJ2-7,CASSLGQAYEQYF\n",
    }
    results = {}
    for label, row in rows.items():
        source = tmp_path / f"{label}.csv"
        source.write_text(header + row)
        result, _ = score_repertoire(source, panel, tmp_path / label, decoder_dir=tmp_path,
                                    python_executable="unused")
        results[label] = result
        persisted = pl.read_parquet(tmp_path / label / "results.parquet")
        assert persisted.schema == pl.Schema(RESULT_SCHEMA)
        assert result.schema == persisted.schema
    assert results["eligible"].schema == results["skipped"].schema
    assert results["eligible"]["rank"].to_list() == [1]
    skipped = results["skipped"]
    assert skipped["status"].to_list() == ["Unresolved"]
    assert skipped["name"].to_list() == [None]
    assert skipped["TRAV"].to_list() == [None]
    assert skipped["rank"].to_list() == [None]
    assert skipped["provenance"].to_list() == ["not_scored"]


def test_repertoire_schema_preserves_additional_backend_audits():
    from tcr_workbench.workflows import RESULT_SCHEMA, _normalize_result_schema

    raw = pl.DataFrame({"score": [-2.0], "custom_reason": ["extra source audit"],
                        "future_flag": [True]})
    result = _normalize_result_schema(raw)
    assert result.columns[:len(RESULT_SCHEMA)] == list(RESULT_SCHEMA)
    assert result["custom_reason"].to_list() == ["extra source audit"]
    assert result["future_flag"].dtype == pl.Boolean


@pytest.mark.parametrize("device", ["cpu", "apple"])
def test_relative_configured_interpreters_keep_the_virtual_environment(tmp_path, device):
    base = tmp_path / "base-python"
    base.touch()
    decoder_python = tmp_path / "decoder-env/bin/python"
    decoder_python.parent.mkdir(parents=True)
    decoder_python.symlink_to(base)
    mlx_python = tmp_path / "mlx-env/bin/python"
    mlx_python.parent.mkdir(parents=True)
    mlx_python.symlink_to(base)
    checkpoint = tmp_path / ("bundle" if device == "apple" else "weights.ckpt")
    checkpoint.mkdir() if device == "apple" else checkpoint.touch()
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"decoder_dir":"decoder", "python_executable":"decoder-env/bin/python",
        "device":device, "checkpoint":checkpoint.name,
        "mlx_python":"mlx-env/bin/python" if device == "apple" else None}))
    result = resolve_settings(SimpleNamespace(config=path))
    assert result["python_executable"] == str(decoder_python)
    assert result["python_executable"] != str(decoder_python.resolve())
    if device == "apple":
        assert result["mlx_python"] == str(mlx_python)
        assert result["mlx_python"] != str(mlx_python.resolve())
