import html
import json

import polars as pl

from tcr_workbench.cli import main
from tcr_workbench.report import output_bundle, sha256_file


def test_demo_end_to_end(tmp_path):
    demo = tmp_path / "demo"
    assert main(["example", "--out", str(demo)]) == 0
    out = demo / "results"
    args = ["screen", "--input", str(demo / "receptors.csv"), "--reference",
            str(demo / "reference.csv"), "--panel", str(demo / "panel.csv"),
            "--donors", str(demo / "donors.csv"), "--out", str(out)]
    assert main(args) == 0
    receptors = pl.read_parquet(out / "receptors.parquet")
    evidence = pl.read_parquet(out / "evidence.parquet")
    assert receptors.height == 3
    assert evidence["receptor_id"].n_unique() == 3
    assert "Unresolved" in evidence["status"].to_list()
    assert "Candidate" in evidence["status"].to_list()
    run = json.loads((out / "manifest.json").read_text())
    assert run["inputs"]["receptors"]["sha256"] == sha256_file(demo / "receptors.csv")
    for name, metadata in run["outputs"].items():
        assert sha256_file(out / name) == metadata["sha256"]
    original = (out / "manifest.json").read_bytes()
    assert main(args) == 2
    assert (out / "manifest.json").read_bytes() == original


def test_atomic_failure_removes_partial_directory(tmp_path):
    output = tmp_path / "failed"
    try:
        with output_bundle(output) as staging:
            (staging / "partial.txt").write_text("partial")
            raise ValueError("test failure")
    except ValueError:
        pass
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_report_escapes_reference_text(tmp_path):
    demo = tmp_path / "demo"
    main(["example", "--out", str(demo)])
    ref = demo / "reference.csv"
    ref.write_text(ref.read_text().replace("SYNTHETIC_DEMO_NOT_BIOLOGICAL_EVIDENCE", "<script>alert(1)</script>"))
    out = demo / "results"
    assert main(["screen", "--input", str(demo / "receptors.csv"), "--reference", str(ref),
                 "--panel", str(demo / "panel.csv"), "--donors", str(demo / "donors.csv"),
                 "--out", str(out)]) == 0
    report = (out / "report.html").read_text()
    assert "<script>alert(1)</script>" not in report
    assert html.escape("<script>alert(1)</script>") in report


def test_profile_cli(tmp_path):
    panel = tmp_path / "panel.csv"
    panel.write_text("peptide,hla\nGILGFVFTL,HLA-A*02:01\nGILGFVFTV,HLA-A*02:01\n")
    out = tmp_path / "profile"
    assert main(["profile", "--panel", str(panel), "--out", str(out)]) == 0
    frame = pl.read_parquet(out / "results.parquet")
    assert frame.height == 9 * 20


def test_evaluate_quoted_missing_values(tmp_path):
    scores, labels = tmp_path / "scores.csv", tmp_path / "labels.csv"
    scores.write_text('receptor_id,peptide,hla,score\n001,GILGFVFTL,HLA-A*02:01,""\n002,GILGFVFTL,HLA-A*02:01,0.5\n')
    labels.write_text('receptor_id,peptide,hla,label\n001,GILGFVFTL,HLA-A*02:01,1\n002,GILGFVFTL,HLA-A*02:01,""\n')
    out = tmp_path / "evaluation"
    assert main(["evaluate", "--scores", str(scores), "--labels", str(labels),
                 "--out", str(out)]) == 0
    frame = pl.read_parquet(out / "results.parquet")
    assert frame["n_labelled"][0] == 1
    assert frame["n_scored"][0] == 0
    assert frame["score_coverage"][0] == 0
    assert frame["n_unknown_labels"][0] == 1


def test_additional_command_manifests_hash_all_outputs(tmp_path):
    demo = tmp_path / "demo"
    assert main(["example", "--out", str(demo)]) == 0
    for command, flags in (
        ("validate", ["--input", str(demo / "receptors.csv")]),
        ("decoder-export", ["--input", str(demo / "receptors.csv"), "--panel", str(demo / "panel.csv")]),
        ("profile", ["--panel", str(demo / "panel.csv")]),
    ):
        output = tmp_path / command
        assert main([command, *flags, "--out", str(output)]) == 0
        run = json.loads((output / "manifest.json").read_text())
        assert set(run["outputs"]) == {p.name for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
        for name, entry in run["outputs"].items():
            assert sha256_file(output / name) == entry["sha256"]


def test_pmhc_profile_prints_unresolved_reason(tmp_path, capsys):
    out = tmp_path / "profile"
    assert main(["pmhc-profile", "--hla", "HLA-DRB1*04:01", "--length", "15",
                 "--decoder-dir", str(tmp_path), "--python", "unused", "--out", str(out)]) == 0
    assert "Unresolved:" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "Unresolved" and summary["reason"]
