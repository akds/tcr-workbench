"""Optional model references stay separate from panel ranks and CDR3 evidence."""
import json

import polars as pl

from tcr_workbench.cli import main, parser
from tcr_workbench.report import sha256_file, write_workflow_report


def test_optional_reference_parser_defaults():
    args = parser().parse_args(["pmhc-score", "--panel", "panel.csv", "--out", "out"])
    assert args.background_mode == "uniform" and args.background_peptides == 1000
    profile = parser().parse_args(["pmhc-score", "--panel", "panel.csv", "--out", "out",
                                  "--background-mode", "mhc-profile", "--background-peptides", "37"])
    assert profile.background_mode == "mhc-profile" and profile.background_peptides == 37
    screen = parser().parse_args(["screen", "--input", "in.csv", "--panel", "panel.csv",
                                  "--donors", "donors.csv", "--reference", "ref.csv", "--out", "out"])
    assert not screen.embedding_matching and screen.embedding_top_k == 10


def test_profile_reference_report_names_generation_and_scoring_context(tmp_path):
    tested = pl.DataFrame({"receptor_id": ["r1"], "hla": ["HLA-A*02:01"],
                          "peptide": ["GILGFVFTL"], "score": [-2.], "rank": [1],
                          "background_percentile": [50.], "background_n": [3], "status": ["Scored"]})
    background = pl.DataFrame({"receptor_id": ["r1"] * 3, "hla": ["HLA-A*02:01"] * 3,
                              "peptide": ["AAAAAAAAA"] * 3, "score": [-3., -2.5, -1.], "status": ["Scored"] * 3})
    write_workflow_report(tmp_path, "Peptide scores", table=tested, background=background,
                          workflow="tcr-score", background_metadata={"mode": "mhc-profile"},
                          interpretation="Model scores")
    report = (tmp_path / "report.html").read_text()
    assert "MHC-profile reference" in report and "No TCR is used to generate them" in report
    assert "PLL and panel rank remain the main outputs" in report
    assert "Uniform AA20" not in report
    assert "background_percentile" in report and "background_n" in report


def test_empty_embedding_report_identifies_audit_and_does_not_imply_nonbinding(tmp_path):
    write_workflow_report(tmp_path, "Reference matches", embedding_table=pl.DataFrame(),
                          embedding_total_rows=0, interpretation="Reference evidence")
    report = (tmp_path / "report.html").read_text()
    assert "No embedding neighbors were returned" in report
    assert "embedding_audit.csv" in report
    assert "does not establish that no peptide is recognized" in report


def test_cli_embedding_results_do_not_replace_cdr3_evidence(tmp_path, monkeypatch):
    from tcr_workbench import embedding_matching

    monkeypatch.delenv("TCR_WORKBENCH_PREPARE_DIR", raising=False)
    demo = tmp_path / "inputs"
    assert main(["example", "--out", str(demo)]) == 0
    reference = demo / "reference.csv"
    pl.read_csv(reference).with_columns(pl.lit("human").alias("species")).write_csv(reference)
    common = ["screen", "--input", str(demo / "receptors.csv"),
              "--panel", str(demo / "panel.csv"), "--donors", str(demo / "donors.csv"),
              "--reference", str(demo / "reference.csv")]
    assert main([*common, "--out", str(tmp_path / "ordinary")]) == 0

    def fake(receptors, references, panel, donors, output_dir, **kwargs):
        assert kwargs["top_k"] == 2 and kwargs["species"] == "human"
        frame = pl.DataFrame({"receptor_id": [receptors["receptor_id"][0]],
                              "reference_id": ["synthetic-1"], "status": ["Experimental"],
                              "cosine_distance": [0.01], "source": ["<script>alert(1)</script>"]})
        frame.write_parquet(output_dir / "embedding_matches.parquet")
        frame.write_csv(output_dir / "embedding_matches.csv")
        pl.DataFrame({"reason": ["test fixture"]}).write_csv(output_dir / "embedding_audit.csv")
        metadata = {"experimental": True}
        (output_dir / "embedding_metadata.json").write_text(json.dumps(metadata))
        return metadata

    monkeypatch.setattr(embedding_matching, "run_embedding_matching", fake)
    out = tmp_path / "with-embeddings"
    assert main([*common, "--out", str(out), "--embedding-matching", "--embedding-top-k", "2",
                 "--decoder-dir", str(tmp_path), "--python", "unused", "--device", "cpu"]) == 0
    assert (out / "evidence.csv").read_bytes() == (tmp_path / "ordinary/evidence.csv").read_bytes()
    report = (out / "report.html").read_text()
    assert "Experimental embedding neighbors" in report
    assert "<script>alert(1)</script>" not in report and "&lt;script&gt;" in report
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["embedding_matching"]["experimental"]
    for name, record in manifest["outputs"].items():
        assert sha256_file(out / name) == record["sha256"]
