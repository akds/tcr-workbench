"""Optional embedding integration must preserve species and atomic publication."""
import json

import polars as pl

from tcr_workbench.cli import main


def inputs(tmp_path):
    demo = tmp_path / "inputs"
    assert main(["example", "--out", str(demo)]) == 0
    refs = pl.read_csv(demo / "reference.csv").with_columns(pl.lit("human").alias("species"))
    refs.write_csv(demo / "reference.csv")
    return demo, refs


def command(demo, output):
    return ["screen", "--input", str(demo / "receptors.csv"), "--format", "paired",
            "--reference", str(demo / "reference.csv"), "--panel", str(demo / "panel.csv"),
            "--donors", str(demo / "donors.csv"), "--out", str(output),
            "--embedding-matching", "--decoder-dir", str(demo), "--python", "unused",
            "--device", "cpu"]


def test_combined_screen_species_filter_prevents_cross_species_cdr3_evidence(tmp_path, monkeypatch):
    from tcr_workbench import embedding_matching

    monkeypatch.delenv("TCR_WORKBENCH_PREPARE_DIR", raising=False)
    demo, refs = inputs(tmp_path)
    mouse = refs.with_columns(pl.lit("mouse").alias("species"),
                              pl.lit("mouse-same-cdr3").alias("reference_id"),
                              pl.lit(None, dtype=pl.String).alias("hla"))
    pl.concat([refs, mouse]).write_csv(demo / "reference.csv")

    def fake(receptors, references, panel, donors, output_dir, **kwargs):
        # Both original species reach the optional module for its exclusion audit.
        assert set(references["species"]) == {"human", "mouse"}
        assert kwargs["species"] == "human"
        empty = pl.DataFrame(schema={"receptor_id": pl.String, "status": pl.String,
                                     "cosine_distance": pl.Float64})
        empty.write_parquet(output_dir / "embedding_matches.parquet")
        empty.write_csv(output_dir / "embedding_matches.csv")
        metadata = {"status": "Experimental", "match_rows": 0}
        (output_dir / "embedding_metadata.json").write_text(json.dumps(metadata))
        return metadata

    monkeypatch.setattr(embedding_matching, "run_embedding_matching", fake)
    output = tmp_path / "result"
    assert main(command(demo, output)) == 0
    evidence = pl.read_csv(output / "evidence.csv")
    assert "mouse-same-cdr3" not in evidence["reference_id"].drop_nulls().to_list()
    assert "synthetic-1" in evidence["reference_id"].drop_nulls().to_list()


def test_embedding_failure_does_not_publish_partial_cdr3_analysis(tmp_path, monkeypatch):
    from tcr_workbench import embedding_matching

    monkeypatch.delenv("TCR_WORKBENCH_PREPARE_DIR", raising=False)
    demo, _ = inputs(tmp_path)

    def fail(receptors, references, panel, donors, output_dir, **kwargs):
        # Ordinary evidence has already been staged, but remains unpublished.
        assert (output_dir / "evidence.csv").is_file()
        raise ValueError("Synthetic checkpoint identity failure")

    monkeypatch.setattr(embedding_matching, "run_embedding_matching", fail)
    output = tmp_path / "result"
    assert main(command(demo, output)) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".result-*"))
