import os
import stat

import polars as pl
import pytest

from tcr_workbench.report import output_bundle, snapshot_inputs, write_evidence_batches


def test_streamed_formats_and_summary_preserve_all_rows(tmp_path):
    frame = pl.DataFrame({"receptor_id": ["001"] * 250 + ["002"] * 200 + ["003"],
                          "status": ["Candidate"] * 450 + ["Unresolved"],
                          "source": ['quoted, source "A"'] * 451,
                          "distance": [0] * 450 + [None]})
    preview, summary = write_evidence_batches(
        iter([frame.slice(0, 250), frame.slice(250)]), tmp_path / "evidence.csv")
    assert pl.read_parquet(tmp_path / "evidence.parquet").equals(frame)
    assert pl.read_csv(tmp_path / "evidence.csv", schema_overrides=frame.schema).equals(frame)
    assert preview.equals(frame.head(200))
    assert summary == {"evidence_rows": 451, "receptors_with_candidates": 2}


def test_typed_empty_evidence_is_readable(tmp_path):
    frame = pl.DataFrame(schema={"receptor_id": pl.String, "status": pl.String})
    preview, summary = write_evidence_batches(iter([frame]), tmp_path / "evidence.csv")
    assert pl.read_parquet(tmp_path / "evidence.parquet").schema == frame.schema
    assert pl.read_csv(tmp_path / "evidence.csv").is_empty()
    assert preview.is_empty()
    assert summary["evidence_rows"] == 0


def test_changed_input_prevents_publication(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("original")
    snapshot = snapshot_inputs([source])
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="Input changed during analysis"):
        with output_bundle(destination, input_snapshot=snapshot) as staging:
            (staging / "result.csv").write_text("finished")
            source.write_text("mutation")  # same size, so the content hash is necessary
    assert not destination.exists()
    assert not list(tmp_path.glob(".result-*"))


@pytest.mark.parametrize("mask,expected", [(0o022, 0o755), (0o077, 0o700)])
def test_published_directory_respects_umask(tmp_path, mask, expected):
    destination = tmp_path / "result"
    previous = os.umask(mask)
    try:
        with output_bundle(destination) as staging:
            (staging / "result.csv").write_text("finished")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(destination.stat().st_mode) & 0o777 == expected
    assert not (destination / ".directory-permissions").exists()
