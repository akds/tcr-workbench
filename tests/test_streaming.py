"""Streaming screening preserves complete evidence and deterministic global order."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tcr_workbench.matching import (
    _OUTPUT_SCHEMA,
    _iter_screen_validated,
    _rank_evidence,
    _screen_validated,
    screen,
)


def fixtures(n_receptors=3, n_references=4):
    receptors = pl.DataFrame(
        {
            "receptor_id": [f"tcr_{i:06}" for i in reversed(range(n_receptors))],
            "donor_id": ["d1"] * n_receptors,
            "cdr3a": pl.Series([None] * n_receptors, dtype=pl.String),
            "cdr3b": ["CASSLGQETQYF"] * n_receptors,
            "pairing_status": ["single_beta"] * n_receptors,
        },
        schema_overrides={
            "receptor_id": pl.String,
            "cdr3b": pl.String,
            "donor_id": pl.String,
            "pairing_status": pl.String,
        },
    )
    references = pl.DataFrame(
        {
            "reference_id": [f"ref_{i:06}" for i in reversed(range(n_references))],
            "cdr3b": ["CASSLGQETQYF"] * n_references,
            "peptide": ["GILGFVFTL"] * n_references,
            "hla": ["HLA-A*02:01"] * n_references,
            "source": ["synthetic"] * n_references,
            "evidence": ["curated"] * n_references,
        }
    )
    panel = pl.DataFrame({"peptide": ["GILGFVFTL"], "hla": ["HLA-A*02:01"]})
    donors = pl.DataFrame({"donor_id": ["d1"], "hla": ["HLA-A*02:01"]})
    return receptors, references, panel, donors


def test_iterator_matches_public_and_materialized_api_with_scrambled_input():
    frames = fixtures()
    batches = list(_iter_screen_validated(*frames, max_distance=0))
    actual = pl.concat(batches, rechunk=False)
    assert_frame_equal(actual, screen(*frames, max_distance=0))
    assert_frame_equal(actual, _screen_validated(*frames, max_distance=0))
    assert actual["receptor_id"].to_list() == [f"tcr_{i:06}" for i in range(3) for _ in range(4)]
    assert actual["reference_id"].to_list() == [f"ref_{i:06}" for _ in range(3) for i in range(4)]
    assert actual["evidence_rank"].to_list() == [1, 2, 3, 4] * 3


def test_many_small_groups_form_bounded_batches_without_splitting_receptors():
    frames = fixtures(n_receptors=150, n_references=100)
    batches = list(_iter_screen_validated(*frames, max_distance=0))
    assert len(batches) > 1
    assert all(0 < batch.height <= 8192 for batch in batches)
    identities = [set(batch["receptor_id"]) for batch in batches]
    assert sum(map(len, identities)) == len(set.union(*identities)) == 150
    actual = pl.concat(batches, rechunk=False)
    assert actual.height == 15_000
    assert_frame_equal(actual, _rank_evidence(actual))
    ranks = actual.group_by("receptor_id").agg(
        pl.col("evidence_rank").min().alias("first"),
        pl.col("evidence_rank").max().alias("last"),
    )
    assert ranks["first"].unique().to_list() == [1]
    assert ranks["last"].unique().to_list() == [100]


def test_large_single_receptor_block_is_complete_and_ranked():
    frames = fixtures(n_receptors=1, n_references=9001)
    batches = list(_iter_screen_validated(*frames, max_distance=0))
    assert len(batches) == 1
    result = batches[0]
    assert result.height == 9001
    assert result["reference_id"].n_unique() == 9001
    assert result["evidence_rank"].to_list() == list(range(1, 9002))
    assert_frame_equal(result, _rank_evidence(result))


def test_smaller_internal_buffer_chunks_cannot_drop_already_flushed_rows(monkeypatch):
    import tcr_workbench.matching as matching

    frames = fixtures(n_receptors=3, n_references=100)
    expected = _screen_validated(*frames, max_distance=0)
    original = matching._EvidenceBuffer
    monkeypatch.setattr(matching, "_EvidenceBuffer", lambda: original(batch_size=10))
    actual = pl.concat(list(_iter_screen_validated(*frames, max_distance=0)), rechunk=False)
    assert actual.height == 300
    assert_frame_equal(actual, expected)


def test_empty_input_yields_one_self_describing_frame():
    frames = fixtures(n_receptors=0)
    batches = list(_iter_screen_validated(*frames))
    assert len(batches) == 1
    assert batches[0].height == 0
    assert batches[0].schema == _OUTPUT_SCHEMA
    assert_frame_equal(batches[0], screen(*frames))


@pytest.mark.parametrize("invalid", [-1, 4, True, 1.1])
def test_iterator_checks_distance_before_yielding(invalid):
    with pytest.raises(ValueError, match="max_distance"):
        list(_iter_screen_validated(*fixtures(), max_distance=invalid))
