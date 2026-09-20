"""Model-free checks of experimental filtering, exact search and provenance."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from tcr_workbench import embedding_matching as em


ALPHA = "CAVRPGGAGPFFVVF"
BETA = "CASSLGQAYEQYF"


def query(**updates):
    row = dict(receptor_id="q1", donor_id="d1", species="human", trav="TRAV21",
               traj="TRAJ6", cdr3a=ALPHA, trbv="TRBV7-9", trbj="TRBJ2-7", cdr3b=BETA,
               pairing_status="paired", unusable_chain_context="", cell_count=1)
    return {**row, **updates}


def reference(**updates):
    row = {key: value for key, value in query().items() if key in (*em._COMPONENTS, "species")}
    row.update(reference_id="ref1", peptide="GILGFVFTL", hla="HLA-A*02:01",
               source="public_synthetic_fixture", evidence="synthetic_only")
    return {**row, **updates}


def paired_vector(first=(1, 0), second=(1, 0)):
    halves = np.asarray([first, second], dtype=np.float64)
    halves /= np.linalg.norm(halves, axis=1, keepdims=True)
    return (halves.reshape(-1) / np.sqrt(2)).astype(np.float32)


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(calls=[], by_beta={}, aliases={}, fail=set(), corrupt=None)

    def embed(frame, output, **kwargs):
        assert frame.columns == list(em._INPUT_SCHEMA)
        assert not {"peptide", "hla", "source", "evidence"} & set(frame.columns)
        state.calls.append((frame.clone(), kwargs))
        output.mkdir()
        vectors, ids, audit = [], [], []
        for row in frame.iter_rows(named=True):
            failed = row["cdr3b"] in state.fail
            digest = state.aliases.get(row["trbv"], hashlib.sha256(
                (row["cdr3a"] + "|" + row["cdr3b"] + "|" + row["trbv"]).encode()).hexdigest())
            audit.append(dict(embedding_id=row["embedding_id"],
                              status="Unresolved" if failed else "Embedded",
                              reason="full-chain reconstruction failed" if failed else "",
                              reconstructed_receptor_sha256=None if failed else digest))
            if not failed:
                ids.append(row["embedding_id"])
                vectors.append(state.by_beta.get(row["cdr3b"], paired_vector()))
        array = np.stack(vectors) if vectors else np.empty((0, 4), dtype=np.float32)
        result = SimpleNamespace(vectors=array, ids=ids, audit=pl.DataFrame(audit),
                                 metadata={"representation": "decodertcr-chain-mean-v1", "precision": "float32",
                                           "checkpoint_sha256": "a" * 64})
        if state.corrupt:
            state.corrupt(result)
        np.save(output / "receptor_embeddings.npy", result.vectors)
        (output / "embedding_ids.json").write_text(json.dumps(result.ids))
        result.audit.write_csv(output / "embedding_audit.csv")
        (output / "manifest.json").write_text(json.dumps(result.metadata))
        return result

    monkeypatch.setitem(sys.modules, "tcr_workbench.receptor_embeddings",
                        SimpleNamespace(embed_receptors=embed))
    return state


def run(tmp_path, *, queries=None, refs=None, panel=None, donors=None, **kwargs):
    return em.run_embedding_matching(
        pl.DataFrame(queries if queries is not None else [query()]),
        pl.DataFrame(refs if refs is not None else [reference()]),
        pl.DataFrame(panel if panel is not None else [dict(peptide="GILGFVFTL", hla="A*02:01")]),
        pl.DataFrame(donors if donors is not None else [dict(donor_id="d1", hla="A*02:01")]),
        tmp_path, **kwargs)


def test_output_separate_from_exact_matches_and_full_provenance(tmp_path, backend):
    (tmp_path / "evidence.csv").write_text("unchanged exact evidence\n")
    metadata = run(tmp_path, refs=[reference(reference_id="b"),
                                   reference(reference_id="a", evidence="other", note="retained")],
                   top_k=1, model="esmc-300m", device="cpu")
    result = pl.read_parquet(tmp_path / "embedding_matches.parquet")
    assert result["reference_id"].to_list() == ["a", "b"]
    assert result["neighbor_rank"].to_list() == [1, 1]
    assert result["status"].to_list() == ["Experimental"] * 2
    assert result["cosine_similarity"].to_list() == pytest.approx([1, 1], abs=1e-6)
    assert json.loads(result["reference_record_json"][0])["note"] == "retained"
    assert len(backend.calls) == 1 and backend.calls[0][0].height == 1
    assert (tmp_path / "evidence.csv").read_text() == "unchanged exact evidence\n"
    assert metadata["matched_queries"] == 1 and metadata["match_rows"] == 2
    assert "embedding_model_manifest.json" in metadata["output_sha256"]
    assert (tmp_path / "embedding_reconstruction_audit.csv").is_file()
    assert set(metadata["input_sha256"]) == {"receptors", "references", "panel", "donors"}
    assert not (tmp_path / "vectors").exists()


def test_actual_full_chain_identity_deduplicates_gene_aliases(tmp_path, backend):
    same = "f" * 64
    backend.aliases = {"TRBV7-9": same, "TRBV7-8": same}
    other_beta = "CASSLGQAYEQF"
    backend.by_beta[other_beta] = paired_vector((0, 1), (0, 1))
    result_meta = run(tmp_path, refs=[reference(reference_id="a"),
        reference(reference_id="alias", trbv="TRBV7-8"),
        reference(reference_id="other", trbv="TRBV19", cdr3b=other_beta)], top_k=2)
    results = pl.read_parquet(tmp_path / "embedding_matches.parquet")
    assert results["reference_id"].to_list() == ["a", "alias", "other"]
    assert results["neighbor_rank"].to_list() == [1, 1, 2]
    assert result_meta["match_rows"] == 3


def test_species_is_explicit_and_cross_species_is_audited(tmp_path, backend):
    frame = pl.DataFrame([reference()]).drop("species")
    with pytest.raises(ValueError, match="explicit species"):
        em.run_embedding_matching(pl.DataFrame([query()]), frame, pl.DataFrame(), None, tmp_path)
    assert not backend.calls
    run(tmp_path, refs=[reference(), reference(reference_id="mouse", species="mouse")])
    assert backend.calls[0][0]["species"].to_list() == ["human"]
    audit = pl.read_csv(tmp_path / "embedding_audit.csv")
    excluded = audit.filter(pl.col("record_id") == "mouse").row(0, named=True)
    assert excluded["status"] == "Unresolved" and "mouse" in excluded["reason"]


@pytest.mark.parametrize("change", [dict(cdr3a=None), dict(trbv="TRBV7-9,TRBV19"),
    dict(cdr3b="CASSXGF"), dict(pairing_status="dual_alpha"),
    dict(unusable_chain_context="beta:nonproductive"), dict(species="mouse")])
def test_ineligible_queries_have_explicit_audit_and_no_inference(tmp_path, backend, change):
    metadata = run(tmp_path, queries=[query(**change)])
    assert not backend.calls and metadata["match_rows"] == 0
    result = pl.read_parquet(tmp_path / "embedding_matches.parquet")
    assert result.is_empty() and result.schema == em._MATCH_SCHEMA
    audit = pl.read_csv(tmp_path / "embedding_audit.csv")
    item = audit.filter(pl.col("source") == "query").row(0, named=True)
    assert item["status"] == "Unresolved" and item["reason"]


def test_panel_filter_precedes_top_k_and_preserves_unresolved_donor(tmp_path, backend):
    beta = "CASSLGQAYEQF"
    backend.by_beta[beta] = paired_vector((0, 1), (0, 1))
    run(tmp_path, refs=[reference(reference_id="closest_wrong_mhc", hla="B*07:02"),
                       reference(reference_id="allowed", cdr3b=beta)],
        donors=[dict(donor_id="d1", hla="B*07:02")], top_k=1)
    result = pl.read_parquet(tmp_path / "embedding_matches.parquet")
    assert result["reference_id"].to_list() == ["allowed"]
    assert result["donor_hla_status"].to_list() == ["unresolved"]
    assert result["hla_status"].to_list() == ["unresolved"]
    assert "does not cover the reference HLA locus" in result["reason"][0]
    assert result["cosine_similarity"][0] == 0


def test_negative_similarity_is_not_a_binding_classification(tmp_path, backend):
    beta = "CASSLGQAYEQF"
    backend.by_beta[beta] = paired_vector((-1, 0), (-1, 0))
    run(tmp_path, refs=[reference(cdr3b=beta)])
    result = pl.read_parquet(tmp_path / "embedding_matches.parquet")
    assert result["cosine_similarity"][0] == pytest.approx(-1, abs=1e-6)
    assert result["cosine_distance"][0] == pytest.approx(2, abs=1e-6)
    assert result["status"][0] == "Experimental"
    assert "not transferred labels" in result["reason"][0]


def test_backend_failure_retained_in_audit(tmp_path, backend):
    backend.fail.add(BETA)
    meta = run(tmp_path)
    assert meta["match_rows"] == 0
    audit = pl.read_csv(tmp_path / "embedding_audit.csv")
    assert set(audit["reason"]) == {"full-chain reconstruction failed"}


@pytest.mark.parametrize("corrupt,match", [
    (lambda r: setattr(r, "vectors", r.vectors.astype(np.float16)), "FP32"),
    (lambda r: r.vectors.fill(np.nan), "nonfinite"),
    (lambda r: r.vectors.fill(0), "unit norm"),
    (lambda r: setattr(r, "ids", ["unknown"]), "requested"),
    (lambda r: setattr(r, "audit", r.audit.head(0)), "every requested"),
    (lambda r: setattr(r, "audit", r.audit.with_columns(pl.lit("Unresolved").alias("status"))), "contradict"),
    (lambda r: r.metadata.update(precision="float16"), "precision"),
    (lambda r: r.metadata.update(representation="peptide-conditioned"), "representation"),
    (lambda r: setattr(r, "audit", r.audit.with_columns(pl.lit(None).alias("reconstructed_receptor_sha256"))), "SHA256"),
])
def test_invalid_backend_output_never_publishes_matches(tmp_path, backend, corrupt, match):
    backend.corrupt = corrupt
    with pytest.raises(ValueError, match=match):
        run(tmp_path)
    assert not (tmp_path / "embedding_matches.csv").exists()
    assert not list(tmp_path.glob(".embedding-matching-*"))


def test_chain_normalization_is_checked():
    with pytest.raises(ValueError, match="contribute equally"):
        em._validate_vectors(np.asarray([[1, 0, 0, 0]], np.float32), ["a"], {"a"})


@pytest.mark.parametrize("k", [0, -1, True, 1.5, 1001])
def test_invalid_neighbor_counts_fail_before_inference(tmp_path, backend, k):
    with pytest.raises(ValueError, match="top_k"):
        run(tmp_path, top_k=k)
    assert not backend.calls


def test_approximate_precision_rejected(tmp_path, backend):
    with pytest.raises(ValueError, match="float32"):
        run(tmp_path, precision="float16")
    assert not backend.calls


def test_duplicate_query_ids_fail(tmp_path, backend):
    with pytest.raises(ValueError, match="receptor_id must be unique"):
        run(tmp_path, queries=[query(), query()])


def test_blocked_search_matches_dense_oracle_with_ties():
    rng = np.random.default_rng(12)
    vectors = rng.normal(size=(37, 12)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors[18] = vectors[17]
    queries, references = [0, 3, 7, 10, 17], list(range(11, 37))
    dense = vectors[queries].astype(np.float64) @ vectors[references].astype(np.float64).T
    expected = np.lexsort((np.broadcast_to(np.arange(len(references)), dense.shape), -dense), axis=1)
    for block_q, block_r in [(1, 1), (2, 4), (32, 1024)]:
        result = list(em._iter_top_k(vectors, queries, references, top_k=7,
                                    query_block_size=block_q, reference_block_size=block_r))
        for row, (query_index, indices, scores) in enumerate(result):
            assert query_index == queries[row]
            np.testing.assert_array_equal(indices, np.asarray(references)[expected[row, :7]])
            np.testing.assert_allclose(scores, np.clip(dense[row, expected[row, :7]], -1, 1), atol=1e-12)


def test_search_empty_reference_returns_no_neighbor():
    result = list(em._iter_top_k(np.ones((1, 2), np.float32), [0], [], top_k=10))
    assert len(result) == 1 and len(result[0][1]) == len(result[0][2]) == 0


def test_reference_reordering_preserves_ties_and_provenance(tmp_path, backend):
    rows = [reference(reference_id="a"), reference(reference_id="b", trbv="TRBV19")]
    run(tmp_path / "first", refs=rows, top_k=1)
    run(tmp_path / "second", refs=rows[::-1], top_k=1)
    first = pl.read_parquet(tmp_path / "first/embedding_matches.parquet")
    second = pl.read_parquet(tmp_path / "second/embedding_matches.parquet")
    assert first.equals(second)


def test_existing_optional_outputs_are_protected(tmp_path, backend):
    (tmp_path / "embedding_audit.csv").write_text("original")
    with pytest.raises(ValueError, match="already exist"):
        run(tmp_path)
    assert not backend.calls
    assert (tmp_path / "embedding_audit.csv").read_text() == "original"


def test_wide_reference_rows_materialized_once_across_donor_groups(tmp_path, backend, monkeypatch):
    """Eligibility uses only HLA; retained wide provenance is reused in a bounded cache."""
    rows = [reference(reference_id=f"ref{i}", trbv=f"TRBV{i + 1}",
                      note="wide provenance " * 1000) for i in range(3)]
    queries = [query(receptor_id=f"q{i}", donor_id=f"donor{i % 3}") for i in range(6)]
    donors = [dict(donor_id=f"donor{i}", hla=hla)
              for i, hla in enumerate(["A*02:01", "A*03:01", "B*07:02"])]
    original = pl.DataFrame.row
    expanded = []

    def tracked(frame, *args, **kwargs):
        if kwargs.get("named") and "reference_id" in frame.columns:
            expanded.append(args[0])
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "row", tracked)
    run(tmp_path, queries=queries, refs=rows, donors=donors, top_k=3)
    results = pl.read_parquet(tmp_path / "embedding_matches.parquet")
    assert results.height == 18
    assert sorted(expanded) == [0, 1, 2]
    assert results.group_by("receptor_id").len()["len"].to_list() == [3] * 6
    assert all(json.loads(value)["note"] == rows[0]["note"]
               for value in results["reference_record_json"])
