import polars as pl
import pytest

from tcr_workbench.evaluation import evaluate_scores


def inputs(values, labels):
    base = {"receptor_id": [str(i) for i in range(len(labels))],
            "peptide": ["GILGFVFTL"] * len(labels), "hla": ["HLA-A*02:01"] * len(labels)}
    return pl.DataFrame({**base, "score": values}), pl.DataFrame({**base, "label": labels})


@pytest.mark.parametrize("values,expected_auc,expected_ap", [
    ([4., 3., 2., 1.], 1., 1.), ([1., 2., 3., 4.], 0., (1 / 3 + 2 / 4) / 2),
    ([1., 1., 1., 1.], .5, .5), ([2., 1., 2., 1.], .5, .5),
])
def test_rank_metrics_and_ties(values, expected_auc, expected_ap):
    scores, labels = inputs(values, [1, 1, 0, 0])
    row = evaluate_scores(scores, labels).row(0, named=True)
    assert row["AUROC"] == pytest.approx(expected_auc)
    assert row["average_precision"] == pytest.approx(expected_ap)


def test_conflicts_missing_and_coverage():
    scores, labels = inputs([4., None, 2., float("nan")], [1, 1, 0, 0])
    extra = labels.head(1).with_columns(pl.lit(0, dtype=pl.Int64).alias("label"))
    result = evaluate_scores(scores, pl.concat([labels, extra])).row(0, named=True)
    assert result["n_conflicting_labels"] == 1
    assert result["n_scored"] == 1
    assert result["score_coverage"] == pytest.approx(1 / 3)
    assert result["AUROC"] is None


def test_rejects_duplicate_score_identity():
    scores, labels = inputs([1., 2.], [0, 1])
    with pytest.raises(ValueError, match="one row"):
        evaluate_scores(pl.concat([scores, scores.head(1)]), labels)


def test_scores_without_labels_are_counted_as_unknown():
    scores, labels = inputs([1., 2., 3.], [0, 1, 1])
    result = evaluate_scores(scores, labels.head(2)).row(0, named=True)
    assert result["n_receptors"] == 3
    assert result["n_unknown_labels"] == 1
    assert result["n_labelled"] == 2
    assert result["score_coverage"] == 1.


def test_normalizes_peptide_and_hla_before_identity_join():
    scores, labels = inputs([1., 2.], [0, 1])
    labels = labels.with_columns(pl.lit(" gilgfvftl ").alias("peptide"),
                                  pl.lit("A*02:01").alias("hla"))
    result = evaluate_scores(scores, labels).row(0, named=True)
    assert result["n_receptors"] == 2
    assert result["score_coverage"] == 1.
    assert result["AUROC"] == 1.
def test_evaluation_rows_preserve_score_provenance_and_direction():
    scores = pl.DataFrame({"receptor_id": ["001", "002"], "peptide": ["GILGFVFTL"] * 2,
                           "hla": ["HLA-A*02:01"] * 2, "score": [0.1, 0.9],
                           "model": ["example-v1"] * 2, "metric": ["affinity"] * 2})
    labels = scores.select("receptor_id", "peptide", "hla").with_columns(
        pl.Series("label", [1, 0]))
    result = evaluate_scores(scores, labels, higher_is_better=False).row(0, named=True)
    assert result["model"] == "example-v1"
    assert result["metric"] == "affinity"
    assert result["higher_is_better"] is False
    assert result["AUROC"] == 1.0
