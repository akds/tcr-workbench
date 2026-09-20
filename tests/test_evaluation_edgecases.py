"""Evaluation boundaries and relocation-safe CLI bundles, without model inference."""

import random

import polars as pl
import pytest

from tcr_workbench.cli import main
from tcr_workbench.evaluation import evaluate_scores


def data(scores, labels, **score_metadata):
    identity = {
        "receptor_id": [str(index) for index in range(len(labels))],
        "peptide": ["GILGFVFTL"] * len(labels),
        "hla": ["HLA-A*02:01"] * len(labels),
    }
    return (
        pl.DataFrame({**identity, "score": scores, **score_metadata}),
        pl.DataFrame({**identity, "label": labels}),
    )


def test_empty_evaluation_retains_a_typed_output_contract():
    identity = {"receptor_id": pl.String, "peptide": pl.String, "hla": pl.String}
    scores = pl.DataFrame(schema={**identity, "score": pl.Float64})
    labels = pl.DataFrame(schema={**identity, "label": pl.Int64})
    result = evaluate_scores(scores, labels)
    assert result.height == 0
    assert {"peptide", "hla", "n_receptors", "AUROC", "average_precision"} <= set(result.columns)
    assert result.schema["AUROC"] == pl.Float64
    assert result.schema["average_precision"] == pl.Float64


def test_all_unresolved_evaluation_keeps_float_types_even_with_no_numeric_metrics():
    scores, labels = data([None, float("nan")], [None, None])
    labels = labels.with_columns(pl.col("label").cast(pl.Int64))
    result = evaluate_scores(scores, labels)
    assert result["status"].to_list() == ["Unresolved"]
    for name in ("AUROC", "average_precision", "prevalence", "score_coverage"):
        assert result.schema[name] == pl.Float64
        assert result[name][0] is None


@pytest.mark.parametrize(
    "column,values",
    [
        ("metric", ["pll_model_a", "binding_affinity_nm"]),
        ("model", ["model_a", "model_b"]),
    ],
)
def test_incomparable_score_scales_are_not_pooled(column, values):
    scores, labels = data([0.9, 1.0], [1, 0], **{column: values})
    with pytest.raises(ValueError, match="metric|model|score"):
        evaluate_scores(scores, labels)


def test_one_known_plus_missing_label_does_not_create_a_label_conflict():
    scores, labels = data([0.9, 0.1], [1, 0])
    missing = labels.head(1).with_columns(pl.lit(None, dtype=pl.Int64).alias("label"))
    row = evaluate_scores(scores, pl.concat([labels, missing])).row(0, named=True)
    assert row["n_labelled"] == 2
    assert row["n_conflicting_labels"] == 0
    assert row["AUROC"] == 1.0


def test_tie_metrics_match_pairwise_auc_and_threshold_ap_for_random_small_sets():
    rng = random.Random(89)
    for _ in range(20):
        values = [rng.randrange(4) for _ in range(10)]
        targets = [1, 0] + [rng.randrange(2) for _ in range(8)]
        positive = [value for value, label in zip(values, targets) if label]
        negative = [value for value, label in zip(values, targets) if not label]
        expected_auc = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
        expected_auc /= len(positive) * len(negative)
        expected_ap = 0.0
        for threshold in sorted(set(values), reverse=True):
            newly_positive = sum(
                value == threshold and label == 1 for value, label in zip(values, targets)
            )
            included = [
                (value, label) for value, label in zip(values, targets) if value >= threshold
            ]
            precision = sum(label for _, label in included) / len(included)
            expected_ap += newly_positive / len(positive) * precision
        scores, labels = data(values, targets)
        row = evaluate_scores(scores, labels).row(0, named=True)
        assert row["AUROC"] == pytest.approx(expected_auc)
        assert row["average_precision"] == pytest.approx(expected_ap)
        inverted = scores.with_columns((-pl.col("score")).alias("score"))
        lower_row = evaluate_scores(inverted, labels, higher_is_better=False).row(0, named=True)
        assert lower_row["AUROC"] == pytest.approx(expected_auc)
        assert lower_row["average_precision"] == pytest.approx(expected_ap)


def test_evaluate_cli_preserves_numeric_looking_receptor_identities(tmp_path):
    scores = tmp_path / "scores.csv"
    labels = tmp_path / "labels.csv"
    scores.write_text(
        "receptor_id,peptide,hla,score\n"
        "001,GILGFVFTL,HLA-A*02:01,0.9\n"
        "1,GILGFVFTL,HLA-A*02:01,0.1\n"
    )
    labels.write_text(
        "receptor_id,peptide,hla,label\n001,GILGFVFTL,HLA-A*02:01,1\n1,GILGFVFTL,HLA-A*02:01,0\n"
    )
    out = tmp_path / "evaluation"
    assert (
        main(["evaluate", "--scores", str(scores), "--labels", str(labels), "--out", str(out)]) == 0
    )
    row = pl.read_parquet(out / "results.parquet").row(0, named=True)
    assert row["n_receptors"] == 2
    assert row["AUROC"] == 1.0


def test_validate_bundle_preserves_cell_and_donor_identifiers(tmp_path):
    source = tmp_path / "paired.csv"
    source.write_text(
        "cell_id,donor_id,cdr3a,cdr3b\n"
        "001,001,CAVRDSNYQLIW,CASSLGQETQYF\n"
        "1,001,CAVRDSNYQLIW,CASSLGQETQYF\n"
    )
    out = tmp_path / "validated"
    assert main(["validate", "--input", str(source), "--out", str(out)]) == 0
    cells = pl.read_parquet(out / "cells.parquet")
    assert set(cells["cell_id"]) == {"001", "1"}
    assert cells["donor_id"].unique().to_list() == ["001"]
    assert cells["receptor_id"].n_unique() == 1


def test_decoder_mapping_survives_atomic_bundle_rename_and_import(tmp_path):
    demo = tmp_path / "demo"
    exported = tmp_path / "exported"
    assert main(["example", "--out", str(demo)]) == 0
    assert (
        main(
            [
                "decoder-export",
                "--input",
                str(demo / "receptors.csv"),
                "--panel",
                str(demo / "panel.csv"),
                "--donors",
                str(demo / "donors.csv"),
                "--out",
                str(exported),
            ]
        )
        == 0
    )
    inputs = pl.read_csv(exported / "decoder_input.csv")
    scores = tmp_path / "mock_scores.csv"
    inputs.with_columns(pl.lit(-1.5).alias("pll_DecoderTCR-ESMC_300M")).write_csv(scores)
    imported = tmp_path / "imported"
    assert (
        main(
            [
                "decoder-import",
                "--input",
                str(exported / "decoder_input.csv"),
                "--scores",
                str(scores),
                "--out",
                str(imported),
            ]
        )
        == 0
    )
    results = pl.read_parquet(imported / "results.parquet")
    assert results.height == inputs.height
    assert results["receptor_id"].null_count() == 0
    cells = pl.read_parquet(exported / "cells.parquet")
    assert cells["cell_id"].n_unique() == 4
    assert set(results["receptor_id"]) == set(cells["receptor_id"])
