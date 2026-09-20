"""Clone-level ranking metrics with explicit missing-score and label coverage."""
from __future__ import annotations

import numpy as np
import polars as pl

from .hla import normalize_hla

EVALUATION_SCHEMA = {
    "peptide": pl.String, "hla": pl.String, "n_receptors": pl.Int64,
    "n_labelled": pl.Int64, "n_scored": pl.Int64, "n_conflicting_labels": pl.Int64,
    "n_unknown_labels": pl.Int64, "score_coverage": pl.Float64, "positives": pl.Int64,
    "prevalence": pl.Float64, "AUROC": pl.Float64, "average_precision": pl.Float64,
    "status": pl.String, "reason": pl.String,
    "model": pl.String, "metric": pl.String, "higher_is_better": pl.Boolean,
}


def _normalize_keys(frame: pl.DataFrame) -> pl.DataFrame:
    """Canonical pMHC identities prevent silent score/label join mismatches."""
    for column in ("receptor_id", "peptide", "hla"):
        if frame.schema[column] != pl.String:
            raise ValueError(f"{column} must be text; preserve identifiers when loading the table")
    frame = frame.with_columns(pl.col("receptor_id", "hla").str.strip_chars(),
                               pl.col("peptide").str.strip_chars().str.to_uppercase())
    if frame.filter((pl.col("receptor_id") == "") |
                    ~pl.col("peptide").str.contains(r"^[ACDEFGHIKLMNPQRSTVWY]+$")).height:
        raise ValueError("evaluation requires nonempty identities and standard amino-acid peptides")
    names = frame["hla"].unique().to_list()
    lookup = {name: normalize_hla(name) for name in names}
    if lookup:
        frame = frame.with_columns(pl.col("hla").replace_strict(lookup, return_dtype=pl.String))
    return frame


def evaluate_scores(scores: pl.DataFrame, labels: pl.DataFrame, *,
                    higher_is_better: bool = True) -> pl.DataFrame:
    """Evaluate one row per receptor/peptide/HLA; no cell-expansion weighting.

    Labels require receptor_id, peptide, hla, label (0/1/null). Conflicting labels
    for the same key are excluded and counted; they are never majority-voted.
    Missing scores stay in the denominator for coverage. AP uses threshold-wise
    ties, equivalent to non-interpolated average precision, not trapezoidal PR area.
    This is descriptive evaluation, not evidence of held-out generalization.
    """
    keys = ["receptor_id", "peptide", "hla"]
    for frame, required, name in ((scores, keys + ["score"], "scores"),
                                   (labels, keys + ["label"], "labels")):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if frame.select(pl.any_horizontal(pl.col(k).is_null() for k in keys).any()).item():
            raise ValueError(f"{name} contains missing identities")
    scores, labels = _normalize_keys(scores), _normalize_keys(labels)
    if scores.unique(keys).height != scores.height:
        raise ValueError("scores must have one row per receptor/peptide/HLA")
    for column in ("metric", "model"):
        if column in scores.columns and scores.height:
            if scores[column].null_count() or scores[column].n_unique() != 1:
                raise ValueError(f"scores must contain one non-null {column}; evaluate runs separately")
    if labels.filter(pl.col("label").is_not_null() & ~pl.col("label").is_in([0, 1])).height:
        raise ValueError("labels must be 0, 1 or null")
    scores = scores.with_columns(pl.col("score").cast(pl.Float64, strict=True))
    provenance = {column: scores[column][0] if column in scores.columns and scores.height else None
                  for column in ("model", "metric")}
    provenance["higher_is_better"] = higher_is_better
    grouped = labels.group_by(keys, maintain_order=True).agg(
        pl.col("label").drop_nulls().n_unique().alias("n_labels"),
        pl.col("label").drop_nulls().first().alias("label"))
    frame = grouped.join(scores.select(keys + ["score"]), on=keys, how="full",
                         coalesce=True, validate="1:1", maintain_order="left_right").with_columns(
        pl.col("n_labels").fill_null(0))
    output = []
    for (peptide, hla), group in frame.group_by("peptide", "hla", maintain_order=True):
        known = group.filter(pl.col("n_labels") == 1)
        usable = known.filter(pl.col("score").is_not_null() & pl.col("score").is_finite())
        y = usable["label"].to_numpy().astype(np.int64)
        score = usable["score"].to_numpy()
        if not higher_is_better:
            score = -score
        npos = int(y.sum())
        nneg = len(y) - npos
        ap = auc = None
        if npos and nneg:
            order = np.argsort(-score, kind="stable")
            ranked_y, ranked_score = y[order], score[order]
            ends = np.r_[np.flatnonzero(np.diff(ranked_score)) + 1, len(y)]
            tp = np.cumsum(ranked_y)[ends - 1]
            fp = ends - tp
            ap = float(np.sum(np.diff(np.r_[0, tp]) / npos * (tp / ends)))
            # ROC trapezoids at distinct score thresholds; ties contribute 0.5.
            auc = float(np.sum(np.diff(np.r_[0, fp]) / nneg *
                               (np.r_[0, tp[:-1]] + tp) / (2 * npos)))
        output.append({**provenance, "peptide": peptide, "hla": hla, "n_receptors": group.height,
                       "n_labelled": known.height, "n_scored": usable.height,
                       "n_conflicting_labels": group.filter(pl.col("n_labels") > 1).height,
                       "n_unknown_labels": group.filter(pl.col("n_labels") == 0).height,
                       "score_coverage": usable.height / known.height if known.height else None,
                       "positives": npos, "prevalence": npos / len(y) if len(y) else None,
                       "AUROC": auc, "average_precision": ap,
                       "status": "Evaluated" if npos and nneg else "Unresolved",
                       "reason": "descriptive clone-level ranking; training overlap not assessed"
                       if npos and nneg else "requires scored positive and negative labels"})
    return pl.DataFrame(output, schema=EVALUATION_SCHEMA)
