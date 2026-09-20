"""Quantitative, bounded, offline motif and matched score report behavior."""
from html.parser import HTMLParser
import math
import re

import polars as pl
import pytest

from tcr_workbench.report import _AA, _histogram, _profile_logo, _score_preview, write_workflow_report


class Elements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def profile(*distributions):
    return pl.DataFrame({"position": range(1, len(distributions) + 1),
                         **{aa: [row.get(aa, 0.0) for row in distributions] for aa in _AA}})


def test_logo_heights_encode_information_not_signed_pssm():
    result = _profile_logo(profile({"A": 1.0}, {"C": 0.5, "W": 0.5}, {aa: 0.05 for aa in _AA}))
    elements = Elements(result).elements
    letters = [attrs for tag, attrs in elements if tag == "g" and "aria-label" in attrs]
    assert len(letters) == 3  # uniform position has zero information
    assert "4.322 bits" in letters[0]["aria-label"]
    assert all("1.661 bits" in letter["aria-label"] for letter in letters[1:])
    paths = [attrs for tag, attrs in elements if tag == "path" and "transform" in attrs]
    heights = [float(re.search(r"scale\(32 ([\d.]+)\)", path["transform"])[1]) for path in paths]
    assert heights[0] == pytest.approx(math.log2(20) * 46, abs=1e-5)
    assert sum(heights[1:]) == pytest.approx((math.log2(20) - 1) * 46, abs=1e-5)
    assert 'class="consensus">AXX</code>' in result
    assert "Letter heights are not signed PSSM scores" in result
    assert "blank or tiny stack" in result


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1, 0.25])
def test_logo_does_not_display_invalid_or_unnormalized_profile(probability):
    result = _profile_logo(profile({"A": probability}))
    assert "motif is unavailable" in result
    assert "<svg" not in result


def test_logo_is_bounded_and_position_labels_are_escaped():
    frame = profile(*[{"A": 1.0}] * 201).with_columns(pl.col("position").cast(pl.String))
    frame[0, "position"] = '<img src=x onerror="alert(1)">'
    result = _profile_logo(frame)
    assert "Showing the first 200 of 201 positions" in result
    assert result.count('aria-label="Information-content') == 8
    assert '<img src=' not in result and '&lt;img src=' in result
    assert '>201</text>' not in result


def test_score_top10_distinct_ranked_and_partitioned_by_context_length():
    tested = pl.DataFrame({"receptor_id": ["r1"] * 15 + ["r2", "r1", "r1"],
                           "hla": ["HLA-A*02:01"] * 18,
                           "peptide": ["A" * 8 + aa for aa in _AA[:15]] + ["A" * 9, "A" * 8, "A" * 8 + _AA[0]],
                           "score": [float(i - 15) for i in range(15)] + [-0.1, -0.1, -15.0],
                           "status": ["Scored"] * 18, "reason": [""] * 18})
    result, columns = _score_preview(tested, None, None)
    assert "Top 10 of 15 distinct" in result
    assert result.count("Top 1 of 1 distinct") == 2
    assert "AAAAAAAAA</td>" in result  # r2 retained, not pooled with r1
    assert "AAAAAAAAC</td>" not in result  # low-scoring r1 peptide omitted
    assert "AAAAAAAAQ</td>" in result  # ranked above cutoff
    assert "rank" in columns and "peptide_length" in columns
    assert result.count("Random-peptide reference unavailable") == 3


def test_score_ties_share_minimum_rank_and_unresolved_reason_survives():
    tested = pl.DataFrame({"hla": ["H-2-Kb"] * 4, "peptide": ["AAAAAAAA", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE"],
                           "score": [-1.0, -1.0, -2.0, None],
                           "status": ["Scored", "Scored", "Scored", "Unresolved"],
                           "reason": ["", "", "", '<script>missing context</script>']})
    result, _ = _score_preview(tested, None, None)
    assert result.count('<td class="nowrap">1</td><td><span class="status status-good">') == 2
    assert '<td class="nowrap">3</td><td><span class="status status-good">' in result
    assert "1 input rows were not scored" in result
    assert "&lt;script&gt;missing context&lt;/script&gt;" in result
    assert "<script>" not in result


def test_distribution_only_uses_finite_scored_same_context_length():
    tested = pl.DataFrame({"receptor_id": ["r1"], "hla": ["H-2-Kb"], "peptide": ["AAAAAAAA"],
                           "score": [-1.0], "status": ["Scored"]})
    background = pl.DataFrame({"receptor_id": ["r1", "r2", "r1", "r1", "r1"],
                              "hla": ["H-2-Kb", "H-2-Kb", "H-2-Kb", "H-2-Db", "H-2-Kb"],
                              "peptide": ["CCCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCC", "DDDDDDDD"],
                              "score": [-2., -999., -999., -999., float("nan")], "status": ["Scored"] * 5})
    result, _ = _score_preview(tested, None, background)
    assert "Random reference · n = 1" in result
    assert "-999" not in result and "nan" not in result
    assert "Random-peptide reference unavailable" not in result
    assert "Fraction of each set" in result


def test_missing_receptor_in_background_is_not_silently_pooled():
    tested = pl.DataFrame({"receptor_id": ["r1"], "hla": ["H-2-Kb"], "peptide": ["AAAAAAAA"],
                           "score": [-1.0], "status": ["Scored"]})
    result, _ = _score_preview(tested, None, tested.drop("receptor_id"))
    assert "Random-peptide reference unavailable" in result
    assert "<svg" not in result


def test_histogram_identical_scores_and_relative_frequencies():
    tested = pl.DataFrame({"score": [-2.0]})
    background = pl.DataFrame({"score": [-2.0] * 100})
    result = _histogram(tested, background)
    rectangles = [attrs for tag, attrs in Elements(result).elements if tag == "rect"]
    positive = [attrs for attrs in rectangles if float(attrs["height"]) > 0]
    assert len(positive) == 2
    assert positive[0]["height"] == positive[1]["height"]
    assert "Random reference · n = 100" in result and "Tested peptides · n = 1" in result
    assert "nan" not in result and "inf" not in result


def test_more_than_twenty_groups_is_bounded_and_explicit():
    tested = pl.DataFrame({"receptor_id": [f"r{i}" for i in range(21)], "hla": ["H-2-Kb"] * 21,
                           "peptide": ["AAAAAAAA"] * 21, "score": [-1.] * 21, "status": ["Scored"] * 21})
    result, _ = _score_preview(tested, None, None)
    assert "Showing 20 of 21 comparison groups" in result
    assert "receptor id: r20" not in result


@pytest.mark.parametrize("workflow", ["pmhc-profile", "tcr-profile"])
def test_profiles_replace_preview_with_motif_and_keep_heatmap_guide(tmp_path, workflow):
    write_workflow_report(tmp_path, "Profile", table=pl.DataFrame({"position": [1]}),
                          profile=profile({"A": 1.0}), workflow=workflow,
                          interpretation="Research preferences", context={"Species": "mouse", "MHC": "H-2-Kb"})
    page = (tmp_path / "report.html").read_text()
    assert 'id="profile"' in page and 'class="heatmap"' in page
    assert "Peptide preference motif" in page and "Result preview" not in page
    assert "Profile definitions" in page and "peptide masked simultaneously" in page
    assert "mouse" in page and "H-2-Kb" in page


def test_repertoire_has_explicit_glossary_for_all_eight_columns(tmp_path):
    frame = pl.DataFrame({"receptor_id": ["r1"], "hla": ["H-2-Kb"], "peptide": ["AAAAAAAA"],
                          "peptide_length": [8], "score": [-2.], "rank": [1],
                          "status": ["Scored"], "reason": [""]})
    write_workflow_report(tmp_path, "Repertoire", table=frame, workflow="repertoire-score", interpretation="Hypotheses")
    page = (tmp_path / "report.html").read_text()
    for column in frame.columns:
        assert f"<code>{column}</code>" in page
    assert "Human molecules use HLA names" in page and "average the natural log" in page
    assert "mask all peptide residues simultaneously" in page
    assert "mask each peptide residue in turn" not in page
    assert "probability of binding" in page and "Unresolved" in page


def test_failed_random_draws_are_visible_and_not_filled_in():
    tested = pl.DataFrame({"hla": ["H-2-Kb"], "peptide": ["AAAAAAAA"], "score": [-1.], "status": ["Scored"]})
    background = pl.DataFrame({"hla": ["H-2-Kb"] * 3, "peptide": ["AAAAAAAA"] * 3,
                              "score": [-2., None, float("inf")], "status": ["Scored", "Unresolved", "Scored"]})
    result, _ = _score_preview(tested, None, background)
    assert "2 random-reference rows in this group were unresolved" in result
    assert "Random reference · n = 1" in result
    result, _ = _score_preview(tested, None, background.tail(2))
    assert "2 random-reference rows in this group were unresolved" in result
    assert "Random-peptide reference unavailable" in result


def test_repertoire_guide_does_not_claim_missing_distribution(tmp_path):
    write_workflow_report(tmp_path, "Repertoire", table=pl.DataFrame(), workflow="repertoire-score", interpretation="Hypotheses")
    page = (tmp_path / "report.html").read_text()
    assert "Receptor and cell mapping" in page
    assert "The distribution compares" not in page



def test_real_tcr_model_hypothesis_status_is_scored_and_not_unresolved():
    tested = pl.DataFrame({"hla": ["H-2-Kb"], "receptor_id": ["r1"], "peptide": ["AAAAAAAA"],
                           "score": [-1.], "status": ["ModelHypothesis"], "reason": [""]})
    result, _ = _score_preview(tested, None, tested.with_columns(pl.lit(-2.).alias("score")))
    assert "Top 1 of 1 distinct successfully scored" in result
    assert 'class="status status-good">ModelHypothesis' in result
    assert "Random reference · n = 1" in result
    assert "No finite scored peptide" not in result
    assert "input rows were not scored" not in result
    assert "excluded from the distribution" not in result
def test_profile_reports_selected_genes_and_upstream_assumptions(tmp_path):
    from tcr_workbench.report import profile_reconstruction_context, write_workflow_report
    requested = dict(trav="TRAV21", traj="TRAJ6", trbv="TRBV12", trbj="TRBJ2-7")
    context = profile_reconstruction_context(requested, {"reconstruction": {
        "TRAV":"TRAV21", "TRAJ":"TRAJ6", "TRBV":"TRBV12-3", "TRBJ":"TRBJ2-7",
        "tcr_reason":"<unsafe> upstream message"}})
    assert context["Reconstructed TRBV (reported)"] == "TRBV12-3"
    assert "upstream selected TRBV=TRBV12-3 for input trbv=TRBV12" in context["Gene reconstruction check"]
    assert "default IMGT alleles" in context["Gene reconstruction check"]
    write_workflow_report(tmp_path,"Profile",context=context,interpretation="Model preferences")
    assert "&lt;unsafe&gt;" in (tmp_path/"report.html").read_text()
    assert "<unsafe>" not in (tmp_path/"report.html").read_text()


def test_profile_missing_gene_report_never_invents_allele():
    from tcr_workbench.report import profile_reconstruction_context
    genes = dict(trav="TRAV21*01", traj="TRAJ6*01", trbv="TRBV7-9*01", trbj="TRBJ2-7*01")
    context = profile_reconstruction_context(genes,{})
    assert context["Reconstructed TRAV (reported)"] == "Not reported"
    assert "not reported" in context["Gene reconstruction check"]
    exact = profile_reconstruction_context(genes,{"reconstruction":{k.upper():v for k,v in genes.items()}})
    assert exact["Gene reconstruction check"] == "Reported V/J genes match the supplied calls."
