"""Offline report safety, truthful empty states and workflow publication checks."""
from html.parser import HTMLParser
import json
from pathlib import Path

import polars as pl
import pytest

from tcr_workbench import decoder_pmhc, prediction
from tcr_workbench.cli import main
from tcr_workbench.report import write_workflow_report


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        self.links.extend(value for key, value in attrs if key in {"href", "src"})


def assert_offline_safe(page):
    document = Document(page)
    assert not any(tag in {"script", "iframe", "object", "embed"} for tag, _ in document.tags)
    assert not any(key.lower().startswith("on") for _, attrs in document.tags for key in attrs)
    assert not any(link.startswith(("http:", "https:", "//", "javascript:", "data:")) for link in document.links)
    return document


def test_all_dynamic_fields_are_escaped_and_preview_is_bounded(tmp_path):
    payload = '<script>alert("unsafe")</script><img src=x onerror=alert(1)>'
    frame = pl.DataFrame({"status": [payload] * 201, "reason": [payload] * 200 + ["LAST_ROW_HIDDEN"]})
    (tmp_path / "results.csv").write_text("full data")
    write_workflow_report(tmp_path, payload, table=frame, summary={payload: payload},
                          metadata={payload: payload}, context={payload: payload},
                          interpretation=payload, reason=payload)
    page = (tmp_path / "report.html").read_text()
    document = assert_offline_safe(page)
    assert "&lt;script&gt;" in page and "LAST_ROW_HIDDEN" not in page
    assert "Showing 200 of 201 rows" in page
    assert "results.csv" in document.links
    assert "results.parquet" not in document.links and "manifest.json" not in document.links
    assert sum(tag == "tr" for tag, _ in document.tags) == 201  # heading plus 200 rows


def test_empty_profile_retains_reason_and_does_not_invent_probabilities(tmp_path):
    profile = pl.DataFrame(schema={"position": pl.Int64, **{aa: pl.Float64 for aa in prediction.AA}})
    write_workflow_report(tmp_path, "Empty profile", table=pl.DataFrame(), profile=profile,
                          summary={"status": "Unresolved", "profile_positions": 0},
                          interpretation="Unresolved does not mean non-binding.", reason="Missing HLA partner")
    page = (tmp_path / "report.html").read_text()
    document = assert_offline_safe(page)
    assert "Missing HLA partner" in page and "No amino-acid profile was produced" in page
    assert not any(attrs.get("class") == "heatmap" for _, attrs in document.tags)
    assert 'class="card card-warn"' in page


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.2, 1.5, None])
def test_invalid_profile_values_have_explicit_fallback(tmp_path, value):
    profile = pl.DataFrame({"position": [1], **{aa: [value if aa == "A" else 0.05] for aa in prediction.AA}})
    write_workflow_report(tmp_path, "Profile", profile=profile, interpretation="Preferences only")
    page = (tmp_path / "report.html").read_text()
    assert "heatmap is unavailable" in page
    assert '<table class="heatmap">' not in page


def test_profile_heatmap_keeps_units_anchors_and_zero_values(tmp_path):
    profile = pl.DataFrame({"position": [1, 2], **{aa: [1.0 if aa == "A" else 0.0, 0.05]
                                                     for aa in prediction.AA}})
    write_workflow_report(tmp_path, "Profile", profile=profile,
                          interpretation="Conditional model preferences")
    page = (tmp_path / "report.html").read_text()
    document = assert_offline_safe(page)
    assert "one-based" in page and "20 canonical amino acids" in page
    assert "100.0%" in page and "0.0%" in page and "5.0%" in page
    assert "0.05" in page and "10⁻¹²" in page and "floor applies only to log-odds" in page
    assert "No class II binding-core register is inferred" in page
    assert sum(tag == "td" for tag, _ in document.tags) == 40


def fake_score(source, output, **kwargs):
    frame = pl.read_csv(source).with_columns(pl.lit(-2.0).alias("pll_" + prediction.DEFAULT_MODEL))
    frame.write_csv(output)
    run = {"schema_version": 1, "model": prediction.DEFAULT_MODEL, "device": "cpu", "precision": "float32",
           "input_sha256": prediction.file_sha256(source), "output_sha256": prediction.file_sha256(output)}
    prediction._json_write(prediction._sidecar(output), run)
    return run


def fake_profile(components, output, **kwargs):
    length = kwargs["length"]
    pl.DataFrame({"position": range(1, length + 1), **{aa: [0.05] * length for aa in prediction.AA}}).write_csv(output)
    return {"model": prediction.DEFAULT_MODEL, "device": "cpu", "precision": "float32", "status": "Profiled"}


def fake_pmhc(source, output, options, *, profile=False):
    frame = pl.read_csv(source)
    if profile:
        return fake_profile({}, output, length=len(frame["peptide"][0]))
    frame.with_columns(pl.lit(True).alias("ok"), pl.lit("").alias("hla_reason"),
                      pl.lit("").alias("inference_reason"),
                      pl.lit(-2.0).alias("pll_" + prediction.DEFAULT_MODEL)).write_csv(output)
    return {"model": prediction.DEFAULT_MODEL, "device": "cpu", "precision": "float32", "status": "Scored"}


@pytest.mark.parametrize("command", ["pmhc-score", "pmhc-profile", "tcr-score", "tcr-profile", "repertoire-score", "screen"])
def test_all_workflow_reports_are_offline_and_in_manifest(tmp_path, monkeypatch, command):
    monkeypatch.setattr(prediction, "run_decoder", fake_score)
    monkeypatch.setattr(prediction, "run_decoder_profile", fake_profile)
    monkeypatch.setattr(decoder_pmhc, "_backend", fake_pmhc)
    examples = Path(__file__).resolve().parents[1] / "examples"
    out = tmp_path / command
    flags = [command, "--out", str(out)]
    if command in {"pmhc-score", "repertoire-score", "screen"}:
        flags += ["--panel", str(examples / "panel.csv")]
    if command in {"repertoire-score", "screen"}:
        flags += ["--input", str(examples / "paired.csv"), "--format", "paired", "--donors", str(examples / "donors.csv")]
    if command == "screen":
        flags += ["--reference", str(examples / "reference_synthetic.csv")]
    else:
        flags += ["--decoder-dir", str(tmp_path), "--python", "unused"]
    if command in {"tcr-score", "tcr-profile"}:
        flags += ["--trav", "TRAV21", "--traj", "TRAJ6", "--cdr3a", "CAVRPGGAGPFFVVF",
                  "--trbv", "TRBV7-9", "--trbj", "TRBJ2-7", "--cdr3b", "CASSLGQAYEQYF"]
    if command in {"pmhc-profile", "tcr-profile", "tcr-score"}:
        flags += ["--hla", "HLA-A*02:01"]
        flags += ["--peptide", "GILGFVFTL"] if command == "tcr-score" else ["--length", "9"]
    assert main(flags) == 0
    page = (out / "report.html").read_text()
    document = assert_offline_safe(page)
    run = json.loads((out / "manifest.json").read_text())
    assert run["outputs"]["report.html"]["sha256"] == prediction.file_sha256(out / "report.html")
    for link in document.links:
        assert link.startswith("#") or (out / link).is_file(), link
    if command != "screen":
        assert prediction.DEFAULT_MODEL in page and "float32" in page and "cpu" in page
    if command.endswith("profile"):
        assert '<table class="heatmap">' in page and "pssm.csv" in document.links
    if command == "repertoire-score":
        assert "Unresolved" in page and "same receptor, MHC and peptide length" in page
