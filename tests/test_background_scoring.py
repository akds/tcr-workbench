"""Matched random comparators, provenance and panel-scoring contracts."""
import json
import io

import numpy as np
import polars as pl
import pytest

from tcr_workbench.background import AA, background_percentiles, random_panel
from tcr_workbench import decoder_pmhc, prediction
from tcr_workbench.tcr_scoring import score_tcr

COMPONENTS = dict(trav="TRAV21", traj="TRAJ6", cdr3a="CAVRPGGAGPFFVVF",
                  trbv="TRBV7-9", trbj="TRBJ2-7", cdr3b="CASSLGQAYEQYF", hla="HLA-A*02:01")
BACKEND = dict(decoder_dir="unused", python_executable="unused")


def test_sampling_matches_contexts_and_is_order_invariant():
    contexts = pl.DataFrame({"hla": ["HLA-A*02:01", "HLA-A*02:01", "HLA-B*07:02"],
                             "peptide_length": [8, 9, 9]})
    frame, meta = random_panel(contexts, peptides=1000, seed=7)
    assert frame.height == 3000 and frame["name"].n_unique() == 3000
    assert frame.equals(random_panel(contexts.reverse(), peptides=1000, seed=7)[0])
    assert not frame.equals(random_panel(contexts, peptides=1000, seed=8)[0])
    assert frame.select((pl.col("peptide").str.len_chars() == pl.col("peptide_length")).all()).item()
    letters = np.frombuffer("".join(frame["peptide"].to_list()).encode(), dtype="S1")
    counts = np.array([(letters == aa.encode()).sum() for aa in AA])
    assert np.all(np.abs(counts / len(letters) - .05) < .01)
    assert meta["alphabet"] == AA and meta["sampling"].startswith("Independent uniform")
    one = random_panel(contexts.head(1), peptides=1000, seed=7)[0]
    assert one.equals(frame.filter(pl.col("peptide_length") == 8))


def test_disabled_and_with_replacement_small_space():
    contexts = pl.DataFrame({"hla": ["H-2-Kb"], "peptide_length": [1]})
    zero, metadata = random_panel(contexts, peptides=0)
    assert zero.is_empty() and metadata["sampled_rows"] == 0
    sampled, _ = random_panel(contexts, peptides=100)
    assert sampled["peptide"].n_unique() < sampled.height
    assert set(sampled["peptide"].to_list()) <= set(AA)


@pytest.mark.parametrize("kwargs", [{"peptides":-1}, {"peptides":True}, {"seed":-1}, {"seed":2**32}])
def test_bad_sampler_options_rejected(kwargs):
    with pytest.raises(ValueError):
        random_panel(pl.DataFrame({"hla":["A"], "peptide_length":[9]}), **kwargs)


def test_large_background_fails_before_allocation():
    with pytest.raises(ValueError, match="Reduce --background-peptides"):
        random_panel(pl.DataFrame({"hla":["A", "B"], "peptide_length":[9,9]}), peptides=100, max_rows=100)


def test_pmhc_scores_background_together_but_keeps_tested_rows_separate(tmp_path, monkeypatch):
    captured = []
    def backend(source, output, options, **kwargs):
        data = pl.read_csv(source)
        captured.append(data)
        data.with_columns(pl.lit(True).alias("ok"), pl.lit("").alias("hla_reason"),
            pl.lit("").alias("inference_reason"),
            pl.lit(-2.0).alias("pll_" + prediction.DEFAULT_MODEL)).write_csv(output)
        return {"status":"Scored"}
    monkeypatch.setattr(decoder_pmhc, "_backend", backend)
    panel = pl.DataFrame({"peptide":["GILGFVFTL", "GILGFVFTL", "AAAAAAAA"], "hla":["HLA-A*02:01"]*3})
    results, record = decoder_pmhc.score_pmhc(panel, tmp_path/"out", background_peptides=13, **BACKEND)
    assert len(captured) == 1 and captured[0].height == 2 + 26
    assert results.height == 3
    background = pl.read_parquet(tmp_path/"out/background_scores.parquet")
    assert background.height == 26 and background["peptide_length"].unique().sort().to_list() == [8,9]
    assert record["background"]["scored_rows"] == 26
    assert "PLL distribution" in (tmp_path/"out/report.html").read_text()
    for name, value in record["outputs"].items():
        assert prediction.file_sha256(tmp_path/"out"/name) == value["sha256"]


def fake_tcr(source, output, **kwargs):
    data = pl.read_csv(source)
    assert set(data.columns) == {"name", *prediction.COMPONENTS}
    # A transparent test oracle, deliberately unrelated to a model benchmark.
    data.with_columns((-3.0 + pl.col("peptide").str.count_matches("A") / 10).alias(
        "pll_" + prediction.DEFAULT_MODEL)).write_csv(output)
    record = {"schema_version":1, "model":prediction.DEFAULT_MODEL, "input_sha256":prediction.file_sha256(source),
              "output_sha256":prediction.file_sha256(output), "species":kwargs.get("species", "human")}
    prediction._json_write(prediction._sidecar(output),record)
    return record


def test_tcr_panel_preserves_duplicates_and_separates_background(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "run_decoder", fake_tcr)
    panel = tmp_path/"panel.csv"
    panel.write_text("peptide\nAAAAAAAAA\nAAAAAAAAA\nGILGFVFTL\nAAAAAAAA\n")
    result, record = score_tcr(COMPONENTS, tmp_path/"out", panel=panel, background_peptides=17, **BACKEND)
    assert result.height == 4
    assert result["rank"].to_list() == [1,1,2,1]
    assert result["receptor_id"].n_unique() == 1
    assert pl.read_parquet(tmp_path/"out/background_scores.parquet").height == 34
    assert record["background"]["scored_rows"] == 34
    assert "Top 2 of 2 distinct" in (tmp_path/"out/report.html").read_text()
    for name, value in record["outputs"].items():
        assert prediction.file_sha256(tmp_path/"out"/name) == value["sha256"]


def test_tcr_panel_wrong_context_and_changed_file_fail(tmp_path, monkeypatch):
    panel = tmp_path/"panel.csv"
    panel.write_text("peptide,hla\nGILGFVFTL,HLA-B*07:02\n")
    with pytest.raises(ValueError, match="must match"):
        score_tcr(COMPONENTS, tmp_path/"wrong", panel=panel, **BACKEND)
    panel.write_text("peptide\nGILGFVFTL\n")
    def mutate(*args, **kwargs):
        record = fake_tcr(*args, **kwargs)
        panel.write_text("peptide\nAAAAAAAAA\n")
        return record
    monkeypatch.setattr(prediction, "run_decoder", mutate)
    with pytest.raises(ValueError, match="Input changed"):
        score_tcr(COMPONENTS, tmp_path/"mutated", panel=panel, background_peptides=0, **BACKEND)
    assert not (tmp_path/"mutated").exists()


def test_background_disabled_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "run_decoder", fake_tcr)
    score_tcr(COMPONENTS, tmp_path/"out", peptide="GILGFVFTL", background_peptides=0, **BACKEND)
    record = json.loads((tmp_path/"out/background_metadata.json").read_text())
    assert record["sampled_rows"] == 0
    assert "reference unavailable" in (tmp_path/"out/report.html").read_text()


def profile(length, **probabilities):
    return pl.DataFrame({"position": list(range(1, length + 1)),
                         **{aa: [probabilities.get(aa, 0.0)] * length for aa in AA}})


def test_profile_sampling_keeps_replacement_and_reuses_mhc_draws_across_receptors():
    contexts = pl.DataFrame({"hla": ["HLA-A*02:01"] * 3, "peptide_length": [8, 8, 9],
                             "receptor_id": ["r1", "r2", "r1"]})
    profiles = {("HLA-A*02:01", n): profile(n, A=.7, C=.3) for n in (8, 9)}
    sampled, meta = random_panel(contexts, peptides=1000, seed=31, mode="mhc-profile", profiles=profiles)
    assert sampled.equals(random_panel(contexts.reverse(), peptides=1000, seed=31,
                                       mode="mhc-profile", profiles=profiles)[0])
    first = sampled.filter((pl.col("receptor_id") == "r1") & (pl.col("peptide_length") == 8))
    other = sampled.filter(pl.col("receptor_id") == "r2")
    assert first["peptide"].equals(other["peptide"])
    assert first["peptide"].n_unique() < first.height and sampled["name"].n_unique() == 3000
    assert abs("".join(sampled["peptide"]).count("A") / 25000 - .7) < .015
    assert meta["with_replacement"] and meta["temperature"] == 1.0
    assert "both TCR chains absent" in meta["generator_context"]
    deterministic, _ = random_panel(contexts, peptides=23, mode="mhc-profile",
        profiles={key: profile(key[1], A=1.0) for key in profiles})
    assert deterministic["peptide"].unique().sort().to_list() == ["A" * 8, "A" * 9]
    assert deterministic.height == 69


def test_profile_sampling_missing_context_never_falls_back_to_uniform():
    contexts = pl.DataFrame({"hla": ["HLA-A*02:01", "HLA-B*07:02"], "peptide_length": [8, 9]})
    available = {("HLA-A*02:01", 8): profile(8, A=1.0)}
    with pytest.raises(ValueError, match="profile or explicit failure"):
        random_panel(contexts, mode="mhc-profile", profiles=available)
    sampled, meta = random_panel(contexts, peptides=11, mode="mhc-profile", profiles=available,
        profile_failures={("HLA-B*07:02", 9): "MHC sequence unavailable"})
    assert sampled.height == 11 and sampled["hla"].unique().to_list() == ["HLA-A*02:01"]
    assert meta["requested_rows"] == 22 and meta["sampled_rows"] == 11
    assert meta["failed_contexts"] == [{"hla": "HLA-B*07:02", "peptide_length": 9,
        "reason": "MHC sequence unavailable", "requested_peptides": 11}]
    empty, meta = random_panel(contexts, peptides=0, mode="mhc-profile")
    assert empty.is_empty() and meta["failed_contexts"] == []


@pytest.mark.parametrize("invalid", [
    profile(2, A=.9), profile(2, A=float("nan")), profile(2, A=-.1, C=1.1),
    profile(2, A=1.0).drop("W"), profile(2, A=1.0).with_columns(pl.lit(1).alias("position")),
    profile(2, A=1.0).with_columns(pl.col("A").cast(pl.String)),
])
def test_profile_sampling_rejects_corrupt_probabilities(invalid):
    with pytest.raises(ValueError, match="Background profile"):
        random_panel(pl.DataFrame({"hla": ["HLA-A*02:01"], "peptide_length": [2]}),
                     mode="mhc-profile", profiles={("HLA-A*02:01", 2): invalid})


def test_percentile_includes_ties_preserves_order_and_ignores_other_contexts():
    tested = pl.DataFrame({"receptor_id": ["r1"] * 5, "hla": ["HLA-A*02:01"] * 5,
        "peptide": ["AA", "CC", "DD", "EE", "FF"], "score": [-1., -2., -3., -5., None],
        "status": ["ModelHypothesis"] * 4 + ["Unresolved"], "rank": [1, 2, 3, 4, None]})
    background = pl.DataFrame({"receptor_id": ["r1"] * 5 + ["r2"], "hla": ["HLA-A*02:01"] * 6,
        "peptide": ["AA", "AA", "CC", "CCC", "EE", "AA"],
        "score": [-2., -2., -4., -100., float("nan"), -999.], "status": ["Scored"] * 6})
    result = background_percentiles(tested, background)
    assert result.select(tested.columns).equals(tested)
    assert result["background_percentile"].to_list() == [25., 75., 75., 100., None]
    assert result["background_n"].to_list() == [3, 3, 3, 3, 3]
    with pytest.raises(ValueError, match="missing required comparison context"):
        background_percentiles(tested, background.drop("receptor_id"))


def fake_profile_and_score(captured, *, fail_length=None, checkpoint_change=False):
    def run(source, output, options, *, profile=False):
        data = pl.read_csv(source)
        captured.append((profile, data, options))
        assert data.columns == ["name", "hla", "peptide"] or set(data.columns) == {"name", "hla", "peptide"}
        identity = {"model": options.model, "species": options.species,
                    "checkpoint_sha256": "a" * 64, "precision": options.precision, "device": options.device}
        if profile == "batch":
            from tcr_workbench.backends.profile_batch_worker import write_profiles
            rows = []
            for row in data.iter_rows(named=True):
                length = len(row["peptide"])
                assert row["peptide"] == "A" * length
                rows.append({**row, "ok": "false" if length == fail_length else "true",
                             "hla_reason": "MHC profile unavailable" if length == fail_length else ""})
            with output.open("w") as target:
                counts = write_profiles(rows, target, lambda row: np.tile([1.0] + [0.0] * 19, (len(row["peptide"]), 1)))
            return {**identity, **counts, "mode": "profiles"}
        assert profile is False
        data.with_columns(pl.lit(True).alias("ok"), pl.lit("").alias("hla_reason"),
            pl.lit("").alias("inference_reason"),
            (-3.0 + pl.col("peptide").str.count_matches("A") / 10).alias("pll_" + options.model)).write_csv(output)
        return {**identity, "checkpoint_sha256": "b" * 64 if checkpoint_change else "a" * 64,
                "status": "Scored"}
    return run


def test_pmhc_profile_reference_preserves_scores_and_records_every_generator(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(decoder_pmhc, "_backend", fake_profile_and_score(captured))
    panel = pl.DataFrame({"peptide": ["C" * 8, "A" * 8, "A" * 8, "C" * 9], "hla": ["HLA-A*02:01"] * 4})
    baseline, _ = decoder_pmhc.score_pmhc(panel, tmp_path/"plain", background_peptides=0, **BACKEND)
    captured.clear()
    result, run = decoder_pmhc.score_pmhc(panel, tmp_path/"sampled", background_mode="mhc-profile",
                                        background_peptides=19, background_seed=9, **BACKEND)
    assert result.drop("background_percentile", "background_n").equals(baseline.drop("background_percentile", "background_n"))
    assert sum(is_profile == "batch" for is_profile, _, _ in captured) == 1
    assert captured[0][1].height == 2
    assert captured[-1][0] is False and captured[-1][1].height == 3 + 38
    assert result["background_n"].to_list() == [19] * 4
    assert result["background_percentile"].to_list() == [100.] * 4
    metadata = run["background"]
    assert metadata["mode"] == "mhc-profile" and len(metadata["profile_runs"]) == 2
    for item in metadata["profile_runs"]:
        assert prediction.file_sha256(tmp_path/"sampled"/item["profile_file"]) == item["profile_sha256"]
        assert prediction.file_sha256(tmp_path/"sampled"/item["input_file"]) == item["input_sha256"]
    for name, record in run["outputs"].items():
        assert prediction.file_sha256(tmp_path/"sampled"/name) == record["sha256"]


def test_failed_mhc_profile_preserves_tested_results_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(decoder_pmhc, "_backend", fake_profile_and_score([], fail_length=9))
    panel = pl.DataFrame({"peptide": ["C" * 8, "C" * 9], "hla": ["HLA-A*02:01"] * 2})
    result, run = decoder_pmhc.score_pmhc(panel, tmp_path/"out", background_mode="mhc-profile",
                                        background_peptides=7, **BACKEND)
    assert result["status"].to_list() == ["Scored", "Scored"]
    assert result["background_n"].to_list() == [7, 0]
    assert result["background_percentile"].to_list() == [100., None]
    assert run["background"]["sampled_rows"] == 7 and run["background"]["requested_rows"] == 14
    assert run["background"]["failed_contexts"][0]["reason"] == "MHC profile unavailable"
    assert pl.read_parquet(tmp_path/"out/background_scores.parquet")["peptide_length"].unique().to_list() == [8]


def test_tcr_reference_generator_omits_tcr_but_scoring_uses_query(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(decoder_pmhc, "_backend", fake_profile_and_score(captured))
    scores = []
    def run_tcr(source, output, **kwargs):
        data = pl.read_csv(source)
        scores.append(data)
        for key, value in COMPONENTS.items():
            assert data[key].unique().to_list() == [value]
        return {**fake_tcr(source, output, **kwargs), "checkpoint_sha256": "a" * 64,
                "precision": "float32", "device": "cpu"}
    monkeypatch.setattr(prediction, "run_decoder", run_tcr)
    panel = tmp_path/"panel.csv"
    pl.DataFrame({"peptide": ["C" * 8, "A" * 8, "A" * 8, "C" * 9]}).write_csv(panel)
    baseline, _ = score_tcr(COMPONENTS, tmp_path/"plain", panel=panel, background_peptides=0, **BACKEND)
    result, run = score_tcr(COMPONENTS, tmp_path/"sampled", panel=panel, background_mode="mhc-profile",
                            background_peptides=13, **BACKEND)
    assert len(captured) == 1 and captured[0][0] == "batch" and captured[0][1].height == 2
    assert scores[-1].height == 4 + 26
    assert result.drop("background_percentile", "background_n").equals(baseline.drop("background_percentile", "background_n"))
    assert run["background"]["scored_rows"] == 26


def test_changed_generator_checkpoint_rejects_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(decoder_pmhc, "_backend", fake_profile_and_score([], checkpoint_change=True))
    with pytest.raises(ValueError, match="differ in checkpoint_sha256"):
        decoder_pmhc.score_pmhc(pl.DataFrame({"peptide": ["AA"], "hla": ["HLA-A*02:01"]}),
            tmp_path/"out", background_mode="mhc-profile", background_peptides=3, **BACKEND)
    assert not (tmp_path/"out").exists()


def test_profile_mode_disabled_and_limit_precede_inference(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(decoder_pmhc, "_backend", fake_profile_and_score(calls))
    panel = pl.DataFrame({"peptide": ["AA", "AAA"], "hla": ["HLA-A*02:01"] * 2})
    result, run = decoder_pmhc.score_pmhc(panel, tmp_path/"disabled", background_mode="mhc-profile",
                                        background_peptides=0, **BACKEND)
    assert len(calls) == 1 and not calls[0][0]
    assert run["background"]["profile_runs"] == [] and result["background_percentile"].null_count() == 2
    calls.clear()
    with pytest.raises(ValueError, match="limit 100,000"):
        decoder_pmhc.score_pmhc(panel, tmp_path/"too-large", background_mode="mhc-profile",
                                background_peptides=100000, **BACKEND)
    assert calls == [] and not (tmp_path/"too-large").exists()


def batch_fixture(tmp_path):
    from tcr_workbench.backends.profile_batch_worker import write_profiles
    rows = [dict(name="one", hla="HLA-A*02:01", peptide="AA", ok="true"),
            dict(name="two", hla="HLA-B*07:02", peptide="AAA", ok="false", hla_reason="missing reference")]
    calls = []
    def predict(row):
        calls.append(row["name"])
        return [[.05] * 20] * len(row["peptide"])
    output = tmp_path/"profiles.csv"
    with output.open("w") as target:
        counts = write_profiles(rows, target, predict)
    assert calls == ["one"]
    return pl.DataFrame(rows).select("name", "hla", "peptide"), output, counts


def test_profile_batch_stream_preserves_unresolved_coverage(tmp_path):
    from tcr_workbench.backends.pmhc_decoder import validate_input, validate_output
    source, output, counts = batch_fixture(tmp_path)
    assert counts == {"rows": 2, "scored": 1, "unresolved": 1}
    source.write_csv(tmp_path/"input.csv")
    assert validate_input(tmp_path/"input.csv", profile="batch").equals(source)
    validate_output(source, output, prediction.DEFAULT_MODEL, counts, profile="batch")
    frame = pl.read_csv(output)
    assert frame.height == 3 and frame.filter(pl.col("name") == "two")["reason"].item() == "missing reference"


@pytest.mark.parametrize("mutation", ["missing_context", "wrong_mhc", "wrong_length", "duplicate_position",
                                      "unknown_status", "invalid_probability", "failure_with_probability",
                                      "failure_without_reason", "missing_column", "wrong_count"])
def test_profile_batch_validation_rejects_corrupt_coverage(tmp_path, mutation):
    from tcr_workbench.backends.pmhc_decoder import validate_output
    source, output, counts = batch_fixture(tmp_path)
    frame = pl.read_csv(output)
    if mutation == "missing_context":
        frame = frame.filter(pl.col("name") == "one")
    elif mutation == "wrong_mhc":
        frame = frame.with_columns(pl.lit("HLA-C*07:02").alias("hla"))
    elif mutation == "wrong_length":
        frame = frame.with_columns(pl.lit(9).alias("peptide_length"))
    elif mutation == "duplicate_position":
        frame = frame.with_columns(pl.when(pl.col("name") == "one").then(1).otherwise(None).alias("position"))
    elif mutation == "unknown_status":
        frame = frame.with_columns(pl.lit("Scored").alias("status"))
    elif mutation == "invalid_probability":
        frame = frame.with_columns(pl.when(pl.col("name") == "one").then(float("nan")).otherwise(None).alias("A"))
    elif mutation == "failure_with_probability":
        frame = frame.with_columns(pl.col("A").fill_null(.05))
    elif mutation == "failure_without_reason":
        frame = frame.with_columns(pl.lit("").alias("reason"))
    elif mutation == "missing_column":
        frame = frame.drop("Y")
    elif mutation == "wrong_count":
        counts["scored"] = 2
    frame.write_csv(output)
    with pytest.raises(ValueError):
        validate_output(source, output, prediction.DEFAULT_MODEL, counts, profile="batch")


@pytest.mark.parametrize("bad", [[], [[.05] * 19], [[float("nan")] + [.05] * 19], [[.1] * 20]])
def test_profile_batch_worker_rejects_invalid_model_probabilities(bad):
    from tcr_workbench.backends.profile_batch_worker import write_profiles
    with pytest.raises(ValueError, match="Profile output"):
        write_profiles([dict(name="one", hla="HLA-A*02:01", peptide="A", ok="true")],
                       io.StringIO(), lambda row: bad)


def test_profile_batch_worker_rejects_tcr_conditioning():
    from tcr_workbench.backends.profile_batch_worker import write_profiles
    with pytest.raises(ValueError, match="must not contain TCR"):
        write_profiles([dict(name="one", hla="HLA-A*02:01", peptide="A", ok="true", TCR_a="CAAF")],
                       io.StringIO(), lambda row: [[.05] * 20])
