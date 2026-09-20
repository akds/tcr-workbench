"""HLA-only and explicit class-II backend contracts without model downloads."""
import csv

import numpy as np
import polars as pl
import pytest

from tcr_workbench import prediction as p
from tcr_workbench.backends import mlx_worker as w
from tcr_workbench.backends import pmhc_decoder as backend
from tcr_workbench.backends.reconstruct_worker import exact_hla_key, reconstruct_pmhc
from tcr_workbench.backends.torch_cache import OutputCache


@pytest.mark.parametrize("hla,key", [
    ("HLA-A*02:01", "A0201__B2M"), ("HLA-A*02:256", "A02256__B2M"),
    ("HLA-DRA*01:01/HLA-DRB1*04:01", "DRA0101__DRB10401"),
    ("HLA-DQA1*03:01/HLA-DQB1*02:01", "DQA10301__DQB10201"),
    ("HLA-DPA1*01:03/HLA-DPB1*04:01", "DPA10103__DPB10401"),
])
def test_exact_compact_lookup_retains_complete_molecule(hla, key):
    assert exact_hla_key(hla) == key


@pytest.mark.parametrize("hla", ["HLA-DRB1*04:01", "HLA-A*102:56", "HLA-A*02:01:01",
    "HLA-A*02:01N", "HLA-DQA1*01:01/HLA-DRB1*07:01", "HLA-DQA1*03:01", "HLA-E*01:03"])
def test_lookup_never_guesses_missing_chains_resolution_or_class(hla):
    with pytest.raises(KeyError):
        exact_hla_key(hla)


def test_pmhc_reconstruction_uses_exact_pairs_and_audits_unknowns(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "out.csv"
    pl.DataFrame({"name": ["001", "002", "003"], "peptide": ["AC", "AA", "CA"],
        "hla": ["HLA-DQA1*03:01/HLA-DQB1*02:01", "HLA-DQA1*03:01/HLA-DQB1*02:02",
                "HLA-A*02:01:01"]}).write_csv(source)
    reconstruct_pmhc(source, output, {"DQA10301__DQB10201": {"HLA_a": "AC", "HLA_b": "DE"}})
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["HLA_a"] == "AC" and rows[0]["HLA_b"] == "DE"
    assert rows[1]["ok"] == rows[2]["ok"] == "False"
    assert "absent" in rows[1]["hla_reason"]
    assert "two-field" in rows[2]["hla_reason"]
    assert "TCR_a" not in rows[0]


def test_hla_only_context_is_explicit_and_cannot_accidentally_condition_on_tcr():
    row = {"HLA_a": "AC", "HLA_b": "DE", "peptide": "FG"}
    key = w.context_key(row, "pmhc")
    assert key == ("AC", "DE", "", "", 2)

    class Tokenizer:
        def encode(self, sequence, **kwargs):
            return [0] + [5] * len(sequence) + [2]

    tokens, positions = w.masked_tokens(key, Tokenizer(), 2048, "pmhc")
    assert len(tokens) == 8 and positions.tolist() == [5, 6]
    with pytest.raises(ValueError, match="full chain is empty"):
        w.masked_tokens(key, Tokenizer(), 2048, "tcr-pmhc")
    with pytest.raises(ValueError, match="must not contain"):
        w.context_key({**row, "TCR_a": "AC"}, "pmhc")


def test_paired_adapter_accepts_explicit_classII_only():
    for hla in ("HLA-DRA*01:01/HLA-DRB1*04:01", "HLA-DQA1*03:01/HLA-DQB1*02:01",
                "HLA-DPA1*01:03/HLA-DPB1*04:01"):
        assert p._Pair.supported_hla(hla) == hla
    for hla in ("HLA-DRB1*04:01", "HLA-DQA1*03:01", "HLA-DPA1*01:03"):
        with pytest.raises(ValueError):
            p._Pair.supported_hla(hla)


def test_paired_compact_alias_collision_is_retained_in_skipped_audit(tmp_path):
    receptor = pl.DataFrame([dict(receptor_id="r1", donor_id="d1", pairing_status="paired",
        trav="TRAV21", traj="TRAJ6", cdr3a="CAVRPGGAGPFFVVF", trbv="TRBV7-9",
        trbj="TRBJ2-7", cdr3b="CASSLGQAYEQYF")])
    panel = pl.DataFrame({"peptide": ["AC", "AC", "AC"],
                          "hla": ["HLA-A*02:256", "HLA-A*022:56", "HLA-A*02:01:01"]})
    target = tmp_path / "input.csv"
    p.export_decoder_input(receptor, panel, target)
    eligible = pl.read_csv(target)
    skipped = pl.read_csv(str(target) + ".skipped.csv")
    assert eligible["hla"].to_list() == ["HLA-A*02:256"]
    assert skipped["hla"].to_list() == ["HLA-A*022:56", "HLA-A*02:01:01"]
    assert skipped["reason"].str.contains("two-field").all()


def test_pmhc_lowlevel_atomic_output_and_cache(tmp_path, monkeypatch):
    source, output = tmp_path / "input.csv", tmp_path / "out.csv"
    pl.DataFrame({"name": ["001", "002"], "peptide": ["AC", "CA"],
                  "hla": ["HLA-A*02:01", "HLA-A*99:99"]}).write_csv(source)
    monkeypatch.setattr(p, "_model_fingerprint", lambda *a, **k: dict(model=p.DEFAULT_MODEL,
        checkpoint_sha256="a" * 64, decoder_dir=str(tmp_path), python_executable="python"))
    calls = []

    def execute(source, candidate, temporary, provenance, **kwargs):
        calls.append(provenance)
        pl.read_csv(source, infer_schema=False).with_columns(
            pl.Series("HLA_a", ["AC", ""]), pl.Series("HLA_b", ["DE", ""]),
            pl.Series("ok", [True, False]), pl.Series("hla_reason", ["", "absent"]),
            pl.lit("").alias("inference_reason"), pl.Series("pll_" + p.DEFAULT_MODEL, [-3., None])
        ).write_csv(candidate)
        return dict(rows=2, scored=1, unresolved=1)

    monkeypatch.setattr(backend, "execute_torch", execute)
    args = dict(decoder_dir=tmp_path, python_executable="python", model="esmc-300m")
    result = backend.run_pmhc_backend(source, output, **args)
    assert result["context_type"] == "pmhc" and result["cache_hit"] is False
    assert backend.run_pmhc_backend(source, output, **args)["cache_hit"] is True
    assert len(calls) == 1
    original = output.read_bytes()
    monkeypatch.setattr(backend, "execute_torch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        backend.run_pmhc_backend(source, output, force=True, **args)
    assert output.read_bytes() == original


def test_pmhc_profile_unresolved_requires_empty_output_and_reason(tmp_path):
    source = pl.DataFrame({"name": ["one"], "peptide": ["A" * 9], "hla": ["HLA-A*99:99"]})
    output = tmp_path / "profile.csv"
    pl.DataFrame(schema={"position": pl.Int64, **{aa: pl.Float64 for aa in p.AA}}).write_csv(output)
    backend.validate_output(source, output, p.DEFAULT_MODEL,
                            {"status": "Unresolved", "reason": "absent"}, profile=True)
    with pytest.raises(ValueError):
        backend.validate_output(source, output, p.DEFAULT_MODEL,
                                {"status": "Unresolved", "reason": ""}, profile=True)
    profile = pl.DataFrame({"position": list(range(1, 10)),
                            **{aa: np.full(9, 0.05) for aa in p.AA}})
    profile.write_csv(output)
    backend.validate_output(source, output, p.DEFAULT_MODEL, {"status": "ModelHypothesis"}, profile=True)


def test_pmhc_input_never_accepts_label_or_tcr_columns(tmp_path):
    source = tmp_path / "input.csv"
    pl.DataFrame({"name": ["a"], "peptide": ["AC"], "hla": ["HLA-A*02:01"], "label": [1]}).write_csv(source)
    with pytest.raises(ValueError, match="only name"):
        backend.validate_input(source)


def test_torch_output_cache_byte_bound_eviction_and_disabled_mode():
    cache = OutputCache(1800)
    first, second = (b"first token sequence",), (b"second token sequence",)
    cache.put(first, "one", 1000)
    assert cache.get(first) == "one"
    cache.put(second, "two", 1000)
    assert cache.get(first) is None and cache.get(second) == "two"
    assert cache.bytes <= 1800 and cache.peak_bytes <= 1800
    disabled = OutputCache(0)
    disabled.put(first, "one", 1000)
    assert disabled.get(first) is None and disabled.bytes == 0
