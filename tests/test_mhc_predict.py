"""Fast adapter contracts; real optional-model smoke results are recorded separately."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from tcr_workbench import mhc_predict as mhc
from tcr_workbench.models import InputError


@pytest.fixture
def backends(monkeypatch):
    created = []

    class FakeBackend:
        def __init__(self, options):
            self.calls = []
            self.lengths = (5, 30)
            self.metadata = {"predictor": "fake", "predictor_version": "1", "source": "test", "license": "test"}
            created.append(self)

        def has_model(self, allele):
            return "99:99" not in allele

        def model_hash(self, allele):
            return "a" * 64

        def predict(self, allele, peptides):
            self.calls.append((allele, peptides))
            return [float(len(peptide)) for peptide in peptides]

    monkeypatch.setattr(mhc, "_FlurryBackend", FakeBackend)
    monkeypatch.setattr(mhc, "_NuggetsBackend", FakeBackend)
    return created


def panel(rows):
    return mhc._validated_panel(pl.DataFrame(rows, schema={"peptide": pl.String, "hla": pl.String}, orient="row"))


def test_mixed_classes_preserve_uncertainty_and_load_each_backend_once(backends):
    inputs = panel([
        ("GILGFVFTL", "A*02:01"), ("NLVPMVATV", "A*02:01"),
        ("PKYVKQNTLKLAT", "DRB1*01:01"),
        ("PKYVKQNTLKLAT", "DQB1*02:01/DQA1*05:01"),
        ("PKYVKQNTLKLAT", "DPA1*01:03/DPB1*04:01"),
        ("GILGFVFTL", "A*99:99"), ("PKYVKQNTLKLAT", "DRB1*99:99"),
        ("GILGFVFTL", None), ("GILGFVFTL", "A*02:01N"),
        ("GILGFVFTL", "A*02:01:01G"), ("GILGFVFTL", "A*02:01P"),
        ("GILGFVFTL", "A*02:01|A*02:02"), ("PKYVKQNTLKLAT", "DQB1*02:01"),
        ("AAAA", "A*02:01"), ("A" * 31, "DRB1*01:01"),
    ])
    result, metadata = mhc._predict_local(inputs, mhc.MHCOptions(batch_size=1))
    assert result.height == inputs.height
    assert_frame_equal(result.select("peptide", "hla"), inputs)
    assert result["status"].to_list() == ["Predicted"] * 5 + ["Unresolved"] * 10
    assert result.filter(pl.col("status") == "Unresolved")["score"].null_count() == 10
    assert result["metric"].unique().to_list() == ["predicted_ic50"]
    assert result["units"].unique().to_list() == ["nM"]
    assert result["higher_is_better"].unique().to_list() == [False]
    assert len(backends) == 2
    assert sum(len(backend.calls) for backend in backends) == 5
    assert all(len(peptides) <= 1 for backend in backends for _, peptides in backend.calls)
    assert "HLA-DQA105:01-DQB102:01" in result["model_allele"]
    assert metadata["predicted_pairs"] == 5
    assert "Not a binding probability" in metadata["calibration"]


@pytest.mark.parametrize("hla", ["DQA1*05:01", "DPA1*01:03", "DRA*01:01", "A*02", "DRB1*01:01L", "A*02:01Q"])
def test_insufficient_or_aberrant_hla_never_loads_model(backends, hla):
    result, _ = mhc._predict_local(panel([("PKYVKQNTLKLAT", hla)]), mhc.MHCOptions())
    assert result["status"].item() == "Unresolved"
    assert result["score"].item() is None
    assert backends == []


def test_model_name_preserves_pairing_and_records_protein_resolution():
    assert mhc._model_context("HLA-A*02:01:01")[1] == "HLA-A*02:01"
    assert mhc._model_context("HLA-DRA*01:01/HLA-DRB1*01:01")[1] == "HLA-DRA01:01-DRB101:01"


def test_nonfinite_and_nonpositive_predictor_scores_are_unresolved(backends):
    backend = mhc._FlurryBackend(mhc.MHCOptions())
    inputs = panel([("GILGFVFTL", "A*02:01"), ("NLVPMVATV", "A*02:01"), ("LLFGYPVYV", "A*02:01")])
    result = mhc._chunk(inputs.with_row_index("_pair_index"), reason="test", backend=backend,
                        allele="HLA-A*02:01", scores=[float("nan"), float("inf"), -1.0]).drop("_pair_index")
    mhc._validate_result(result, inputs)
    assert result["score"].null_count() == 3
    assert result["status"].to_list() == ["Unresolved"] * 3
    with pytest.raises(InputError, match="number of scores"):
        mhc._chunk(inputs.with_row_index("_pair_index"), reason="test", scores=[1.0])


@pytest.mark.parametrize("field", ["metric", "units", "higher_is_better", "reason", "predictor", "predictor_version", "source", "license", "model_allele", "model_sha256"])
def test_null_required_or_predicted_provenance_fields_fail(backends, field):
    inputs = panel([("GILGFVFTL", "A*02:01")])
    result, _ = mhc._predict_local(inputs, mhc.MHCOptions())
    broken = result.with_columns(pl.lit(None, dtype=mhc.RESULT_SCHEMA[field]).alias(field))
    with pytest.raises(InputError, match="invalid scores"):
        mhc._validate_result(broken, inputs)


@pytest.mark.parametrize("field,value", [("reason", " "), ("metric", "probability"), ("units", "percent"), ("higher_is_better", True), ("model_sha256", "bad"), ("score", 0.0)])
def test_invalid_claim_contract_fails(backends, field, value):
    inputs = panel([("GILGFVFTL", "A*02:01")])
    result, _ = mhc._predict_local(inputs, mhc.MHCOptions())
    with pytest.raises(InputError, match="invalid scores"):
        mhc._validate_result(result.with_columns(pl.lit(value).alias(field)), inputs)


def test_result_cannot_drop_duplicate_or_reorder_pairs(backends):
    inputs = panel([("GILGFVFTL", "A*02:01"), ("NLVPMVATV", "A*02:01")])
    result, _ = mhc._predict_local(inputs, mhc.MHCOptions())
    for bad in (result.head(1), result.reverse(), pl.concat([result, result])):
        with pytest.raises(InputError, match="altered, omitted or reordered"):
            mhc._validate_result(bad, inputs)


def test_public_subprocess_parquet_protocol_and_hashes(monkeypatch, backends):
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        monkeypatch.setenv(name, "/wrong/environment")

    def run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert kwargs["env"]["TF_USE_LEGACY_KERAS"] == "1"
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert not {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"} & kwargs["env"].keys()
        assert command[1] == "-I"
        mhc._worker(Path(command[-1]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    inputs = pl.DataFrame({"peptide": ["gilgfvftl", "GILGFVFTL"], "hla": ["A*02:01", "HLA-A*02:01"]})
    result, metadata = mhc.predict_peptide_mhc(inputs, python=sys.executable)
    assert result.height == 1
    assert metadata["input_rows"] == 2
    assert metadata["unique_pairs"] == 1
    assert metadata["adapter_sources"] == mhc._adapter_sources()
    assert len(metadata["executable_sha256"]) == 64


@pytest.mark.parametrize("field", ["adapter_sources", "executable_sha256", "input_sha256", "output_sha256"])
def test_public_rejects_stale_worker_or_corrupt_provenance(monkeypatch, backends, field):
    def run(command, **kwargs):
        root = Path(command[-1])
        mhc._worker(root)
        path = root / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata[field] = "corrupt"
        path.write_text(json.dumps(metadata))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(InputError):
        mhc.predict_peptide_mhc(panel([("GILGFVFTL", "A*02:01")]), python=sys.executable)


def test_worker_rejects_old_installed_adapter_before_model_loading(tmp_path, backends):
    inputs = panel([("GILGFVFTL", "A*02:01")])
    inputs.write_parquet(tmp_path / "panel.parquet")
    (tmp_path / "request.json").write_text(json.dumps({"options": {}, "input_sha256": mhc._sha256(tmp_path / "panel.parquet"), "adapter_sources": {}}))
    with pytest.raises(InputError, match="different workbench"):
        mhc._worker(tmp_path)
    assert backends == []


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": True}, {"max_pairs": 0}, {"cpu_threads": 0}, {"timeout": float("nan")}, {"timeout": False}])
def test_options_are_strict(kwargs):
    with pytest.raises(ValueError):
        mhc.predict_peptide_mhc(panel([("GILGFVFTL", "A*02:01")]), python=sys.executable, **kwargs)


def test_guard_and_malformed_panel_fail_before_model_load():
    with pytest.raises(InputError, match="exceeds"):
        mhc.predict_peptide_mhc(panel([("GILGFVFTL", "A*02:01"), ("NLVPMVATV", "A*02:01")]), python=sys.executable, max_pairs=1)
    for inputs in (pl.DataFrame({"peptide": [123], "hla": ["A*02:01"]}),
                   pl.DataFrame({"peptide": ["GILX"], "hla": ["A*02:01"]})):
        with pytest.raises(ValueError):
            mhc.predict_peptide_mhc(inputs, python=sys.executable)


def test_empty_panel_needs_no_models(backends):
    result, metadata = mhc._predict_local(pl.DataFrame(schema={"peptide": pl.String, "hla": pl.String}), mhc.MHCOptions())
    assert result.schema == mhc.RESULT_SCHEMA
    assert result.is_empty()
    assert metadata["predicted_pairs"] == 0
    assert backends == []


def test_real_installed_allele_filename_fixture_preserves_exact_dr_heterodimer():
    fixture = json.loads((Path(__file__).parent / "fixtures/mhcnuggets_2.4.1_model_names.json").read_text())
    for hla in ("DRB1*01:01", "DRA*01:01/DRB1*01:01", "DQA1*05:01/DQB1*02:01", "DPA1*01:03/DPB1*04:01"):
        _, name, reason = mhc._model_context(hla)
        assert reason == ""
        assert name + "_BA.h5" in fixture["files"]
    _, alternate, _ = mhc._model_context("DRA*01:02/DRB1*01:01")
    assert alternate == "HLA-DRA01:02-DRB101:01"
    assert alternate + "_BA.h5" not in fixture["files"]


def test_missing_exact_dr_alpha_model_never_falls_back_to_beta(backends, monkeypatch):
    monkeypatch.setattr(mhc._NuggetsBackend, "has_model", lambda self, name: name == "HLA-DRB101:01")
    inputs = panel([("PKYVKQNTLKLAT", "DRA*01:02/DRB1*01:01"), ("PKYVKQNTLKLAT", "DRB1*01:01")])
    result, metadata = mhc._predict_local(inputs, mhc.MHCOptions())
    assert result["status"].to_list() == ["Unresolved", "Predicted"]
    assert "not replaced by a DRB-only model" in result["reason"][0]
    assert metadata["backend_counts"]["mhcnuggets"] == {"routed": 2, "predicted": 1, "no_model": 1, "unsupported_length": 0, "failed_score": 0}
    assert metadata["comparability_scope"] == ["predictor", "predictor_version", "model_allele"]


def test_seven_mer_is_rejected_before_loading_class_one_backend(backends):
    result, metadata = mhc._predict_local(panel([("GILGFVF", "A*02:01")]), mhc.MHCOptions())
    assert result["status"].item() == "Unresolved"
    assert metadata["backend_counts"]["mhcflurry"]["unsupported_length"] == 1
    assert backends == []


def test_adapter_digest_covers_entire_python_package():
    files = mhc._adapter_sources()
    root = Path(mhc.__file__).parent
    assert set(files) == {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
    assert "__init__.py" in files
    assert "matching.py" in files


@pytest.mark.parametrize("failure", ["missing", "bad_parquet", "bad_json", "wrong_json_type"])
def test_missing_or_unreadable_worker_results_are_input_errors(monkeypatch, backends, failure):
    def run(command, **kwargs):
        root = Path(command[-1])
        if failure != "missing":
            mhc._worker(root)
        if failure == "bad_parquet":
            (root / "evidence.parquet").write_text("broken")
        if failure in ("bad_json", "wrong_json_type"):
            (root / "metadata.json").write_text("{" if failure == "bad_json" else "[]")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(InputError, match="readable evidence and provenance"):
        mhc.predict_peptide_mhc(panel([("GILGFVFTL", "A*02:01")]), python=sys.executable)


def test_worker_timeout_becomes_an_actionable_input_error(monkeypatch):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(InputError, match="exceeded its 1s timeout"):
        mhc.predict_peptide_mhc(panel([("GILGFVFTL", "A*02:01")]), python=sys.executable, timeout=1)


def test_public_normalization_contract_is_suitable_for_panel_joins():
    raw = pl.DataFrame({"peptide": [" gilgfvftl ", "GILGFVFTL"], "hla": ["A*02:01", "HLA-A*02:01"]})
    normalized = mhc.normalize_peptide_panel(raw)
    assert normalized.to_dicts() == [{"peptide": "GILGFVFTL", "hla": "HLA-A*02:01"}]


def test_nuggets_failed_load_invalidates_previous_weight_identity(monkeypatch, tmp_path):
    # Exercise the real adapter method without importing TensorFlow in the core suite.
    import types

    dataset = types.ModuleType("mhcnuggets.src.dataset")
    dataset.mask_peptides = lambda peptides, max_len: (peptides, peptides)
    dataset.tensorize_keras = lambda peptides, embed_type: np.zeros((len(peptides), 30, 21))
    monkeypatch.setitem(sys.modules, "mhcnuggets", types.ModuleType("mhcnuggets"))
    monkeypatch.setitem(sys.modules, "mhcnuggets.src", types.ModuleType("mhcnuggets.src"))
    monkeypatch.setitem(sys.modules, "mhcnuggets.src.dataset", dataset)

    class Model:
        def __init__(self):
            self.loads = []
            self.fail = False

        def load_weights(self, path):
            self.loads.append(Path(path).name)
            if self.fail:
                self.fail = False
                raise RuntimeError("partial load failed")

        def __call__(self, encoded, training):
            assert training is False
            return np.full((len(encoded), 1), 0.5)

    backend = mhc._NuggetsBackend.__new__(mhc._NuggetsBackend)
    backend.directory = tmp_path
    backend.current_allele = None
    backend.model = Model()
    first = backend.predict("alleleA", ["PKYVKQNTLKLAT"])
    assert first[0] == pytest.approx(50000 ** 0.5)
    backend.model.fail = True
    with pytest.raises(RuntimeError, match="partial load"):
        backend.predict("alleleB", ["PKYVKQNTLKLAT"])
    assert backend.current_allele is None
    backend.predict("alleleA", ["PKYVKQNTLKLAT"])
    assert backend.model.loads == ["alleleA_BA.h5", "alleleB_BA.h5", "alleleA_BA.h5"]


@pytest.mark.parametrize("values", [[-0.1], [1.1], [float("nan")], [float("inf")], [0.5, 0.5]])
def test_nuggets_raw_regression_range_and_shape_are_checked(monkeypatch, tmp_path, values):
    import types

    dataset = types.ModuleType("mhcnuggets.src.dataset")
    dataset.mask_peptides = lambda peptides, max_len: (peptides, peptides)
    dataset.tensorize_keras = lambda peptides, embed_type: np.zeros((len(peptides), 30, 21))
    for name in ("mhcnuggets", "mhcnuggets.src"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mhcnuggets.src.dataset", dataset)
    backend = mhc._NuggetsBackend.__new__(mhc._NuggetsBackend)
    backend.directory = tmp_path
    backend.current_allele = "alleleA"
    backend.model = lambda encoded, training: np.array(values)
    with pytest.raises(InputError, match="raw affinity"):
        backend.predict("alleleA", ["PKYVKQNTLKLAT"])
