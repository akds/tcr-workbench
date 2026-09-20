"""Precision selection is explicit, auditable, and cannot reuse a different mode."""
import json
import subprocess
import sys
from types import SimpleNamespace

import polars as pl
import pytest
from pydantic import ValidationError

from tcr_workbench import decoder_pmhc, prediction, workflows
from tcr_workbench.backends import mlx_decoder, mlx_worker, pmhc_decoder
from tcr_workbench.backends.precision_contract import precision_metadata
from tcr_workbench.cli import main
from tcr_workbench.model_registry import validate_backend
from tcr_workbench.runtime_settings import resolve_settings


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:2"])
@pytest.mark.parametrize("operation", ["paired", "profile", "pmhc", "pmhc_panel", "pmhc_profile", "repertoire"])
def test_unsupported_precision_rejected_before_inference_even_when_unresolved(tmp_path, device, operation):
    options = dict(decoder_dir=tmp_path, python_executable="unused", device=device, precision="float16")
    source, panel = tmp_path / "input.csv", tmp_path / "panel.csv"
    source.write_text("cell_id,donor_id,cdr3a,cdr3b\nc1,d1,,CASSF\n")
    panel.write_text("peptide,hla\nGILGFVFTL,\n")
    out = tmp_path / "output"
    calls = {
        "paired": lambda: prediction.run_decoder(source, out, **options),
        "profile": lambda: prediction.run_decoder_profile({}, out, length=9, **options),
        "pmhc": lambda: pmhc_decoder.run_pmhc_backend(panel, out, **options),
        "pmhc_panel": lambda: decoder_pmhc.score_pmhc(panel, out, **options),
        "pmhc_profile": lambda: decoder_pmhc.profile_pmhc("A*02", 9, out, **options),
        "repertoire": lambda: workflows.score_repertoire(source, panel, out, **options),
    }
    with pytest.raises(ValueError, match="float16 precision is supported only"):
        calls[operation]()
    assert not out.exists()


@pytest.mark.parametrize("precision", ["bfloat16", "half", None, True, 16])
def test_precision_contract_rejects_unknown_modes(precision):
    with pytest.raises(ValueError, match="precision"):
        validate_backend("esmc-300m", "apple", "bundle", "python", precision)


@pytest.mark.parametrize("model", ["esmc-600m", "esmc-6b"])
def test_fp16_other_model_is_rejected(model):
    with pytest.raises(ValueError, match="currently requires float32"):
        validate_backend(model, "apple", "bundle", "python", "float16")


def test_settings_backward_compatibility_and_explicit_cpu_override(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(decoder_dir="decoder", python_executable="python")))
    assert resolve_settings(SimpleNamespace(config=path))["precision"] == "float32"
    path.write_text(json.dumps(dict(decoder_dir="decoder", python_executable="python", device="apple",
        checkpoint="bundle", mlx_python="mlx", precision="float16")))
    assert resolve_settings(SimpleNamespace(config=path))["precision"] == "float16"
    with pytest.raises(ValueError, match="float16 precision"):
        resolve_settings(SimpleNamespace(config=path, device="cpu", checkpoint="weights.ckpt"))
    options = resolve_settings(SimpleNamespace(config=path, device="cpu", checkpoint="weights.ckpt", precision="float32"))
    assert options["precision"] == "float32" and options["mlx_python"] is None


def test_fp16_configure_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "decoder").mkdir()
    (tmp_path / "bundle").mkdir()
    (tmp_path / "python").touch()
    assert main(["configure", "--decoder-dir", "decoder", "--python", "python", "--device", "apple",
                 "--checkpoint", "bundle", "--mlx-python", "python", "--precision", "float16"]) == 0
    assert resolve_settings(SimpleNamespace(config=None))["precision"] == "float16"


def test_runtime_handshake_requires_consistent_precision():
    runtime = dict(backend="mlx", device="apple", mode="scores", context_type="pmhc",
        **precision_metadata("float16"), tokenizer_variant="decodertcr-esm1b",
        source_checkpoint_sha256="a" * 64, mlx_peak_memory_bytes=10, model_load_seconds=0.1,
        inference_seconds=0.1, rows=1, scored=1, unresolved=0, unique_context_forwards=1)
    assert mlx_decoder.AppleRuntime.model_validate(runtime).approximate is True
    for key, value in (("precision", "float32"), ("approximate", False), ("dtype", "float32"),
                       ("rope_frequency_dtype", "float16"), ("score_reduction_dtype", "float16")):
        with pytest.raises(ValidationError):
            mlx_decoder.AppleRuntime.model_validate({**runtime, key: value})


def test_host_rejects_valid_but_different_worker_precision(tmp_path, monkeypatch):
    runtime = dict(backend="mlx", device="apple", mode="scores", context_type="pmhc",
        **precision_metadata("float32"), tokenizer_variant="decodertcr-esm1b",
        source_checkpoint_sha256="a" * 64, mlx_peak_memory_bytes=10, model_load_seconds=0.1,
        inference_seconds=0.1, rows=1, scored=1, unresolved=0, unique_context_forwards=1)
    def execute(command, *args):
        if "--runtime" in command:
            assert command[command.index("--precision") + 1] == "float16"
            (tmp_path / "runtime.json").write_text(json.dumps(runtime))
    monkeypatch.setattr(prediction, "_execute", execute)
    provenance = dict(decoder_dir=str(tmp_path), python_executable="unused", context_type="pmhc",
        mlx_python_executable="unused", bundle_path=str(tmp_path), model=prediction.DEFAULT_MODEL,
        batch_size=1, token_budget=4096, cache_bytes=0, checkpoint_sha256="a"*64, **precision_metadata("float16"))
    with pytest.raises(ValueError, match="inconsistent model/device provenance"):
        mlx_decoder.execute("input", "output", tmp_path, provenance, timeout=None)


def test_cast_is_once_and_preserves_frequency_dtype():
    events = []
    param = SimpleNamespace(dtype="f32")
    rope = SimpleNamespace(_freqs=SimpleNamespace(dtype="f32"))
    class Model:
        transformer = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(rope=rope))])
        def parameters(self):
            return [("weight", param)]
        def set_dtype(self, value):
            events.append("cast")
            param.dtype = value
    mx = SimpleNamespace(float16="f16", float32="f32", eval=lambda *a: events.append("eval"),
                         clear_cache=lambda: events.append("clear"))
    model = Model()
    assert not mlx_worker.prepare_model_precision(model, "float32", mx, lambda v:v)["approximate"]
    assert events == []
    assert mlx_worker.prepare_model_precision(model, "float16", mx, lambda v:v)["approximate"]
    assert events == ["cast", "eval", "clear"]
    assert rope._freqs.dtype == "f32"
    with pytest.raises(RuntimeError, match="parameter dtype"):
        mlx_worker.prepare_model_precision(model, "float32", mx, lambda v:v)
    param.dtype = "f32"
    rope._freqs.dtype = "f16"
    with pytest.raises(RuntimeError, match="RoPE"):
        mlx_worker.prepare_model_precision(model, "float32", mx, lambda v:v)


def test_isolated_worker_can_load_its_precision_contract():
    # -I deliberately excludes the worker directory, as in production. Exercise
    # the standalone import branch without importing MLX or loading any model.
    code = "import runpy; f=runpy.run_path(%r)['prepare_model_precision']; f(None,'invalid',None,None)" % mlx_worker.__file__
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode != 0 and "precision must be float32 or float16" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("kind", ["paired", "pmhc"])
def test_apple_precision_changes_cache_identity(tmp_path, monkeypatch, kind):
    source, out = tmp_path / "input.csv", tmp_path / "scores.csv"
    pl.DataFrame([dict(name="1", trav="TRAV21", traj="TRAJ6", cdr3a="CAVRPGGAGPFF",
        trbv="TRBV7-9", trbj="TRBJ2-7", cdr3b="CASSLGQAYEQYF", hla="HLA-A*02:01", peptide="AC")]).write_csv(source)
    monkeypatch.setattr(mlx_decoder, "fingerprint", lambda *a, **kw: dict(model=prediction.DEFAULT_MODEL,
        backend="mlx", device="apple", **precision_metadata(kw["precision"])))
    if kind == "pmhc":
        pl.read_csv(source).select("name", "peptide", "hla").write_csv(source)
    calls = []
    def execute(source, output, temporary, provenance, **kwargs):
        calls.append(provenance["precision"])
        pl.read_csv(source, infer_schema=False).with_columns(
            pl.lit(-2.0).alias("pll_"+prediction.DEFAULT_MODEL), pl.lit(True).alias("ok"),
            pl.lit("AC").alias("HLA_a"), pl.lit("DE").alias("HLA_b"),
            pl.lit("").alias("hla_reason"), pl.lit("").alias("inference_reason")).write_csv(output)
        return {**precision_metadata(provenance["precision"]), "rows": 1, "scored": 1}
    monkeypatch.setattr(mlx_decoder, "execute", execute)
    opts=dict(decoder_dir=tmp_path, python_executable="unused", device="apple", checkpoint=tmp_path, mlx_python="unused")
    run = prediction.run_decoder if kind == "paired" else pmhc_decoder.run_pmhc_backend
    assert run(source, out, **opts)["precision"] == "float32"
    with pytest.raises(ValueError, match="provenance"):
        run(source, out, precision="float16", **opts)
    result = run(source, out, precision="float16", force=True, **opts)
    assert result["approximate"] and result["dtype"] == "float16"
    assert run(source, out, precision="float16", **opts)["cache_hit"]
    assert calls == ["float32", "float16"]


def test_fp16_unresolved_pmhc_records_request_without_claiming_execution(tmp_path):
    frame, run = decoder_pmhc.score_pmhc(pl.DataFrame({"peptide":["AC"],"hla":["A*02"]}), tmp_path/"out",
        decoder_dir=tmp_path, python_executable="unused", device="apple", checkpoint=tmp_path,
        mlx_python="unused", precision="float16")
    assert run["summary"]["approximate"] and run["summary"]["precision"] == "float16"
    assert run["backend"]["status"] == "not_run"
    assert "approximate FP16" in frame["interpretation"][0]


def test_fp16_profile_cli_propagates_precision_and_labels_artifacts(tmp_path, monkeypatch, capsys):
    def profile(components, output, **kwargs):
        assert kwargs["precision"] == "float16"
        pl.DataFrame({"position":[1,2], **{aa:[0.05,0.05] for aa in prediction.AA}}).write_csv(output)
        return {"model":prediction.DEFAULT_MODEL, **precision_metadata("float16")}
    monkeypatch.setattr(prediction,"run_decoder_profile",profile)
    out = tmp_path/"profile"
    assert main(["tcr-profile","--trav","TRAV21","--traj","TRAJ6","--cdr3a","CAVRPGGAGPFFVVF",
        "--trbv","TRBV7-9","--trbj","TRBJ2-7","--cdr3b","CASSLGQAYEQYF","--hla","A*02:01",
        "--length","2","--out",str(out),"--decoder-dir",str(tmp_path),"--python","unused",
        "--device","apple","--mlx-python","unused","--checkpoint",str(tmp_path),"--precision","float16"]) == 0
    assert "Approximate Apple FP16" in capsys.readouterr().err
    metadata = json.loads((out/"pssm_metadata.json").read_text())
    assert metadata["precision"] == "float16" and metadata["approximate"]
    assert all("approximate FP16" in value for value in pl.read_parquet(out/"pssm.parquet")["interpretation"])


def test_fp16_unresolved_repertoire_report_labels_precision(tmp_path):
    source, panel, out = tmp_path/"input.csv", tmp_path/"panel.csv", tmp_path/"out"
    source.write_text("cell_id,donor_id,cdr3a,cdr3b\nc1,d1,,CASSF\n")
    panel.write_text("peptide,hla\nGILGFVFTL,A*02:01\n")
    _, run = workflows.score_repertoire(source,panel,out,decoder_dir=tmp_path,python_executable="unused",
        device="apple",checkpoint=tmp_path,mlx_python="unused",precision="float16")
    assert run["execution"]["status"] == "not_run_no_eligible_pairs"
    assert run["parameters"]["precision"] == "float16" and run["summary"]["approximate"]
    assert "float16 (approximate)" in (out/"report.html").read_text()
