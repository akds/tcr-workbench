"""Cheap contracts exercise batching/scoring independently of optional model weights."""
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tcr_workbench import prediction as p
from tcr_workbench.backends import mlx_decoder as host
from tcr_workbench.backends import mlx_worker as w
from tcr_workbench.backends.precision_contract import precision_metadata
from tcr_workbench.backends.reconstruct_worker import reconstruct
from tcr_workbench.backends.torch_decoder import checkpoint_arguments
from tcr_workbench.backends.torch_checkpoint_worker import validate_inventory
from tcr_workbench.cli import parser
from tcr_workbench.model_registry import normalize_device, resolve_model, validate_backend


class Tokenizer:
    def encode(self, sequence, add_special_tokens=True):
        encoded = [dict(zip(w.AA, w.AA_IDS))[aa] for aa in sequence]
        return [0] + encoded + [2] if add_special_tokens else encoded


def row(name, peptide="AC", *, ok=True, extension=""):
    return dict(name=name, HLA_a="AC" + extension, HLA_b="DE", TCR_a="FG", TCR_b="HI",
                peptide=peptide, ok=str(ok), tcr_reason="" if ok else "failed stitching")


def test_registry_separates_model_size_and_backend():
    assert resolve_model("esmc-300m").name == p.DEFAULT_MODEL
    assert resolve_model("esmc-600m").checkpoint.endswith("600M.ckpt")
    assert normalize_device("MLX") == "apple"
    assert normalize_device("gpu") == "cuda"
    assert validate_backend("esmc-600m", "apple", "bundle", "python").arch == "DecoderTCRC_600M"
    assert validate_backend("esmc-6b", "apple", "bundle", "python").arch == "DecoderTCRC_6B"
    for model in ("esm2-650m", "esm2-3b"):
        with pytest.raises(ValueError, match="ESM-C 300M, 600M and 6B architectures only"):
            validate_backend(model, "apple", "bundle", "python")
    with pytest.raises(ValueError, match="unsupported device"):
        normalize_device("mps")
    with pytest.raises(ValueError, match="applies only"):
        validate_backend("esmc-300m", "cpu", "bundle", "python")
    with pytest.raises(ValueError, match="unsupported model"):
        resolve_model("invented")


def test_checkpoint_override_has_explicit_architecture(tmp_path):
    result = checkpoint_arguments("esmc-600m", tmp_path / "custom.ckpt")
    assert result == ["--checkpoint", str(tmp_path / "custom.ckpt"), "--backbone", "esmc",
                      "--arch", "DecoderTCRC_600M"]
    assert checkpoint_arguments(p.DEFAULT_MODEL, None) == []


def test_checkpoint_size_guard_checks_actual_tensor_inventory():
    class Tensor:
        shape = (64, 960)
    state = {"model.model.embed.weight": Tensor()}
    state.update({f"model.model.transformer.blocks.{i}.weight": None for i in range(30)})
    validate_inventory({"state_dict": state}, "DecoderTCRC_300M")
    with pytest.raises(ValueError, match="does not match"):
        validate_inventory({"state_dict": state}, "DecoderTCRC_600M")
    del state["model.model.transformer.blocks.29.weight"]
    with pytest.raises(ValueError, match="does not match"):
        validate_inventory({"state_dict": state}, "DecoderTCRC_300M")
    with pytest.raises(ValueError, match="Lightning"):
        validate_inventory({}, "DecoderTCRC_300M")


def test_decoder_cli_exposes_model_checkpoint_and_runtime():
    args = parser().parse_args(["decoder-run", "--input", "i", "--output", "o",
        "--decoder-dir", "upstream", "--python", "reconstruction", "--device", "apple",
        "--checkpoint", "bundle", "--mlx-python", "mlx", "--model", "esmc-300m",
        "--batch-size", "4", "--token-budget", "5000", "--cache-bytes", "100000"])
    assert args.checkpoint == Path("bundle")
    assert args.mlx_python == "mlx"
    assert args.batch_size == 4


def test_context_assembly_masks_only_peptide_positions():
    tokens, positions = w.masked_tokens(w.context_key(row("a")), Tokenizer(), 2048)
    assert positions.tolist() == [5, 6]
    assert tokens.tolist() == [0, 5, 23, 13, 9, 32, 32, 18, 6, 21, 12, 2]
    with pytest.raises(ValueError, match="maximum"):
        w.masked_tokens(w.context_key(row("a")), Tokenizer(), 11)


def test_pll_uses_all_64_channels_profile_only_20():
    logits = np.zeros((2, 64), dtype=np.float32)
    logits[:, 63] = 5
    expected = -np.log(63 + np.exp(5))
    scores = w.score_peptides(["AC", "CA"], logits)
    np.testing.assert_allclose(scores, expected, rtol=1e-6)
    np.testing.assert_allclose(w.profile_probabilities(logits), 0.05, atol=1e-7)
    assert expected < -np.log(20)
    with pytest.raises(ValueError, match="non-finite"):
        w.log_softmax(logits * np.nan)
    with pytest.raises(ValueError, match="standard amino acids"):
        w.score_peptides(["XX"], logits)
    with pytest.raises(ValueError, match="lengths"):
        w.score_peptides(["A", "AAA"], logits)


def test_batching_preserves_order_failures_padding_and_context_reuse():
    source = [row("a", "AC"), row("b", "CA"), row("bad", ok=False),
              row("long", "AC", extension="AAA"), row("other_length", "ACD"), row("again", "AA")]
    calls = []
    output = []

    def infer(tokens, positions):
        calls.append((tokens.copy(), positions.copy()))
        return np.zeros((*positions.shape, 64), np.float32)

    stats = w.process_rows(iter(source), output.append, infer=infer, tokenizer=Tokenizer(),
                           max_length=2048, batch_size=3, token_budget=32, cache_bytes=100000,
                           chunk_size=5)
    assert [r["name"] for r in output] == ["a", "b", "bad", "long", "other_length", "again"]
    assert output[2]["_score"] == "" and output[2]["tcr_reason"] == "failed stitching"
    assert stats["scored"] == 5 and stats["unique_context_forwards"] == 3
    assert stats["cache_hits"] == 1 and stats["batches"] == 2
    assert calls[0][0].shape == (2, 15)
    assert calls[0][0][0, -3:].tolist() == [1, 1, 1]
    np.testing.assert_allclose([r["_score"] for r in output if r["_score"] != ""], -np.log(64))


def test_cache_budget_eviction_and_no_cache_mode():
    cache = w.LogitCache(1800)
    logits = np.zeros((2, 64), np.float32)
    one, two = w.context_key(row("a")), w.context_key(row("b", extension="A"))
    cache.put(one, logits)
    assert cache.get(one) is not None
    cache.put(two, logits)
    assert cache.get(one) is None and cache.get(two) is not None
    assert cache.nbytes <= 1800
    zero = w.LogitCache(0)
    zero.put(one, logits)
    assert zero.get(one) is None and zero.nbytes == 0
    view = logits[:, :1]
    cache.put(one, view)
    stored, _ = cache.get(one)
    assert stored.flags.owndata and not np.shares_memory(stored, logits)


def test_oversize_context_abstains_without_forward():
    output = []
    stats = w.process_rows(iter([row("too_long")]), output.append,
        infer=lambda *a: pytest.fail("unexpected model call"), tokenizer=Tokenizer(),
        max_length=10, batch_size=1, token_budget=4096, cache_bytes=0)
    assert stats["unresolved"] == 1
    assert output[0]["ok"] == "False" and "maximum" in output[0]["inference_reason"]


def test_inference_rejects_wrong_shape_or_nonfinite_output():
    for bad in (np.zeros((1, 2, 33)), np.full((1, 2, 64), np.nan)):
        with pytest.raises(ValueError):
            w.process_rows(iter([row("a")]), lambda _: None, infer=lambda *a: bad,
                tokenizer=Tokenizer(), max_length=2048, batch_size=1,
                token_budget=4096, cache_bytes=0)


def test_reconstruction_deduplicates_tcr_across_hla_and_peptides_and_retains_ids(tmp_path):
    source, target = tmp_path / "input.csv", tmp_path / "output.csv"
    genes = dict(trav="TRAV1", traj="TRAJ1", cdr3a="CAF", trbv="TRBV1", trbj="TRBJ1", cdr3b="CFW")
    rows = [dict(genes, name="0001", hla="known", peptide="AC"),
            dict(genes, name="0002", hla="unknown", peptide="CA"),
            dict(genes, name="0003", hla="known", peptide="AA")]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    calls = []
    alleles = []

    def stitch(batch):
        calls.append(batch)
        return {r["name"]: dict(TCR_a="AC", TCR_b="DE", ok=True, reason="", TRAV="TRAV1*01")
                for r in batch}

    def lookup(allele):
        alleles.append(allele)
        if allele == "unknown":
            raise KeyError("unknown HLA")
        return "FG", "HI"

    reconstruct(source, target, stitch=stitch, lookup=lookup, chunk_size=2)
    with target.open() as handle:
        result = list(csv.DictReader(handle))
    assert [r["name"] for r in result] == ["0001", "0002", "0003"]
    assert len(calls) == 1 and len(calls[0]) == 1
    assert alleles == ["known", "unknown"]
    assert result[1]["ok"] == "False" and "unknown HLA" in result[1]["hla_reason"]
    assert result[0]["TRAV"] == "TRAV1*01"


def test_bundle_fingerprint_changes_with_content_not_only_stats(tmp_path):
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (tmp_path / name).write_text("aaaa")
    (tmp_path / "manifest.json").write_text(json.dumps({"source_sha256": "a" * 64}))
    before = host.bundle_fingerprint(tmp_path)
    (tmp_path / "model.safetensors").write_text("bbbb")
    after = host.bundle_fingerprint(tmp_path)
    assert before["bundle_sha256"] != after["bundle_sha256"]
    assert before["checkpoint_sha256"] == "a" * 64


@pytest.mark.parametrize("options", [(0, 4096, 0), (True, 4096, 0), (1, 2, 0), (1, 4096, -1)])
def test_apple_resource_parameters_fail_closed(options):
    with pytest.raises(ValueError):
        host.validate_options(*options)


def test_worker_handshake_rejects_silent_backend_or_precision_changes():
    from pydantic import ValidationError
    runtime = dict(backend="mlx", device="apple", mode="scores", context_type="tcr-pmhc", **precision_metadata("float32"),
        tokenizer_variant="decodertcr-esm1b",
        source_checkpoint_sha256="a" * 64, mlx_peak_memory_bytes=10,
        model_load_seconds=0.1, inference_seconds=0.1,
        rows=1, scored=1, unresolved=0, unique_context_forwards=1)
    host.AppleRuntime.model_validate(runtime)
    for key, value in (("device", "cpu"), ("dtype", "float16"), ("rows", True),
                       ("math_precision", "tf32"), ("unknown", "anything")):
        with pytest.raises(ValidationError):
            host.AppleRuntime.model_validate({**runtime, key: value})


def test_apple_adapter_dispatch_preserves_manifest_import_and_cache(tmp_path, monkeypatch):
    source, output = tmp_path / "input.csv", tmp_path / "scores.csv"
    pl.DataFrame([dict(name="0001", trav="TRAV21", traj="TRAJ6", cdr3a="CAVRPGGAGPFF",
        trbv="TRBV7-9", trbj="TRBJ2-7", cdr3b="CASSLGQAYEQYF", hla="HLA-A*02:01",
        peptide="AC")]).write_csv(source)
    monkeypatch.setattr(host, "fingerprint", lambda *a, **k: dict(model=p.DEFAULT_MODEL,
        backend="mlx", device="apple", python_executable="reconstruct", checkpoint_sha256="abc"))
    calls = []

    def execute(source, candidate, temporary, fingerprint, **kwargs):
        calls.append(kwargs)
        pl.read_csv(source, infer_schema=False).with_columns(
            pl.lit(-2.0).alias("pll_" + p.DEFAULT_MODEL), pl.lit(True).alias("ok")
        ).write_csv(candidate)
        return {"backend": "mlx", "device": "apple", "dtype": "float32"}

    monkeypatch.setattr(host, "execute", execute)
    args = dict(decoder_dir=tmp_path, python_executable="reconstruct", device="apple",
                checkpoint=tmp_path, mlx_python="mlx", model="esmc-300m")
    first = p.run_decoder(source, output, **args)
    assert first["backend"] == "mlx" and first["cache_hit"] is False
    assert p.run_decoder(source, output, **args)["cache_hit"] is True
    assert len(calls) == 1
    result = p.import_decoder_scores(source, output, model="esmc-300m", require_manifest=True)
    assert result["name"].to_list() == ["0001"]
    assert result["status"].to_list() == ["ModelHypothesis"]
