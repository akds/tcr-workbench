import csv
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from tcr_workbench import receptor_embeddings as host
from tcr_workbench.backends import receptor_embedding_worker as worker
from tcr_workbench.models import InputError


COMPONENTS = dict(trav="TRAV21", traj="TRAJ6", cdr3a="CAVRPGGAGPFFVVF",
                  trbv="TRBV7-9", trbj="TRBJ2-7", cdr3b="CASSLGQAYEQYF")
OPTIONS = dict(decoder_dir="/unused", python_executable="/unused/python", model="esmc-300m")
FINGERPRINT = dict(decoder_dir="/unused", python_executable="/unused/python",
                   checkpoint_sha256="a" * 64, checkpoint_path="/unused/model")


def receptor_frame():
    return pl.DataFrame([dict(embedding_id="a", species="human", **COMPONENTS),
                         dict(embedding_id="b", species="human", **COMPONENTS)])


def fake_stitch(rows):
    return {row["name"]: dict(TCR_a="ACDE", TCR_b="FGHIK", ok=True, reason="",
                              **{key: row[key.lower()] for key in worker.SELECTED}) for row in rows}


def worker_args(input_path, output_path, **kwargs):
    return SimpleNamespace(input=str(input_path), output=str(output_path), model="DecoderTCR-ESMC_300M",
        device="cpu", species="human", checkpoint_sha256="a" * 64, batch_size=4, token_budget=4096, **kwargs)


def fake_encoder(args, calls=None):
    def encode(sequences):
        if calls is not None:
            calls.append(sequences)
        vectors = np.zeros((len(sequences), 960), np.float32)
        for index, sequence in enumerate(sequences):
            vectors[index, 0 if sequence.startswith("A") else 1] = 1
        return vectors
    return encode, lambda: None, dict(backend="torch", device="cpu", tokenizer_variant="decodertcr-esm1b", model_load_seconds=0.0)


def mock_worker(input_path, staging, options, fingerprint):
    reconstructed = staging / ".reconstructed.csv"
    worker.reconstruct(input_path, reconstructed, options.species, stitch=fake_stitch)
    worker.infer(worker_args(reconstructed, staging), encoder_loader=fake_encoder)
    reconstructed.unlink()


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(host, "_fingerprint", lambda options: FINGERPRINT)
    monkeypatch.setattr(host, "_run_worker", mock_worker)


def test_end_to_end_mock_deduplicates_and_retains_audit(tmp_path, fake_runtime):
    frame = receptor_frame().vstack(pl.DataFrame([dict(embedding_id="bad", species="human", **(COMPONENTS | {"trav": None}))]))
    result = host.embed_receptors(frame, tmp_path / "out", **OPTIONS)
    assert result.ids == ["a", "b"]
    assert result.vectors.shape == (2, 1920)
    np.testing.assert_allclose(result.vectors[0], result.vectors[1])
    np.testing.assert_allclose(result.vectors[:, [0, 961]], 2 ** -.5)
    assert result.audit["status"].to_list() == ["Embedded", "Embedded", "Unresolved"]
    assert "trav" in result.audit["reason"][2]
    assert result.audit["reconstructed_receptor_sha256"][0] == worker.chain_hashes("ACDE", "FGHIK")["reconstructed_receptor_sha256"]
    assert result.metadata["runtime"]["unique_chains"] == 2
    assert result.metadata["runtime"]["batches"] == 1
    assert "gene alleles" in result.audit["reason"][0]
    assert result.metadata["representation"] == "decodertcr-chain-mean-v1"
    loaded = host.load_receptor_embeddings(tmp_path / "out")
    np.testing.assert_array_equal(loaded.vectors, result.vectors)
    assert not list((tmp_path / "out").glob(".*"))


@pytest.mark.parametrize("change", [
    lambda frame: frame.with_columns(pl.lit("A*02:01").alias("hla")),
    lambda frame: frame.with_columns(pl.lit("A").alias("peptide")),
    lambda frame: frame.with_columns(pl.lit("a").alias("embedding_id")),
    lambda frame: frame.with_columns(pl.lit(None).alias("embedding_id")),
    lambda frame: frame.with_columns(pl.Series("species", ["human", "mouse"])),
    lambda frame: frame.with_columns(pl.lit("Human").alias("species")),
    lambda frame: frame.with_columns(pl.lit(1).alias("trav")),
])
def test_bad_schema_fails_before_model(change, tmp_path):
    with pytest.raises(InputError):
        host.embed_receptors(change(receptor_frame()), tmp_path / "out", **OPTIONS)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("options", [dict(species="mouse"), dict(precision="float16"),
    dict(mhc_reference="unneeded.json"), dict(model="esm2-650m"), dict(device="mps"), dict(batch_size=True)])
def test_unsupported_option_fails_before_model(options, tmp_path):
    with pytest.raises(InputError):
        host.embed_receptors(receptor_frame(), tmp_path / "out", **(OPTIONS | options))


@pytest.mark.parametrize("frame", [
    pl.DataFrame([dict(embedding_id="bad", species="human", **(COMPONENTS | {"cdr3a": "AAA"}))]),
])
def test_unresolved_never_loads_model(frame, tmp_path, monkeypatch, fake_runtime):
    def no_model(args):
        raise AssertionError("No successful reconstructed receptor should load weights")
    def run(input_path, staging, options, fingerprint):
        reconstructed = staging / "temp.csv"
        worker.reconstruct(input_path, reconstructed, options.species, stitch=fake_stitch)
        worker.infer(worker_args(reconstructed, staging), encoder_loader=no_model)
        reconstructed.unlink()
    monkeypatch.setattr(host, "_run_worker", run)
    result = host.embed_receptors(frame, tmp_path / "out", **OPTIONS)
    assert result.vectors.shape == (0, 1920)
    assert not result.ids
    assert result.audit.height == frame.height


def test_empty_input_rejected_before_model(tmp_path):
    frame = pl.DataFrame(schema={key: pl.String for key in host.INPUT_COLUMNS})
    with pytest.raises(InputError, match="at least one"):
        host.embed_receptors(frame, tmp_path / "out", **OPTIONS)
    assert not (tmp_path / "out").exists()


def test_reconstruction_keeps_failures_and_selected_genes(tmp_path):
    original = host._input_frame(receptor_frame())
    original.write_csv(tmp_path / "input.csv")
    seen = []
    def stitch(rows):
        seen.extend(rows)
        return {rows[0]["name"]: dict(TCR_a="AAA*", TCR_b="ACD", ok=True, reason="warning", TRAV="TRAV21*01")}
    worker.reconstruct(tmp_path / "input.csv", tmp_path / "out.csv", "human", stitch=stitch)
    audit = pl.read_csv(tmp_path / "out.csv", infer_schema=False)
    assert len(seen) == 1
    assert audit["status"].to_list() == ["Unresolved"] * 2
    assert all("AA20" in value for value in audit["reason"])
    assert audit["TRAV"].to_list() == ["TRAV21*01"] * 2


def test_length_batches_bound_padding_cost_and_single_sequence_limit():
    items = [(i, "A" * length) for i, length in enumerate([3, 4, 5, 7, 8])]
    batches = list(worker.length_batches(iter(items), batch_size=3, token_budget=20))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert all(len(batch) * (len(batch[-1][1]) + 2) <= 20 for batch in batches)
    with pytest.raises(ValueError, match="token-budget"):
        list(worker.length_batches(iter([(0, "A" * 30)]), 1, 20))


def test_tokens_retain_only_residues_and_one_cls_eos():
    vocabulary = "LAGVSERTIDPKQNFYMHWC"
    lookup = {aa: position + 4 for position, aa in enumerate(vocabulary)}
    def encode(sequence):
        return [0, *(lookup[aa] for aa in sequence), 2]
    tokens = worker.token_array(["AC", "A"], encode, 1)
    assert tokens.tolist() == [[0, lookup["A"], lookup["C"], 2], [0, lookup["A"], 2, 1]]
    with pytest.raises(ValueError, match="Tokenizer"):
        worker.token_array(["A"], lambda _: [0, 3, 2], 1)
    with pytest.raises(ValueError, match="PAD token ID"):
        worker.token_array(["A"], encode, 3)


def test_reconstruction_failure_keeps_original_reason(tmp_path):
    host._input_frame(receptor_frame()).write_csv(tmp_path / "input.csv")
    def stitch(rows):
        return {row["name"]: dict(TCR_a="", TCR_b="", ok=False,
                                  reason="Supplied alpha gene is absent from the reference") for row in rows}
    worker.reconstruct(tmp_path / "input.csv", tmp_path / "out.csv", "human", stitch=stitch)
    audit = pl.read_csv(tmp_path / "out.csv", infer_schema=False)
    assert audit["reason"].to_list() == ["Supplied alpha gene is absent from the reference"] * 2


def test_chain_budget_checked_before_model_load(tmp_path):
    host._input_frame(receptor_frame()).write_csv(tmp_path / "input.csv")
    worker.reconstruct(tmp_path / "input.csv", tmp_path / "reconstructed.csv", "human", stitch=fake_stitch)
    args = worker_args(tmp_path / "reconstructed.csv", tmp_path / "out")
    args.token_budget = 6  # ACDE fits; FGHIK needs 7 tokens.
    def no_model(args):
        raise AssertionError("Token budget must be checked before loading weights")
    with pytest.raises(ValueError, match="needs 7 tokens"):
        worker.infer(args, encoder_loader=no_model)
    assert not (tmp_path / "out/.chain-vectors.sqlite").exists()
    args.token_budget = 7  # Small budgets remain valid when each chain fits.
    worker.infer(args, encoder_loader=fake_encoder)
    runtime = json.loads((tmp_path / "out/embedding_runtime.json").read_text())
    assert runtime["batches"] == 2 and runtime["unique_chains"] == 2


@pytest.mark.parametrize("corruption", ["ids", "vector", "species", "dimension", "checkpoint", "hash", "status"])
def test_corrupt_worker_cannot_publish(corruption, tmp_path, monkeypatch, fake_runtime):
    def bad_worker(input_path, staging, options, fingerprint):
        mock_worker(input_path, staging, options, fingerprint)
        if corruption == "ids":
            (staging / "embedding_ids.json").write_text('["b","a"]')
        elif corruption == "vector":
            vectors = np.load(staging / "receptor_embeddings.npy")
            vectors[0, 0] = np.nan
            np.save(staging / "receptor_embeddings.npy", vectors)
        elif corruption in ("dimension", "checkpoint"):
            path = staging / "embedding_runtime.json"
            runtime = json.loads(path.read_text())
            runtime["vector_dimension" if corruption == "dimension" else "source_checkpoint_sha256"] = 42 if corruption == "dimension" else "b" * 64
            path.write_text(json.dumps(runtime))
        else:
            path = staging / "embedding_audit.csv"
            frame = pl.read_csv(path, infer_schema=False)
            changes = {"species": ("species", "mouse"), "hash": ("alpha_sha256", ""), "status": ("status", "Scored")}
            key, value = changes[corruption]
            frame.with_columns(pl.lit(value).alias(key)).write_csv(path)
    monkeypatch.setattr(host, "_run_worker", bad_worker)
    with pytest.raises(InputError):
        host.embed_receptors(receptor_frame(), tmp_path / "out", **OPTIONS)
    assert not (tmp_path / "out").exists()


def test_checkpoint_changed_during_run_cannot_publish(tmp_path, monkeypatch, fake_runtime):
    fingerprints = iter([FINGERPRINT, FINGERPRINT | {"checkpoint_sha256": "b" * 64}])
    monkeypatch.setattr(host, "_fingerprint", lambda _: next(fingerprints))
    with pytest.raises(InputError, match="changed during"):
        host.embed_receptors(receptor_frame(), tmp_path / "out", **OPTIONS)
    assert not (tmp_path / "out").exists()


def test_saved_bundle_renaming_and_corruption(tmp_path, fake_runtime):
    folder = tmp_path / "out"
    host.embed_receptors(receptor_frame(), folder, **OPTIONS)
    (folder / "manifest.json").rename(folder / "embedding_model_manifest.json")
    for extension in ("csv", "parquet"):
        (folder / f"embedding_audit.{extension}").rename(folder / f"embedding_reconstruction_audit.{extension}")
    loaded = host.load_receptor_embeddings(folder, manifest_name="embedding_model_manifest.json", audit_name="embedding_reconstruction_audit.csv")
    assert loaded.ids == ["a", "b"]
    (folder / "embedding_ids.json").write_text("[]")
    with pytest.raises(InputError, match="changed"):
        host.load_receptor_embeddings(folder, manifest_name="embedding_model_manifest.json", audit_name="embedding_reconstruction_audit.csv")


def test_worker_deduplication_across_distinct_receptors(tmp_path):
    frame = host._input_frame(receptor_frame())
    frame.write_csv(tmp_path / "input.csv")
    worker.reconstruct(tmp_path / "input.csv", tmp_path / "reconstructed.csv", "human", stitch=fake_stitch)
    calls = []
    worker.infer(worker_args(tmp_path / "reconstructed.csv", tmp_path / "out"),
                 encoder_loader=lambda args: fake_encoder(args, calls))
    assert calls == [["ACDE", "FGHIK"]]
    with (tmp_path / "out/embedding_audit.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_torch_pool_matches_reference_without_logits_or_layer_stack(monkeypatch):
    """Optional tiny oracle; no checkpoints/downloads or large model allocation."""
    torch = pytest.importorskip("torch")
    esmc = pytest.importorskip("esmc.models.esmc")
    tokenizer_module = pytest.importorskip("esmc.tokenization")
    torch.manual_seed(19)
    model = esmc.ESMC(64, 1, 2, tokenizer_module.EsmSequenceTokenizer()).eval()
    tokens = torch.tensor(worker.token_array(["ACDE", "FG"], model.tokenizer.encode, 1)).long()
    with torch.inference_mode():
        official = model(sequence_tokens=tokens).embeddings
        expected = torch.stack([official[0, 1:5].mean(0), official[1, 1:3].mean(0)])
        expected = expected / torch.linalg.vector_norm(expected, dim=1, keepdim=True)
        def forbidden(*args, **kwargs):
            raise AssertionError("Embedding inference must not compute logits or stack all layers")
        monkeypatch.setattr(model.sequence_head, "forward", forbidden)
        monkeypatch.setattr(torch, "stack", forbidden)
        actual = worker.torch_pool(model, tokens, torch)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)


def test_direct_loader_rejects_invalid_norms_even_with_updated_file_hash(tmp_path, fake_runtime):
    folder = tmp_path / "out"
    host.embed_receptors(receptor_frame(), folder, **OPTIONS)
    path = folder / "receptor_embeddings.npy"
    vectors = np.load(path)
    vectors[0] *= 2
    np.save(path, vectors)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["outputs"][path.name]["sha256"] = host.file_sha256(path)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(InputError, match="normalized"):
        host.load_receptor_embeddings(folder)
