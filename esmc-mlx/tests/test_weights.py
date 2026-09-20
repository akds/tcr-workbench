import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from esmc_mlx.config import (ModelConfig, config_300m, config_600m, config_6b,
                             validate_decoder_config)
from esmc_mlx.tokenizer import Tokenizer
from esmc_mlx.weights import (BundleError, BundleManifest, convert_tensors,
    expected_shapes, source_mapping, sha256_file, verify_bundle, SOURCE_PINS)


def tiny_config(variant="biohub-esmc-published-v1"):
    return ModelConfig(model_id="synthetic-test", source_variant=variant,
        tokenizer_variant="biohub-esmc" if variant.startswith("biohub") else "decodertcr-esm1b",
        hidden_size=8, num_attention_heads=2, num_hidden_layers=2, intermediate_size=12)


def tiny_source(config):
    output = {}
    for dest, names in source_mapping(config).items():
        shape = expected_shapes(config)[dest]
        if len(names) > 1:
            shape = (shape[0] // len(names), *shape[1:])
        for i, name in enumerate(names):
            output[name] = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + (i + 1) / 10)
    return output


@pytest.mark.parametrize("variant", ["biohub-esmc-published-v1", "decodertcr-lightning-v03"])
def test_inventory_fusion_order_biases_no_transpose(variant):
    config = tiny_config(variant)
    source = tiny_source(config)
    tensors, records = convert_tensors(source, config)
    assert set(tensors) == set(expected_shapes(config))
    for dest, names in source_mapping(config).items():
        assert records[dest]["source_keys"] == list(names)
        np.testing.assert_array_equal(tensors[dest], np.concatenate([source[k] for k in names], axis=0) if len(names) > 1 else source[names[0]])
    assert np.count_nonzero(tensors["transformer.blocks.0.attn.ln_qkv.bias"]) == 8
    # Matrices use [out,in], with gate/up rows in that order.
    assert tensors["transformer.blocks.0.ffn.fc1.weight"].shape == (24, 8)


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "shape", "nan", "inf"])
def test_corrupt_sources_fail_loudly(mutation):
    config = tiny_config()
    source = tiny_source(config)
    key = "esmc.layers.0.self_attn.k_proj.weight"
    if mutation == "missing":
        del source[key]
    elif mutation == "extra":
        source["unrecognized.weight"] = np.zeros(1, np.float32)
    elif mutation == "dtype":
        source[key] = source[key].astype(np.float16)
    elif mutation == "shape":
        source[key] = source[key][:1]
    else:
        source[key][0, 0] = np.nan if mutation == "nan" else np.inf
    with pytest.raises(BundleError):
        convert_tensors(source, config)


def make_bundle(root: Path):
    c = tiny_config()
    weights, records = convert_tensors(tiny_source(c), c)
    root.mkdir()
    (root / "config.json").write_text(c.model_dump_json())
    (root / "tokenizer.json").write_text(Tokenizer(c.tokenizer_variant).config.model_dump_json())
    save_file(weights, root / "model.safetensors")
    (root / "licenses").mkdir()
    (root / "licenses/LICENSE").write_text("Synthetic fixture; no pretrained weights.")
    files = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    m = BundleManifest(model_id=c.model_id, source_variant=c.source_variant,
        source_name="synthetic.safetensors", source_sha256="0" * 64, source_bytes=1,
        **(SOURCE_PINS[c.source_variant] | {"model_revision": None, "model_url": None}),
        checkpoint_identity="user-supplied-unvalidated", source_tensor_count=sum(map(len,source_mapping(c).values())),
        parameter_count=sum(int(np.prod(s)) for s in expected_shapes(c).values()),
        discarded_training_keys=[], tensors=records, files=files,
        conversion="exact-fp32-row-concatenation-no-transpose", software={"test":"true"})
    (root / "manifest.json").write_text(m.model_dump_json())
    return c, m


def test_bundle_roundtrip_and_tamper(tmp_path):
    root = tmp_path / "bundle"
    c, m = make_bundle(root)
    checked, manifest, weights = verify_bundle(root)
    assert checked == c and weights == root / "model.safetensors"
    assert manifest["checkpoint_identity"] == "user-supplied-unvalidated"
    (root / "config.json").write_text((root / "config.json").read_text()+" ")
    with pytest.raises(BundleError, match="hash mismatch"):
        verify_bundle(root)


@pytest.mark.parametrize("mutation", ["shape", "source_keys", "tokenizer", "duplicate_json", "extra_file", "symlink", "nan"])
def test_bundle_rejects_corrupt_schema_with_rehashed_file(tmp_path, mutation):
    root = tmp_path / "bundle"
    c, m = make_bundle(root)
    manifest = m.model_dump(mode="json")
    key = next(iter(manifest["tensors"]))
    if mutation in ("shape", "source_keys"):
        manifest["tensors"][key][mutation] = [42] if mutation == "shape" else ["incorrect.key"]
    elif mutation == "tokenizer":
        p = root / "tokenizer.json"
        data = json.loads(p.read_text())
        data["vocabulary"][4] = "WRONG"
        p.write_text(json.dumps(data))
        manifest["files"]["tokenizer.json"] = sha256_file(p)
    elif mutation == "duplicate_json":
        p = root / "config.json"
        p.write_text('{"model_id":"ignored",' + p.read_text()[1:])
        manifest["files"]["config.json"] = sha256_file(p)
    elif mutation == "extra_file":
        (root / "untracked.txt").write_text("extra")
    elif mutation == "symlink":
        p = root / "licenses/LICENSE"
        p.unlink()
        p.symlink_to(root / "config.json")
        manifest["files"]["licenses/LICENSE"] = sha256_file(p)
    else:
        tensors, records = convert_tensors(tiny_source(c), c)
        tensors[key].flat[0] = np.nan
        save_file(tensors, root / "model.safetensors")
        manifest["files"]["model.safetensors"] = sha256_file(root / "model.safetensors")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError):
        verify_bundle(root)


def test_real_safetensors_conversion_atomic_custom_identity(tmp_path, monkeypatch):
    import esmc_mlx.weights as module
    c = tiny_config()
    monkeypatch.setattr(module, "config_300m", lambda variant: c)
    source = tmp_path / "input.safetensors"
    save_file(tiny_source(c), source)
    out = tmp_path / "converted"
    with pytest.raises(BundleError, match="explicit --model-id"):
        module.convert_checkpoint(source, out, c.source_variant)
    with pytest.raises(BundleError, match="SHA-256"):
        module.convert_checkpoint(source, out, c.source_variant, expected_sha256="f" * 64, model_id="custom-test")
    manifest = module.convert_checkpoint(source, out, c.source_variant, expected_sha256=sha256_file(source), model_id="custom-test")
    assert manifest["checkpoint_identity"] == "user-supplied-unvalidated"
    assert manifest["model_revision"] is None and manifest["model_url"] is None
    checked, _, _ = verify_bundle(out)
    assert checked.model_id == "custom-test"
    with pytest.raises(BundleError, match="already exists"):
        module.convert_checkpoint(source, out, c.source_variant, model_id="custom-test")


def test_lightning_safe_load_discards_only_named_training_state(tmp_path, monkeypatch):
    import esmc_mlx.weights as module
    torch = pytest.importorskip("torch")
    c = tiny_config("decodertcr-lightning-v03")
    monkeypatch.setattr(module, "config_300m", lambda variant: c)
    source = tmp_path / "checkpoint.ckpt"
    state = {key: torch.from_numpy(value) for key, value in tiny_source(c).items()}
    torch.save({"state_dict": state, "optimizer_states": [{"weight": torch.ones(17)}], "epoch": 42}, source)
    manifest = module.convert_checkpoint(source, tmp_path / "converted", c.source_variant, model_id="custom-test")
    assert manifest["discarded_training_keys"] == ["epoch", "optimizer_states"]
    torch.save({"state_dict": state, "extra_model": torch.ones(3)}, source)
    with pytest.raises(BundleError, match="Unexpected Lightning metadata"):
        module.convert_checkpoint(source, tmp_path / "bad", c.source_variant, model_id="custom-test")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad-*"))


def test_manifest_cannot_falsely_attribute_custom_weights(tmp_path):
    root = tmp_path / "bundle"
    _, m = make_bundle(root)
    data = m.model_dump(mode="json")
    data["checkpoint_identity"] = "pinned-release"
    (root / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(BundleError, match="pinned-release identity"):
        verify_bundle(root)


def test_600m_architecture_and_requested_model_are_bound():
    config = config_600m("decodertcr-lightning-v03")
    assert (config.hidden_size, config.num_attention_heads, config.num_hidden_layers,
            config.intermediate_size, config.head_dim) == (1152, 18, 36, 3072, 64)
    assert config.residue_scaling_factor == 1.0
    assert len(expected_shapes(config)) == 368
    assert expected_shapes(config)["transformer.blocks.35.ffn.fc1.weight"] == (6144, 1152)
    validate_decoder_config(config, "DecoderTCR-ESMC_600M")
    validate_decoder_config(config_300m("decodertcr-lightning-v03"), "DecoderTCR-ESMC_300M")
    for model in ("DecoderTCR-ESMC_300M", "DecoderTCR-ESMC_6B", "DecoderTCR_650M"):
        with pytest.raises(ValueError):
            validate_decoder_config(config, model)
    with pytest.raises(ValueError, match="Lightning"):
        config_600m("biohub-esmc-published-v1")
    for changes in ({"intermediate_size": 3071}, {"residue_scaling_base": 30.0},
                    {"pre_norm_bias": False}, {"max_position_embeddings": 1024}):
        changed = ModelConfig.model_validate(config.model_dump() | changes)
        with pytest.raises(ValueError, match="architecture/tokenizer"):
            validate_decoder_config(changed, "DecoderTCR-ESMC_600M")


@pytest.mark.parametrize("size", ["600m", "6b"])
def test_large_conversion_and_known_bytes_cannot_be_mislabeled(tmp_path, monkeypatch, size):
    import esmc_mlx.weights as module
    torch = pytest.importorskip("torch")
    config = tiny_config("decodertcr-lightning-v03")
    monkeypatch.setattr(module, "config_" + size, lambda variant: config)
    source = tmp_path / (size + ".ckpt")
    state = {key: torch.from_numpy(value) for key, value in tiny_source(config).items()}
    torch.save({"state_dict": state}, source)
    monkeypatch.setitem(module.KNOWN_CHECKPOINTS_BY_SIZE[config.source_variant],
                        size, sha256_file(source))
    with pytest.raises(BundleError, match="select --model-size " + size):
        module.convert_checkpoint(source, tmp_path / "wrong", config.source_variant)
    assert not (tmp_path / "wrong").exists()
    manifest = module.convert_checkpoint(source, tmp_path / "right", config.source_variant,
                                         model_size=size)
    assert manifest["checkpoint_identity"] == "pinned-release"
    assert verify_bundle(tmp_path / "right")[0] == config
    with pytest.raises(BundleError, match="Unsupported model_size"):
        module.convert_checkpoint(source, tmp_path / "unsupported", config.source_variant,
                                  model_size="12b")


def test_6b_architecture_contract_without_allocating_weights():
    config = config_6b("decodertcr-lightning-v03")
    assert (config.hidden_size, config.num_attention_heads, config.num_hidden_layers,
            config.intermediate_size, config.head_dim) == (2560, 40, 80, 6912, 64)
    assert config.residue_scaling_factor == np.sqrt(80 / 36)
    shapes = expected_shapes(config)
    assert len(shapes) == 808
    assert shapes["transformer.blocks.79.ffn.fc1.weight"] == (13824, 2560)
    assert shapes["transformer.blocks.79.attn.qkv.weight"] == (7680, 2560)
    validate_decoder_config(config, "DecoderTCR-ESMC_6B")
    for model in ("DecoderTCR-ESMC_300M", "DecoderTCR-ESMC_600M", "DecoderTCR_650M"):
        with pytest.raises(ValueError):
            validate_decoder_config(config, model)
    with pytest.raises(ValueError, match="Lightning"):
        config_6b("biohub-esmc-published-v1")
    changed = ModelConfig.model_validate(config.model_dump() | {"intermediate_size": 6826})
    with pytest.raises(ValueError, match="architecture/tokenizer"):
        validate_decoder_config(changed, "DecoderTCR-ESMC_6B")


def test_deepspeed_consolidation_metadata_requires_complete_exact_inventory():
    from types import SimpleNamespace
    from esmc_mlx.weights import _validate_consolidated_metadata
    state = {"weight": SimpleNamespace(shape=(2, 3))}
    valid = {"param_shapes": [{"weight": (2, 3)}], "buffer_names": [],
             "frozen_param_shapes": {}, "frozen_param_fragments": {}, "shared_params": {}}
    _validate_consolidated_metadata(valid, state)
    for changes in ({"param_shapes": [{"weight": (3, 2)}]},
                    {"param_shapes": [{"other": (2, 3)}]},
                    {"param_shapes": [{"weight": (2, 3)}, {"weight": (2, 3)}]},
                    {"frozen_param_fragments": {"extra": [1]}},
                    {"buffer_names": ["unhandled_buffer"]}):
        with pytest.raises(BundleError, match="DeepSpeed"):
            _validate_consolidated_metadata(valid | changes, state)
