"""Strict, separate checkpoint adapters and integrity-checked local MLX bundles.

Published split-projection mappings adapt Biohub's MIT checkpoint_layout.py at
43b4548b86762edfa747b07d5f440aad3c33acee. Unlike upstream's compatibility
loader, these adapters reject every missing or unexpected model tensor. Torch
is imported only when converting the explicitly selected Lightning format.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from .config import ModelConfig, ModelSize, SourceVariant, config_300m, config_600m, config_6b
from .tokenizer import Tokenizer, TokenizerConfig

SOURCE_PINS = {
    "biohub-esmc-published-v1": {
        "source_revision": "43b4548b86762edfa747b07d5f440aad3c33acee",
        "source_url": "https://github.com/Biohub/esm",
        "model_revision": "e4bac860f0c502cd80f3aeac3d0ce6524c6627cb",
        "model_url": "https://huggingface.co/biohub/ESMC-300M",
    },
    "decodertcr-lightning-v03": {
        "source_revision": "3e3d9889d26d79635f674940ddab039c4dc6f9f9",
        "source_url": "https://github.com/Biohub/DecoderTCR",
        "model_revision": "803fed3bbf3dcc40ed481f7d50a20f668db0ef01",
        "model_url": "https://huggingface.co/biohub/DecoderTCR",
    },
}
KNOWN_CHECKPOINT_HASHES = {
    "biohub-esmc-published-v1": "b06a03aa359c29a889151da4a07a517f2c307c1fcc61c8b767ec5360dd40cb49",
    "decodertcr-lightning-v03": "18d47c169d0ce992152838b8229e1c682b0a681f55057b471dcdcbf07d2fcad9",
}
KNOWN_CHECKPOINTS_BY_SIZE = {
    "biohub-esmc-published-v1": {"300m": KNOWN_CHECKPOINT_HASHES["biohub-esmc-published-v1"]},
    "decodertcr-lightning-v03": {
        "300m": KNOWN_CHECKPOINT_HASHES["decodertcr-lightning-v03"],
        "600m": "4d3c84f30e3781c023e412eb3f3c098ef9fe64fd1423fd34df4b40b9dcbac0aa",
        "6b": "b6bd3170b55a9a06d9c1472e248bd083e092afc7924439b444bf7379b74ab692",
    },
}


def _size_config(source_variant: SourceVariant, model_size: ModelSize) -> ModelConfig:
    if model_size == "300m":
        return config_300m(source_variant)
    if model_size == "600m":
        return config_600m(source_variant)
    if model_size == "6b":
        return config_6b(source_variant)
    raise BundleError("Unsupported model_size; choose 300m, 600m or 6b")


def _known_size(source_variant: SourceVariant, source_hash: str) -> str | None:
    return next((size for size, digest in KNOWN_CHECKPOINTS_BY_SIZE[source_variant].items()
                 if digest == source_hash), None)

_TRAINER_KEYS = {
    "epoch", "global_step", "pytorch-lightning_version", "loops", "callbacks",
    "optimizer_states", "lr_schedulers", "hparams_name", "hyper_parameters", "training_config",
}
_DEEPSPEED_KEYS = {
    "buffer_names", "param_shapes", "frozen_param_shapes", "shared_params",
    "frozen_param_fragments", "data_sampler", "random_ltd", "sparse_tensor_module_names",
    "global_samples", "ds_config", "ds_version",
}
_TRAINER_KEYS |= _DEEPSPEED_KEYS


class BundleError(ValueError):
    """An unsupported, corrupt or internally inconsistent model artifact."""


class TensorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    shape: list[int]
    dtype: Literal["float32"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_keys: list[str] = Field(min_length=1)


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    format_version: Literal[1] = 1
    converter_version: Literal["1"] = "1"
    model_id: str
    source_variant: SourceVariant
    source_name: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_bytes: int = Field(gt=0)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_url: str
    model_revision: str | None = Field(pattern=r"^[0-9a-f]{40}$")
    model_url: str | None
    checkpoint_identity: Literal["pinned-release", "user-supplied-unvalidated"]
    source_tensor_count: int = Field(gt=0)
    parameter_count: int = Field(gt=0)
    discarded_training_keys: list[str]
    tensors: dict[str, TensorRecord]
    files: dict[str, str]
    conversion: Literal["exact-fp32-row-concatenation-no-transpose"]
    software: dict[str, str]

    @model_validator(mode="after")
    def valid_files(self):
        required = {"config.json", "tokenizer.json", "model.safetensors"}
        if not required <= self.files.keys():
            raise ValueError("bundle is missing required file hashes")
        for name, digest in self.files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or name == "manifest.json":
                raise ValueError("invalid relative bundle file name")
            if not (name in required or name.startswith("licenses/")):
                raise ValueError(f"unrecognized bundle file {name}")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid file SHA-256")
        pins = SOURCE_PINS[self.source_variant]
        if self.source_revision != pins["source_revision"] or self.source_url != pins["source_url"]:
            raise ValueError("source adapter revision does not match the implemented schema")
        if self.checkpoint_identity == "pinned-release":
            if (_known_size(self.source_variant, self.source_sha256) is None
                    or self.model_revision != pins["model_revision"] or self.model_url != pins["model_url"]):
                raise ValueError("pinned-release identity does not match checkpoint bytes and revision")
        elif self.model_revision is not None or self.model_url is not None:
            raise ValueError("user-supplied checkpoints cannot claim a known model release")
        if set(self.discarded_training_keys) - _TRAINER_KEYS:
            raise ValueError("unrecognized discarded training-state keys")
        if self.source_variant == "biohub-esmc-published-v1" and self.discarded_training_keys:
            raise ValueError("safetensors sources cannot contain discarded training-state keys")
        if not any(name.startswith("licenses/") for name in self.files):
            raise ValueError("bundle must retain source license notices")
        return self


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BundleError(f"Duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), object_pairs_hook=unique)
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Cannot read bundle JSON {path}: {exc}") from exc


def expected_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    d, h, v = config.hidden_size, config.intermediate_size, config.vocab_size
    shapes = {"embed.weight": (v, d), "transformer.norm.weight": (d,),
              "sequence_head.0.weight": (d, d), "sequence_head.0.bias": (d,),
              "sequence_head.2.weight": (d,), "sequence_head.2.bias": (d,),
              "sequence_head.3.weight": (v, d), "sequence_head.3.bias": (v,)}
    if config.final_norm_bias:
        shapes["transformer.norm.bias"] = (d,)
    for index in range(config.num_hidden_layers):
        prefix = f"transformer.blocks.{index}."
        leaves = {"attn.ln_qkv.weight": (d,), "attn.qkv.weight": (3 * d, d),
                  "attn.q_ln.weight": (d,), "attn.k_ln.weight": (d,),
                  "attn.out_proj.weight": (d, d), "ffn.ln.weight": (d,),
                  "ffn.fc1.weight": (2 * h, d), "ffn.fc2.weight": (d, h)}
        if config.pre_norm_bias:
            leaves.update({"attn.ln_qkv.bias": (d,), "ffn.ln.bias": (d,)})
        if config.qk_norm_bias:
            leaves.update({"attn.q_ln.bias": (d,), "attn.k_ln.bias": (d,)})
        shapes.update({prefix + key: shape for key, shape in leaves.items()})
    return shapes


def source_mapping(config: ModelConfig) -> dict[str, tuple[str, ...]]:
    """Each destination maps to ordered source parts; fusion is along axis zero."""
    mapping = {}
    for destination in expected_shapes(config):
        if config.source_variant == "decodertcr-lightning-v03":
            name = destination
            for before, after in ((".attn.ln_qkv.", ".attn.layernorm_qkv.0."),
                                  (".attn.qkv.", ".attn.layernorm_qkv.1."),
                                  (".ffn.ln.", ".ffn.0."),
                                  (".ffn.fc1.", ".ffn.1."),
                                  (".ffn.fc2.", ".ffn.3.")):
                name = name.replace(before, after)
            mapping[destination] = ("model.model." + name,)
            continue
        if destination == "embed.weight":
            mapping[destination] = ("esmc.embed_tokens.weight",)
        elif destination.startswith("transformer.norm."):
            mapping[destination] = (destination.replace("transformer.norm.", "esmc.norm."),)
        elif destination.startswith("sequence_head."):
            name = destination.replace("sequence_head.0.", "lm_head.dense.").replace("sequence_head.2.", "lm_head.layer_norm.").replace("sequence_head.3.", "lm_head.decoder.")
            mapping[destination] = (name,)
        else:
            pieces = destination.split(".")
            prefix = f"esmc.layers.{pieces[2]}."
            leaf = ".".join(pieces[3:])
            if leaf == "attn.qkv.weight":
                leaves = tuple(f"self_attn.{part}_proj.weight" for part in ("q", "k", "v"))
            elif leaf == "ffn.fc1.weight":
                leaves = ("mlp.gate_proj.weight", "mlp.up_proj.weight")
            else:
                rename = {"attn.ln_qkv.": "input_layernorm.", "attn.q_ln.": "self_attn.q_norm.",
                          "attn.k_ln.": "self_attn.k_norm.", "attn.out_proj.": "self_attn.o_proj.",
                          "ffn.ln.": "post_attention_layernorm.", "ffn.fc2.": "mlp.down_proj."}
                for before, after in rename.items():
                    leaf = leaf.replace(before, after)
                leaves = (leaf,)
            mapping[destination] = tuple(prefix + leaf for leaf in leaves)
    flat = [key for keys in mapping.values() for key in keys]
    if len(set(flat)) != len(flat):
        raise BundleError("Internal source adapter maps a source tensor more than once")
    return mapping


def convert_tensors(source: Mapping[str, np.ndarray], config: ModelConfig) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Validate every source tensor and fuse only the explicitly named parts.

    Primarily useful for small conversion tests; real conversion supplies a lazy
    safetensors or memory-mapped Torch mapping to avoid eagerly copying inputs.
    """
    mapping = source_mapping(config)
    expected = {key for keys in mapping.values() for key in keys}
    found = set(source)
    if expected != found:
        raise BundleError(f"Source tensor inventory mismatch; missing={sorted(expected-found)}, unexpected={sorted(found-expected)}")
    output, records = {}, {}
    shapes = expected_shapes(config)
    for destination, keys in mapping.items():
        shape = shapes[destination]
        parts = []
        for key in keys:
            array = np.asarray(source[key])
            source_shape = (shape[0] // len(keys), *shape[1:]) if len(keys) > 1 else shape
            if array.shape != source_shape:
                raise BundleError(f"{key}: expected shape {source_shape}, found {array.shape}")
            if array.dtype != np.dtype("float32"):
                raise BundleError(f"{key}: expected float32, found {array.dtype}; implicit casts are forbidden")
            if not np.isfinite(array).all():
                raise BundleError(f"{key}: non-finite weights")
            parts.append(array)
        tensor = np.concatenate(parts, axis=0) if len(parts) > 1 else np.ascontiguousarray(parts[0])
        output[destination] = tensor
        records[destination] = {"shape": list(shape), "dtype": "float32", "sha256": _array_sha256(tensor), "source_keys": list(keys)}
    return output, records


class _SafeSource(Mapping):
    def __init__(self, source):
        self.source = source
        self.keys_ = source.keys()
    def __len__(self):
        return len(self.keys_)
    def __iter__(self):
        return iter(self.keys_)
    def __getitem__(self, key):
        return self.source.get_tensor(key)


class _TorchSource(Mapping):
    def __init__(self, state):
        self.state = state
    def __len__(self):
        return len(self.state)
    def __iter__(self):
        return iter(self.state)
    def __getitem__(self, key):
        import torch
        tensor = self.state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise BundleError(f"{key}: expected a dense tensor")
        if tensor.dtype != torch.float32:
            raise BundleError(f"{key}: expected float32, found {tensor.dtype}")
        return tensor.detach().numpy()


def _validate_consolidated_metadata(checkpoint: Mapping, state: Mapping) -> None:
    """Accept inventoried DeepSpeed metadata, never discard extra model fragments."""
    for key in ("buffer_names", "frozen_param_shapes", "shared_params",
                "frozen_param_fragments", "sparse_tensor_module_names"):
        value = checkpoint.get(key)
        if value is not None and (not isinstance(value, (Mapping, list, tuple, set)) or len(value)):
            raise BundleError(f"Unsupported nonempty DeepSpeed model metadata: {key}")
    if "param_shapes" not in checkpoint:
        return
    groups = checkpoint["param_shapes"]
    if not isinstance(groups, list) or not groups or any(not isinstance(g, Mapping) for g in groups):
        raise BundleError("DeepSpeed param_shapes must be a nonempty list of mappings")
    shapes = {}
    for group in groups:
        for key, shape in group.items():
            if key in shapes or not isinstance(shape, (list, tuple)) or any(type(d) is not int for d in shape):
                raise BundleError("Invalid or repeated DeepSpeed param_shapes entry")
            shapes[key] = tuple(shape)
    if set(shapes) != set(state) or any(
        getattr(state[key], "shape", None) is None or shapes[key] != tuple(state[key].shape)
        for key in shapes
    ):
        raise BundleError("DeepSpeed param_shapes disagree with consolidated state_dict")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def convert_checkpoint(source: str | Path, output: str | Path, source_variant: SourceVariant,
                       expected_sha256: str | None = None, model_id: str | None = None,
                       model_size: ModelSize = "300m") -> dict:
    """Convert one checked supported artifact atomically; never overwrite a bundle."""
    import importlib.metadata
    import platform

    source, output = Path(source).resolve(), Path(output).absolute()
    config = _size_config(source_variant, model_size)
    if output.exists() or output.is_symlink():
        raise BundleError(f"Output already exists: {output}")
    source_hash = sha256_file(source)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise BundleError("Source checkpoint SHA-256 does not match expected_sha256")
    known_size = _known_size(source_variant, source_hash)
    if known_size is not None and known_size != model_size:
        raise BundleError(f"Pinned checkpoint is {known_size}; select --model-size {known_size}")
    known_release = known_size is not None
    if not known_release and model_id is None:
        raise BundleError("Unrecognized checkpoint bytes: supply an explicit --model-id for a user-supplied, unvalidated checkpoint of the selected architecture")
    if model_id is not None:
        config = ModelConfig.model_validate(config.model_dump() | {"model_id": model_id})
    pins = dict(SOURCE_PINS[source_variant])
    if not known_release:
        pins.update(model_revision=None, model_url=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    training_keys: list[str] = []
    try:
        if source_variant == "biohub-esmc-published-v1":
            with safe_open(source, framework="np", device="cpu") as reader:
                source_count = len(reader.keys())
                tensors, records = convert_tensors(_SafeSource(reader), config)
        else:
            import torch
            # Explicitly safe: no unrestricted pickle fallback or auto-allowlisted
            # globals. mmap avoids faulting optimizer storage into resident RAM.
            checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise BundleError("DecoderTCR Lightning source must contain state_dict")
            training_keys = sorted(set(checkpoint) - {"state_dict"})
            if set(training_keys) - _TRAINER_KEYS:
                raise BundleError(f"Unexpected Lightning metadata: {sorted(set(training_keys)-_TRAINER_KEYS)}")
            state = checkpoint["state_dict"]
            if not isinstance(state, Mapping):
                raise BundleError("DecoderTCR state_dict must be a tensor mapping")
            _validate_consolidated_metadata(checkpoint, state)
            source_count = len(state)
            tensors, records = convert_tensors(_TorchSource(state), config)
        save_file(tensors, staging / "model.safetensors", metadata={"format": "esmc_mlx_v1"})
        # Reopen serialized FP32 arrays and check every byte after write. No
        # transpose/cast/quantization is accepted by this baseline converter.
        with safe_open(staging / "model.safetensors", framework="np") as saved:
            for key, tensor in tensors.items():
                if _array_sha256(saved.get_tensor(key)) != records[key]["sha256"]:
                    raise BundleError(f"Serialized tensor differs: {key}")
        del tensors
        if sha256_file(source) != source_hash:
            raise BundleError("Source checkpoint changed during conversion")
        _write_json(staging / "config.json", config.model_dump(mode="json"))
        tok = Tokenizer(config.tokenizer_variant, config.max_position_embeddings)
        _write_json(staging / "tokenizer.json", tok.config.model_dump(mode="json"))
        license_root = Path(__file__).resolve().parent / "licenses"
        if not license_root.is_dir():
            license_root = Path(__file__).resolve().parents[1] / "licenses"
        shutil.copytree(license_root, staging / "licenses")
        if not any((staging / "licenses").iterdir()):
            raise BundleError("Source license snapshots are missing")
        files = {str(p.relative_to(staging)): sha256_file(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = BundleManifest(
            model_id=config.model_id, source_variant=source_variant, source_name=source.name,
            source_sha256=source_hash, source_bytes=source.stat().st_size,
            source_tensor_count=source_count, parameter_count=sum(int(np.prod(s)) for s in expected_shapes(config).values()),
            discarded_training_keys=training_keys, tensors=records, files=files,
            conversion="exact-fp32-row-concatenation-no-transpose", **pins,
            checkpoint_identity="pinned-release" if known_release else "user-supplied-unvalidated",
            software={"python": platform.python_version(), "numpy": np.__version__, "safetensors": importlib.metadata.version("safetensors")},
        )
        _write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        verify_bundle(staging)
        # rename is same-filesystem and fails for an already populated output;
        # explicitly recheck before publishing rather than overwrite user files.
        if output.exists() or output.is_symlink():
            raise BundleError(f"Output appeared during conversion: {output}")
        os.rename(staging, output)
        return manifest.model_dump(mode="json")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_bundle(path: str | Path) -> tuple[ModelConfig, dict, Path]:
    """Verify hashes, schemas, inventory and finite FP32 tensors without MLX/Torch."""
    root = Path(path).resolve()
    try:
        manifest = BundleManifest.model_validate(_read_json(root / "manifest.json"))
        actual_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
        if actual_files != set(manifest.files):
            raise BundleError("Bundle files differ from the manifest inventory")
        for name, digest in manifest.files.items():
            file = root / name
            if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
                raise BundleError(f"Bundle file is missing or links outside the bundle: {name}")
            if sha256_file(file) != digest:
                raise BundleError(f"Bundle hash mismatch: {name}")
        config = ModelConfig.model_validate(_read_json(root / "config.json"))
        if manifest.checkpoint_identity == "pinned-release":
            size = _known_size(manifest.source_variant, manifest.source_sha256)
            expected_config = _size_config(manifest.source_variant, size)
            if config.model_dump(exclude={"model_id"}) != expected_config.model_dump(exclude={"model_id"}):
                raise BundleError("Pinned checkpoint architecture differs from its published model size")
        # JSON arrays intentionally become tuples only through Pydantic JSON mode.
        tokenizer = TokenizerConfig.model_validate_json((root / "tokenizer.json").read_text())
        if (manifest.model_id != config.model_id or manifest.source_variant != config.source_variant
                or tokenizer.variant != config.tokenizer_variant or tokenizer.max_length != config.max_position_embeddings):
            raise BundleError("Manifest, architecture and tokenizer identities disagree")
        shapes, mapping = expected_shapes(config), source_mapping(config)
        if set(manifest.tensors) != set(shapes):
            raise BundleError("Manifest tensor inventory differs from architecture")
        if manifest.parameter_count != sum(int(np.prod(s)) for s in shapes.values()):
            raise BundleError("Manifest parameter count differs from architecture")
        if manifest.source_tensor_count != sum(map(len, mapping.values())):
            raise BundleError("Manifest source tensor count differs from source adapter")
        for key, record in manifest.tensors.items():
            if record.shape != list(shapes[key]) or record.source_keys != list(mapping[key]):
                raise BundleError(f"Manifest tensor schema differs: {key}")
        with safe_open(root / "model.safetensors", framework="np", device="cpu") as reader:
            if set(reader.keys()) != set(shapes):
                raise BundleError("Safetensors inventory differs from architecture")
            for key, shape in shapes.items():
                tensor = reader.get_tensor(key)
                if tensor.shape != shape or tensor.dtype != np.dtype("float32"):
                    raise BundleError(f"Invalid tensor shape/dtype: {key}")
                if not np.isfinite(tensor).all() or _array_sha256(tensor) != manifest.tensors[key].sha256:
                    raise BundleError(f"Non-finite or mismatched tensor: {key}")
        return config, manifest.model_dump(mode="json"), root / "model.safetensors"
    except BundleError:
        raise
    except (ValueError, TypeError, OSError, SafetensorError) as exc:
        raise BundleError(f"Invalid MLX model bundle {root}: {exc}") from exc
