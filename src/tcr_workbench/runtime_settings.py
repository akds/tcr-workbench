"""Explicit local runtime defaults so routine biology commands need fewer flags."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .backends.precision_contract import Precision
from .model_registry import normalize_device, resolve_model, validate_backend

DEFAULTS = {"precision": "float32", "model": "DecoderTCR-ESMC_300M", "device": "cpu", "batch_size": 1,
            "token_budget": 4096, "cache_bytes": 64 * 1024 * 1024}
KEYS = ("precision", "decoder_dir", "python_executable", "model", "device", "checkpoint", "mlx_python",
        "batch_size", "token_budget", "cache_bytes", "timeout")
PATHS = {"decoder_dir", "python_executable", "checkpoint", "mlx_python"}


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate runtime setting: {key}")
        result[key] = value
    return result


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decoder_dir: str
    python_executable: str
    model: str = "DecoderTCR-ESMC_300M"
    device: str = "cpu"
    precision: Precision = "float32"
    checkpoint: Optional[str] = None
    mlx_python: Optional[str] = None
    batch_size: int = Field(default=1, ge=1, le=128)
    token_budget: int = Field(default=4096, ge=3, le=262144)
    cache_bytes: int = Field(default=64 * 1024 * 1024, ge=0, le=1024**3)
    timeout: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


def resolve_settings(args):
    """Explicit flags override a named config, or the local configured defaults."""
    supplied = getattr(args, "config", None)
    config = Path(supplied) if supplied is not None else Path(".tcr-workbench.json")
    values = dict(DEFAULTS)
    if supplied is not None or config.exists():
        loaded = RuntimeSettings.model_validate(json.loads(
            config.read_text(), object_pairs_hook=_unique_keys)).model_dump()
        for key in PATHS:
            if loaded.get(key) and not Path(loaded[key]).is_absolute():
                # Interpreter paths may be virtual-environment symlinks. Resolving
                # their target would run the base Python and lose its packages.
                loaded[key] = str((config.resolve().parent / loaded[key]).absolute())
        values.update(loaded)
    for key in KEYS:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = str(value) if key in PATHS else value
    for required in ("decoder_dir", "python_executable"):
        if not values.get(required):
            raise ValueError("DecoderTCR needs --decoder-dir and --python, or saved defaults from configure")
    options = RuntimeSettings.model_validate(values).model_dump()
    options["device"] = normalize_device(options["device"])
    options["model"] = resolve_model(options["model"]).name
    if options["device"] != "apple" and getattr(args, "mlx_python", None) is None:
        options["mlx_python"] = None
    if (options["device"] != "apple" and options.get("checkpoint")
            and Path(options["checkpoint"]).is_dir()):
        raise ValueError("Saved checkpoint is an Apple bundle; supply --checkpoint with the matching PyTorch file for CPU/CUDA")
    validate_backend(options["model"], options["device"], options["checkpoint"], options["mlx_python"], options["precision"])
    for key, value in options.items():
        setattr(args, key, value)
    return options


def save_settings(args):
    from .prediction import _json_write
    options = resolve_settings(args)
    output = Path(args.settings_out)
    if output.exists() and not args.force:
        raise ValueError(f"Settings already exist: {output}; use --force to replace them")
    for key in PATHS:
        if options.get(key):
            options[key] = str(Path(options[key]).absolute())
    if not Path(options["decoder_dir"]).is_dir():
        raise ValueError("DecoderTCR directory does not exist")
    for key in ("python_executable", "mlx_python"):
        if options.get(key) and not Path(options[key]).is_file():
            raise ValueError(f"Runtime interpreter does not exist: {options[key]}")
    if options.get("checkpoint") and not Path(options["checkpoint"]).exists():
        raise ValueError("Checkpoint or MLX bundle does not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    _json_write(output, options)
    return options
