"""Isolated CPU peptide–MHC affinity inference with conservative model coverage.

The normal workbench process never imports Torch or TensorFlow. Published model
weights and their optional environment are installed separately, never fetched
by a prediction. MHCnuggets is restricted to exact class-II BA model files; its
nearest-allele and ligand-model substitutions are deliberately not invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Optional, Union

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from .hla import parse_hla
from .models import InputError
from .prediction import _validated_panel

PathLike = Union[str, Path]
FLURRY_SOURCE = "https://github.com/openvax/mhcflurry/tree/v2.2.1"
NUGGETS_SOURCE = "https://pypi.org/project/mhcnuggets/2.4.1/"
NUGGETS_LICENSE = "JHU Academic Software License; non-commercial research only"
RESULT_SCHEMA = {
    "peptide": pl.String, "hla": pl.String, "status": pl.String,
    "score": pl.Float64, "metric": pl.String, "units": pl.String,
    "higher_is_better": pl.Boolean, "predictor": pl.String,
    "predictor_version": pl.String, "model_allele": pl.String,
    "model_sha256": pl.String, "source": pl.String, "license": pl.String,
    "reason": pl.String,
}


class MHCOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    mhcflurry_models: Optional[str] = None
    mhcnuggets_models: Optional[str] = None
    batch_size: int = Field(default=1024, ge=1, le=65536)
    max_pairs: int = Field(default=100000, ge=1, le=10000000)
    cpu_threads: int = Field(default=4, ge=1, le=64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(directory: Path, paths: list[Path]) -> dict:
    files = {str(path.relative_to(directory)): _sha256(path) for path in sorted(paths)}
    if not files:
        raise InputError(f"No model/source files found in {directory}")
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def _adapter_sources() -> dict:
    root = Path(__file__).parent
    return {path.relative_to(root).as_posix(): _sha256(path) for path in sorted(root.rglob("*.py"))}


def _require_version(package: str, expected: str) -> str:
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise InputError(f"Validated {package} version is {expected}; this environment has {actual}")
    return actual


def _model_context(hla: Optional[str]) -> tuple[str, Optional[str], str]:
    if hla is None:
        return "", None, "HLA is missing"
    molecule = parse_hla(hla)
    if molecule.kind == "ambiguous" or any(len(a.fields) < 2 or a.suffix for a in molecule.alleles):
        return "", None, "HLA ambiguity, group or expression suffix is not resolved for prediction"
    alleles = molecule.alleles
    # Higher fields describe synonymous/noncoding differences; report the exact
    # two-field protein-level model name separately from the supplied HLA.
    names = [a.locus + "*" + ":".join(a.fields[:2]) for a in alleles]
    if len(alleles) == 1 and alleles[0].locus in {"A", "B", "C"}:
        return "mhcflurry", "HLA-" + names[0], ""
    if len(alleles) == 1 and not alleles[0].locus.startswith("DRB"):
        return "", None, "Class-II prediction requires an explicit DQ/DP pair or DRB restriction"
    return "mhcnuggets", "HLA-" + "-".join(name.replace("*", "") for name in names), ""


class _FlurryBackend:
    def __init__(self, options: MHCOptions):
        import mhcflurry
        import torch
        from mhcflurry import Class1AffinityPredictor
        from mhcflurry.common import configure_pytorch, get_pytorch_device
        from mhcflurry.downloads import get_default_class1_models_dir

        version = _require_version("mhcflurry", "2.2.1")
        configure_pytorch(backend="cpu", gpu_device_nums=[], num_threads=options.cpu_threads)
        directory = Path(options.mhcflurry_models or get_default_class1_models_dir()).resolve()
        if not (directory / "manifest.csv").is_file():
            raise InputError(f"MHCflurry model directory needs manifest.csv: {directory}")
        self.inventory = _inventory(directory, [p for p in directory.rglob("*") if p.is_file()])
        self.model = Class1AffinityPredictor.load(str(directory))
        self.alleles = frozenset(self.model.supported_alleles)
        self.lengths = self.model.supported_peptide_lengths
        package = Path(mhcflurry.__file__).parent
        self.metadata = {
            "predictor": "MHCflurry", "predictor_version": version,
            "source": FLURRY_SOURCE, "license": "Apache-2.0",
            "implementation": _inventory(package, list(package.rglob("*.py"))),
            "model_directory": str(directory), "model_inventory": self.inventory,
            "model_hash_kind": "SHA-256 of model-directory file-hash inventory (ensemble/model-set identity)",
            "runtime": {"torch": torch.__version__, "device": str(get_pytorch_device())},
        }

    def has_model(self, allele: str) -> bool:
        return allele in self.alleles

    def model_hash(self, allele: str) -> str:
        return self.inventory["sha256"]

    def predict(self, allele: str, peptides: list[str]):
        return self.model.predict(peptides=peptides, allele=allele, throw=True)


class _NuggetsBackend:
    def __init__(self, options: MHCOptions):
        # Legacy Keras is necessary for the published HDF5 recurrent weights.
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import mhcnuggets
        import tensorflow as tf
        from mhcnuggets.src.models import mhcnuggets_lstm

        version = _require_version("mhcnuggets", "2.4.1")
        _require_version("tensorflow", "2.16.2")
        _require_version("tf-keras", "2.16.0")
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(options.cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        package = Path(mhcnuggets.__file__).parent
        self.directory = Path(options.mhcnuggets_models or package / "saves" / "production").resolve()
        if not self.directory.is_dir():
            raise InputError(f"MHCnuggets model directory does not exist: {self.directory}")
        self.model = mhcnuggets_lstm((30, 21))
        self.current_allele = None
        self.lengths = (9, 30)  # Conservative adapter bounds: one class-II core through upstream maximum.
        self.hashes = {}
        self.metadata = {
            "predictor": "MHCnuggets", "predictor_version": version,
            "source": NUGGETS_SOURCE, "license": NUGGETS_LICENSE,
            "implementation": _inventory(package, list((package / "src").glob("*.py"))),
            "model_directory": str(self.directory), "model_files": self.hashes,
            "model_hash_kind": "SHA-256 of the exact requested BA weight file",
            "runtime": {"tensorflow": tf.__version__, "tf-keras": "2.16.0", "device": "CPU"},
            "license_source": "https://github.com/KarchinLab/mhcnuggets/blob/b666fea3a54a1d357efba4ea4d8550ce5dd50aba/LICENSE",
            "model_policy": "Exact requested allele BA weights only; no closest-allele or ligand substitution",
        }

    def _weight(self, allele: str) -> Path:
        return self.directory / (allele + "_BA.h5")

    def has_model(self, allele: str) -> bool:
        return self._weight(allele).is_file()

    def model_hash(self, allele: str) -> str:
        name = self._weight(allele).name
        if name not in self.hashes:
            self.hashes[name] = _sha256(self._weight(allele))
        return self.hashes[name]

    def predict(self, allele: str, peptides: list[str]):
        import numpy as np
        from mhcnuggets.src.dataset import mask_peptides, tensorize_keras

        if self.current_allele != allele:
            # A failed load may have partially replaced weights. Never identify
            # that mixed state as the previously loaded allele on a later call.
            self.current_allele = None
            self.model.load_weights(str(self._weight(allele)))
            self.current_allele = allele
        padded, retained = mask_peptides(peptides, max_len=30)
        if retained != peptides:
            raise InputError("MHCnuggets altered/dropped peptides after boundary validation")
        encoded = tensorize_keras(padded, embed_type="softhot")
        raw = np.asarray(self.model(np.asarray(encoded), training=False)).reshape(-1)
        if len(raw) != len(peptides) or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
            raise InputError("MHCnuggets returned invalid raw affinity regression outputs")
        # Upstream inverse affinity transform; raw sigmoid is NOT a probability.
        return np.power(50000.0, 1.0 - raw.astype(np.float64))


def _chunk(frame: pl.DataFrame, *, reason: str, backend=None, allele=None, scores=None) -> pl.DataFrame:
    metadata = backend.metadata if backend is not None else {}
    values = pl.Series("score", [None] * frame.height if scores is None else scores, dtype=pl.Float64)
    if len(values) != frame.height:
        raise InputError("Predictor returned a different number of scores than peptides")
    valid = values.is_not_null() & values.is_finite() & (values > 0)
    output = frame.select("_pair_index", "peptide", "hla").with_columns(
        values, valid.alias("_valid"),
        pl.lit("predicted_ic50").alias("metric"), pl.lit("nM").alias("units"),
        pl.lit(False).alias("higher_is_better"),
        *[pl.lit(metadata.get(name), dtype=pl.String).alias(name)
          for name in ("predictor", "predictor_version", "source", "license")],
        pl.lit(allele, dtype=pl.String).alias("model_allele"),
        pl.lit(backend.model_hash(allele) if backend is not None and scores is not None else None,
               dtype=pl.String).alias("model_sha256"),
    ).with_columns(
        pl.when(pl.col("_valid")).then(pl.lit("Predicted")).otherwise(pl.lit("Unresolved")).alias("status"),
        pl.when(pl.col("_valid") | pl.lit(scores is None)).then(pl.lit(reason))
        .otherwise(pl.lit("Predictor returned a missing/nonfinite/nonpositive affinity")).alias("reason"),
        pl.when(pl.col("_valid")).then(pl.col("score")).otherwise(None).alias("score"),
    )
    return output.select("_pair_index", *RESULT_SCHEMA)


def _validate_result(frame: pl.DataFrame, panel: pl.DataFrame) -> None:
    if dict(frame.schema) != RESULT_SCHEMA:
        raise InputError("MHC predictor response has an invalid column/dtype contract")
    if not frame.select("peptide", "hla").equals(panel.select("peptide", "hla")):
        raise InputError("MHC predictor response altered, omitted or reordered peptide/HLA pairs")
    predicted = pl.col("status") == "Predicted"
    valid_score = (pl.col("score").is_not_null() & pl.col("score").is_finite() & (pl.col("score") > 0)).fill_null(False)
    required = ("peptide", "status", "metric", "units", "higher_is_better", "reason")
    predicted_fields = ("hla", "predictor", "predictor_version", "source", "license", "model_allele", "model_sha256")
    bad = frame.filter(
        pl.any_horizontal(pl.col(name).is_null() for name in required)
        | ~pl.col("status").is_in(["Predicted", "Unresolved"])
        | (predicted != valid_score).fill_null(True)
        | (~predicted & pl.col("score").is_not_null())
        | (pl.col("metric") != "predicted_ic50") | (pl.col("units") != "nM")
        | pl.col("higher_is_better")
        | (pl.col("reason").str.strip_chars() == "")
        | (predicted & pl.any_horizontal((pl.col(name).is_null() | (pl.col(name).str.strip_chars() == "")) for name in predicted_fields))
        | (predicted & (~pl.col("model_sha256").str.contains(r"^[a-f0-9]{64}$")).fill_null(True))
    )
    if bad.height:
        raise InputError("MHC predictor response has invalid scores, status or model provenance")


def _predict_local(panel: pl.DataFrame, options: MHCOptions) -> tuple[pl.DataFrame, dict]:
    """Worker implementation; backend constructors are replaceable in unit tests."""
    started = time.perf_counter()
    indexed = panel.with_row_index("_pair_index")
    contexts = {hla: _model_context(hla) for hla in panel["hla"].unique()}
    chunks = []
    metadata = {"schema_version": 1, "feature": "peptide_mhc_affinity", "metric": "predicted_ic50", "units": "nM",
                "higher_is_better": False, "calibration": "Not a binding probability or TCR-recognition score",
                "comparability_scope": ["predictor", "predictor_version", "model_allele"],
                "comparability_note": "No calibrated cross-predictor or cross-model ranking; no common binding threshold",
                "models": {}, "python": sys.version, "executable": sys.executable,
                "executable_sha256": _sha256(Path(sys.executable)), "adapter_sources": _adapter_sources(),
                "parameters": options.model_dump(), "backend_counts": {}}
    for hla, (backend_name, allele, reason) in contexts.items():
        if not backend_name:
            rows = indexed.filter(pl.col("hla").is_null() if hla is None else pl.col("hla") == hla)
            chunks.append(_chunk(rows, reason=reason))
    for name, constructor, bounds in (
        ("mhcflurry", _FlurryBackend, (8, 15)), ("mhcnuggets", _NuggetsBackend, (9, 30))
    ):
        names = [hla for hla, context in contexts.items() if context[0] == name]
        counts = {"routed": 0, "predicted": 0, "no_model": 0, "unsupported_length": 0, "failed_score": 0}
        metadata["backend_counts"][name] = counts
        if not names:
            continue
        rows = indexed.filter(pl.col("hla").is_in(names))
        counts["routed"] = rows.height
        in_bounds = pl.col("peptide").str.len_chars().is_between(*bounds)
        invalid = rows.filter(~in_bounds)
        counts["unsupported_length"] += invalid.height
        if invalid.height:
            chunks.append(_chunk(invalid, reason=f"Unsupported peptide length for {name}: adapter accepts {bounds[0]}–{bounds[1]} residues"))
        rows = rows.filter(in_bounds)
        if not rows.height:
            continue
        backend = constructor(options)
        metadata["models"][name] = backend.metadata
        backend.metadata["counts"] = counts
        for hla, group in rows.partition_by("hla", as_dict=True).items():
            allele = contexts[hla[0]][1]
            if not backend.has_model(allele):
                counts["no_model"] += group.height
                reason = "No exact supported allele/model; no nearest-allele substitution"
                if name == "mhcnuggets" and allele.startswith("HLA-DRA"):
                    reason = "No exact DR alpha/beta model; supplied DRA context is not replaced by a DRB-only model"
                chunks.append(_chunk(group, reason=reason, backend=backend, allele=allele))
                continue
            supported_length = pl.col("peptide").str.len_chars().is_between(*backend.lengths)
            unavailable = group.filter(~supported_length)
            counts["unsupported_length"] += unavailable.height
            if unavailable.height:
                chunks.append(_chunk(unavailable, reason="Peptide length outside installed model support", backend=backend, allele=allele))
            supported = group.filter(supported_length)
            score_parts = []
            for batch in supported.iter_slices(options.batch_size):
                scores = pl.Series("score", backend.predict(allele, batch["peptide"].to_list()), dtype=pl.Float64)
                if len(scores) != batch.height:
                    raise InputError("Predictor returned a different number of scores than peptides")
                score_parts.append(scores)
            if score_parts:
                # Keep only compact numeric buffers between inference batches.
                # Constant provenance strings are constructed once per HLA group.
                chunk = _chunk(supported, reason="Predicted peptide–MHC affinity; requires experimental validation; not TCR recognition",
                               backend=backend, allele=allele, scores=pl.concat(score_parts))
                n_predicted = chunk.filter(pl.col("status") == "Predicted").height
                counts["predicted"] += n_predicted
                counts["failed_score"] += chunk.height - n_predicted
                chunks.append(chunk)
    frame = (pl.concat(chunks).sort("_pair_index").drop("_pair_index")
             if chunks else pl.DataFrame(schema=RESULT_SCHEMA))
    _validate_result(frame, panel)
    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    metadata["unique_pairs"] = panel.height
    metadata["predicted_pairs"] = frame.filter(pl.col("status") == "Predicted").height
    return frame, metadata


def normalize_peptide_panel(panel: pl.DataFrame) -> pl.DataFrame:
    """Return unique normalized peptide/HLA keys used by prediction results.

    Peptides are stripped/uppercased; HLA names are canonicalized; empty HLA is
    null. Normalize raw panel keys with this function before joining results.
    Missing/nonstandard peptide sequences and malformed HLA syntax fail loudly.
    """
    return _validated_panel(panel)


def predict_peptide_mhc(
    panel: pl.DataFrame, *, python: PathLike,
    mhcflurry_models: Optional[PathLike] = None, mhcnuggets_models: Optional[PathLike] = None,
    batch_size: int = 1024, timeout: float = 600, max_pairs: int = 100000, cpu_threads: int = 4,
) -> tuple[pl.DataFrame, dict]:
    """Predict unique panel pairs using the separately installed, pinned CPU environment.

    Returns typed ``Predicted``/``Unresolved`` evidence and JSON-serializable model
    provenance. Malformed input, unavailable installations and inference errors
    fail loudly. Unsupported biology is retained as an explicit unresolved row.
    """
    options = MHCOptions(
        mhcflurry_models=str(Path(mhcflurry_models).resolve()) if mhcflurry_models is not None else None,
        mhcnuggets_models=str(Path(mhcnuggets_models).resolve()) if mhcnuggets_models is not None else None,
        batch_size=batch_size, max_pairs=max_pairs, cpu_threads=cpu_threads,
    )
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise InputError("MHC prediction timeout must be finite and positive")
    if panel.height > options.max_pairs:
        raise InputError(f"Panel exceeds {options.max_pairs} rows; partition the panel or increase max_pairs")
    normalized = normalize_peptide_panel(panel)
    executable = Path(python).absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise InputError(f"MHC Python executable is missing or not executable: {executable}")
    with tempfile.TemporaryDirectory(prefix="tcr-mhc-") as temporary:
        root = Path(temporary)
        input_path, output_path = root / "panel.parquet", root / "evidence.parquet"
        normalized.write_parquet(input_path)
        request = {"options": options.model_dump(), "input_sha256": _sha256(input_path),
                   "adapter_sources": _adapter_sources(), "executable_sha256": _sha256(executable)}
        (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_USE_LEGACY_KERAS="1", TF_CPP_MIN_LOG_LEVEL="2")
        for variable in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
            environment.pop(variable, None)
        environment["PYTHONNOUSERSITE"] = "1"
        try:
            with (root / "worker.log").open("w") as log:
                completed = subprocess.run(
                    [str(executable), "-I", "-m", "tcr_workbench.mhc_predict", "--worker", str(root)],
                    env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise InputError(f"MHC predictor exceeded its {timeout}s timeout") from exc
        if completed.returncode:
            detail = (root / "worker.log").read_text(errors="replace")[-4000:]
            raise InputError(f"MHC predictor failed in its optional environment:\n{detail}")
        try:
            frame = pl.read_parquet(output_path)
            metadata = json.loads((root / "metadata.json").read_text())
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            raise InputError(f"MHC predictor did not return readable evidence and provenance: {exc}") from exc
        _validate_result(frame, normalized)
        if metadata.get("input_sha256") != request["input_sha256"] or metadata.get("output_sha256") != _sha256(output_path):
            raise InputError("MHC predictor provenance hashes do not match input/output")
        if (metadata.get("adapter_sources") != request["adapter_sources"]
                or metadata.get("executable_sha256") != request["executable_sha256"]):
            raise InputError("MHC worker adapter/source or Python executable differs from the requested implementation")
        metadata["input_rows"] = panel.height
        return frame, metadata


def _worker(root: Path) -> None:
    request = json.loads((root / "request.json").read_text())
    options = MHCOptions.model_validate(request["options"])
    input_path = root / "panel.parquet"
    if _sha256(input_path) != request["input_sha256"]:
        raise InputError("MHC worker input changed before inference")
    if _adapter_sources() != request["adapter_sources"]:
        raise InputError("Optional MHC environment has a different workbench implementation; reinstall current workbench there")
    panel = pl.read_parquet(input_path)
    if panel.height > options.max_pairs:
        raise InputError("MHC worker panel exceeds configured pair limit")
    frame, metadata = _predict_local(_validated_panel(panel), options)
    frame.write_parquet(root / "evidence.parquet")
    metadata.update(input_sha256=request["input_sha256"], output_sha256=_sha256(root / "evidence.parquet"))
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    _worker(parser.parse_args().worker)
