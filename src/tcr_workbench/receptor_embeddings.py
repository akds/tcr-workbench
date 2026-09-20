"""Audited experimental receptor embeddings without peptide or MHC inputs.

Framework imports stay in an isolated worker. The representation is the final
postnorm residue mean for each reconstructed chain, normalized independently,
then concatenated alpha/beta and normalized again. It is not a binding score.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .decoder_pmhc import PMHCOptions
from .model_registry import normalize_device, validate_backend, resolve_model
from .models import InputError
from .prediction import (_ReceptorComponents, _annotate_gene_resolution, _execute,
                         _model_fingerprint, file_sha256)
from .report import output_bundle, source_digest, write_frame, write_json


GENES = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b")
INPUT_COLUMNS = ("embedding_id", "species", *GENES)
FEATURE_VERSION = "independent-chain-postnorm-mean-l2-v1"
REPRESENTATION = "decodertcr-chain-mean-v1"
WIDTHS = {"DecoderTCR-ESMC_300M": 960, "DecoderTCR-ESMC_600M": 1152,
          "DecoderTCR-ESMC_6B": 2560}
HASH_FIELDS = ("alpha_sha256", "beta_sha256", "reconstructed_receptor_sha256")
ARTIFACTS = ("receptor_embeddings.npy", "embedding_ids.json", "embedding_audit.csv",
             "embedding_audit.parquet", "embedding_runtime.json", "embedding_input.csv",
             "embedding_input.parquet")


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    ids: list[str]
    audit: pl.DataFrame
    metadata: dict


class EmbeddingRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    backend: Literal["torch", "mlx"]
    device: str
    model: str
    source_checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_version: Literal["independent-chain-postnorm-mean-l2-v1"]
    precision: Literal["float32"]
    math_precision: Literal["float32-no-tf32"]
    tokenizer_variant: Literal["decodertcr-esm1b"]
    model_load_seconds: float = Field(ge=0, allow_inf_nan=False)
    inference_seconds: float = Field(ge=0, allow_inf_nan=False)
    rows: int = Field(ge=0)
    embedded: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    unique_chains: int = Field(ge=0)
    chain_dimension: int = Field(ge=1)
    vector_dimension: int = Field(ge=2)
    batches: int = Field(ge=0)
    accelerator_peak_memory_bytes: Optional[int] = Field(default=None, ge=0)


def _options(frame: pl.DataFrame, kwargs: dict) -> PMHCOptions:
    if not isinstance(frame, pl.DataFrame) or set(frame.columns) != set(INPUT_COLUMNS):
        raise InputError("Receptor embeddings require exactly embedding_id,species,trav,traj,cdr3a,trbv,trbj,cdr3b; peptide/MHC columns are not allowed")
    if frame.is_empty():
        raise InputError("Receptor embeddings require at least one explicitly identified receptor")
    if any(frame[field].dtype not in (pl.String, pl.Null) for field in INPUT_COLUMNS):
        raise InputError("Receptor embedding input columns must contain strings")
    if frame["embedding_id"].null_count() or frame["embedding_id"].n_unique() != frame.height:
        raise InputError("Receptor embedding_id values must be non-null and unique")
    if any(not value.strip() or any(character in value for character in "\r\n\x00")
           for value in frame["embedding_id"].to_list()):
        raise InputError("Receptor embedding_id values must be nonempty single-line strings")
    species = frame["species"].unique().to_list()
    if len(species) > 1 or any(value not in ("human", "mouse") for value in species):
        raise InputError("One explicit human or mouse species is required per embedding call")
    try:
        options = PMHCOptions.model_validate({**kwargs, "species": kwargs.get("species", species[0] if species else "human")})
        if species and species[0] != options.species:
            raise ValueError("Embedding species differs from configured species")
        if options.mhc_reference is not None:
            raise ValueError("Receptor embeddings do not use MHC references")
        if options.precision != "float32":
            raise ValueError("Experimental receptor embeddings currently require float32 precision")
        device = normalize_device(options.device)
        model = validate_backend(options.model, device, options.checkpoint, options.mlx_python, options.precision)
        if model.backbone != "esmc":
            raise ValueError("Receptor embeddings support DecoderTCR ESM-C 300M, 600M and 6B only")
        return PMHCOptions.model_validate(options.model_dump() | {"model": model.name, "device": device})
    except (ValueError, TypeError) as exc:
        raise InputError(f"Invalid receptor embedding options: {exc}") from exc


def _input_frame(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for row in frame.select(INPUT_COLUMNS).iter_rows(named=True):
        try:
            normalized = _ReceptorComponents.model_validate(row).model_dump()
            rows.append({**row, **normalized, "status": "Pending", "reason": ""})
        except ValidationError as exc:
            # Pydantic error messages retain the specific missing/ambiguous
            # gene or anchor problem; unsuccessful receptors remain auditable.
            reason = "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors())
            rows.append({**row, "status": "Unresolved", "reason": reason})
    return pl.DataFrame(rows, schema={field: pl.String for field in (*INPUT_COLUMNS, "status", "reason")})


def _fingerprint(options: PMHCOptions) -> dict:
    if options.device == "apple":
        from .backends.mlx_decoder import fingerprint
        result = fingerprint(options.decoder_dir, options.python_executable, options.model,
            options.checkpoint, options.mlx_python, batch_size=options.batch_size,
            token_budget=options.token_budget, cache_bytes=options.cache_bytes, precision=options.precision)
    else:
        checkpoint = (Path(options.checkpoint).resolve() if options.checkpoint is not None else
                      Path(options.decoder_dir).resolve() / resolve_model(options.model).checkpoint)
        result = _model_fingerprint(options.decoder_dir, options.python_executable,
                                    options.model, checkpoint=checkpoint)
        result.update(checkpoint_path=str(checkpoint), backend="torch", device=options.device)
    return {**result, "context_type": "independent-receptor-chains", "species": options.species,
            "representation": REPRESENTATION, "feature_version": FEATURE_VERSION,
            "precision": "float32", "math_precision": "float32-no-tf32"}


def _run_worker(input_path: Path, staging: Path, options: PMHCOptions, fingerprint: dict) -> None:
    worker = Path(__file__).resolve().parent / "backends" / "receptor_embedding_worker.py"
    root = Path(fingerprint["decoder_dir"])
    reconstructed = staging / ".embedding-reconstructed.csv"
    _execute([fingerprint["python_executable"], str(worker), "--phase", "reconstruct",
              "--input", str(input_path), "--output", str(reconstructed), "--species", options.species],
             root, options.timeout)
    if options.device == "apple":
        executable = fingerprint["mlx_python_executable"]
        checkpoint = fingerprint["bundle_path"]
    else:
        executable = fingerprint["python_executable"]
        checkpoint = fingerprint["checkpoint_path"]
    _execute([executable, "-I", str(worker), "--phase", "infer", "--input", str(reconstructed),
              "--output", str(staging), "--species", options.species, "--model", options.model,
              "--device", options.device, "--checkpoint", checkpoint,
              "--checkpoint-sha256", fingerprint["checkpoint_sha256"],
              "--batch-size", str(options.batch_size), "--token-budget", str(options.token_budget)],
             root, options.timeout)
    reconstructed.unlink()


def _validate_vectors(vectors: np.ndarray, rows: int, width: int) -> None:
    if vectors.shape != (rows, 2 * width) or vectors.dtype != np.float32:
        raise InputError("Embedding vectors have invalid dimensions or dtype")
    for start in range(0, rows, 1024):
        chunk = vectors[start:start + 1024]
        if not np.isfinite(chunk).all() or not np.allclose(np.linalg.norm(chunk, axis=1), 1, atol=2e-5, rtol=0):
            raise InputError("Embedding vectors must be finite and L2-normalized")
        if any(not np.allclose(np.linalg.norm(part, axis=1), 2 ** -.5, atol=2e-5, rtol=0)
               for part in (chunk[:, :width], chunk[:, width:])):
            raise InputError("Embedding vectors must give equal norm to alpha and beta chains")


def _validate(staging: Path, expected: pl.DataFrame, options: PMHCOptions, fingerprint: dict):
    try:
        runtime = EmbeddingRuntime.model_validate_json((staging / "embedding_runtime.json").read_text()).model_dump()
        audit = pl.read_csv(staging / "embedding_audit.csv", infer_schema=False, missing_utf8_is_empty_string=True)
        ids = json.loads((staging / "embedding_ids.json").read_text())
        vectors = np.load(staging / "receptor_embeddings.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise InputError(f"Invalid receptor embedding worker output: {exc}") from exc
    width = WIDTHS[options.model]
    if (runtime["model"] != options.model or runtime["device"] != options.device
            or runtime["backend"] != ("mlx" if options.device == "apple" else "torch")
            or runtime["source_checkpoint_sha256"] != fingerprint["checkpoint_sha256"]
            or runtime["chain_dimension"] != width or runtime["vector_dimension"] != 2 * width):
        raise InputError("Embedding worker changed model, device, checkpoint or vector dimensions")
    required = {*INPUT_COLUMNS, "status", "reason", "TRAV", "TRAJ", "TRBV", "TRBJ", *HASH_FIELDS}
    if not required.issubset(audit.columns) or audit.height != expected.height:
        raise InputError("Embedding worker audit has missing columns or changed row coverage")
    if not audit.select(INPUT_COLUMNS).equals(expected.select(INPUT_COLUMNS).fill_null("")):
        raise InputError("Embedding worker changed receptor identities, species or input components")
    if (~audit["status"].is_in(["Embedded", "Unresolved"])).any():
        raise InputError("Embedding worker returned unknown statuses")
    embedded = audit.filter(pl.col("status") == "Embedded")
    unresolved = audit.filter(pl.col("status") == "Unresolved")
    if unresolved.filter(pl.col("reason").str.strip_chars() == "").height:
        raise InputError("Unresolved embedding receptors must retain an explicit reason")
    if audit.filter((expected["status"] == "Unresolved") & (pl.col("status") != "Unresolved")).height:
        raise InputError("Embedding worker inferred components for an ineligible receptor")
    if any(embedded.filter(~pl.col(field).str.contains(r"^[a-f0-9]{64}$")).height for field in HASH_FIELDS):
        raise InputError("Successful receptor embeddings require full-chain identity hashes")
    if (not isinstance(ids, list) or ids != embedded["embedding_id"].to_list()
            or vectors.shape != (embedded.height, 2 * width) or vectors.dtype != np.float32
            or runtime["rows"] != audit.height or runtime["embedded"] != embedded.height
            or runtime["unresolved"] != unresolved.height
            or runtime["unique_chains"] != len(set(embedded["alpha_sha256"]) | set(embedded["beta_sha256"]))
            or (runtime["batches"] == 0) != (embedded.height == 0)):
        raise InputError("Embedding vectors, identities or runtime row coverage disagree")
    _validate_vectors(vectors, len(ids), width)
    return vectors, ids, audit, runtime


def embed_receptors(frame: pl.DataFrame, output_dir, **decoder_kwargs) -> EmbeddingResult:
    """Reconstruct and embed a strictly identified single-species receptor table.

    Invalid components and reconstruction failures remain Unresolved audit rows.
    Repeated full chains are inferred once. No MHC/peptide conditioning occurs.
    The destination must be new; all artifacts publish together after validation.
    """
    options = _options(frame, decoder_kwargs)
    source = source_digest()
    normalized = _input_frame(frame)
    provenance = _fingerprint(options)
    destination = Path(output_dir).resolve()
    with output_bundle(destination) as staging:
        write_frame(frame.select(INPUT_COLUMNS), staging / "embedding_input.csv")
        worker_input = staging / ".embedding-input.csv"
        normalized.write_csv(worker_input)
        _run_worker(worker_input, staging, options, provenance)
        worker_input.unlink()
        vectors, ids, audit, runtime = _validate(staging, normalized, options, provenance)
        del vectors
        audit = _annotate_gene_resolution(audit.with_columns(
            pl.when(pl.col("status") == "Embedded").then(pl.lit("ModelHypothesis"))
            .otherwise(pl.col("status")).alias("status"))).with_columns(
            pl.when(pl.col("status") == "ModelHypothesis").then(pl.lit("Embedded"))
            .otherwise(pl.col("status")).alias("status"))
        write_frame(audit, staging / "embedding_audit.csv")
        if source_digest() != source or _fingerprint(options) != provenance:
            raise InputError("Embedding source, checkpoint, environment or germlines changed during inference")
        metadata = {"schema_version": 1, "task": "receptor-embeddings", "experimental": True,
                    "representation": REPRESENTATION, "feature_version": FEATURE_VERSION,
                    "species": options.species, "model": options.model, "source_sha256": source,
                    "precision": "float32",
                    "dimension": runtime["vector_dimension"], "chain_dimension": runtime["chain_dimension"],
                    "pooling": "FP32 mean of final postnorm residues; CLS/EOS/PAD excluded; independent-chain L2, alpha/beta concatenation, final L2",
                    "interpretation": "Uncalibrated receptor representation similarity; not antigen recognition or binding probability",
                    "backend": provenance, "runtime": runtime,
                    "parameters": {"batch_size": options.batch_size, "token_budget": options.token_budget},
                    "outputs": {name: {"sha256": file_sha256(staging / name), "size_bytes": (staging / name).stat().st_size}
                                for name in ARTIFACTS}}
        write_json(staging / "manifest.json", metadata)
    return EmbeddingResult(np.load(destination / "receptor_embeddings.npy", mmap_mode="r", allow_pickle=False),
                           ids, audit, metadata)


def load_receptor_embeddings(output_dir, *, manifest_name="manifest.json", audit_name="embedding_audit.csv") -> EmbeddingResult:
    """Verify a saved bundle, including a flat screen export with renamed audit."""
    root = Path(output_dir)
    if any(not isinstance(name, str) or Path(name).name != name for name in (manifest_name, audit_name)):
        raise InputError("Embedding manifest/audit overrides must be filenames within the bundle")
    metadata = json.loads((root / manifest_name).read_text())
    if (metadata.get("schema_version") != 1 or metadata.get("representation") != REPRESENTATION
            or metadata.get("feature_version") != FEATURE_VERSION or set(metadata.get("outputs", {})) != set(ARTIFACTS)):
        raise InputError("Unsupported receptor embedding manifest")
    runtime = EmbeddingRuntime.model_validate(metadata["runtime"])
    if (metadata.get("model") not in WIDTHS or metadata.get("precision") != "float32"
            or metadata.get("chain_dimension") != WIDTHS[metadata["model"]]
            or metadata.get("dimension") != 2 * WIDTHS[metadata["model"]]
            or runtime.model != metadata["model"] or runtime.vector_dimension != metadata["dimension"]):
        raise InputError("Saved embedding architecture or numerical mode is inconsistent")
    for name, identity in metadata["outputs"].items():
        actual = audit_name if name == "embedding_audit.csv" else name
        if name == "embedding_audit.parquet" and audit_name != "embedding_audit.csv":
            actual = str(Path(audit_name).with_suffix(".parquet"))
        path = root / actual
        if path.stat().st_size != identity["size_bytes"] or file_sha256(path) != identity["sha256"]:
            raise InputError(f"Embedding artifact changed: {actual}")
    ids = json.loads((root / "embedding_ids.json").read_text())
    vectors = np.load(root / "receptor_embeddings.npy", mmap_mode="r", allow_pickle=False)
    audit = pl.read_csv(root / audit_name, infer_schema=False, missing_utf8_is_empty_string=True)
    if (ids != audit.filter(pl.col("status") == "Embedded")["embedding_id"].to_list()
            or len(set(ids)) != len(ids) or runtime.rows != audit.height or runtime.embedded != len(ids)):
        raise InputError("Saved embedding identities or dimensions disagree")
    _validate_vectors(vectors, len(ids), metadata["chain_dimension"])
    return EmbeddingResult(vectors, ids, audit, metadata)
