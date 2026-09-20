"""Auditable HLA-only DecoderTCR execution on CPU, CUDA, or Apple MLX."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import polars as pl

from ..hla import parse_hla
from ..model_registry import normalize_device, validate_backend
from . import mlx_decoder
from .torch_decoder import execute_torch



def validate_input(path, *, profile=False, species="human"):
    from ..prediction import _read_csv, AA
    frame = _read_csv(path)
    if set(frame.columns) != {"name", "peptide", "hla"} or not frame.height:
        raise ValueError("pMHC input requires only name,peptide,hla and at least one row")
    if (frame["name"].null_count() or frame["name"].n_unique() != frame.height
            or frame.filter(pl.col("name").str.strip_chars() == "").height):
        raise ValueError("pMHC names must be nonempty and unique")
    if frame.filter(pl.col("peptide").is_null()
                    | ~pl.col("peptide").str.contains("^[" + AA + "]+$")).height:
        raise ValueError("pMHC peptides must be normalized standard amino-acid sequences")
    for value in frame["hla"].unique().to_list():
        from ..species import model_mhc
        if species == "mouse":
            if model_mhc(value, species) != value:
                raise ValueError("pMHC input MHC must already be normalized")
            continue
        molecule = parse_hla(value)
        if (molecule.name != value or molecule.kind == "ambiguous"
                or any(allele.suffix or len(allele.fields) < 2 for allele in molecule.alleles)
                or (molecule.kind == "allele" and molecule.alleles[0].locus not in ("A", "B", "C"))):
            raise ValueError("pMHC input requires an explicit expressed class I allele or complete class II molecule")
    if profile not in (False, True, "batch"):
        raise ValueError("Unknown profile mode")
    if profile == "batch":
        if not frame["peptide"].str.len_chars().is_between(1, 50).all():
            raise ValueError("Batched pMHC profiles require lengths between 1 and 50")
    elif profile and (frame.height != 1 or not 1 <= len(frame["peptide"][0]) <= 50):
        raise ValueError("pMHC profile requires one context and length between 1 and 50")
    return frame


def validate_output(source, output, model, runtime, *, profile=False):
    from ..prediction import _read_csv, AA
    if profile == "batch":
        from ..background import _probabilities
        from .profile_batch_worker import FIELDS
        frame = pl.read_csv(output)
        if (frame.columns != FIELDS or frame["name"].null_count()
                or not frame.select("name").unique().sort("name").equals(source.select("name").sort("name"))):
            raise ValueError("Batched profile output changed context identities or columns")
        observed = frame.partition_by("name", as_dict=True)
        scored = unresolved = 0
        for row in source.iter_rows(named=True):
            part = observed[(row["name"],)]
            length = len(row["peptide"])
            if (part["hla"].null_count() or part["hla"].unique().to_list() != [row["hla"]]
                    or part["peptide_length"].null_count()
                    or not part["peptide_length"].dtype.is_integer()
                    or part["peptide_length"].unique().to_list() != [length]):
                raise ValueError("Batched profile output changed MHC or peptide length")
            statuses = part["status"].unique().to_list()
            if statuses == ["Unresolved"]:
                if (part.height != 1 or not part["reason"][0]
                        or part.select("position", *AA).null_count().sum_horizontal().item() != 21):
                    raise ValueError("Unresolved batched profile requires one empty row and a reason")
                unresolved += 1
            elif statuses == ["Profiled"]:
                _probabilities(part.select("position", *AA), length)
                scored += 1
            else:
                raise ValueError("Batched profile has inconsistent status")
        if runtime.get("rows") != source.height or runtime.get("scored") != scored or runtime.get("unresolved") != unresolved:
            raise ValueError("Batched profile runtime coverage differs from output")
        return
    if profile:
        frame = pl.read_csv(output)
        if frame.columns != ["position"] + list(AA):
            raise ValueError("pMHC profile has unexpected columns")
        if runtime.get("status") == "Unresolved":
            if frame.height or not runtime.get("reason"):
                raise ValueError("unresolved pMHC profile requires an empty profile and reason")
            return
        length = len(source["peptide"][0])
        values = frame.select(list(AA)).to_numpy()
        if (frame["position"].to_list() != list(range(1, length + 1))
                or not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("pMHC profile is not a normalized length-matched AA20 distribution")
        return
    scored = _read_csv(output)
    required = ["name", "peptide", "hla", "HLA_a", "HLA_b", "ok", "hla_reason", "inference_reason", "pll_" + model]
    if any(key not in scored.columns for key in required) or scored.height != source.height:
        raise ValueError("pMHC output must retain every input row and required audit columns")
    if not scored.select("name", "peptide", "hla").equals(source.select("name", "peptide", "hla")):
        raise ValueError("pMHC output changed row identity, ordering or components")
    values = scored["pll_" + model].cast(pl.Float64, strict=True).to_numpy()
    good = scored["ok"].str.to_lowercase().is_in(["true", "1"]).to_numpy()
    if (not scored["ok"].str.to_lowercase().is_in(["true", "false", "1", "0"]).all()
            or not np.array_equal(np.isfinite(values), good)):
        raise ValueError("pMHC output score/failure status is inconsistent")
    if runtime.get("scored") != int(good.sum()) or runtime.get("rows") != source.height:
        raise ValueError("pMHC worker row counts do not match the output")



def run_pmhc_backend(input_path, output_path, *, decoder_dir, python_executable,
                     model="DecoderTCR-ESMC_300M", checkpoint=None, device="cpu", mlx_python=None, precision="float32",
                     batch_size=1, token_budget=4096, cache_bytes=67108864,
                     timeout=None, force=False, profile=False, species="human", mhc_reference=None):
    from ..prediction import _model_fingerprint, file_sha256, _json_write, _sidecar
    device = normalize_device(device)
    model = validate_backend(model, device, checkpoint, mlx_python, precision).name
    mlx_decoder.validate_options(batch_size, token_budget, cache_bytes)
    from ..species import biological_fingerprint, verify_biological_fingerprint
    biology = biological_fingerprint(species, mhc_reference)
    frame = validate_input(input_path, profile=profile, species=species)
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output:
        raise ValueError("input and output paths must differ")
    if device == "apple":
        fingerprint = mlx_decoder.fingerprint(decoder_dir, python_executable, model,
            checkpoint, mlx_python, batch_size=batch_size, token_budget=token_budget, cache_bytes=cache_bytes, precision=precision)
    else:
        fingerprint = _model_fingerprint(decoder_dir, python_executable, model, checkpoint=checkpoint)
        fingerprint.update(backend="torch", device=device, dtype="float32", cache_bytes=cache_bytes)
        if checkpoint is not None:
            fingerprint["checkpoint_path"] = str(Path(checkpoint).resolve())
    fingerprint.update(**biology, precision=precision, approximate=precision == "float16", schema_version=1, context_type="pmhc", mode="profiles" if profile == "batch" else "profile" if profile else "scores",
                       input_sha256=file_sha256(source),
                       pmhc_adapter_sha256=hashlib.sha256(b"".join(path.read_bytes() for path in
                           sorted(Path(__file__).parent.glob("*.py")))).hexdigest())
    if output.exists() and not force:
        previous = json.loads(_sidecar(output).read_text()) if _sidecar(output).is_file() else {}
        if (all(previous.get(key) == value for key, value in fingerprint.items())
                and previous.get("output_sha256") == file_sha256(output)):
            validate_output(frame, output, model, previous, profile=profile)
            return {**previous, "cache_hit": True}
        raise ValueError("existing pMHC output has different provenance; use force to replace")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pmhc-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "output.csv"
        executor = mlx_decoder.execute if device == "apple" else execute_torch
        runtime = executor(source, candidate, temporary, fingerprint, timeout=timeout, profile=profile)
        verify_biological_fingerprint(fingerprint)
        validate_output(frame, candidate, model, runtime, profile=profile)
        if file_sha256(source) != fingerprint["input_sha256"]:
            raise ValueError("pMHC input changed during inference")
        fingerprint.update(runtime)
        fingerprint["output_sha256"] = file_sha256(candidate)
        candidate.replace(output)
    fingerprint["interpretation"] = "HLA-conditioned DecoderTCR token likelihood; not binding affinity or probability"
    _json_write(_sidecar(output), fingerprint)
    return {**fingerprint, "cache_hit": False}
