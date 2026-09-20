"""Reproducible random peptides for matched, explicitly uncalibrated comparisons."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

AA = "ACDEFGHIKLMNPQRSTVWY"


class BackgroundOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    peptides: int = Field(default=1000, ge=0, le=100000)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    mode: Literal["uniform", "mhc-profile"] = "uniform"


def background_groups(contexts: pl.DataFrame, *, peptides: int = 1000, seed: int = 0,
                      mode: str = "uniform", max_rows: int = 100000) -> tuple[pl.DataFrame, BackgroundOptions]:
    """Validate the complete allocation before generation or model execution."""
    options = BackgroundOptions(peptides=peptides, seed=seed, mode=mode)
    keys = [key for key in ("receptor_id", "hla", "peptide_length") if key in contexts.columns]
    if not {"hla", "peptide_length"} <= set(keys):
        raise ValueError("Background requires MHC context and peptide length")
    groups = contexts.select(keys).unique().sort(keys)
    if groups.height and (groups.null_count().sum_horizontal().sum()
            or not groups["peptide_length"].dtype.is_integer()
            or not groups["peptide_length"].is_between(1, 50).all()):
        raise ValueError("Background contexts must be complete with integer lengths 1–50")
    count = groups.height * options.peptides
    if count > max_rows:
        raise ValueError(f"Random background would require {count:,} peptides (limit {max_rows:,}). "
                         "Reduce --background-peptides or split the panel; use 0 to disable.")
    return groups, options


def _probabilities(profile: pl.DataFrame, length: int) -> np.ndarray:
    if (not {"position", *AA} <= set(profile.columns) or profile.height != length
            or not profile["position"].dtype.is_integer()
            or profile["position"].null_count()
            or profile["position"].sort().to_list() != list(range(1, length + 1))
            or any(not profile.schema[aa].is_numeric() for aa in AA)):
        raise ValueError("Background profile requires complete one-based positions and numeric AA20 columns")
    values = profile.sort("position").select(list(AA)).to_numpy().astype(np.float64)
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError("Background profile probabilities must be finite, nonnegative and sum to one")
    # Only correct floating-point row-sum rounding; do not floor, temper or truncate.
    return values / values.sum(axis=1, keepdims=True)


def random_panel(contexts: pl.DataFrame, *, peptides: int = 1000, seed: int = 0,
                 max_rows: int = 100000, mode: str = "uniform",
                 profiles: dict[tuple[str, int], pl.DataFrame] | None = None,
                 profile_failures: dict[tuple[str, int], str] | None = None) -> tuple[pl.DataFrame, dict]:
    """Sample independent AA20 positions with replacement per context.

    Contexts must include hla and peptide_length, and may include receptor_id.
    Stable per-context seeds make unrelated panel reordering/additions harmless.
    This is a synthetic comparator, never a set of known nonbinders.
    """
    groups, options = background_groups(contexts, peptides=peptides, seed=seed, mode=mode, max_rows=max_rows)
    keys = groups.columns
    sampler = "uniform-aa20-v1" if options.mode == "uniform" else "mhc-profile-aa20-v1"
    alphabet = np.frombuffer(AA.encode("ascii"), dtype="S1")
    frames = []
    failures = []
    distributions = {}
    for context in groups.iter_rows(named=True):
        if not options.peptides:
            break
        identity = json.dumps(context, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{sampler}:{options.seed}:{identity}".encode()).digest()
        length = context["peptide_length"]
        if options.mode == "uniform":
            rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))
            residues = alphabet[rng.integers(0, 20, size=(options.peptides, length))]
        else:
            key = (context["hla"], length)
            if key in (profile_failures or {}):
                reason = profile_failures[key]
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("Failed background profiles require an explicit reason")
                if key in (profiles or {}):
                    raise ValueError("Background profile cannot be both available and failed")
                failures.append({**context, "reason": reason, "requested_peptides": options.peptides})
                continue
            if key not in (profiles or {}):
                raise ValueError("MHC-profile background requires a profile or explicit failure for every context")
            if key not in distributions:
                distributions[key] = _probabilities(profiles[key], length)
            # Reuse identical draws across query receptors with the same MHC/length.
            generator_identity = json.dumps({"hla": key[0], "peptide_length": length},
                                            sort_keys=True, separators=(",", ":"))
            generator_digest = hashlib.sha256(f"{sampler}:{options.seed}:{generator_identity}".encode()).digest()
            rng = np.random.Generator(np.random.PCG64(int.from_bytes(generator_digest[:16], "big")))
            indices = np.empty((options.peptides, length), dtype=np.uint8)
            cdf = distributions[key].cumsum(axis=1)
            cdf[:, -1] = 1.0
            for position in range(length):
                indices[:, position] = np.searchsorted(cdf[position], rng.random(options.peptides), side="right")
            residues = alphabet[indices]
        sequences = residues.view(f"S{length}").reshape(-1).astype(str).tolist()
        prefix = digest.hex()[:24]
        frames.append(pl.DataFrame({
            **{key: [value] * options.peptides for key, value in context.items()},
            "name": [f"background_{prefix}_{i + 1}" for i in range(options.peptides)],
            "peptide": sequences,
        }))
    schema = {**{key: groups.schema[key] for key in keys}, "name": pl.String, "peptide": pl.String}
    frame = pl.concat(frames).cast(schema) if frames else pl.DataFrame(schema=schema)
    return frame, {
        "mode": options.mode, "sampler": sampler, "alphabet": AA, "numpy_version": np.__version__,
        "rng": "PCG64 with SHA-256-derived per-context seed",
        "seed": options.seed, "peptides_per_context": options.peptides,
        "contexts": groups.height, "requested_rows": groups.height * options.peptides,
        "sampled_rows": frame.height, "failed_contexts": failures,
        "with_replacement": True, "temperature": 1.0,
        "generator_context": "Uniform AA20" if options.mode == "uniform" else "MHC only; both TCR chains absent",
        "sampling": ("Independent uniform draws from 20 amino acids, with replacement; length matched. "
                     if options.mode == "uniform" else
                     "Independent draws from each MHC-only AA20 profile position at temperature 1, with replacement; length matched. "
                     "Profile rows are normalized only for floating-point sum rounding. ")
                    + "Duplicates and overlap with tested peptides are retained as random draws; no top-k or deduplication.",
        "interpretation": "Synthetic comparison, not known nonbinders, proteome frequencies, "
                          "binding probabilities, or a calibrated significance test.",
    }


def background_percentiles(results: pl.DataFrame, background: pl.DataFrame) -> pl.DataFrame:
    """Descriptive upper-tail percentage, including ties; never a binding p-value."""
    keys = [key for key in ("receptor_id", "hla") if key in results.columns] + ["peptide_length"]
    original = results.columns
    candidate = results.with_columns(pl.col("peptide").str.len_chars().alias("peptide_length"))
    reference = background.with_columns(pl.col("peptide").str.len_chars().alias("peptide_length"))
    if reference.height and not set(keys) <= set(reference.columns):
        raise ValueError("Reference scores are missing required comparison context columns")
    tail = np.full(results.height, np.nan, dtype=np.float64)
    counts = np.zeros(results.height, dtype=np.uint32)
    valid = pl.col("score").is_finite() & pl.col("status").is_in(["Scored", "ModelHypothesis"])
    if set(keys) <= set(reference.columns):
        # Sort once per bounded context; searchsorted includes exact ties in the upper tail.
        refs = {key: np.sort(part["score"].to_numpy())
                for key, part in reference.filter(valid).partition_by(keys, as_dict=True).items()}
        for key, part in candidate.with_row_index("_background_row").partition_by(keys, as_dict=True).items():
            values = refs.get(key)
            if values is None or not len(values):
                continue
            counts[part["_background_row"].to_numpy()] = len(values)
            part = part.filter(valid)
            indices = part["_background_row"].to_numpy()
            tail[indices] = 100 * (1 + len(values) - np.searchsorted(values, part["score"].to_numpy(), side="left")) / (len(values) + 1)
    return results.with_columns(pl.Series("background_percentile", tail, nan_to_null=True),
                                pl.Series("background_n", counts)).select(*original, "background_percentile", "background_n")
