"""Columnar obs-only bridge for the supplied AnnData benchmark.

Only selected obs datasets are read. X, raw, layers, obsm and embeddings are
never accessed. Experimental labels remain separate from model input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from .prediction import GENES, PathLike, export_decoder_input, file_sha256


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install tcr-workbench[dataset] to read h5ad obs metadata") from exc
    return h5py


def _text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_obs_column(node: Any) -> pl.Series:
    h5py = _h5py()
    if isinstance(node, h5py.Dataset):
        if node.ndim != 1:
            raise ValueError(f"obs column {node.name} must be one-dimensional")
        if h5py.check_string_dtype(node.dtype) is not None:
            # Bound temporary Python string objects; retain only native columnar chunks.
            chunks = [pl.Series(node.asstr()[start:start + 65536], dtype=pl.String)
                      for start in range(0, node.shape[0], 65536)]
            return pl.concat(chunks, rechunk=False) if chunks else pl.Series([], dtype=pl.String)
        return pl.Series(node[:])
    encoding = _text(node.attrs.get("encoding-type", ""))
    if encoding == "categorical" and set(node.keys()) >= {"categories", "codes"}:
        categories = _read_obs_column(node["categories"])
        codes = np.asarray(node["codes"][:])
        if codes.ndim != 1 or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError(f"invalid categorical codes in {node.name}")
        if ((codes < -1) | (codes >= len(categories))).any():
            raise ValueError(f"categorical code outside category range in {node.name}")
        indices = pl.Series(codes).cast(pl.Int64).set(pl.Series(codes == -1), None)
        return categories.gather(indices)
    if encoding in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
        values = _read_obs_column(node["values"])
        mask = node["mask"][:]
        if mask.ndim != 1 or mask.dtype != np.bool_ or len(mask) != len(values):
            raise ValueError(f"invalid nullable mask shape, dtype or length in {node.name}")
        return values.set(pl.Series(mask), None)
    raise ValueError(f"unsupported obs encoding {encoding!r} at {node.name}")


def inspect_h5ad(path: PathLike) -> Dict[str, Any]:
    """Inspect only obs names/shapes; do not load expression or obs values."""
    with _h5py().File(path, "r") as handle:
        if "obs" not in handle:
            raise ValueError("h5ad file has no obs group")
        obs = handle["obs"]
        index_key = _text(obs.attrs.get("_index", "_index"))
        if index_key not in obs:
            raise ValueError("h5ad obs index dataset is missing")
        node = obs[index_key]
        n_cells = node.shape[0] if hasattr(node, "shape") else node["codes"].shape[0]
        return {
            "path": str(Path(path).resolve()),
            "n_cells": int(n_cells),
            "columns": [key for key in obs.keys() if key != index_key],
            "index_key": index_key,
            "expression_loaded": False,
        }


def read_h5ad_obs(path: PathLike, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """Read selected scalar obs columns and the cell index using h5py only."""
    with _h5py().File(path, "r") as handle:
        if "obs" not in handle:
            raise ValueError("h5ad file has no obs group")
        obs = handle["obs"]
        index_key = _text(obs.attrs.get("_index", "_index"))
        selected = columns if columns is not None else [k for k in obs if k != index_key]
        missing = set(selected) - set(obs.keys())
        if missing:
            raise ValueError(f"h5ad obs is missing columns: {sorted(missing)}")
        values = {"cell_id": _read_obs_column(obs[index_key])}
        for key in selected:
            if key == "cell_id":
                # AnnData's index is authoritative; an additional conflicting
                # cell_id column must not silently overwrite it.
                if not _read_obs_column(obs[key]).equals(values["cell_id"]):
                    raise ValueError("obs cell_id column conflicts with AnnData index")
            else:
                values[key] = _read_obs_column(obs[key])
        if any(len(value) != len(values["cell_id"]) for value in values.values()):
            raise ValueError("obs column lengths do not match the cell index")
    result = pl.DataFrame(values, strict=True)
    for key in ("cell_id", "donor", "donor_id"):
        if key in result.columns:
            invalid = result[key].cast(pl.String).str.strip_chars().is_in([""]).fill_null(True)
            if invalid.any():
                row = int(np.flatnonzero(invalid.to_numpy())[0])
                raise ValueError(f"AnnData obs {key} is empty or missing at row {row}")
    if result["cell_id"].null_count() or result["cell_id"].n_unique() != result.height:
        raise ValueError("AnnData cell index must be non-null and unique")
    return result


def build_dataset(
    path: PathLike, output_dir: PathLike, *, panel: Optional[pl.DataFrame] = None
) -> Dict[str, Any]:
    """Export paired receptors, cell mappings, separate labels, and model pairs.

    The default panel reproduces the supplied GILGFVFTL/ELAGIGILTV A*02:01
    benchmark. This restriction is supplied panel context, not inferred donor
    typing. Mixed experimental labels are retained as conflicting, never voted.
    """
    from .ingest import read_receptors

    info = inspect_h5ad(path)
    columns = GENES + ["donor", "benchmark_label"]
    obs = read_h5ad_obs(path, columns=columns)
    obs = obs.with_columns([pl.col(c).cast(pl.String) for c in columns + ["cell_id"]])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired = obs.select("cell_id", pl.col("donor").alias("donor_id"), *GENES)
    paired_path = output / "paired_cells.csv"
    paired.write_csv(paired_path)
    ingested = read_receptors(paired_path, format="paired")
    ingested.receptors.write_csv(output / "receptors.csv")
    ingested.chains.write_csv(output / "chains.csv")
    labels = obs.select("cell_id", pl.col("donor").alias("donor_id"), "benchmark_label")
    # Keep original AnnData index values for notebook reindexing while matching
    # the whitespace-normalized ingestion keys. Ambiguous normalization is an
    # error, never a silently missing cell-to-receptor mapping.
    keyed = labels.with_columns(
        pl.col("cell_id").str.strip_chars().alias("_join_cell"),
        pl.col("donor_id").str.strip_chars().alias("_join_donor"),
    )
    if keyed.select(pl.struct("_join_cell", "_join_donor").is_duplicated().any()).item():
        raise ValueError("AnnData cell/donor keys collide after whitespace normalization")
    mapped = ingested.cells.rename({"cell_id": "_join_cell", "donor_id": "_join_donor"})
    cells = keyed.join(
        mapped,
        on=["_join_cell", "_join_donor"],
        how="left",
        maintain_order="left",
        validate="1:1",
    ).drop("_join_cell", "_join_donor")
    cells = cells.with_columns(
        pl.col("cell_id").alias("cell"), pl.col("receptor_id").alias("tcr_id")
    )
    if cells.height != obs.height or cells["receptor_id"].null_count():
        raise ValueError(
            "ingestion must retain exactly one receptor mapping per AnnData cell, "
            "including unresolved placeholder receptors"
        )
    cells.drop("benchmark_label").write_csv(output / "cell_tcr_map.csv")
    labels.write_csv(output / "cell_labels.csv")
    if panel is None:
        panel = pl.DataFrame(
            {"peptide": ["GILGFVFTL", "ELAGIGILTV"], "hla": ["HLA-A*02:01", "HLA-A*02:01"]}
        )
    exported = export_decoder_input(ingested.receptors, panel, output / "decoder_input.csv")
    # One row per observed label and clone preserves discordant measurements.
    clone_labels = cells.group_by("receptor_id", "benchmark_label", maintain_order=True).len(
        name="n_cells"
    )
    label_counts = clone_labels.group_by("receptor_id", maintain_order=True).agg(
        pl.col("benchmark_label").drop_nulls().n_unique().alias("n_distinct_labels")
    )
    clone_labels = clone_labels.join(label_counts, on="receptor_id", how="left",
                                     maintain_order="left").with_columns(
        (pl.col("n_distinct_labels") > 1).alias("conflicting_labels")
    )
    clone_labels.write_csv(output / "clone_labels.csv")
    # This hash intentionally covers metadata only; hashing the HDF5 file itself
    # would scan expression matrices and violate the obs-only memory/IO contract.
    obs_path = output / "obs_metadata.csv"
    obs.write_csv(obs_path)
    obs_hash = file_sha256(obs_path)
    report = {
        **info,
        "obs_sha256": obs_hash,
        "n_receptors": ingested.receptors.height,
        "n_mapped_cells": cells.filter(pl.col("receptor_id").is_not_null()).height,
        "n_conflicting_clones": label_counts.filter(pl.col("n_distinct_labels") > 1).height,
        "output_dir": ".",
        "decoder_export": exported,
        "qc": ingested.qc,
        "labels_in_model_input": False,
        "notebook_note": "Map scored name through decoder_input.csv.mapping.csv before "
        "joining receptor_id to tcr_id; labels live in separate files.",
    }
    (output / "dataset_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
