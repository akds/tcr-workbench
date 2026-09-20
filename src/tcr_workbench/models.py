"""Validated boundaries shared by TCR Workbench modules.

Large repertoire tables are validated with column expressions, rather than by
allocating a Python/Pydantic object for every chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputError(ValueError):
    """An actionable input/schema error, safe to display at the CLI boundary."""


class IngestOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: Path
    format: Literal["auto", "10x", "airr", "paired"] = "auto"
    donor_id: Optional[str] = None
    species: Literal["human", "mouse"] = "human"
    max_pairings_per_cell: int = Field(default=16, ge=1, le=10000)

    @field_validator("donor_id")
    @classmethod
    def valid_donor(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value or any(ord(c) < 32 for c in value):
            raise ValueError("donor_id must be nonempty and contain no control characters")
        return value


RECEPTOR_SCHEMA = {
    "receptor_id": pl.String,
    "donor_id": pl.String,
    "cdr3a": pl.String,
    "cdr3b": pl.String,
    "trav": pl.String,
    "traj": pl.String,
    "trbv": pl.String,
    "trbj": pl.String,
    "pairing_status": pl.String,
    "unusable_chain_context": pl.String,
    "cell_count": pl.UInt64,
    "observation_count": pl.UInt64,
}
CELL_SCHEMA = {"cell_id": pl.String, "donor_id": pl.String, "receptor_id": pl.String}
CHAIN_SCHEMA = {
    "chain_id": pl.String,
    "donor_id": pl.String,
    "cell_id": pl.String,
    "locus": pl.String,
    "cdr3": pl.String,
    "v_call": pl.String,
    "j_call": pl.String,
    "productive": pl.Boolean,
    "high_confidence": pl.Boolean,
    "is_cell": pl.Boolean,
    "source_row": pl.UInt32,
    "sequence_ambiguous": pl.Boolean,
    "junction_incomplete": pl.Boolean,
    "eligible": pl.Boolean,
    "chain_observed": pl.Boolean,
}


class IngestResult(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", frozen=True, strict=True
    )

    chains: pl.DataFrame
    receptors: pl.DataFrame
    cells: pl.DataFrame
    qc: list[dict[str, Any]]
    source: str = ""
    format: str = ""
    species: Literal["human", "mouse"] = "human"

    @field_validator("chains")
    @classmethod
    def validate_chains(cls, frame: pl.DataFrame) -> pl.DataFrame:
        _validate_frame(frame, CHAIN_SCHEMA, ("chain_id", "donor_id", "locus", "source_row"))
        return frame

    @field_validator("receptors")
    @classmethod
    def validate_receptors(cls, frame: pl.DataFrame) -> pl.DataFrame:
        _validate_frame(
            frame,
            RECEPTOR_SCHEMA,
            ("receptor_id", "donor_id", "pairing_status", "unusable_chain_context"),
        )
        if frame["receptor_id"].n_unique() != frame.height:
            raise ValueError("receptor_id must be unique")
        return frame

    @field_validator("cells")
    @classmethod
    def validate_cells(cls, frame: pl.DataFrame) -> pl.DataFrame:
        _validate_frame(frame, CELL_SCHEMA, ("donor_id", "receptor_id"))
        return frame


def _validate_frame(frame: pl.DataFrame, schema: dict, nonnull: tuple) -> None:
    missing = set(schema) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing output columns: {', '.join(sorted(missing))}")
    wrong = [name for name, dtype in schema.items() if frame.schema[name] != dtype]
    if wrong:
        raise ValueError(f"Invalid output dtypes: {', '.join(wrong)}")
    if frame.select(pl.any_horizontal(pl.col(name).is_null() for name in nonnull).any()).item():
        raise ValueError(f"Null values in required output columns: {', '.join(nonnull)}")
