"""Experimental receptor-only nearest neighbours, separate from antigen evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import tempfile
import time

import numpy as np
import polars as pl
from pydantic import ValidationError

from .hla import compare_hla, donor_compatibility
from .matching import (DonorRow, PanelRow, ReferenceRow, _ReceptorRow,
                       _unique_references, _validate_frame)
from .prediction import _ReceptorComponents
from .report import sha256_file, write_json
from .species import normalize_mhc, validate_species

_COMPONENTS = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b")
_INPUT_SCHEMA = {key: pl.String for key in ("embedding_id", "species", *_COMPONENTS)}
_MATCH_SCHEMA = {
    **{key: pl.String for key in (
        "receptor_id", "donor_id", "status", "evidence_type", "reference_receptor_id",
        "reference_id", "source", "reference_evidence", "peptide", "hla", "panel_hla",
        "hla_status", "donor_hla_status", "panel_hla_status")},
    "cosine_similarity": pl.Float64, "cosine_distance": pl.Float64, "neighbor_rank": pl.Int64,
    "reason": pl.String, "reference_record_json": pl.String,
}
_AUDIT_FIELDS = ("source", "record_id", "embedding_id", "status", "reason")
_INTERPRETATION = (
    "Experimental receptor embedding similarity; no validated binding threshold or antigen "
    "assignment. Reference annotations are provenance, not transferred labels. Alpha and beta "
    "chains are encoded separately without peptide or MHC inputs."
)


def _frame_hash(frame: pl.DataFrame) -> str:
    """Hash exact typed input content without a full serialized copy."""
    digest = hashlib.sha256(str(frame.schema).encode())
    for batch in frame.iter_slices(2048):
        digest.update(batch.write_json().encode())
    return digest.hexdigest()


def _eligible(row: dict, species: str, *, query: bool) -> tuple[dict | None, str]:
    observed_species = row.get("species", species if query else None)
    if observed_species != species:
        return None, f"Explicit species is {observed_species!r}; requested species is {species}"
    if (query or "pairing_status" in row) and row.get("pairing_status") != "paired":
        return None, f"Unambiguous paired receptor required; pairing_status={row.get('pairing_status')}"
    if row.get("unusable_chain_context"):
        return None, "Unusable chain context: " + row["unusable_chain_context"]
    try:
        components = _ReceptorComponents.model_validate(row).model_dump()
    except ValidationError as exc:
        errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return None, "Full-chain reconstruction requires complete unambiguous annotations: " + errors
    body = {"species": species, **components}
    identity = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {"embedding_id": "receptor-" + identity, **body}, ""


def _validate_vectors(vectors, ids, expected: set[str]) -> dict[str, int]:
    if (not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32
            or vectors.ndim != 2 or vectors.shape[1] == 0 or vectors.shape[1] % 2):
        raise ValueError("Embedding vectors must be a nonempty-width FP32 matrix of paired chains")
    if len(ids) != len(vectors) or len(ids) != len(set(ids)) or not set(ids) <= expected:
        raise ValueError("Embedding IDs must be unique, requested and aligned with vectors")
    for start in range(0, len(vectors), 2048):
        block = np.asarray(vectors[start:start + 2048], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Embedding vectors contain nonfinite values")
        if not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-5, rtol=1e-5):
            raise ValueError("Embedding vectors must have unit norm; zero vectors are invalid")
        halves = block.reshape(len(block), 2, -1)
        if not np.allclose(np.linalg.norm(halves, axis=2), 1 / np.sqrt(2),
                           atol=1e-5, rtol=1e-5):
            raise ValueError("Alpha and beta must contribute equally after separate normalization")
    return {identity: i for i, identity in enumerate(ids)}


def _iter_top_k(vectors, query_indices, reference_indices, *, top_k: int,
                query_block_size: int = 32, reference_block_size: int = 1024):
    """Exact cosine blocks; reference order supplies the deterministic tie break.

    References must be sorted by stable receptor identity. Memory is bounded by
    the two feature blocks and query_block_size × (reference_block_size + top_k).
    """
    if any(type(n) is not int or n < 1 for n in
           (top_k, query_block_size, reference_block_size)):
        raise ValueError("Search sizes must be positive integers")
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    k = min(top_k, len(reference_indices))
    for start in range(0, len(query_indices), query_block_size):
        queries = query_indices[start:start + query_block_size]
        q = np.asarray(vectors[queries], dtype=np.float64)
        best_scores = np.empty((len(queries), 0), dtype=np.float64)
        best_positions = np.empty((len(queries), 0), dtype=np.int64)
        for offset in range(0, len(reference_indices), reference_block_size):
            indices = reference_indices[offset:offset + reference_block_size]
            r = np.asarray(vectors[indices], dtype=np.float64)
            scores = q @ r.T
            positions = np.broadcast_to(np.arange(offset, offset + len(indices)), scores.shape)
            scores = np.concatenate((best_scores, scores), axis=1)
            positions = np.concatenate((best_positions, positions), axis=1)
            order = np.lexsort((positions, -scores), axis=1)[:, :k]
            best_scores = np.take_along_axis(scores, order, axis=1)
            best_positions = np.take_along_axis(positions, order, axis=1)
        for i, query_index in enumerate(queries):
            yield query_index, reference_indices[best_positions[i]], np.clip(best_scores[i], -1, 1)


def run_embedding_matching(receptors, references, panel, donors, output_dir, *, top_k=10,
                           species="human", **decoder_kwargs) -> dict:
    """Write optional neighbours without changing ordinary exact/edit evidence.

    HLA incompatibility excludes a reference before ranking. Incomplete donor or
    panel typing remains explicitly unresolved, following ordinary screen rules.
    The caller owns the surrounding atomic analysis output directory.
    """
    validate_species(species)
    if type(top_k) is not int or not 1 <= top_k <= 1000:
        raise ValueError("embedding top_k must be an integer from 1 to 1000")
    if "species" not in references.columns:
        raise ValueError("Embedding matching requires an explicit species column in references")
    if decoder_kwargs.get("precision") not in (None, "float32"):
        raise ValueError("Experimental embedding matching requires float32 precision")
    receptors = _validate_frame(receptors, _ReceptorRow, "receptors")
    references = _unique_references(_validate_frame(references, ReferenceRow, "references"))
    panel = _validate_frame(panel, PanelRow, "panel")
    donors = (_validate_frame(donors, DonorRow, "donors") if donors is not None else
              pl.DataFrame(schema={"donor_id": pl.String, "hla": pl.String}))
    if receptors.height and receptors["receptor_id"].is_duplicated().any():
        raise ValueError("receptor_id must be unique")
    for frame in (panel, donors):
        for restriction in frame["hla"].drop_nulls().unique():
            normalize_mhc(restriction, species)
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    managed_names = {"embedding_matches.csv", "embedding_matches.parquet", "embedding_audit.csv",
                     "embedding_metadata.json", "embedding_model_manifest.json",
                     "embedding_reconstruction_audit.csv", "embedding_reconstruction_audit.parquet",
                     "receptor_embeddings.npy", "embedding_ids.json", "embedding_input.csv",
                     "embedding_input.parquet", "embedding_runtime.json"}
    if any((output / name).exists() for name in managed_names):
        raise ValueError("Embedding output files already exist; choose a new output directory")
    panel_by_peptide = defaultdict(set)
    for row in panel.iter_rows(named=True):
        panel_by_peptide[row["peptide"]].add(row.get("hla"))
    donor_by_id = defaultdict(set)
    for row in donors.iter_rows(named=True):
        donor_by_id[row["donor_id"]].add(row["hla"])

    @lru_cache(maxsize=32768)
    def panel_contexts(peptide, restriction):
        contexts = []
        for hla in sorted(panel_by_peptide.get(peptide, ()), key=lambda x: x or ""):
            result = compare_hla(restriction, hla)
            if result.status != "incompatible":
                contexts.append((hla, result.status, result.reason))
        return tuple(contexts)

    @lru_cache(maxsize=32768)
    def donor_check(restriction, alleles):
        return donor_compatibility(restriction, alleles)

    requested = {}
    query_map, reference_map = {}, {}
    excluded = {}
    references_by_embedding = defaultdict(list)
    for query, frame, id_column in ((True, receptors, "receptor_id"),
                                    (False, references, "reference_id")):
        for index, row in enumerate(frame.iter_rows(named=True)):
            entity, reason = _eligible(row, species, query=query)
            if entity is not None and not query:
                try:
                    if row.get("hla"):
                        normalize_mhc(row["hla"], species)
                except ValueError as exc:
                    reason, entity = str(exc), None
                if entity is not None and not panel_contexts(row["peptide"], row.get("hla")):
                    reason, entity = "No compatible peptide/MHC restriction in the supplied panel", None
            if entity is None:
                excluded[(query, row[id_column])] = reason
                continue
            identity = entity["embedding_id"]
            requested[identity] = entity
            if query:
                query_map[row[id_column]] = identity
            else:
                reference_map[index] = identity
                references_by_embedding[identity].append(index)

    with tempfile.TemporaryDirectory(prefix=".embedding-matching-", dir=output) as temporary:
        stage = Path(temporary)
        embedding_metadata = {}
        backend_audit = {}
        indices = {}
        vectors = np.empty((0, 2), dtype=np.float32)
        if query_map and references_by_embedding:
            from .receptor_embeddings import embed_receptors
            frame = pl.DataFrame([requested[key] for key in sorted(requested)], schema=_INPUT_SCHEMA)
            result = embed_receptors(frame, stage / "vectors", **decoder_kwargs)
            indices = _validate_vectors(result.vectors, result.ids, set(requested))
            vectors, embedding_metadata = result.vectors, result.metadata
            if (embedding_metadata.get("representation") != "decodertcr-chain-mean-v1"
                    or embedding_metadata.get("precision") != "float32"):
                raise ValueError("Embedding backend returned a different representation or precision")
            if not {"embedding_id", "status", "reason"} <= set(result.audit.columns):
                raise ValueError("Embedding reconstruction audit is missing IDs, status or reason")
            for row in result.audit.iter_rows(named=True):
                identity = row["embedding_id"]
                if identity not in requested or identity in backend_audit:
                    raise ValueError("Embedding reconstruction audit has invalid or duplicate IDs")
                if row["status"] not in {"Embedded", "Unresolved"}:
                    raise ValueError("Embedding reconstruction audit has an invalid status")
                if (identity in indices) != (row["status"] == "Embedded"):
                    raise ValueError("Embedding vectors contradict reconstruction audit status")
                if row["status"] == "Unresolved" and not row["reason"]:
                    raise ValueError("Unresolved embedding audit requires a reason")
                backend_audit[identity] = row
            if set(backend_audit) != set(requested):
                raise ValueError("Embedding reconstruction audit does not cover every requested receptor")
        reference_groups = defaultdict(list)
        reference_vector_indices = {}
        for identity, rows in references_by_embedding.items():
            if identity not in indices:
                continue
            digest = backend_audit[identity].get("reconstructed_receptor_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Successful embedding audit requires a reconstructed receptor SHA256")
            group = "reconstructed-" + digest
            previous = reference_vector_indices.get(group)
            if previous is not None and not np.allclose(
                    vectors[previous], vectors[indices[identity]], atol=1e-6, rtol=1e-6):
                raise ValueError("Identical reconstructed receptors have inconsistent embeddings")
            reference_vector_indices[group] = indices[identity]
            reference_groups[group].extend(rows)
        reference_ids = references.get_column("reference_id")
        reference_hla = (references.get_column("hla") if "hla" in references.columns
                         else pl.Series("hla", [], dtype=pl.String))
        reference_restrictions = {}
        for identity, rows in reference_groups.items():
            rows.sort(key=lambda i: reference_ids[i])
            reference_restrictions[identity] = tuple(sorted(
                {reference_hla[i] for i in rows}, key=lambda value: value or ""))

        @lru_cache(maxsize=256)
        def expanded_reference(index):
            row = references.row(index, named=True)
            return row, json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)

        query_ids = receptors.get_column("receptor_id")
        query_donors = (receptors.get_column("donor_id") if "donor_id" in receptors.columns
                        else pl.Series("donor_id", [], dtype=pl.String))
        query_groups = defaultdict(list)
        for index, (query_id, donor_id) in enumerate(zip(query_ids, query_donors)):
            identity = query_map.get(query_id)
            if identity in indices:
                alleles = tuple(sorted(donor_by_id.get(donor_id, ())))
                query_groups[alleles].append(index)
        matched_queries = set()
        matches = 0
        with (stage / "embedding_matches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_MATCH_SCHEMA))
            writer.writeheader()
            for alleles, queries in sorted(query_groups.items()):
                allowed = []
                for identity in sorted(reference_groups):
                    if any(donor_check(restriction, alleles).status != "incompatible"
                           for restriction in reference_restrictions[identity]):
                        allowed.append(identity)
                # One actual reconstructed receptor consumes one neighbour slot.
                by_query_vector = defaultdict(list)
                for query_index in sorted(queries, key=lambda i: query_ids[i]):
                    identity = query_map[query_ids[query_index]]
                    by_query_vector[indices[identity]].append(query_index)
                ref_indices = [reference_vector_indices[identity] for identity in allowed]
                identities_by_index = {reference_vector_indices[identity]: identity for identity in allowed}
                for q_index, neighbors, similarities in _iter_top_k(
                        vectors, list(by_query_vector), ref_indices, top_k=top_k):
                    for query_index in by_query_vector[q_index]:
                        query_id, donor_id = query_ids[query_index], query_donors[query_index]
                        for rank, (neighbor, similarity) in enumerate(zip(neighbors, similarities), 1):
                            identity = identities_by_index[int(neighbor)]
                            for i in reference_groups[identity]:
                                reference, reference_json = expanded_reference(i)
                                donor = donor_check(reference.get("hla"), alleles)
                                if donor.status == "incompatible":
                                    continue
                                for panel_hla, panel_status, panel_reason in panel_contexts(
                                        reference["peptide"], reference.get("hla")):
                                    reason = _INTERPRETATION
                                    if donor.status != "compatible":
                                        reason += " " + donor.reason
                                    if panel_status != "compatible":
                                        reason += " " + panel_reason
                                    writer.writerow(dict(
                                        receptor_id=query_id, donor_id=donor_id,
                                        status="Experimental", evidence_type="embedding_neighbor",
                                        reference_receptor_id=identity, reference_id=reference["reference_id"],
                                        source=reference["source"], reference_evidence=reference["evidence"],
                                        peptide=reference["peptide"], hla=reference.get("hla"),
                                        panel_hla=panel_hla, panel_hla_status=panel_status,
                                        donor_hla_status=donor.status,
                                        hla_status=("compatible" if donor.status == panel_status == "compatible"
                                                    else "unresolved"),
                                        cosine_similarity=float(similarity), cosine_distance=float(1 - similarity),
                                        neighbor_rank=rank,
                                        reason=reason, reference_record_json=reference_json))
                                    matches += 1
                                    matched_queries.add(query_id)
        pl.scan_csv(stage / "embedding_matches.csv", schema=_MATCH_SCHEMA).sink_parquet(
            stage / "embedding_matches.parquet", compression="zstd")
        audit_counts = defaultdict(int)
        with (stage / "embedding_audit.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_AUDIT_FIELDS)
            writer.writeheader()
            for query, frame, id_column in ((True, receptors, "receptor_id"),
                                            (False, references, "reference_id")):
                for index, row in enumerate(frame.iter_rows(named=True)):
                    identity = query_map.get(row[id_column]) if query else reference_map.get(index)
                    reason = excluded.get((query, row[id_column]), "")
                    status = "Unresolved"
                    if identity in indices:
                        if query and row[id_column] not in matched_queries:
                            reason = "No embedded reference with retained panel/donor MHC context"
                        else:
                            status = "Experimental"
                            reason = "Embedded receptor; similarity does not assign antigen specificity"
                    elif identity and backend_audit:
                        item = backend_audit[identity]
                        reason = item.get("reason") or item.get("tcr_reason") or "Full-chain reconstruction unavailable"
                    elif identity:
                        reason = "Embedding skipped: no eligible query/reference comparison"
                    writer.writerow(dict(source="query" if query else "reference", record_id=row[id_column],
                                         embedding_id=identity, status=status, reason=reason))
                    audit_counts[("query" if query else "reference") + "_" + status.lower()] += 1
        bundle = stage / "vectors"
        if bundle.exists():
            names = {"manifest.json": "embedding_model_manifest.json",
                     "embedding_audit.csv": "embedding_reconstruction_audit.csv",
                     "embedding_audit.parquet": "embedding_reconstruction_audit.parquet"}
            for source in bundle.iterdir():
                if not source.is_file():
                    raise ValueError("Embedding backend returned an unexpected nested output")
                target = stage / names.get(source.name, source.name)
                if target.exists() or (output / target.name).exists():
                    raise ValueError("Embedding backend output filename collision")
                source.rename(target)
            bundle.rmdir()
        metadata = dict(
            schema_version=1, status="Experimental", method="exact_blocked_cosine",
            representation="decodertcr-chain-mean-v1", top_k=top_k, species=species,
            matching_source_sha256=sha256_file(Path(__file__)),
            neighbor_identity="reconstructed full alpha/beta sequence pair SHA256",
            interpretation=_INTERPRETATION, query_receptors=receptors.height,
            reference_records=references.height, embedded_receptors=len(indices),
            matched_queries=len(matched_queries), match_rows=matches, audit_counts=dict(audit_counts),
            input_sha256={name: _frame_hash(frame) for name, frame in (
                ("receptors", receptors), ("references", references), ("panel", panel), ("donors", donors))},
            embedding=embedding_metadata, elapsed_seconds=round(time.perf_counter() - started, 4),
            output_sha256={path.name: sha256_file(path) for path in stage.iterdir() if path.is_file()})
        write_json(stage / "embedding_metadata.json", metadata)
        for source in stage.iterdir():
            if (output / source.name).exists():
                raise ValueError("Embedding output appeared during execution")
        for source in stage.iterdir():
            source.rename(output / source.name)
    return metadata
