"""Adversarial wet-lab failure modes across ingestion and evidence screening."""

import polars as pl
import pytest

from tcr_workbench.ingest import read_receptors
from tcr_workbench.matching import screen

ALPHA = "CAVRDSNYQLIW"
BETA = "CASSLGQETQYF"
ALT_ALPHA = "CAVSDRQAGTALIF"
ALT_BETA = "CASSIRSSYEQYF"
PEPTIDE = "GILGFVFTL"


def contig(cell="cell1", locus="TRA", sequence=ALPHA, **kwargs):
    row = {
        "barcode": cell,
        "donor_id": "donor1",
        "chain": locus,
        "cdr3": sequence,
        "productive": "true",
        "high_confidence": "true",
        "is_cell": "true",
        "v_gene": "TRAV1-2" if locus == "TRA" else "TRBV19",
        "j_gene": "TRAJ33" if locus == "TRA" else "TRBJ2-7",
    }
    row.update(kwargs)
    return row


def reference(**kwargs):
    row = {
        "reference_id": "ref1",
        "cdr3a": ALPHA,
        "cdr3b": BETA,
        "trav": "TRAV1-2",
        "traj": "TRAJ33",
        "trbv": "TRBV19",
        "trbj": "TRBJ2-7",
        "peptide": PEPTIDE,
        "hla": "A*02:01",
        "source": "synthetic-test",
        "evidence": "functional",
    }
    row.update(kwargs)
    return row


def evaluate(tmp_path, rows, fmt="10x", references=None):
    path = tmp_path / "input.tsv"
    pl.DataFrame(rows).write_csv(path, separator="\t")
    ingested = read_receptors(path, format=fmt)
    evidence = screen(
        ingested.receptors,
        pl.DataFrame(references if references is not None else [reference()]),
        pl.DataFrame([{"peptide": PEPTIDE, "hla": "A*02:01"}]),
        pl.DataFrame([{"donor_id": "donor1", "hla": "A*02:01"}]),
    )
    return ingested, evidence


@pytest.mark.parametrize("legacy_source", ["panel", "reference", "donor"])
def test_legacy_hla_keeps_sequence_supported_reference_in_cli_report(tmp_path, legacy_source):
    from tcr_workbench.cli import main

    source, references = tmp_path / "contigs.csv", tmp_path / "references.csv"
    panel, donors, output = tmp_path / "panel.csv", tmp_path / "donors.csv", tmp_path / "result"
    pl.DataFrame([contig(), contig(locus="TRB", sequence=BETA)]).write_csv(source)
    pl.DataFrame([reference(hla="A*0201" if legacy_source == "reference" else "A*02:01")]).write_csv(references)
    pl.DataFrame([{"peptide": PEPTIDE, "hla": "A*0201" if legacy_source == "panel" else "A*02:01"}]).write_csv(panel)
    pl.DataFrame([{"donor_id": "donor1", "hla": "A*0201" if legacy_source == "donor" else "A*02:01"}]).write_csv(donors)
    assert main(["screen", "--input", str(source), "--reference", str(references),
                 "--panel", str(panel), "--donors", str(donors), "--out", str(output)]) == 0
    evidence = pl.read_parquet(output / "evidence.parquet")
    assert evidence["reference_id"].to_list() == ["ref1"]
    assert evidence["evidence_type"].to_list() == ["exact_paired"]
    assert evidence["status"].to_list() == ["Unresolved"]
    assert "colon-delimited" in evidence["reason"].item()
    assert pl.read_parquet(output / "cells.parquet")["cell_id"].to_list() == ["cell1"]
    assert pl.read_parquet(output / "chains.parquet").height == 2


@pytest.mark.parametrize("flag", ["productive", "high_confidence", "is_cell"])
def test_explicit_negative_quality_keeps_cells_but_cannot_support_antigen(tmp_path, flag):
    rows = [contig(**{flag: "false"}), contig(locus="TRB", sequence=BETA, **{flag: "false"})]
    ingested, evidence = evaluate(tmp_path, rows)
    assert ingested.chains.height == 2
    assert ingested.cells["cell_id"].unique().to_list() == ["cell1"]
    assert evidence["status"].to_list() == ["Unresolved"]
    assert evidence["reference_id"].to_list() == [None]


@pytest.mark.parametrize("sequence", ["CA*RDSNYQLIW", "CAXRDSNYQLIW", "AVRDSNYQLI", None])
def test_dropout_ambiguous_or_nonproductive_alpha_preserves_usable_beta(tmp_path, sequence):
    rows = [contig(sequence=sequence), contig(locus="TRB", sequence=BETA)]
    ingested, evidence = evaluate(tmp_path, rows)
    assert ingested.chains.height == 2
    assert ingested.receptors["cdr3a"].to_list() == [None]
    assert evidence["evidence_type"].to_list() == ["exact_single_chain"]
    assert "Only beta chain" in evidence["reason"][0]
    assert "exact_paired" not in evidence["evidence_type"].to_list()
    if sequence is not None:
        assert evidence["unusable_chain_context"][0]
        assert "Excluded observed chains:" in evidence["reason"][0]


def test_airr_without_cell_identifiers_does_not_manufacture_a_paired_match(tmp_path):
    rows = [
        {
            "sequence_id": "a",
            "donor_id": "donor1",
            "locus": "TRA",
            "junction_aa": ALPHA,
            "v_call": "TRAV1-2",
            "j_call": "TRAJ33",
            "productive": "T",
        },
        {
            "sequence_id": "b",
            "donor_id": "donor1",
            "locus": "TRB",
            "junction_aa": BETA,
            "v_call": "TRBV19",
            "j_call": "TRBJ2-7",
            "productive": "T",
        },
    ]
    ingested, evidence = evaluate(tmp_path, rows, fmt="airr")
    assert ingested.receptors.height == 2
    assert ingested.receptors["pairing_status"].to_list() == ["unpaired_airr", "unpaired_airr"]
    assert evidence["evidence_type"].to_list() == ["exact_single_chain", "exact_single_chain"]


def test_dual_alpha_alternatives_survive_with_one_supported_and_one_unresolved(tmp_path):
    rows = [contig(), contig(sequence=ALT_ALPHA), contig(locus="TRB", sequence=BETA)]
    ingested, evidence = evaluate(tmp_path, rows)
    assert ingested.receptors.height == 2
    assert ingested.chains.height == 3
    assert ingested.cells.height == 2
    assert ingested.receptors["pairing_status"].to_list() == ["dual_alpha", "dual_alpha"]
    assert sorted(evidence["status"].to_list()) == ["Candidate", "Unresolved"]
    supported = evidence.filter(pl.col("status") == "Candidate")
    assert "dual_alpha" in supported["reason"][0]


def test_same_barcode_in_different_donors_never_forms_paired_reference(tmp_path):
    rows = [contig(), contig(locus="TRB", sequence=BETA, donor_id="donor2")]
    ingested, evidence = evaluate(tmp_path, rows)
    assert ingested.receptors.height == 2
    assert all(
        value is None
        for value in ingested.receptors.filter(pl.col("donor_id") == "donor1")["cdr3b"]
    )
    assert evidence["evidence_type"].to_list().count("exact_single_chain") == 2
    assert "exact_paired" not in evidence["evidence_type"].to_list()
    assert evidence.filter(pl.col("donor_id") == "donor2")["status"].to_list() == ["Unresolved"]


def test_explicit_paired_rows_never_recombine_to_create_an_unsupported_pair(tmp_path):
    rows = [
        {"cell_id": "cell1", "donor_id": "donor1", "cdr3a": ALPHA, "cdr3b": ALT_BETA},
        {"cell_id": "cell1", "donor_id": "donor1", "cdr3a": ALT_ALPHA, "cdr3b": BETA},
    ]
    ingested, evidence = evaluate(tmp_path, rows, fmt="paired")
    assert ingested.receptors.height == 2
    assert evidence["status"].to_list() == ["Unresolved", "Unresolved"]
    assert evidence["reference_id"].null_count() == 2


def test_gene_ambiguity_survives_ingestion_and_does_not_become_a_conflict(tmp_path):
    rows = [contig(), contig(locus="TRB", sequence=BETA, v_gene="TRBV6-5, TRBV19")]
    ingested, evidence = evaluate(tmp_path, rows)
    assert ingested.receptors["trbv"].to_list() == ["TRBV6-5, TRBV19"]
    assert evidence["gene_status"].to_list() == ["unresolved"]
    assert evidence["status"].to_list() == ["Candidate"]


def test_all_sequence_uncertainty_retained_as_one_explicit_unresolved_cell(tmp_path):
    ingested, evidence = evaluate(
        tmp_path, [contig(sequence="CAXRDSNYQLIW"), contig(locus="TRB", sequence="CASSXGQETQYF")]
    )
    assert ingested.chains.height == 2
    assert ingested.cells.height == 1
    assert evidence["status"].to_list() == ["Unresolved"]
    assert "No usable" in evidence["reason"][0]
    assert "alpha:ambiguous_sequence" in evidence["unusable_chain_context"][0]
    assert "beta:ambiguous_sequence" in evidence["reason"][0]
