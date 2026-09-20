"""Biological edge cases and native-search equivalence, deliberately small/fast."""

import random

import polars as pl
import pytest
from rapidfuzz.distance import Levenshtein

from tcr_workbench.hla import compare_hla, donor_compatibility, normalize_hla
from tcr_workbench.matching import (
    _SequenceIndex,
    read_donors,
    read_panel,
    read_references,
    screen,
)

ALPHA = "CAVRDSNYQLIW"
BETA = "CASSLGQETQYF"
PEPTIDE = "GILGFVFTL"


def receptor(**updates):
    row = {
        "receptor_id": "r1",
        "donor_id": "d1",
        "cdr3a": ALPHA,
        "cdr3b": BETA,
        "trav": "TRAV1-2",
        "traj": "TRAJ33",
        "trbv": "TRBV19",
        "trbj": "TRBJ2-7",
        "pairing_status": "paired",
        "cell_count": 1,
    }
    row.update(updates)
    return row


def reference(**updates):
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
        "source": "test-source",
        "evidence": "functional",
    }
    row.update(updates)
    return row


def run(receptors=None, references=None, panel=None, donors=None, **kwargs):
    return screen(
        pl.DataFrame(receptors if receptors is not None else [receptor()]),
        pl.DataFrame(references if references is not None else [reference()]),
        pl.DataFrame(panel if panel is not None else [{"peptide": PEPTIDE, "hla": "A*02:01"}]),
        pl.DataFrame(donors if donors is not None else [{"donor_id": "d1", "hla": "A*02:01"}]),
        **kwargs,
    )


@pytest.mark.parametrize(
    "left,right,status",
    [
        ("A*02:01", "HLA-A*02:01:01:01", "compatible"),
        ("A*02:01:01", "A*02:01:02", "compatible"),
        ("A*02:01", "A*02:02", "incompatible"),
        ("A*02:01", "B*02:01", "incompatible"),
        ("A*02", "A*02:01", "unresolved"),
        ("A*02", "A*03:01", "incompatible"),
        ("A*02:01:01G", "A*02:01", "unresolved"),
        # G-group representative prefixes are not membership tables.
        ("A*02:01:01G", "A*02:99", "unresolved"),
        ("A*02:01P", "A*02:01P", "unresolved"),
        ("A*02:01N", "A*02:01N", "incompatible"),
        ("A*02:01S", "A*02:01", "incompatible"),
        ("A*02:01C", "A*02:01", "incompatible"),
        ("A*02:01Q", "A*02:01", "unresolved"),
        ("A*02:01L", "A*02:01", "unresolved"),
        ("A*02:01|A*02:02", "A*02:01", "unresolved"),
        ("A*02:01/A*02:02", "A*03:01", "incompatible"),
        (None, "A*02:01", "unresolved"),
        ("DRB1*04:01", "DRB1*04:01", "compatible"),
        ("DQB1*02:01", "DQB1*02:01", "unresolved"),
        ("DQA1*05:01/DQB1*02:01", "DQB1*02:01/DQA1*05:01", "compatible"),
        ("DQA1*05:01/DQB1*02:01", "DQA1*02:01/DQB1*02:01", "incompatible"),
        ("DPA1*01:03/DPB1*04:01", "DPA1*01:03/DPB1*04:01:01", "compatible"),
        ("DPA1*01:03/DPB1*04:01", "DPB1*04:01", "unresolved"),
    ],
)
def test_hla_resolution_expression_groups_and_class_ii(left, right, status):
    assert compare_hla(left, right).status == status
    assert compare_hla(right, left).status == status


@pytest.mark.parametrize(
    "invalid",
    [
        "HLA-A0201",
        "A*2:1",
        "DQA1*05:01/DPB1*02:01",
        "A*02:01/B*07:02",
        "A*02:01G",
        "A*02:01:01P",
        "",
    ],
)
def test_invalid_hla_is_explicit_error(invalid):
    with pytest.raises(ValueError):
        normalize_hla(invalid)


def test_donor_pairs_never_inferred_from_individually_typed_chains():
    restriction = "DQA1*05:01/DQB1*02:01"
    assert donor_compatibility(restriction, ["DQA1*05:01", "DQB1*02:01"]).status == "unresolved"
    assert donor_compatibility(restriction, [restriction]).status == "compatible"
    assert donor_compatibility(restriction, []).status == "unresolved"


@pytest.mark.parametrize("group", ["A*02:01:01G", "A*02:01P"])
def test_explicit_null_expression_veto_precedes_group_uncertainty(group):
    assert compare_hla(group, "A*02:01N").status == "incompatible"
    assert compare_hla("A*02:01N", group).status == "incompatible"


def test_drb_only_convention_does_not_imply_a_precisely_typed_alpha_allele():
    pair = "DRA*01:01/DRB1*04:01"
    assert compare_hla("DRB1*04:01", "DRB1*04:01").status == "compatible"
    for left, right in [(pair, "DRB1*04:01"), ("DRB1*04:01", pair)]:
        result = compare_hla(left, right)
        assert result.status == "unresolved"
        assert "missing alpha/beta typing" in result.reason
    assert compare_hla("DRA*01:01N/DRB1*04:01", "DRB1*04:01").status == "incompatible"


def test_fully_specified_dr_pairs_with_different_beta_genes_explain_gene_difference():
    first, second = "DRA*01:01/DRB1*04:01", "DRA*01:01/DRB5*01:01"
    for left, right in [(first, second), (second, first)]:
        result = compare_hla(left, right)
        assert result.status == "unresolved"
        assert "different chain genes" in result.reason
        assert "DRB1" in result.reason and "DRB5" in result.reason
        assert "partially specified" not in result.reason


def test_dra_only_typing_never_identifies_a_dr_restriction():
    assert compare_hla("DRA*01:01", "DRA*01:01").status == "unresolved"
    assert donor_compatibility("DRA*01:01", ["DRA*01:01"]).status == "unresolved"
    row = run(
        references=[reference(hla="DRA*01:01")],
        panel=[{"peptide": PEPTIDE, "hla": "DRA*01:01"}],
        donors=[{"donor_id": "d1", "hla": "DRA*01:01"}],
    ).row(0, named=True)
    assert row["status"] == "Unresolved"
    assert row["reference_id"] == "ref1"
    assert "both alpha and beta" in row["reason"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("A*02:01|A*02:01", "compatible"),
        ("A*02:01:01G|A*02:01:01G", "unresolved"),
        ("DRA*01:01/DRA*01:01", "unresolved"),
        ("A*02:01N,A*02:01N", "incompatible"),
        ("A*02:01;A*03:01;A*02:01", "unresolved"),
    ],
)
def test_repeated_identical_hla_alternatives_normalize_idempotently(value, expected):
    canonical = normalize_hla(value)
    assert normalize_hla(canonical) == canonical
    assert compare_hla(value, canonical).status == expected
    assert compare_hla(value, value).status == compare_hla(canonical, canonical).status == expected


@pytest.mark.parametrize("value", ["A*02:1000", "DPB1*1000:01", "A*02:1000N"])
def test_hla_fields_support_current_four_digit_allele_numbers(value):
    assert normalize_hla(value) == "HLA-" + value
    expected = (
        "incompatible"
        if value.endswith("N")
        else ("unresolved" if value.startswith("DPB1") else "compatible")
    )
    assert compare_hla(value, value).status == expected
    assert compare_hla("DPA1*01:03/DPB1*1000:01", "DPA1*01:03/DPB1*1000:01").status == "compatible"


@pytest.mark.parametrize("legacy,modern", [
    ("A*0201", "A*02:01"),
    ("A*020101", "A*02:01:01"),
    ("A*02010101", "A*02:01:01:01"),
    ("A*10256", "A*10:256"),
    ("A*10256", "A*102:56"),
    ("DRA*0101/DRB1*0401", "DRA*01:01/DRB1*04:01"),
])
def test_delimiter_free_hla_retains_ambiguity_without_inferred_fields(legacy, modern):
    for left, right in ((legacy, modern), (modern, legacy), (legacy, legacy)):
        result = compare_hla(left, right)
        assert result.status == "unresolved"
        assert "colon-delimited" in result.reason
    assert normalize_hla(legacy).replace("HLA-", "") == legacy


def test_legacy_hla_guard_preserves_explicit_modern_fields_and_known_exclusions():
    assert compare_hla("A*102:56", "A*102:56").status == "compatible"
    assert compare_hla("A*102:56", "A*10:256").status == "incompatible"
    assert compare_hla("DPB1*100:01", "DPB1*100:01").status == "unresolved"
    assert compare_hla("DPA1*01:03/DPB1*1000:01", "DPA1*01:03/DPB1*1000:01").status == "compatible"
    assert compare_hla("A*0201", "B*02:01").status == "incompatible"
    assert compare_hla("A*0201N", "A*02:01").status == "incompatible"


@pytest.mark.parametrize("alpha", ["DRA*01:01", "DRA*01:02"])
def test_typed_donor_dr_molecule_supports_a_beta_level_restriction(alpha):
    pair = alpha + "/DRB1*04:01"
    assert donor_compatibility("DRB1*04:01", [pair]).status == "compatible"
    assert donor_compatibility(pair, ["DRB1*04:01"]).status == "unresolved"
    assert compare_hla("DRB1*04:01", pair).status == "unresolved"


@pytest.mark.parametrize("alpha", ["DRA*01:01N", "DRA*01:01S", "DRA*01:01L", "DRA*01"])
def test_drb_positive_evidence_rule_never_bypasses_alpha_expression_or_resolution(alpha):
    assert donor_compatibility("DRB1*04:01", [alpha + "/DRB1*04:01"]).status == "unresolved"


@pytest.mark.parametrize(
    "chain,pair",
    [
        ("DQB1*02:01", "DQA1*05:01/DQB1*02:01"),
        ("DPB1*04:01", "DPA1*01:03/DPB1*04:01"),
    ],
)
def test_drb_directional_rule_does_not_generalize_to_partial_dq_dp(chain, pair):
    assert donor_compatibility(chain, [pair]).status == "unresolved"


def test_exact_paired_is_candidate_with_auditable_provenance():
    result = run().row(0, named=True)
    assert result["status"] == "Candidate"
    assert result["evidence_type"] == "exact_paired"
    assert result["gene_status"] == "compatible"
    assert result["source"] == "test-source"
    assert result["reference_evidence"] == "functional"
    assert result["distance"] == 0
    assert result["evidence_rank"] == 1
    assert "requires experimental validation" in result["reason"]


def test_single_chain_reference_is_preserved_and_identified():
    result = run(references=[reference(cdr3a=None, trav=None, traj=None)]).row(0, named=True)
    assert result["evidence_type"] == "exact_single_chain"
    assert result["alpha_distance"] is None
    assert "Only beta chain" in result["reason"]


def test_known_discordant_second_chain_cannot_be_hidden():
    result = run(references=[reference(cdr3a="CAAAAAAAAAAW")]).row(0, named=True)
    assert result["status"] == "Unresolved"
    assert result["reference_id"] is None


def test_total_edit_budget_applies_across_both_chains():
    refs = [reference(cdr3a="CAVRDSNYQIIW", cdr3b="CASSLGQETQFF")]
    assert run(references=refs, max_distance=1)["evidence_type"].to_list() == ["none"]
    row = run(references=refs, max_distance=2).row(0, named=True)
    assert (row["distance"], row["alpha_distance"], row["beta_distance"]) == (2, 1, 1)
    assert row["evidence_type"] == "similar_paired"


def test_indels_and_length_cutoff():
    result = run(references=[reference(cdr3b="CASSLGQETQYAF")]).row(0, named=True)
    assert result["distance"] == 1
    assert run(references=[reference(cdr3b="CASSLGQETQYAAAF")])["status"].to_list() == [
        "Unresolved"
    ]


def test_hla_ambiguity_retains_support_as_unresolved():
    row = run(donors=[{"donor_id": "d1", "hla": "A*02:01:01G"}]).row(0, named=True)
    assert row["reference_id"] == "ref1"
    assert row["status"] == "Unresolved"
    assert row["hla_status"] == "unresolved"
    assert "G/P-group" in row["reason"]


def test_partial_donor_typing_retains_support_but_wrong_panel_excludes_it():
    mismatch = run(donors=[{"donor_id": "d1", "hla": "A*03:01"}]).row(0, named=True)
    assert mismatch["status"] == "Unresolved"
    assert mismatch["reference_id"] == "ref1"
    assert "does not assert genotype completeness" in mismatch["reason"]
    assert run(panel=[{"peptide": "NLVPMVATV", "hla": "A*02:01"}])["reference_id"].to_list() == [
        None
    ]
    assert run(panel=[{"peptide": PEPTIDE, "hla": "B*07:02"}])["reference_id"].to_list() == [None]


@pytest.mark.parametrize(
    "restriction",
    ["B*07:02", "C*07:01", "DRB1*04:01", "DQA1*05:01/DQB1*02:01", "DPA1*01:03/DPB1*04:01"],
)
def test_untyped_donor_locus_retains_reference_provenance(restriction):
    row = run(
        references=[reference(hla=restriction)], panel=[{"peptide": PEPTIDE, "hla": restriction}]
    ).row(0, named=True)
    assert row["status"] == "Unresolved"
    assert row["reference_id"] == "ref1"
    assert row["source"] == "test-source"
    assert row["donor_hla_status"] == "unresolved"
    assert row["panel_hla_status"] == "compatible"
    assert "does not cover" in row["reason"]


def test_partial_locus_coverage_never_overrides_explicit_null_restriction():
    assert donor_compatibility("B*07:02N", ["A*02:01"]).status == "incompatible"
    assert donor_compatibility("A*02:01N", []).status == "incompatible"


@pytest.mark.parametrize(
    "alleles",
    [["A*03:01"], ["A*03:01", "A*24:02"], ["A*03:01", "A*24:02", "B*07:02"]],
)
def test_reported_alleles_do_not_assert_genotype_completeness(alleles):
    assert compare_hla("A*02:01", alleles[0]).status == "incompatible"
    result = donor_compatibility("A*02:01", alleles)
    assert result.status == "unresolved"
    assert "completeness" in result.reason
    assert donor_compatibility("A*02:01", alleles + ["A*02:01"]).status == "compatible"


def test_explicit_class_ii_pairs_do_not_assert_exhaustive_molecules():
    result = donor_compatibility(
        "DQA1*05:01/DQB1*02:01", ["DQA1*01:01/DQB1*05:01", "DQA1*02:01/DQB1*03:01"]
    )
    assert result.status == "unresolved"
    assert "completeness" in result.reason


@pytest.mark.parametrize(
    "receptor_gene,reference_gene,expected",
    [
        ("TRAV14/DV4", "TRAV14", "unresolved"),
        ("TRAV38-2/DV8*01", "TRAV38-2*01", "unresolved"),
        ("TRAV14/DV4*01", "TRAV14/DV4*01", "compatible"),
        ("TRAV14/DV4*01", "TRAV14/DV4*02", "conflict"),
    ],
)
def test_shared_alpha_gene_names_preserve_nomenclature_uncertainty(
    receptor_gene, reference_gene, expected
):
    row = run(
        receptors=[receptor(trav=receptor_gene)], references=[reference(trav=reference_gene)]
    ).row(0, named=True)
    assert row["gene_status"] == expected


@pytest.mark.parametrize(
    "query,reference_call,expected",
    [
        ("TRBV12-3/TRBV12-4", "TRBV12-4", "unresolved"),
        ("TRBV12-3*01/TRBV12-4*01", "TRBV12-4*01", "unresolved"),
        ("TRBV20/OR9-2", "TRBV20/OR9-2", "compatible"),
        ("TRBV20/OR9-2", "TRBV20", "conflict"),
    ],
)
def test_slash_alternatives_are_distinct_from_orphan_gene_names(query, reference_call, expected):
    row = run(receptors=[receptor(trbv=query)], references=[reference(trbv=reference_call)]).row(
        0, named=True
    )
    assert row["gene_status"] == expected


@pytest.mark.parametrize(
    "query,reference_call,expected",
    [
        ("TRBV6-5*01,TRBV6-5*02", "TRBV6-5*03", "conflict"),
        ("TRBV6-5*01,TRBV6-5*02", "TRBV6-5*02,TRBV6-5*03", "unresolved"),
        ("TRBV6-5*01,TRBV7-9", "TRBV6-5*02", "conflict"),
        ("TRBV6-5*01,TRBV6-5", "TRBV6-5*03", "unresolved"),
        ("TRBV6-5*01,TRBV7-9*01", "TRBV6-5*02,TRBV7-9*02", "conflict"),
    ],
)
def test_gene_alternatives_require_at_least_one_possible_allele_agreement(
    query, reference_call, expected
):
    for first, second in [(query, reference_call), (reference_call, query)]:
        row = run(receptors=[receptor(trbv=first)], references=[reference(trbv=second)]).row(
            0, named=True
        )
        assert row["gene_status"] == expected
        assert row["status"] == ("Unresolved" if expected == "conflict" else "Candidate")


@pytest.mark.parametrize(
    "field,value",
    [
        ("trbj", "NA"),
        ("trbj", "N/A"),
        ("trbv", "NONE"),
        ("trbv", "UNKNOWN"),
        ("trbv", "TRBV19 (F)"),
        ("trbv", "TRBV19*"),
        ("trbv", "TRAV19"),
    ],
)
def test_unrecognized_reference_gene_text_is_retained_as_uncertainty(tmp_path, field, value):
    path = tmp_path / "refs.csv"
    pl.DataFrame([reference(**{field: value})]).write_csv(path)
    loaded = read_references(path)
    assert loaded[field].item() == value
    row = run(references=loaded.to_dicts()).row(0, named=True)
    assert row["gene_status"] == "unresolved"
    assert row["status"] == "Candidate"
    assert row["reference_id"] == "ref1"
    assert "V/J gene calls: unresolved" in row["reason"]


def test_unrecognized_public_receptor_gene_text_cannot_establish_identity():
    annotated = receptor(trbv="TRBV19 (F)")
    for ref in (reference(), reference(trbv="TRBV19 (F)")):
        row = run(receptors=[annotated], references=[ref]).row(0, named=True)
        assert row["gene_status"] == "unresolved"
        assert row["status"] == "Candidate"
    # Uncertainty at V cannot erase a separately established J-gene conflict.
    row = run(receptors=[annotated], references=[reference(trbj="TRBJ1-2")]).row(0, named=True)
    assert row["gene_status"] == "conflict"
    assert row["status"] == "Unresolved"


def test_unknown_alternative_does_not_claim_all_known_alleles_conflict():
    row = run(
        receptors=[receptor(trbv="TRBV6-5*01,NA")], references=[reference(trbv="TRBV6-5*03")]
    ).row(0, named=True)
    assert row["gene_status"] == "unresolved"


def test_identical_reported_typings_share_cache_across_donor_ids(monkeypatch):
    import tcr_workbench.matching as matching

    original = matching.donor_compatibility
    calls = []

    def recorded(restriction, alleles):
        calls.append((restriction, alleles))
        return original(restriction, alleles)

    monkeypatch.setattr(matching, "donor_compatibility", recorded)
    rows = run(
        receptors=[
            receptor(receptor_id="r1", donor_id="d1"),
            receptor(receptor_id="r2", donor_id="d2"),
        ],
        donors=[
            {"donor_id": identity, "hla": allele}
            for identity, alleles in [
                ("d1", ["A*02:01", "B*07:02", "A*02:01"]),
                ("d2", ["B*07:02", "A*02:01"]),
            ]
            for allele in alleles
        ],
    )
    assert rows["status"].to_list() == ["Candidate", "Candidate"]
    assert rows["donor_id"].to_list() == ["d1", "d2"]
    assert len(calls) == 1


def test_missing_donor_and_unknown_hla_keep_evidence_unresolved():
    rows = run(receptors=[receptor(donor_id="unknown")]).to_dicts()
    assert rows[0]["status"] == "Unresolved"
    assert rows[0]["reference_id"] == "ref1"
    assert "typing is missing" in rows[0]["reason"]
    assert run(references=[reference(hla=None)])["hla_status"].to_list() == ["unresolved"]


def test_class_ii_requires_both_dq_chains():
    hla = "DQA1*05:01/DQB1*02:01"
    kwargs = {"references": [reference(hla=hla)], "panel": [{"peptide": PEPTIDE, "hla": hla}]}
    assert run(donors=[{"donor_id": "d1", "hla": hla}], **kwargs)["status"].to_list() == [
        "Candidate"
    ]
    assert run(donors=[{"donor_id": "d1", "hla": "DQB1*02:01"}], **kwargs)["status"].to_list() == [
        "Unresolved"
    ]


@pytest.mark.parametrize(
    "gene,status",
    [
        ("TRBV19,TRBV6-5", "unresolved"),
        (None, "unresolved"),
        ("TRBV6-5", "conflict"),
        ("TRBV19*01", "unresolved"),
    ],
)
def test_gene_ambiguity_and_conflicts_are_not_erased(gene, status):
    row = run(receptors=[receptor(trbv=gene)]).row(0, named=True)
    assert row["gene_status"] == status
    assert status in row["reason"]
    if status == "conflict":
        assert row["status"] == "Unresolved"


def test_dual_alpha_alternatives_and_nonproductive_sequences_are_visible():
    row = run(receptors=[receptor(pairing_status="dual_alpha")]).row(0, named=True)
    assert "dual_alpha" in row["reason"]
    row = run(references=[reference(cdr3a="CA*RDSNYQLIW")]).row(0, named=True)
    assert row["evidence_type"] == "exact_single_chain"
    assert "nonproductive" in row["reason"]
    row = run(receptors=[receptor(cdr3a=None, cdr3b=None)]).row(0, named=True)
    assert row["status"] == "Unresolved"
    assert "No usable" in row["reason"]


def test_deduplicated_search_preserves_independent_reference_rows_and_ranks():
    refs = [
        reference(reference_id="z", evidence="multimer"),
        reference(reference_id="a", evidence="functional"),
        reference(reference_id="single", cdr3a=None),
        reference(reference_id="near", cdr3a="CAVRDSNYQIIW"),
    ]
    result = run(references=refs)
    assert result["reference_id"].to_list() == ["a", "z", "single", "near"]
    assert result["evidence_rank"].to_list() == [1, 2, 3, 4]
    assert run(references=list(reversed(refs))).equals(result)


def test_every_receptor_retained_and_donors_not_crossed():
    rows = [
        receptor(),
        receptor(receptor_id="r2", donor_id="d2"),
        receptor(receptor_id="r3", cdr3a=None, cdr3b=None),
    ]
    result = run(receptors=rows)
    assert result["receptor_id"].to_list() == ["r1", "r2", "r3"]
    assert result["status"].to_list() == ["Candidate", "Unresolved", "Unresolved"]


@pytest.mark.parametrize("distance", [-1, 4, True, 1.1])
def test_invalid_distance_boundary(distance):
    with pytest.raises(ValueError, match="max_distance"):
        run(max_distance=distance)


def test_loaders_validate_normalize_and_preserve_extra_metadata(tmp_path):
    refs = tmp_path / "refs.tsv"
    pl.DataFrame([reference(extra_note="kept", hla="hla-a*02:01", cdr3a="cav r")]).write_csv(
        refs, separator="\t"
    )
    with pytest.raises(ValueError, match="row 2"):
        read_references(refs)
    pl.DataFrame([reference(extra_note="kept", hla="hla-a*02:01", cdr3a="ca*rdsnyqliw")]).write_csv(
        refs, separator="\t"
    )
    loaded = read_references(refs)
    assert loaded["extra_note"][0] == "kept"
    assert loaded["cdr3a"][0] == "CA*RDSNYQLIW"
    assert loaded["hla"][0] == "HLA-A*02:01"
    panel_path = tmp_path / "panel.csv"
    pl.DataFrame([{"peptide": PEPTIDE.lower(), "hla": ""}]).write_csv(panel_path)
    assert read_panel(panel_path)["hla"][0] is None
    assert read_panel(panel_path)["peptide"][0] == PEPTIDE
    donor_path = tmp_path / "donors.csv"
    pl.DataFrame([{"donor_id": "d1", "hla": "DQB1*02:01/DQA1*05:01"}]).write_csv(donor_path)
    assert read_donors(donor_path)["hla"][0] == "HLA-DQA1*05:01/HLA-DQB1*02:01"


def test_duplicate_reference_id_fails_with_context():
    with pytest.raises(ValueError, match="reference_id must be unique"):
        run(references=[reference(), reference()])


def test_duplicate_reference_error_identifies_first_duplicate_in_wide_frame():
    from tcr_workbench.matching import _unique_references

    frame = pl.DataFrame(
        {
            "reference_id": ["unique", "first", "second", "first", "second"],
            **{f"extra_{i}": ["metadata"] * 5 for i in range(100)},
        }
    )
    with pytest.raises(ValueError, match="duplicate: 'first'"):
        _unique_references(frame)


def test_reference_loader_accepts_utf8_bom(tmp_path):
    path = tmp_path / "references.csv"
    path.write_text("\ufeff" + pl.DataFrame([reference()]).write_csv())
    assert read_references(path)["reference_id"].to_list() == ["ref1"]


@pytest.mark.parametrize(
    "bad",
    [
        {"hla": "bad HLA"},
        {"cdr3b": "CASS invalid"},
        {"source": ""},
        {"evidence": ""},
        {"cdr3a": None, "cdr3b": None},
    ],
)
def test_off_panel_reference_rows_still_receive_complete_validation(tmp_path, bad):
    path = tmp_path / "references.csv"
    invalid = reference(reference_id="off-panel", peptide="NLVPMVATV", **bad)
    pl.DataFrame([reference(), invalid]).write_csv(path)
    with pytest.raises(ValueError, match="references row 3"):
        read_references(path)
    with pytest.raises(ValueError, match="references row 3"):
        run(references=[reference(), invalid])


def test_multiple_panel_restrictions_preserve_cartesian_evidence_once():
    panel = [{"peptide": PEPTIDE, "hla": hla} for hla in ["A*02:01", "A*03:01", "B*07:02"]]
    rows = run(
        references=[reference(reference_id=str(i), hla=None) for i in range(3)], panel=panel + panel
    )
    assert rows.height == 9
    assert rows["evidence_rank"].to_list() == list(range(1, 10))
    assert rows.select("reference_id", "panel_hla").n_unique() == 9


def test_duplicate_loader_headers_are_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "reference_id,cdr3b,cdr3b,peptide,source,evidence\n"
        "r,CASSF,CAVVF,GILGFVFTL,test,functional\n"
    )
    with pytest.raises(ValueError, match="duplicate column names"):
        read_references(path)


def test_spaced_gene_alternative_can_support_reference_without_conflict():
    row = run(receptors=[receptor(trbv="TRBV6-5, TRBV19")]).row(0, named=True)
    assert row["gene_status"] == "unresolved"
    assert row["status"] == "Candidate"


def test_query_cache_bounds_actual_match_count_without_dropping_provenance():
    index = _SequenceIndex(pl.DataFrame({"cdr3b": [BETA] * 3000}), "cdr3b", 0)
    assert len(index.find(BETA)) == 3000
    assert len(index.find(BETA)) == 3000
    assert index.find.cache_info().currsize == 0
    assert index.find.cache_info().cached_hits == 0

    queries = [BETA, "CASSLGQETQFF", "CASSLGQEAQYF"]
    index = _SequenceIndex(pl.DataFrame({"cdr3b": queries * 4}), "cdr3b", 1)
    index.find.max_hits = 16
    for query in queries:
        assert len(index.find(query)) >= 4
    assert index.find.cache_info().cached_hits <= 16


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_native_bucketed_index_matches_naive_exact_edit_search(budget):
    rng = random.Random(21)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["C" + "".join(rng.choices(alphabet, k=rng.randrange(7, 12))) + "F" for _ in range(80)]
    seqs += [seqs[0], seqs[0][:-1] + "AF", seqs[0][:3] + "A" + seqs[0][4:]]
    index = _SequenceIndex(pl.DataFrame({"cdr3b": seqs}), "cdr3b", budget)
    for query in seqs[:12]:
        expected = {
            (i, Levenshtein.distance(query, ref))
            for i, ref in enumerate(seqs)
            if Levenshtein.distance(query, ref) <= budget
        }
        assert set(index.find(query)) == expected
        assert set(index.find(query)) == expected
    assert index.find.cache_info().hits >= 12


def test_empty_reference_or_panel_returns_unresolved():
    refs = pl.DataFrame(
        schema={
            "reference_id": pl.String,
            "peptide": pl.String,
            "source": pl.String,
            "evidence": pl.String,
        }
    )
    result = screen(
        pl.DataFrame([receptor()]),
        refs,
        pl.DataFrame([{"peptide": PEPTIDE, "hla": "A*02:01"}]),
        pl.DataFrame([{"donor_id": "d1", "hla": "A*02:01"}]),
    )
    assert result["status"].to_list() == ["Unresolved"]
    result = screen(
        pl.DataFrame([receptor()]),
        pl.DataFrame([reference()]),
        pl.DataFrame(schema={"peptide": pl.String, "hla": pl.String}),
        pl.DataFrame([{"donor_id": "d1", "hla": "A*02:01"}]),
    )
    assert result["status"].to_list() == ["Unresolved"]


@pytest.mark.parametrize("mixed_case", [False, True])
def test_prevalidated_cli_path_matches_public_screen(tmp_path, mixed_case):
    from polars.testing import assert_frame_equal

    from tcr_workbench.ingest import read_receptors
    from tcr_workbench.matching import _screen_validated

    receptor_path = tmp_path / "receptors.csv"
    reference_path = tmp_path / "reference.csv"
    panel_path = tmp_path / "panel.csv"
    donor_path = tmp_path / "donors.csv"
    pl.DataFrame([{**receptor(), "cell_id": "c1"}]).write_csv(receptor_path)
    pl.DataFrame([reference(), reference(reference_id="other", cdr3b="CASSLGQETQFF")]).write_csv(
        reference_path
    )
    pl.DataFrame({"peptide": [PEPTIDE], "hla": ["A*02:01"]}).write_csv(panel_path)
    pl.DataFrame({"donor_id": ["d1"], "hla": ["A*02:01"]}).write_csv(donor_path)
    if mixed_case:
        for path in (receptor_path, reference_path, panel_path, donor_path):
            raw = pl.read_csv(path, infer_schema=False)
            columns = [
                name
                for name in ("cdr3a", "cdr3b", "trav", "traj", "trbv", "trbj", "hla", "peptide")
                if name in raw.columns
            ]
            raw.with_columns(
                pl.concat_str(pl.lit(" "), pl.col(columns).str.to_lowercase(), pl.lit(" "))
            ).write_csv(path)
    frames = (
        read_receptors(receptor_path).receptors,
        read_references(reference_path),
        read_panel(panel_path),
        read_donors(donor_path),
    )
    assert_frame_equal(screen(*frames), _screen_validated(*frames))
    assert "Candidate" in _screen_validated(*frames)["status"]
    for invalid in (-1, 4, True):
        with pytest.raises(ValueError, match="max_distance"):
            _screen_validated(*frames, max_distance=invalid)


def test_columnar_evidence_batches_preserve_all_rows_and_unresolved_receptors():
    # Cross the 8192-row buffer boundary and then emit a no-support receptor.
    refs = [reference(reference_id=f"ref-{i:05}") for i in range(8200)]
    result = run(
        references=refs,
        receptors=[receptor(), receptor(receptor_id="r2", cdr3a=None, cdr3b=None)],
        max_distance=0,
    )
    supported = result.filter(pl.col("receptor_id") == "r1")
    assert supported.height == 8200
    assert supported["reference_id"].to_list() == [f"ref-{i:05}" for i in range(8200)]
    assert supported["evidence_rank"].to_list() == list(range(1, 8201))
    unresolved = result.filter(pl.col("receptor_id") == "r2")
    assert unresolved.height == 1
    assert unresolved["status"].item() == "Unresolved"
    assert unresolved["evidence_rank"].item() == 1
