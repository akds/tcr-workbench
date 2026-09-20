import csv
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tcr_workbench.ingest import read_receptors
from tcr_workbench.models import IngestOptions, IngestResult, InputError


def write_table(tmp_path, rows, name="input.csv", delimiter=","):
    path = tmp_path / name
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return path


def chain(cell="c1", locus="TRA", cdr3="CAVF", **kwargs):
    return dict(barcode=cell, chain=locus, cdr3=cdr3, productive="true", **kwargs)


def paired(cell="c1", donor="d1", alpha="CAVF", beta="CASSF", **kwargs):
    return dict(cell_id=cell, donor_id=donor, cdr3a=alpha, cdr3b=beta, **kwargs)


def codes(result):
    return {item["code"]: item["count"] for item in result.qc}


def test_paired_deduplicates_cells_but_preserves_original_records(tmp_path):
    path = write_table(tmp_path, [paired(), paired("c2"), paired("c2")])
    result = read_receptors(path)
    assert result.format == "paired"
    assert result.receptors.height == 1
    assert result.receptors["cell_count"].item() == 2
    assert result.cells.height == 2
    assert result.chains.height == 6
    assert set(result.chains["input_cell_id"]) == {"c1", "c2"}
    assert result.chains["source_row"].n_unique() == 3


def test_deduplication_keeps_donors_and_alleles_separate(tmp_path):
    result = read_receptors(
        write_table(
            tmp_path,
            [
                paired(trav="TRAV1-2*01"),
                paired("c2", trav="TRAV1-2*02"),
                paired("c1", donor="d2", trav="TRAV1-2*01"),
            ],
        )
    )
    assert result.receptors.height == 3
    assert result.cells.height == 3


def test_hash_identity_is_stable_across_row_order(tmp_path):
    rows = [paired(), paired("c2", beta="CASGF"), paired("c3", donor="d2")]
    first = read_receptors(write_table(tmp_path, rows, "first.csv"))
    second = read_receptors(write_table(tmp_path, rows[::-1], "second.csv"))
    assert_frame_equal(first.receptors, second.receptors)
    assert_frame_equal(first.cells, second.cells)


def test_dual_alpha_retains_all_alternatives_and_nonproductive_chain(tmp_path):
    rows = [
        chain(),
        chain(cdr3="CAGF"),
        chain(locus="TRB", cdr3="CASSF"),
        dict(barcode="c1", chain="TRA", cdr3="CAX*F", productive="false"),
    ]
    result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    assert result.chains.height == 4
    assert result.receptors.height == 2
    assert set(result.receptors["cdr3a"]) == {"CAVF", "CAGF"}
    assert set(result.receptors["pairing_status"]) == {"dual_alpha"}
    assert codes(result)["nonproductive"] == 1
    assert codes(result)["dual_alpha"] == 1
    assert result.cells.height == 2


def test_never_pairs_between_cells_or_donors(tmp_path):
    rows = [
        chain("c1", donor_id="d1"),
        chain("c2", "TRB", "CASSF", donor_id="d1"),
        chain("c1", "TRB", "CASGF", donor_id="d2"),
    ]
    result = read_receptors(write_table(tmp_path, rows))
    assert result.receptors.height == 3
    assert not result.receptors.filter(
        pl.col("cdr3a").is_not_null() & pl.col("cdr3b").is_not_null()
    ).height


def test_airr_without_cell_id_remains_unpaired(tmp_path):
    rows = [
        dict(sequence_id="a", locus="TRA", junction_aa="CAVF", productive="T"),
        dict(sequence_id="b", locus="TRB", junction_aa="CASSF", productive="T"),
    ]
    result = read_receptors(write_table(tmp_path, rows, "airr.tsv", "\t"), donor_id="d1")
    assert result.format == "airr"
    assert result.receptors.height == 2
    assert result.cells["cell_id"].null_count() == 2
    assert set(result.receptors["pairing_status"]) == {"unpaired_airr"}
    assert codes(result)["unpaired_airr"] == 2


def test_airr_uses_junction_not_anchor_excluding_cdr3(tmp_path):
    rows = [
        dict(
            sequence_id="a",
            cell_id="c1",
            locus="TRA",
            junction_aa="CAVF",
            cdr3_aa="AV",
            productive="T",
        )
    ]
    result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    assert result.receptors["cdr3a"].item() == "CAVF"
    assert result.chains["input_cdr3_aa"].item() == "AV"
    rows[0].pop("junction_aa")
    with pytest.raises(InputError, match="junction_aa"):
        read_receptors(write_table(tmp_path, rows), donor_id="d1")


@pytest.mark.parametrize("bad", ["AVF", "CAV", "CAXF", "CA*F", ""])
def test_unsafe_sequence_preserved_with_unresolved_placeholder(tmp_path, bad):
    result = read_receptors(write_table(tmp_path, [paired(alpha=bad, beta="")]))
    assert result.receptors.height == 1
    assert result.receptors["cdr3a"].item() is None
    assert result.receptors["cdr3b"].item() is None
    assert result.receptors["pairing_status"].item() == "unresolved_sequence"
    assert result.cells.height == 1
    assert result.chains["input_cdr3a"].item() == (bad or None)


def test_nonproductive_and_unsupported_locus_only_cells_are_not_dropped(tmp_path):
    rows = [
        dict(barcode="c1", chain="TRA", cdr3="CAVF", productive="false"),
        chain("c2", "TRG", "CAGF"),
    ]
    result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    assert result.chains.height == 2
    assert result.cells.height == 2
    assert set(result.receptors["pairing_status"]) == {"no_productive_chains"}
    assert codes(result)["unsupported_locus"] == 1


def test_candidate_expansion_is_bounded_before_join(tmp_path):
    rows = [chain(cdr3="CA" + aa + "F") for aa in "ADEGH"]
    rows += [chain(locus="TRB", cdr3="CA" + aa + "F") for aa in "ADEGH"]
    result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    assert result.chains.height == 10
    assert result.receptors.height == 1
    assert result.receptors["pairing_status"].item() == "excess_chains"
    assert result.receptors["cdr3a"].item() is None
    assert codes(result)["excess_chains"] == 1
    expanded = read_receptors(write_table(tmp_path, rows), donor_id="d1", max_pairings_per_cell=25)
    assert expanded.receptors.height == 25
    assert set(expanded.receptors["pairing_status"]) == {"ambiguous_pairing"}


def test_explicit_pair_rows_do_not_create_unobserved_pairs(tmp_path):
    rows = [paired(alpha="CAVF", beta="CASSF"), paired(alpha="CAGF", beta="CASGF")]
    result = read_receptors(write_table(tmp_path, rows))
    assert set(result.receptors.select("cdr3a", "cdr3b").iter_rows()) == {
        ("CAVF", "CASSF"),
        ("CAGF", "CASGF"),
    }
    assert set(result.receptors["pairing_status"]) == {"ambiguous_input_pairs"}
    assert codes(result)["multiple_paired_rows"] == 1


def test_unknown_productivity_and_allele_ambiguity_are_not_silently_resolved(tmp_path):
    rows = [chain(v_gene="TRAV1-2*01,TRAV1-2*02")]
    rows[0].pop("productive")
    result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    assert result.receptors["trav"].item() == "TRAV1-2*01,TRAV1-2*02"
    assert result.chains["productive"].item() is None
    assert codes(result)["productivity_unknown"] == 1
    assert codes(result)["gene_ambiguity"] == 1


@pytest.mark.parametrize(
    "rows, kwargs, match",
    [
        ([dict(cell_id="c", cdr3a="CAVF", cdr3b="CASSF")], {}, "Missing donor_id"),
        ([paired()], {"donor_id": "different"}, "conflicts"),
        ([dict(barcode="", chain="TRA", cdr3="CAVF")], {"donor_id": "d"}, "barcode"),
        (
            [dict(barcode="c", chain="TRA", cdr3="CAVF", productive="maybe")],
            {"donor_id": "d"},
            "boolean",
        ),
        ([paired(alpha="CA3F")], {}, "amino acid"),
        ([chain(contig_id="same"), chain(contig_id="same")], {"donor_id": "d"}, "Duplicate chain"),
    ],
)
def test_schema_failures_are_actionable(tmp_path, rows, kwargs, match):
    with pytest.raises(InputError, match=match):
        read_receptors(write_table(tmp_path, rows), **kwargs)


def test_duplicate_column_names_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("cell_id,donor_id,cdr3a,cdr3b,cdr3a\nc,d,CAVF,CASSF,CAGF\n")
    with pytest.raises(InputError, match="Duplicate column"):
        read_receptors(path)


def test_source_row_column_preserved_and_internal_names_cannot_collide(tmp_path):
    result = read_receptors(
        write_table(tmp_path, [paired(source_row="original", input_cell_id="data")])
    )
    assert result.chains["source_row"].to_list() == [2, 2]
    assert result.chains["input_source_row"].to_list() == ["original", "original"]
    assert result.chains["input_input_cell_id"].to_list() == ["data", "data"]


def test_boundary_options_reject_unexpected_fields(tmp_path):
    with pytest.raises(ValueError):
        IngestOptions(path=Path("a"), max_pairings_per_cell="16")
    with pytest.raises(ValueError):
        IngestOptions(path=Path("a"), extra="field")
    good = read_receptors(write_table(tmp_path, [paired()]))
    with pytest.raises(ValueError, match="dtypes"):
        IngestResult(
            chains=good.chains,
            receptors=good.receptors.with_columns(pl.col("cell_count").cast(pl.String)),
            cells=good.cells,
            qc=[],
        )


def test_columnar_bulk_ingestion(tmp_path):
    # Many observations of the same clone exercise grouping and prevent per-row validation regressions.
    count = 12000
    frame = pl.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(count)],
            "donor_id": ["d"] * count,
            "cdr3a": ["CAVF"] * count,
            "cdr3b": ["CASSF"] * count,
        }
    )
    path = tmp_path / "large.csv"
    frame.write_csv(path)
    result = read_receptors(path)
    assert result.receptors["cell_count"].item() == count
    assert result.chains.height == 2 * count
    assert result.cells.height == count


@pytest.mark.parametrize("flag", ["is_cell", "high_confidence"])
def test_negative_10x_quality_flags_preserved_but_never_matched(tmp_path, flag):
    result = read_receptors(write_table(tmp_path, [chain(**{flag: "false"})]), donor_id="d1")
    assert result.chains[flag].item() is False
    assert result.chains["eligible"].item() is False
    assert result.receptors["cdr3a"].item() is None
    assert result.receptors["pairing_status"].item() == "excluded_quality"
    assert result.cells.height == 1


def test_paired_productivity_applies_per_chain(tmp_path):
    result = read_receptors(
        write_table(tmp_path, [paired(productive_a="false", productive_b="true")])
    )
    assert result.receptors["cdr3a"].item() is None
    assert result.receptors["cdr3b"].item() == "CASSF"
    assert result.receptors["pairing_status"].item() == "single_beta"
    assert codes(result)["nonproductive"] == 1


def test_locus_inconsistent_gene_fails_at_boundary(tmp_path):
    with pytest.raises(InputError, match="locus-inconsistent"):
        read_receptors(write_table(tmp_path, [chain(v_gene="TRBV1*01")]), donor_id="d1")


def test_missing_airr_cell_ids_count_observations_not_cells(tmp_path):
    result = read_receptors(
        write_table(
            tmp_path,
            [
                {"sequence_id": "one", "locus": "TRA", "junction_aa": "CAVF"},
                {"sequence_id": "two", "locus": "TRA", "junction_aa": "CAVF"},
            ],
        ),
        donor_id="d1",
    )
    assert result.receptors["cell_count"].item() == 0
    assert result.receptors["observation_count"].item() == 2


def test_unknown_receptors_do_not_form_a_synthetic_clone(tmp_path):
    result = read_receptors(
        write_table(
            tmp_path,
            [
                paired("unsafe1", alpha="CAXF", beta=""),
                paired("unsafe2", alpha="CAXF", beta=""),
                paired("empty", alpha="", beta=""),
            ],
        )
    )
    assert result.receptors.height == 3
    assert result.receptors["receptor_id"].n_unique() == 3
    assert result.receptors["cell_count"].to_list() == [1, 1, 1]


def test_excess_chain_cells_do_not_form_a_synthetic_clone(tmp_path):
    rows = [
        chain(cell, locus, "CA" + aa + "F")
        for cell in ("c1", "c2")
        for locus in ("TRA", "TRB")
        for aa in "ADEGH"
    ]
    result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    assert result.receptors.height == 2
    assert result.receptors["receptor_id"].n_unique() == 2
    assert result.receptors["cell_count"].to_list() == [1, 1]


def test_unknown_airr_sequences_are_independent_observations(tmp_path):
    rows = [{"sequence_id": name, "locus": "TRA", "junction_aa": "CAXF"} for name in ("s1", "s2")]
    first = read_receptors(write_table(tmp_path, rows, "first.csv"), donor_id="d1")
    second = read_receptors(write_table(tmp_path, rows[::-1], "second.csv"), donor_id="d1")
    assert first.receptors.height == 2
    assert first.receptors["cell_count"].to_list() == [0, 0]
    assert_frame_equal(first.receptors, second.receptors)


@pytest.mark.parametrize("column", ["sample_id", "library_id"])
def test_cross_library_barcode_collision_fails_loudly(tmp_path, column):
    with pytest.raises(InputError, match="Barcode collision"):
        read_receptors(
            write_table(tmp_path, [paired(**{column: "one"}), paired(**{column: "two"})])
        )


def test_normalized_missing_chain_expression_preserves_mixed_rows(tmp_path):
    rows = [
        paired("nulls", alpha=None, beta=None),
        paired("whitespace", alpha="  ", beta=" \t "),
        paired("beta", alpha=None),
        paired("alpha", beta=" "),
        paired("pair"),
        paired("gene_only", alpha=None, beta=None, trav="TRAV1-2"),
    ]
    result = read_receptors(write_table(tmp_path, rows))
    assert result.chains.height == 7
    assert set(result.cells["cell_id"]) == {row["cell_id"] for row in rows}
    assert "_both_cdr3_missing" not in result.chains.columns
    placeholders = result.chains.filter(pl.col("cell_id").is_in(["nulls", "whitespace"]))
    assert placeholders["cdr3"].to_list() == [None, None]
    assert placeholders["chain_observed"].to_list() == [False, False]
    linked = result.cells.join(result.receptors, on=["donor_id", "receptor_id"])
    assert linked.filter(pl.col("cell_id") == "whitespace")["unusable_chain_context"].item() == ""
    assert (
        linked.filter(pl.col("cell_id") == "gene_only")["unusable_chain_context"].item()
        == "alpha:incomplete_junction"
    )


@pytest.mark.parametrize(
    "alpha,extra,context",
    [
        ("CAXF", {}, "alpha:ambiguous_sequence"),
        ("AVF", {}, "alpha:incomplete_junction"),
        ("CAVF", {"productive_a": "false"}, "alpha:nonproductive"),
    ],
)
def test_excluded_partner_context_distinguishes_missing_and_unsafe_alpha(
    tmp_path, alpha, extra, context
):
    rows = [paired("absent", alpha=None), paired("unsafe", alpha=alpha, **extra)]
    result = read_receptors(write_table(tmp_path, rows))
    assert result.receptors.height == 2
    assert set(result.receptors["pairing_status"]) == {"single_beta"}
    assert set(result.receptors["unusable_chain_context"]) == {"", context}
    assert result.receptors["cell_count"].to_list() == [1, 1]


def test_excluded_beta_context_is_preserved_for_single_alpha(tmp_path):
    result = read_receptors(write_table(tmp_path, [paired(beta="CAXF")]))
    assert result.receptors["pairing_status"].item() == "single_alpha"
    assert result.receptors["unusable_chain_context"].item() == "beta:ambiguous_sequence"
    assert result.receptors["cdr3a"].item() == "CAVF"
    assert result.receptors["cdr3b"].item() is None


@pytest.mark.parametrize(
    "flag,context",
    [
        ("productive", "alpha:nonproductive"),
        ("high_confidence", "alpha:low_confidence"),
        ("is_cell", "alpha:noncell"),
    ],
)
def test_10x_excluded_partner_context_controls_deduplication(tmp_path, flag, context):
    rows = [chain("absent", "TRB", "CASSF")]
    for cell in ("unsafe1", "unsafe2"):
        alpha = chain(cell)
        alpha[flag] = "false"
        rows.extend([alpha, chain(cell, "TRB", "CASSF")])
    result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    assert result.receptors.height == 2
    uncertain = result.receptors.filter(pl.col("unusable_chain_context") == context)
    clean = result.receptors.filter(pl.col("unusable_chain_context") == "")
    assert uncertain["cell_count"].item() == 2
    assert clean["cell_count"].item() == 1
    assert set(result.receptors["pairing_status"]) == {"single_beta"}


def test_unsafe_additional_chain_context_survives_usable_pair(tmp_path):
    rows = [chain(), chain(locus="TRB", cdr3="CASSF"), chain(cdr3="CAXF")]
    result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    assert result.receptors["pairing_status"].item() == "paired"
    assert result.receptors["unusable_chain_context"].item() == "alpha:ambiguous_sequence"
    assert result.chains.height == 3


@pytest.mark.parametrize(
    "row",
    [
        paired("gene_only", alpha=None, beta=None, trav="TRAV1-2"),
        {"barcode": "gene_only", "chain": "TRA", "cdr3": None, "v_gene": "TRAV1-2"},
        {
            "sequence_id": "s1",
            "cell_id": "gene_only",
            "locus": "TRA",
            "junction_aa": None,
            "v_call": "TRAV1-2",
        },
    ],
)
def test_gene_only_unknown_productivity_does_not_assert_nonproductivity(tmp_path, row):
    result = read_receptors(write_table(tmp_path, [row]), donor_id="d1")
    assert result.chains["productive"].item() is None
    assert result.chains["junction_incomplete"].item() is True
    assert result.receptors["pairing_status"].item() == "unresolved_sequence"
    assert result.receptors["unusable_chain_context"].item() == "alpha:incomplete_junction"
    assert "nonproductive" not in codes(result)


def test_cellranger_none_sentinels_are_scoped_audited_and_raw_values_retained(tmp_path):
    row = {
        "barcode": "None", "chain": "TRA", "cdr3": "CAVF", "contig_id": "None",
        "v_gene": "None", "j_gene": " nOnE ", "d_gene": "None", "c_gene": "None",
        "productive": "None", "high_confidence": "None", "is_cell": "None",
    }
    result = read_receptors(write_table(tmp_path, [row]), donor_id="None")
    assert result.receptors["cdr3a"].item() == "CAVF"
    assert result.receptors["donor_id"].item() == "None"
    assert result.chains["cell_id"].item() == "None"
    assert result.chains["chain_id"].item() == "None"
    for column in ("v_call", "j_call", "productive", "high_confidence", "is_cell"):
        assert result.chains[column].item() is None
    for column in ("v_gene", "j_gene", "d_gene", "c_gene", "productive", "high_confidence", "is_cell"):
        assert result.chains["input_" + column].item() == row[column]
    assert codes(result)["cellranger_none_sentinel"] == 7
    assert codes(result)["productivity_unknown"] == 1


@pytest.mark.parametrize(
    "row",
    [paired(trav="None"), paired(productive="None"),
     {"locus": "TRA", "junction_aa": "CAVF", "productive": "None"}],
)
def test_none_sentinel_is_not_guessed_in_other_receptor_formats(tmp_path, row):
    with pytest.raises(InputError):
        read_receptors(write_table(tmp_path, [row]), donor_id="d1")


@pytest.mark.parametrize(
    "alpha,beta,flags,status",
    [
        ("CAVF", "CASSF", {"productive": "false"}, "no_productive_chains"),
        (None, "CASSF", {"productive": "false"}, "no_productive_chains"),
        ("CAVF", None, {"productive": "false"}, "no_productive_chains"),
        ("CAXF", "CASSF", {"productive": "false"}, "unresolved_sequence"),
        (None, None, {}, "unresolved_sequence"),
        ("CAVF", "CASSF", {"high_confidence": "false"}, "excluded_quality"),
        ("CAVF", "CASSF", {"is_cell": "false"}, "excluded_quality"),
        ("CAXF", "CASSF", {"is_cell": "false"}, "excluded_quality"),
    ],
)
def test_exclusion_status_and_context_agree_across_formats(tmp_path, alpha, beta, flags, status):
    paired_result = read_receptors(write_table(tmp_path, [paired(alpha=alpha, beta=beta, **flags)]))
    rows = [
        {"barcode": "c1", "chain": locus, "cdr3": seq, **flags}
        for locus, seq in (("TRA", alpha), ("TRB", beta)) if seq is not None
    ]
    if not rows:
        rows = [{"barcode": "c1", "chain": "TRA", "cdr3": None, **flags}]
    long_result = read_receptors(write_table(tmp_path, rows), donor_id="d1")
    airr_rows = [
        {"cell_id": row["barcode"], "locus": row["chain"], "junction_aa": row["cdr3"], **flags}
        for row in rows
    ]
    airr_result = read_receptors(write_table(tmp_path, airr_rows), donor_id="d1")
    for result in (paired_result, long_result, airr_result):
        assert result.receptors["pairing_status"].item() == status
        assert result.receptors["cdr3a"].item() is None
        assert result.receptors["cdr3b"].item() is None
        assert result.cells.height == 1
    # An explicit empty long-format chain is observed; the paired empty row is a placeholder.
    if alpha is not None or beta is not None:
        assert paired_result.receptors["unusable_chain_context"].item() == long_result.receptors["unusable_chain_context"].item()
        assert long_result.receptors["unusable_chain_context"].item() == airr_result.receptors["unusable_chain_context"].item()
