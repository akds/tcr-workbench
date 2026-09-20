"""Recall regressions for the lossless four-gram Levenshtein prefilter.

The oracle deliberately scans every usable reference with native edit distance;
its independence matters more here than duplicating the prefilter's internals.
"""

import random

import polars as pl
import pytest
from rapidfuzz.distance import Levenshtein

from tcr_workbench import matching
from tcr_workbench.matching import _SequenceIndex


_AMINO = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def small_prefilter(monkeypatch):
    """Exercise full prefilter recall cheaply, independently of tuning thresholds."""
    monkeypatch.setattr(_SequenceIndex, "_PREFILTER_MIN_SEQUENCES", 256)


def _fill_references(sequences, seed, count=320):
    """Keep input order and add distinct usable CDR3s above the activation limit."""
    rng = random.Random(seed)
    result = list(dict.fromkeys(sequences))
    seen = set(result)
    while len(result) < count:
        sequence = "C" + "".join(rng.choices(_AMINO, k=rng.randrange(6, 28))) + "F"
        if sequence not in seen:
            result.append(sequence)
            seen.add(sequence)
    return result


def _assert_oracle(index, sequences, query, budget):
    expected = {}
    for row_index, reference in enumerate(sequences):
        distance = Levenshtein.distance(query, reference)
        if distance <= budget:
            expected[row_index] = distance
    found = index.find(query)
    assert len(found) == len(expected), (query, budget, found, expected)
    assert dict(found) == expected, (query, budget, found, expected)


def _edit(sequence, operation, position, residue="Y"):
    if operation == "insert":
        return sequence[:position] + residue + sequence[position:]
    if operation == "delete":
        return sequence[:position] + sequence[position + 1 :]
    replacement = "A" if sequence[position] == residue else residue
    return sequence[:position] + replacement + sequence[position + 1 :]


@pytest.mark.parametrize("chain", ["cdr3a", "cdr3b"])
@pytest.mark.parametrize("budget", [1, 2, 3])
def test_prefilter_seeded_recall_across_edit_combinations(chain, budget, small_prefilter):
    rng = random.Random(437 + budget)
    sequences = _fill_references([], seed=737 + budget)
    index = _SequenceIndex(pl.DataFrame({chain: sequences}), chain, budget)
    assert index._qgrams is not None
    optimized_queries = 0
    for query_number in range(105):
        source_index = rng.randrange(len(sequences))
        query = sequences[source_index]
        for _ in range(query_number % (budget + 1)):
            operation = rng.choice(("insert", "delete", "substitute"))
            position = rng.randrange(1, len(query) - 1)
            query = _edit(query, operation, position, rng.choice(_AMINO))
        optimized_queries += index._prefilter_choices(query) is not None
        _assert_oracle(index, sequences, query, budget)
        # Every query was produced with at most the allowed number of edits.
        assert source_index in dict(index.find(query))
    assert optimized_queries >= 25


@pytest.mark.parametrize("budget", [1, 2, 3])
@pytest.mark.parametrize("motif", ["A", "AG", "ACDEFGHIKLMNPQRSTVWY"])
def test_prefilter_all_internal_edit_positions_and_partition_boundaries(
    budget, motif, small_prefilter
):
    # Exact activation length makes the chosen segments four residues long.
    # +1 also exercises uneven partition lengths and their boundary insertions.
    minimum_length = 4 * (budget + 1)
    for query_length in sorted({minimum_length, minimum_length + 1, minimum_length + budget}):
        query = "C" + (motif * query_length)[: query_length - 2] + "F"
        variants = [query]
        for position in range(1, len(query)):
            variants.append(_edit(query, "insert", position))
            if position < len(query) - 1:
                variants.append(_edit(query, "delete", position))
                variants.append(_edit(query, "substitute", position))
            # Consecutive insertions/deletions test runs of indels as well as
            # single edits. The longest query keeps even k deletions indexed.
            for operation in ("insert", "delete", "substitute"):
                edited = query
                for edit_number in range(budget):
                    offset = edit_number if operation == "substitute" else 0
                    edit_position = min(position + offset, len(edited) - 2)
                    edited = _edit(edited, operation, edit_position)
                variants.append(edited)
            # Spread edits across partition boundaries. With three edits this
            # combines insertion, substitution, and deletion in the same CDR3.
            mixed = _edit(query, "insert", position)
            if budget >= 2:
                mixed = _edit(mixed, "substitute", 1 + (position + 4) % (len(mixed) - 2))
            if budget >= 3:
                mixed = _edit(mixed, "delete", 1 + (position + 8) % (len(mixed) - 2))
            variants.append(mixed)
        sequences = _fill_references(variants, seed=query_length + budget)
        index = _SequenceIndex(pl.DataFrame({"cdr3b": sequences}), "cdr3b", budget)
        _assert_oracle(index, sequences, query, budget)
        # Reverse the orientation as well: deletions in a reference become
        # insertions in the query, with different query partition boundaries.
        for variant in variants[1::7]:
            _assert_oracle(index, sequences, variant, budget)

        # A dense family can intentionally fall back to full native scanning.
        # This sparse fixture forces the prefilter for the same edit positions,
        # including low-complexity CDR3s, so fallback cannot hide a recall bug.
        sparse_sequences = _fill_references([query], seed=query_length + budget)
        sparse_index = _SequenceIndex(pl.DataFrame({"cdr3b": sparse_sequences}), "cdr3b", budget)
        for variant in variants:
            if len(variant) >= 4 * (budget + 1):
                assert sparse_index._prefilter_choices(variant) is not None
            _assert_oracle(sparse_index, sparse_sequences, variant, budget)


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_prefilter_short_queries_use_full_recall_fallback(budget, small_prefilter):
    query_lengths = (3, 4 * (budget + 1) - 1)
    queries = ["C" + "A" * (length - 2) + "F" for length in query_lengths]
    references = []
    for query in queries:
        references.extend([query, query[:1] + "Y" + query[1:], query[:1] + query[2:]])
    # The shortest deletion loses its usable CDR3 anchor/length and is omitted.
    references = [sequence for sequence in references if len(sequence) >= 3]
    sequences = _fill_references(references, seed=983)
    index = _SequenceIndex(pl.DataFrame({"cdr3b": sequences}), "cdr3b", budget)
    assert index._qgrams is not None
    for query in queries:
        assert index._prefilter_choices(query) is None
        _assert_oracle(index, sequences, query, budget)


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_prefilter_retains_duplicate_provenance_and_original_row_offsets(budget, small_prefilter):
    query = "CASSLGQETQYFAGGF"
    usable = _fill_references([query, query[:7] + "A" + query[7:]], seed=37)
    # Invalid rows are deliberately interspersed: index IDs must refer to the
    # input frame, not a compacted list of usable/distinct sequences.
    sequences = [None, query, "CAXXF", *usable, "AAAAF", query, query, "CASS*"]
    frame = pl.DataFrame(
        {"cdr3b": sequences, "reference_id": [f"evidence-{i}" for i in range(len(sequences))]}
    )
    index = _SequenceIndex(frame, "cdr3b", budget)
    usable_rows = [1, *range(3, 3 + len(usable)), 4 + len(usable), 5 + len(usable)]
    expected = {
        row: Levenshtein.distance(query, sequences[row])
        for row in usable_rows
        if Levenshtein.distance(query, sequences[row]) <= budget
    }
    for _ in range(2):
        found = index.find(query)
        assert len(found) == len(expected)
        assert dict(found) == expected
        assert [row for row, distance in found if distance == 0] == [
            1, 3, 4 + len(usable), 5 + len(usable)
        ]
    if budget == 0:
        assert index._qgrams is None
        assert index.find(query[:4] + "Y" + query[5:]) == ()


def test_prefilter_production_gates_and_large_index_activation():
    # No threshold override: production avoids index construction for small
    # references and known small query workloads that cannot amortize its cost.
    query = "CASSLGQETQYFAGGF"
    sequences = _fill_references([query], seed=407, count=4096)
    frame = pl.DataFrame({"cdr3b": sequences})
    small_index = _SequenceIndex(frame.head(320), "cdr3b", 1)
    assert small_index._qgrams is None
    _assert_oracle(small_index, sequences[:320], query, 1)

    few_queries = _SequenceIndex(frame, "cdr3b", 1, query_count=1023)
    assert few_queries._qgrams is None
    _assert_oracle(few_queries, sequences, query, 1)

    large_index = _SequenceIndex(frame, "cdr3b", 1, query_count=1024)
    assert large_index._qgrams is not None
    assert large_index._prefilter_choices(query) is not None
    _assert_oracle(large_index, sequences, query, 1)


@pytest.mark.parametrize("alpha", [None, "CAF"])
def test_screen_skips_absent_chain_index_but_keeps_short_observed_chain(alpha, monkeypatch):
    constructed = {}

    class RecordingIndex(_SequenceIndex):
        def __init__(self, rows, chain, max_distance, **kwargs):
            super().__init__(rows, chain, max_distance, **kwargs)
            constructed[chain] = (rows.height, kwargs["query_count"], self)

    monkeypatch.setattr(matching, "_SequenceIndex", RecordingIndex)
    beta = "CASSLGQETQYF"
    peptide = "GILGFVFTL"
    hla = "A*02:01"
    references = pl.DataFrame(
        {
            "reference_id": ["matching", "unrelated"],
            "cdr3a": ["CAF", "CAW"],
            "cdr3b": [beta, "CAAAAAAAAF"],
            "peptide": [peptide] * 2,
            "hla": [hla] * 2,
            "source": ["fixture"] * 2,
            "evidence": ["functional"] * 2,
        }
    )
    result = matching.screen(
        pl.DataFrame(
            {"receptor_id": ["query"], "donor_id": ["d1"], "cdr3a": [alpha], "cdr3b": [beta]}
        ),
        references,
        pl.DataFrame({"peptide": [peptide], "hla": [hla]}),
        pl.DataFrame({"donor_id": ["d1"], "hla": [hla]}),
    )
    assert constructed["cdr3a"][0] == (0 if alpha is None else references.height)
    assert constructed["cdr3a"][1] == 0
    assert constructed["cdr3b"][0] == references.height
    row = result.row(0, named=True)
    assert result.height == 1
    assert row["reference_id"] == "matching"
    assert row["status"] == "Candidate"
    assert row["beta_distance"] == 0
    assert row["alpha_distance"] == (None if alpha is None else 0)
    assert row["evidence_type"] == ("exact_single_chain" if alpha is None else "exact_paired")
