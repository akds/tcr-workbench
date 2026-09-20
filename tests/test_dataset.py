import json

import h5py
import numpy as np
import polars as pl
import pytest

from tcr_workbench.dataset import build_dataset, inspect_h5ad, read_h5ad_obs


def make_h5ad(path, *, bad_code=False):
    strings = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=["cell1", "cell2", "cell3"], dtype=strings)
        values = {
            "donor": ["d1", "d1", "d1"],
            "trav": ["TRAV21"] * 3,
            "traj": ["TRAJ6"] * 3,
            "cdr3a": ["CAVRPGGAGPFFVVF"] * 3,
            "trbv": ["TRBV7-9"] * 3,
            "trbj": ["TRBJ2-7"] * 3,
            "cdr3b": ["CASSLGQAYEQYF"] * 3,
        }
        for key, value in values.items():
            obs.create_dataset(key, data=value, dtype=strings)
        label = obs.create_group("benchmark_label")
        label.attrs["encoding-type"] = "categorical"
        label.create_dataset("categories", data=["negative", "GILGFVFTL A0201"], dtype=strings)
        label.create_dataset("codes", data=[0, 1, 9 if bad_code else -1])
        # A dangling external link makes any attempted matrix access fail.
        handle["X"] = h5py.ExternalLink("DOES_NOT_EXIST.h5", "/matrix")


def test_obs_only_categories_and_missing_values(tmp_path):
    path = tmp_path / "benchmark.h5ad"
    make_h5ad(path)
    info = inspect_h5ad(path)
    assert info["n_cells"] == 3
    assert info["expression_loaded"] is False
    frame = read_h5ad_obs(path, columns=["benchmark_label"])
    assert frame.columns == ["cell_id", "benchmark_label"]
    assert frame["benchmark_label"].to_list() == ["negative", "GILGFVFTL A0201", None]


def test_corrupt_categories_fail_loudly(tmp_path):
    path = tmp_path / "bad.h5ad"
    make_h5ad(path, bad_code=True)
    with pytest.raises(ValueError, match="category range"):
        read_h5ad_obs(path, ["benchmark_label"])
    with pytest.raises(ValueError, match="missing columns"):
        read_h5ad_obs(path, ["missing"])


def test_bridge_deduplicates_receptors_preserves_labels_no_leakage(tmp_path):
    path = tmp_path / "benchmark.h5ad"
    make_h5ad(path)
    output = tmp_path / "output"
    result = build_dataset(path, output)
    assert result["n_cells"] == 3
    assert result["n_receptors"] == 1
    assert result["n_mapped_cells"] == 3
    assert result["n_conflicting_clones"] == 1
    assert result["decoder_export"]["exported_pairs"] == 2
    assert "label" not in pl.read_csv(output / "decoder_input.csv").columns
    labels = pl.read_csv(output / "clone_labels.csv")
    assert labels.height == 3
    assert labels["n_cells"].sum() == 3
    assert labels["conflicting_labels"].all()
    cells = pl.read_csv(output / "cell_tcr_map.csv")
    assert cells["tcr_id"].n_unique() == 1
    assert "benchmark_label" not in cells.columns
    assert pl.read_csv(output / "cell_labels.csv")["benchmark_label"].to_list() == ["negative", "GILGFVFTL A0201", None]
    assert json.loads((output / "dataset_manifest.json").read_text())["expression_loaded"] is False


def test_nullable_encoding_and_duplicate_cell_index(tmp_path):
    path = tmp_path / "nullable.h5ad"
    make_h5ad(path)
    with h5py.File(path, "a") as handle:
        node = handle["obs"].create_group("flag")
        node.attrs["encoding-type"] = "nullable-boolean"
        node.create_dataset("values", data=np.array([True, False, True]))
        node.create_dataset("mask", data=np.array([False, True, False]))
    assert read_h5ad_obs(path, ["flag"])["flag"].to_list() == [True, None, True]
    with h5py.File(path, "a") as handle:
        handle["obs/_index"][1] = "cell1"
    with pytest.raises(ValueError, match="unique"):
        read_h5ad_obs(path, ["flag"])


def test_bridge_retains_original_index_and_missing_label_is_not_conflicting(tmp_path):
    path = tmp_path / "whitespace.h5ad"
    make_h5ad(path)
    with h5py.File(path, "a") as handle:
        handle["obs/_index"][0] = " cell1 "
        handle["obs/donor"][0] = " d1 "
        handle["obs/benchmark_label/codes"][:] = [1, 1, -1]
    output = tmp_path / "out"
    result = build_dataset(path, output)
    assert result["n_mapped_cells"] == 3
    assert result["n_receptors"] == 1
    assert result["n_conflicting_clones"] == 0
    cells = pl.read_csv(output / "cell_tcr_map.csv")
    assert cells["cell"][0] == " cell1 "
    assert cells["receptor_id"].null_count() == 0
    labels = pl.read_csv(output / "clone_labels.csv")
    assert labels["benchmark_label"].null_count() == 1
    assert labels["n_distinct_labels"].to_list() == [1, 1]


def test_bridge_rejects_key_normalization_collisions(tmp_path):
    path = tmp_path / "collision.h5ad"
    make_h5ad(path)
    with h5py.File(path, "a") as handle:
        handle["obs/_index"][1] = " cell1 "
    with pytest.raises(ValueError, match="collide"):
        build_dataset(path, tmp_path / "out")


def test_anchor_failing_cells_have_placeholder_receptors_not_null_clones(tmp_path):
    path = tmp_path / "unsafe.h5ad"
    make_h5ad(path)
    with h5py.File(path, "a") as handle:
        handle["obs/cdr3b"][:] = ["ASSLGQAYEQYF"] * 3
        handle["obs/cdr3a"][:] = ["AVRPGGAGPFFVVF"] * 3
    output = tmp_path / "out"
    result = build_dataset(path, output)
    cells = pl.read_csv(output / "cell_tcr_map.csv")
    labels = pl.read_csv(output / "clone_labels.csv")
    assert result["n_mapped_cells"] == 3
    assert cells["receptor_id"].null_count() == 0
    assert labels["receptor_id"].null_count() == 0
    assert labels["conflicting_labels"].null_count() == 0


@pytest.mark.parametrize("column", ["_index", "donor"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_identifiers_fail_at_obs_boundary(tmp_path, column, value):
    path = tmp_path / "empty.h5ad"
    make_h5ad(path)
    with h5py.File(path, "a") as handle:
        handle[f"obs/{column}"][1] = value
    with pytest.raises(ValueError, match="empty or missing at row 1"):
        read_h5ad_obs(path, ["donor"])


def test_clone_label_csv_order_is_reproducible(tmp_path):
    path = tmp_path / "repeat.h5ad"
    make_h5ad(path)
    build_dataset(path, tmp_path / "first")
    build_dataset(path, tmp_path / "second")
    for name in ("clone_labels.csv", "cell_tcr_map.csv", "decoder_input.csv.mapping.csv"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_columnar_obs_handles_empty_categories_and_rejects_corrupt_masks(tmp_path):
    path = tmp_path / "encoded.h5ad"
    make_h5ad(path)
    with h5py.File(path, "a") as handle:
        group = handle["obs"].create_group("unknown")
        group.attrs["encoding-type"] = "categorical"
        group.create_dataset("categories", data=[], dtype=h5py.string_dtype("utf-8"))
        group.create_dataset("codes", data=[-1, -1, -1])
        masked = handle["obs"].create_group("invalid_mask")
        masked.attrs["encoding-type"] = "nullable-integer"
        masked.create_dataset("values", data=[1, 2, 3])
        masked.create_dataset("mask", data=[0, 1, 2])
    assert read_h5ad_obs(path, ["unknown"])["unknown"].to_list() == [None, None, None]
    with pytest.raises(ValueError, match="invalid nullable mask"):
        read_h5ad_obs(path, ["invalid_mask"])
