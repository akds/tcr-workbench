import json
import os
import shutil
import sys
from types import SimpleNamespace
import venv

import numpy as np
import polars as pl
import pytest

from tcr_workbench import prediction as p
from tcr_workbench.backends import torch_decoder


@pytest.fixture
def receptor():
    return pl.DataFrame(
        {
            "receptor_id": ["r1"],
            "donor_id": ["d1"],
            "trav": ["TRAV21"],
            "traj": ["TRAJ6"],
            "cdr3a": ["CAVRPGGAGPFFVVF"],
            "trbv": ["TRBV7-9"],
            "trbj": ["TRBJ2-7"],
            "cdr3b": ["CASSLGQAYEQYF"],
            "pairing_status": ["paired"],
        }
    )


@pytest.fixture
def panel():
    return pl.DataFrame(
        {"peptide": ["GILGFVFTL", "ELAGIGILTV"], "hla": ["HLA-A*02:01", "HLA-A*02:01"]}
    )


def export(tmp_path, receptor, panel):
    path = tmp_path / "input.csv"
    p.export_decoder_input(receptor, panel, path)
    return path


def write_scores(path, target, scores, **columns):
    frame = pl.read_csv(path).with_columns(pl.Series("pll_" + p.DEFAULT_MODEL, scores))
    if columns:
        frame = frame.with_columns([pl.Series(k, v) for k, v in columns.items()])
    frame.write_csv(target)


def test_export_stable_ids_no_labels_and_audits(tmp_path, receptor, panel):
    receptor = receptor.with_columns(pl.lit("positive").alias("label"))
    path = export(tmp_path, receptor, panel)
    one = pl.read_csv(path)
    two_path = tmp_path / "again.csv"
    p.export_decoder_input(receptor, panel.reverse(), two_path)
    two = pl.read_csv(two_path)
    assert "label" not in one.columns
    assert set(one["name"]) == set(two["name"])
    assert one["name"].n_unique() == 2
    manifest = json.loads(p._sidecar(path).read_text())
    assert manifest["input_sha256"] == p.file_sha256(path)
    assert manifest["exported_pairs"] == 2
    assert manifest["skipped_pairs"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"pairing_status": "dual_alpha"},
        {"cdr3a": None},
        {"trav": "TRAV1,TRAV2"},
        {"cdr3b": "ASSLGQAYEQYF"},
    ],
)
def test_ineligible_receptors_explicitly_unresolved(tmp_path, receptor, panel, change):
    receptor = receptor.with_columns([pl.lit(v).alias(k) for k, v in change.items()])
    path = export(tmp_path, receptor, panel)
    skipped = pl.read_csv(str(path) + ".skipped.csv")
    assert skipped.height == 2
    assert skipped["status"].to_list() == ["Unresolved"] * 2
    assert pl.read_csv(path).height == 0


def test_class_ii_and_ambiguous_donor_are_audited(tmp_path, receptor, panel):
    panel = pl.concat(
        [panel, pl.DataFrame({"peptide": ["AAAAAAAAAAAAAAA"], "hla": ["HLA-DRB1*04:01"]})]
    )
    result = p.export_decoder_input(
        receptor,
        panel,
        tmp_path / "pairs.csv",
        donors=pl.DataFrame({"donor_id": ["d1"], "hla": ["A*02"]}),
    )
    assert result["skipped_pairs"] == 3
    result = p.export_decoder_input(
        receptor,
        panel,
        tmp_path / "pairs.csv",
        donors=pl.DataFrame({"donor_id": ["d1"], "hla": ["A*02:01"]}),
    )
    assert result["exported_pairs"] == 2
    assert result["skipped_pairs"] == 1


def test_max_pair_limit_before_cartesian_allocation(tmp_path, receptor, panel):
    with pytest.raises(ValueError, match="max_pairs"):
        p.export_decoder_input(receptor, panel, tmp_path / "input.csv", max_pairs=1)
    assert not (tmp_path / "input.csv").exists()


def test_import_validates_ids_components_and_preserves_failures(tmp_path, receptor, panel):
    path = export(tmp_path, receptor, panel)
    scored = tmp_path / "scores.csv"
    write_scores(path, scored, [-1.2, -0.1], ok=[True, False], tcr_reason=[None, "cannot stitch"])
    result = p.import_decoder_scores(path, scored)
    assert result["status"].to_list() == ["ModelHypothesis", "Unresolved"]
    assert result["score"].to_list() == [-1.2, None]
    assert result["receptor_id"].to_list() == ["r1", "r1"]
    assert result["provenance"].to_list() == ["external_unverified"] * 2
    pl.read_csv(scored).head(1).write_csv(scored)
    result = p.import_decoder_scores(path, scored)
    assert result.height == 2 and result["status"][1] == "Unresolved"
    bad = pl.read_csv(scored).with_columns(pl.lit("VVVVVVVVV").alias("peptide"))
    bad.write_csv(scored)
    with pytest.raises(ValueError, match="changes input components"):
        p.import_decoder_scores(path, scored)


@pytest.mark.parametrize("failure", ["duplicate", "unknown", "invalid_score", "bad_ok"])
def test_import_rejects_corrupted_scores(tmp_path, receptor, panel, failure):
    path = export(tmp_path, receptor, panel)
    scored = tmp_path / "scores.csv"
    write_scores(path, scored, [-1.0, -2.0])
    frame = pl.read_csv(scored)
    if failure == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    elif failure == "unknown":
        frame = frame.with_columns(pl.lit("unknown").alias("name")).head(1)
    elif failure == "invalid_score":
        frame = frame.with_columns(pl.lit("oops").alias("pll_" + p.DEFAULT_MODEL))
    else:
        frame = frame.with_columns(pl.lit("banana").alias("ok"))
    frame.write_csv(scored)
    with pytest.raises(ValueError):
        p.import_decoder_scores(path, scored)


def test_external_input_cannot_leak_labels(tmp_path, receptor, panel):
    path = export(tmp_path, receptor, panel)
    pl.read_csv(path).with_columns(pl.lit(1).alias("label")).write_csv(path)
    with pytest.raises(ValueError, match="keep labels separate"):
        p._input_frame(path)


def test_checkpoint_input_source_and_output_invalidate_cache(
    tmp_path, receptor, panel, monkeypatch
):
    path = export(tmp_path, receptor, panel)
    root = tmp_path / "decoder"
    source = root / "src/DecoderTCR/utils/predict_from_genes.py"
    source.parent.mkdir(parents=True)
    source.write_text("# pretend checkout\n")
    checkpoint = root / p.CHECKPOINTS[p.DEFAULT_MODEL]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake-weights")
    monkeypatch.setattr(
        p,
        "_environment_fingerprint",
        lambda *args: {
            "environment_sha256": "fake_environment",
            "germline_sha256": "fake_germline",
        },
    )
    calls = []

    def fake(source, candidate, temporary, fingerprint, **kwargs):
        calls.append(fingerprint)
        write_scores(source, candidate, [-1.0, -2.0])
        return {}

    monkeypatch.setattr(torch_decoder, "execute_torch", fake)
    scored = tmp_path / "scored.csv"
    kwargs = dict(decoder_dir=root, python_executable=sys.executable)
    first = p.run_decoder(path, scored, **kwargs)
    assert first["cache_hit"] is False
    assert p.run_decoder(path, scored, **kwargs)["cache_hit"] is True
    assert len(calls) == 1
    assert (
        p.import_decoder_scores(path, scored, require_manifest=True)["provenance"][0]
        == "hash_verified"
    )
    checkpoint.write_bytes(b"updated-weights")
    with pytest.raises(ValueError, match="provenance"):
        p.run_decoder(path, scored, **kwargs)
    p.run_decoder(path, scored, force=True, **kwargs)
    source.write_text("# changed source")
    with pytest.raises(ValueError, match="provenance"):
        p.run_decoder(path, scored, **kwargs)
    p.run_decoder(path, scored, force=True, **kwargs)
    scored.write_text(scored.read_text().replace("-1.0", "-5.0"))
    with pytest.raises(ValueError, match="stale"):
        p.import_decoder_scores(path, scored, require_manifest=True)


def test_failure_does_not_replace_existing_output(tmp_path, receptor, panel, monkeypatch):
    path = export(tmp_path, receptor, panel)
    output = tmp_path / "scores.csv"
    output.write_text("old scores")
    monkeypatch.setattr(
        p,
        "_model_fingerprint",
        lambda *args: {"python_executable": sys.executable, "model": p.DEFAULT_MODEL},
    )

    def fail(*args, **kwargs):
        raise RuntimeError("missing germline data")

    monkeypatch.setattr(torch_decoder, "execute_torch", fail)
    with pytest.raises(RuntimeError, match="germline"):
        p.run_decoder(
            path, output, decoder_dir=tmp_path, python_executable=sys.executable, force=True
        )
    assert output.read_text() == "old scores"


def test_empirical_profile_counts_lengths_and_normalization(panel):
    panel = pl.concat([panel, panel.head(1)])
    profile = p.empirical_profile(panel, pseudocount=0.5)
    assert profile.height == (9 + 10) * 20
    assert set(profile["length"]) == {9, 10}
    assert set(profile["n_peptides"]) == {1}
    sums = profile.group_by("hla", "length", "position").agg(pl.col("frequency").sum())
    assert np.allclose(sums["frequency"].to_numpy(), 1)
    first = profile.filter(
        (pl.col("length") == 9) & (pl.col("position") == 1) & (pl.col("amino_acid") == "G")
    )
    assert first["count"][0] == 1
    assert first["frequency"][0] == pytest.approx(1.5 / 11)
    assert "not a binding predictor" in first["interpretation"][0]
    with pytest.raises(ValueError):
        p.empirical_profile(panel, pseudocount=0)


def test_lookup_metadata_unknowns_duplicates_and_hash(tmp_path, panel):
    library = tmp_path / "library.csv"
    panel.head(1).with_columns(pl.lit(13.2).alias("score")).write_csv(library)
    metadata = tmp_path / "metadata.json"
    meta = dict(
        predictor="example",
        version="1",
        metric="IC50",
        units="nM",
        higher_is_better=False,
        source="fixture",
        library_sha256=p.file_sha256(library),
    )
    metadata.write_text(json.dumps(meta))
    results = p.peptide_mhc_lookup(panel, library, metadata)
    assert results["status"].to_list() == ["PrecomputedScore", "Unresolved"]
    assert results["score"].to_list() == [13.2, None]
    assert results["metric"][0] == "IC50"
    library.write_text(library.read_text() + library.read_text().splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="hash"):
        p.peptide_mhc_lookup(panel, library, metadata)
    meta["library_sha256"] = p.file_sha256(library)
    metadata.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="duplicate"):
        p.peptide_mhc_lookup(panel, library, metadata)


def test_direct_profile_adapter_validates_marginals(tmp_path, receptor, monkeypatch):
    components = receptor.to_dicts()[0]
    components["hla"] = "HLA-A*02:01"
    monkeypatch.setattr(
        p,
        "_model_fingerprint",
        lambda *args: {"python_executable": sys.executable, "model": p.DEFAULT_MODEL},
    )

    def fake(source, candidate, temporary, fingerprint, **kwargs):
        assert kwargs["profile"]
        assert fingerprint["context_type"] == "tcr-pmhc"
        pl.DataFrame({"position": [1, 2], **{aa: [0.05, 0.05] for aa in p.AA}}).write_csv(candidate)
        return {}

    monkeypatch.setattr(torch_decoder, "execute_torch", fake)
    result = p.run_decoder_profile(
        components,
        tmp_path / "profile.csv",
        length=2,
        decoder_dir=tmp_path,
        python_executable=sys.executable,
    )
    assert result["length"] == 2
    assert "not binding probabilities" in result["interpretation"]


def test_export_bundle_remains_valid_after_move(tmp_path, receptor, panel):
    staging = tmp_path / "staging"
    staging.mkdir()
    path = export(staging, receptor, panel)
    scored = staging / "scores.csv"
    write_scores(path, scored, [-1.0, -2.0])
    published = tmp_path / "published"
    shutil.move(str(staging), str(published))
    result = p.import_decoder_scores(published / "input.csv", published / "scores.csv")
    assert result["receptor_id"].to_list() == ["r1", "r1"]


def test_peptide_na_is_not_a_null_marker(tmp_path, receptor):
    panel = pl.DataFrame({"peptide": ["NA"], "hla": ["HLA-A*02:01"]})
    path = export(tmp_path, receptor, panel)
    assert p._input_frame(path)["peptide"].to_list() == ["NA"]
    assert p.empirical_profile(panel)["length"].to_list() == [2] * 40


def test_environment_checks_import_origin_and_germline_changes(tmp_path, monkeypatch):
    germline = tmp_path / "Stitchr"
    germline.mkdir()
    data = germline / "TRAV.fasta"
    data.write_text(">TRAV1\nAAAA\n")
    metadata = {
        "decoder_origin": str(tmp_path / "src/DecoderTCR/__init__.py"),
        "germline_roots": [str(germline)],
        "packages": [["torch", "2.9.0"]],
    }

    def inspect(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(metadata))

    monkeypatch.setattr(p.subprocess, "run", inspect)
    first = p._environment_fingerprint(tmp_path / "python", tmp_path)
    data.write_text(">TRAV1\nAAAC\n")
    second = p._environment_fingerprint(tmp_path / "python", tmp_path)
    assert first["germline_sha256"] != second["germline_sha256"]
    assert first["environment_sha256"] == second["environment_sha256"]
    metadata["decoder_origin"] = "/some/other/DecoderTCR/__init__.py"
    with pytest.raises(ValueError, match="supplied checkout"):
        p._environment_fingerprint(tmp_path / "python", tmp_path)


@pytest.mark.parametrize("gene", ["TRAV14/DV4", "TRAV38-2/DV8", "TRAV14/DV4*01"])
def test_shared_alpha_delta_v_gene_is_a_single_gene(tmp_path, receptor, panel, gene):
    receptor = receptor.with_columns(pl.lit(gene).alias("trav"))
    path = export(tmp_path, receptor, panel)
    assert p._input_frame(path)["trav"].to_list() == [gene] * 2


@pytest.mark.parametrize("gene", ["TRAV14/TRAV1", "TRAV14/DV4/TRAV1", "TRAV1/DV4oops"])
def test_slash_ambiguity_is_not_a_shared_gene(tmp_path, receptor, panel, gene):
    receptor = receptor.with_columns(pl.lit(gene).alias("trav"))
    path = export(tmp_path, receptor, panel)
    assert p._read_csv(path).height == 0


def test_duplicate_csv_headers_fail_loudly(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("name,score,score\npair,1,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        p._read_csv(path)


@pytest.mark.parametrize("hla", ["banana", "HLA-A*02", "HLA-A*02:01G", "HLA-DRB1*04:01"])
def test_external_model_input_requires_supported_unambiguous_hla(tmp_path, receptor, panel, hla):
    path = export(tmp_path, receptor, panel)
    pl.read_csv(path).with_columns(pl.lit(hla).alias("hla")).write_csv(path)
    with pytest.raises(ValueError):
        p._input_frame(path)


def test_import_requires_complete_receptor_mapping(tmp_path, receptor, panel):
    path = export(tmp_path, receptor, panel)
    scored = tmp_path / "scored.csv"
    write_scores(path, scored, [-1.0, -2.0])
    mapping = tmp_path / "input.csv.mapping.csv"
    pl.read_csv(mapping).head(1).write_csv(mapping)
    manifest = json.loads(p._sidecar(path).read_text())
    manifest["mapping_sha256"] = p.file_sha256(mapping)
    p._sidecar(path).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cover each pair ID"):
        p.import_decoder_scores(path, scored)


@pytest.mark.parametrize(
    "field,gene",
    [("trav", "TRBV7-9"), ("trbj", "TRAJ6"), ("trav", "banana"), ("trbv", "TRAV14/DV4")],
)
def test_export_audits_gene_locus_swaps(tmp_path, receptor, panel, field, gene):
    receptor = receptor.with_columns(pl.lit(gene).alias(field))
    path = export(tmp_path, receptor, panel)
    assert p._read_csv(path).height == 0
    assert p._read_csv(str(path) + ".skipped.csv").height == 2


def test_export_requires_explicit_pairing_status(tmp_path, receptor, panel):
    path = export(tmp_path, receptor.drop("pairing_status"), panel)
    assert p._read_csv(path).height == 0
    skipped = p._read_csv(str(path) + ".skipped.csv")
    assert "pairing_status" in skipped["reason"][0]


@pytest.mark.parametrize("hla", [None, "", "   "])
def test_blank_panel_hla_stays_auditable(tmp_path, receptor, hla):
    panel = pl.DataFrame({"peptide": ["GILGFVFTL"], "hla": [hla]})
    path = export(tmp_path, receptor, panel)
    skipped = p._read_csv(str(path) + ".skipped.csv")
    assert skipped.height == 1 and "HLA restriction is missing" in skipped["reason"][0]
    profile = p.empirical_profile(panel)
    assert profile.height == 9 * 20
    assert profile["hla"].null_count() == profile.height


def test_export_distinguishes_missing_and_unresolved_donor(tmp_path, receptor, panel):
    path = tmp_path / "input.csv"
    for donor, expected in [("different_donor", "typing not supplied"), ("d1", "unresolved")]:
        p.export_decoder_input(
            receptor, panel, path, donors=pl.DataFrame({"donor_id": [donor], "hla": ["A*03:01"]})
        )
        assert expected in p._read_csv(str(path) + ".skipped.csv")["reason"][0]


def test_lookup_distinguishes_missing_key_missing_value_missing_hla(tmp_path):
    library = tmp_path / "library.csv"
    library.write_text("peptide,hla,score\nGILGFVFTL,HLA-A*02:01,\n")
    meta = dict(
        predictor="fixture",
        version="1",
        metric="score",
        units="arbitrary",
        higher_is_better=True,
        source="fixture",
        library_sha256=p.file_sha256(library),
    )
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps(meta))
    panel = pl.DataFrame(
        {
            "peptide": ["GILGFVFTL", "ELAGIGILTV", "GILGFVFTL"],
            "hla": ["HLA-A*02:01", "HLA-A*02:01", None],
        }
    )
    result = p.peptide_mhc_lookup(panel, library, metadata)
    assert result["status"].to_list() == ["Unresolved"] * 3
    assert "score is missing" in result["reason"][0]
    assert "no score for" in result["reason"][1]
    assert "HLA restriction is missing" in result["reason"][2]


def test_execute_uses_selected_environment_thimble_and_sdpa(tmp_path, monkeypatch):
    seen = {}
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        monkeypatch.setenv(key, "ambient-other-environment")
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    monkeypatch.setenv("USE_FLASH_ATTN", "1")
    monkeypatch.setenv("PATH", "ambient-bin")

    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(p.subprocess, "run", run)
    python = tmp_path / "env/bin/python"
    p._execute([str(python), "-m", "DecoderTCR"], tmp_path, None)
    assert seen["env"]["PATH"].split(p.os.pathsep)[0] == str(python.parent)
    assert seen["env"]["PATH"].split(p.os.pathsep)[1:] == ["ambient-bin"]
    assert seen["env"]["USE_FLASH_ATTN"] == "0"
    assert seen["env"]["PYTHONNOUSERSITE"] == "1"
    assert not {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"} & seen["env"].keys()
    assert p.os.environ["PYTHONPATH"] == "ambient-other-environment"


@pytest.mark.parametrize("positions", [[0, 1], [1, 3]])
def test_direct_profile_rejects_wrong_position_coordinates(
    tmp_path, receptor, monkeypatch, positions
):
    components = {**receptor.to_dicts()[0], "hla": "HLA-A*02:01"}
    monkeypatch.setattr(
        p,
        "_model_fingerprint",
        lambda *args: {"python_executable": sys.executable, "model": p.DEFAULT_MODEL},
    )

    def fake(source, candidate, temporary, fingerprint, **kwargs):
        pl.DataFrame({"position": positions, **{aa: [0.05, 0.05] for aa in p.AA}}).write_csv(candidate)
        return {}

    monkeypatch.setattr(torch_decoder, "execute_torch", fake)
    with pytest.raises(ValueError, match="invalid marginal"):
        p.run_decoder_profile(
            components,
            tmp_path / "profile.csv",
            length=2,
            decoder_dir=tmp_path,
            python_executable=sys.executable,
        )


def test_import_preserves_reconstructed_sequences_and_flags_family_choice(
    tmp_path, receptor, panel
):
    receptor = receptor.with_columns(pl.lit("TRBV12-X").alias("trbv"))
    path = export(tmp_path, receptor, panel)
    scored = tmp_path / "scores.csv"
    write_scores(
        path,
        scored,
        [-1.0, -2.0],
        TRAV=["TRAV21"] * 2,
        TRAJ=["TRAJ6"] * 2,
        TRBV=["TRBV12-3"] * 2,
        TRBJ=["TRBJ2-7"] * 2,
        TCR_a=["CAVRPGGAGPFFVVF"] * 2,
        TCR_b=["CASSLGQAYEQYF"] * 2,
        HLA_a=["ACDEFGHIK"] * 2,
        HLA_b=["LMNPQRSTV"] * 2,
        tcr_ok=[True] * 2,
    )
    result = p.import_decoder_scores(path, scored)
    assert result["trbv"].to_list() == ["TRBV12-X"] * 2
    assert result["TRBV"].to_list() == ["TRBV12-3"] * 2
    assert result["TCR_a"][0] == "CAVRPGGAGPFFVVF"
    assert result["HLA_b"][0] == "LMNPQRSTV"
    assert result["gene_resolution_status"][0] == "inferred_gene_choice"
    assert "upstream selected TRBV=TRBV12-3" in result["reason"][0]
    assert "default IMGT alleles" in result["gene_resolution_reason"][0]


def test_import_distinguishes_gene_spelling_and_explicit_allele_change(tmp_path, receptor, panel):
    receptor = receptor.with_columns(
        [pl.concat_str(pl.col(k), pl.lit("*01")).alias(k) for k in ["trav", "traj", "trbv", "trbj"]]
    )
    receptor = receptor.with_columns(pl.lit("TRBV07-09*01").alias("trbv"))
    path = export(tmp_path, receptor, panel)
    scored = tmp_path / "scores.csv"
    write_scores(
        path,
        scored,
        [-1.0, -2.0],
        TRAV=["TRAV21*01"] * 2,
        TRAJ=["TRAJ6*01"] * 2,
        TRBV=["TRBV7-9*01", "TRBV7-9*02"],
        TRBJ=["TRBJ2-7*01"] * 2,
    )
    result = p.import_decoder_scores(path, scored)
    assert result["gene_resolution_status"].to_list() == [
        "nomenclature_normalized",
        "inferred_gene_choice",
    ]
    assert "input trbv=TRBV07-09*01" in result["gene_resolution_reason"][1]


def test_import_explicitly_marks_unreported_gene_choices(tmp_path, receptor, panel):
    path = export(tmp_path, receptor, panel)
    scored = tmp_path / "scores.csv"
    write_scores(path, scored, [-1.0, -2.0])
    result = p.import_decoder_scores(path, scored)
    assert result["gene_resolution_status"].to_list() == ["unreported"] * 2
    assert "not reported" in result["reason"][0]


def test_germline_fingerprint_includes_stitchr_sibling_data(tmp_path, monkeypatch):
    source = tmp_path / "src"
    for package in ("DecoderTCR", "Stitchr"):
        directory = source / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("")
    data = source / "Data" / "HUMAN"
    data.mkdir(parents=True)
    fasta = data / "TRAV.fasta"
    fasta.write_text(">TRAV1\nAAAA\n")
    (source / "Stitchr" / "stitchrfunctions.py").write_text(f"data_dir = {str(data.parent)!r}\n")
    environment = tmp_path / "env"
    venv.EnvBuilder(with_pip=False).create(environment)
    site_packages = next(environment.rglob("site-packages"))
    (site_packages / "decoder-fixture.pth").write_text(str(source) + "\n")
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    monkeypatch.setenv("PYTHONPATH", "ambient-other-environment")
    before = p._environment_fingerprint(python, tmp_path)
    fasta.write_text(">TRAV1\nAAAC\n")
    after = p._environment_fingerprint(python, tmp_path)
    assert before["germline_sha256"] != after["germline_sha256"]


@pytest.mark.parametrize(
    "device,canonical",
    [("cpu", "cpu"), ("gpu", "cuda"), ("CUDA", "cuda"), ("cuda:0", "cuda:0"), ("cuda:2", "cuda:2")],
)
def test_decoder_device_aliases(device, canonical):
    assert p.normalize_decoder_device(device) == canonical


def test_apple_device_requires_explicit_runtime_and_bundle_before_input_access(tmp_path):
    with pytest.raises(ValueError, match="Apple MLX requires --checkpoint"):
        p.run_decoder(
            tmp_path / "absent.csv",
            tmp_path / "out.csv",
            decoder_dir=tmp_path,
            python_executable=tmp_path / "missing_python",
            device="apple",
        )
    with pytest.raises(ValueError, match="Apple MLX requires --checkpoint"):
        p.run_decoder_profile(
            {},
            tmp_path / "profile.csv",
            length=9,
            decoder_dir=tmp_path,
            python_executable=tmp_path / "missing_python",
            device="apple",
        )


@pytest.mark.parametrize("device", ["banana", "cuda:-1", "cuda:01", "gpu:0", ""])
def test_invalid_device_rejected(device):
    with pytest.raises(ValueError, match="unsupported device"):
        p.normalize_decoder_device(device)


def test_missing_mapping_sidecar_is_explicit_in_import(tmp_path, receptor, panel):
    path = export(tmp_path, receptor, panel)
    scored = tmp_path / "scored.csv"
    write_scores(path, scored, [-1.0, -2.0])
    p._sidecar(path).unlink()
    result = p.import_decoder_scores(path, scored)
    assert result["receptor_id"].null_count() == 2
    assert result["donor_id"].null_count() == 2
    assert result["mapping_status"].to_list() == ["mapping_unavailable"] * 2
    assert "mapping unavailable" in result["reason"][0]


@pytest.mark.parametrize("version", [None, 2, True, 1.0])
def test_manifest_version_is_validated(tmp_path, receptor, panel, version):
    path = export(tmp_path, receptor, panel)
    scored = tmp_path / "scored.csv"
    write_scores(path, scored, [-1.0, -2.0])
    manifest = json.loads(p._sidecar(path).read_text())
    manifest["schema_version"] = version
    p._sidecar(path).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version"):
        p.import_decoder_scores(path, scored)


@pytest.mark.parametrize("alias_kind", ["parent", "symlink"])
def test_model_output_alias_cannot_overwrite_input(tmp_path, receptor, panel, alias_kind):
    path = export(tmp_path, receptor, panel)
    if alias_kind == "parent":
        (tmp_path / "sub").mkdir()
        output = tmp_path / "sub" / ".." / path.name
    else:
        output = tmp_path / "alias.csv"
        output.symlink_to(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="input and output paths must differ"):
        p.run_decoder(
            path, output, decoder_dir=tmp_path, python_executable=sys.executable, force=True
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("hla", [None, "banana"])
def test_invalid_donor_hla_has_typed_context(tmp_path, receptor, panel, hla):
    with pytest.raises(ValueError, match="donor typing row 'd1'"):
        p.export_decoder_input(
            receptor,
            panel,
            tmp_path / "out.csv",
            donors=pl.DataFrame({"donor_id": ["d1"], "hla": [hla]}),
        )


def test_profile_overwrite_requires_force_before_model_loading(tmp_path, receptor):
    output = tmp_path / "profile.csv"
    output.write_text("original")
    components = {**receptor.to_dicts()[0], "hla": "HLA-A*02:01"}
    with pytest.raises(ValueError, match="already exists"):
        p.run_decoder_profile(
            components, output, length=9, decoder_dir=tmp_path, python_executable=sys.executable
        )
    assert output.read_text() == "original"


@pytest.mark.parametrize("case", ["valid", "nonempty", "no_reason", "contradictory_counts", "no_status"])
def test_profile_failed_reconstruction_remains_audited(tmp_path, receptor, monkeypatch, case):
    monkeypatch.setattr(p, "_model_fingerprint", lambda *a: dict(model=p.DEFAULT_MODEL))
    runtime = dict(status="Unresolved", reason="TCR stitching failed", rows=1, scored=0, unresolved=1,
                   reconstruction={"tcr_ok": "False", "tcr_reason": "TCR stitching failed"})
    if case == "no_reason":
        runtime["reason"] = " "
    elif case == "contradictory_counts":
        runtime["scored"] = 1
    elif case == "no_status":
        runtime.pop("status")

    def execute(source, candidate, temporary, provenance, **kwargs):
        frame = pl.DataFrame(schema={"position": pl.Int64, **{aa: pl.Float64 for aa in p.AA}})
        if case == "nonempty":
            frame = pl.DataFrame({"position": [1, 2], **{aa: [.05, .05] for aa in p.AA}})
        frame.write_csv(candidate)
        return runtime

    monkeypatch.setattr(torch_decoder, "execute_torch", execute)
    output = tmp_path / "profile.csv"
    kwargs = dict(length=2, decoder_dir=tmp_path, python_executable=sys.executable)
    components = {**receptor.row(0, named=True), "hla": "HLA-A*02:01"}
    if case == "valid":
        result = p.run_decoder_profile(components, output, **kwargs)
        assert result["status"] == "Unresolved" and result["reason"] == "TCR stitching failed"
        assert pl.read_csv(output).height == 0
        manifest = json.loads(p._sidecar(output).read_text())
        assert manifest["reconstruction"]["tcr_ok"] == "False"
        assert manifest["output_sha256"] == p.file_sha256(output)
    else:
        with pytest.raises(ValueError, match="invalid"):
            p.run_decoder_profile(components, output, **kwargs)
        assert not output.exists()


def test_decoder_cache_fingerprint_includes_transitive_hla_source(tmp_path, monkeypatch):
    from tcr_workbench import report
    root = tmp_path / "decoder"
    upstream = root / "src/DecoderTCR/utils/predict_from_genes.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# pinned upstream")
    package = tmp_path / "workbench"
    package.mkdir()
    (package / "report.py").write_text("# source digest implementation")
    hla = package / "hla.py"
    hla.write_text("# HLA contract v1")
    monkeypatch.setattr(report, "__file__", str(package / "report.py"))
    monkeypatch.setattr(p, "_environment_fingerprint", lambda *a: {"environment_sha256": "fixed"})
    first = p._decoder_fingerprint(root, sys.executable)
    assert first["workbench_adapter_sha256"] == report.source_digest()
    hla.write_text("# HLA contract v2")
    second = p._decoder_fingerprint(root, sys.executable)
    assert first["workbench_adapter_sha256"] != second["workbench_adapter_sha256"]
    assert first["decoder_source_sha256"] == second["decoder_source_sha256"]
