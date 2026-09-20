
import numpy as np
import polars as pl
import pytest

from tcr_workbench import decoder_pmhc as p
from tcr_workbench.models import InputError

KWARGS = {"decoder_dir": "/unused/DecoderTCR", "python_executable": "/unused/python"}
METRIC = "pll_DecoderTCR-ESMC_300M"


def fake_backend(input_path, output_path, options, *, profile=False):
    frame = pl.read_csv(input_path, infer_schema=False)
    if profile:
        length = len(frame["peptide"][0])
        pl.DataFrame({"position": range(1,length+1), **{aa:[0.05]*length for aa in p.AA}}).write_csv(output_path)
        return {"status":"Profiled", "with_tcr":False}
    good = [hla != "HLA-A*99:99" for hla in frame["hla"]]
    frame.with_columns(
        pl.Series("ok",good), pl.Series("hla_reason",["" if ok else "No exact training HLA sequence" for ok in good]),
        pl.lit("").alias("inference_reason"), pl.Series(METRIC,[-2.1 if ok else None for ok in good],dtype=pl.Float64),
    ).write_csv(output_path)
    return {"status":"Scored", "with_tcr":False}


def test_score_all_rows_duplicates_class_ii_and_abstention(tmp_path, monkeypatch):
    inputs = ["A*02:01", " HLA-A*02:01 ", "DRA*01:01/DRB1*04:01", "DQA1*03:01/DQB1*02:01",
              "DPA1*01:03/DPB1*04:01", "DRB1*04:01", "A*02", "A*02:01|A*02:02", "A*02:01N",
              "A*02:01:01G", "A*99:99", None, "A*02:01:01", "A*102:56"]
    panel = pl.DataFrame({"peptide":[" gilgfvftl "]*len(inputs),"hla":inputs,"donor_note":["kept"]*len(inputs)})
    captured=[]
    def recording(*args,**kwargs):
        captured.append(pl.read_csv(args[0]))
        return fake_backend(*args,**kwargs)
    monkeypatch.setattr(p,"_backend",recording)
    frame, manifest = p.score_pmhc(panel,tmp_path/"run",background_peptides=0,**KWARGS)
    assert frame.height == len(inputs)
    assert frame["input_row"].to_list() == list(range(1,len(inputs)+1))
    assert frame["input_hla"].to_list() == inputs
    assert frame["peptide"].unique().to_list() == ["GILGFVFTL"]
    assert frame["status"].to_list() == ["Scored"]*5+["Unresolved"]*9
    assert captured[0].height == 5  # duplicate pair shared, plus unresolved absent allele attempted
    assert "trav" not in captured[0].columns and "cdr3a" not in captured[0].columns
    assert manifest["with_tcr"] is False
    assert manifest["summary"]["scored_rows"] == 5
    assert len(manifest["input"]["logical_sha256"]) == 64
    saved=pl.read_csv(tmp_path/"run/input_panel.csv")
    assert saved["donor_note"].to_list() == ["kept"]*len(inputs)
    for name,identity in manifest["outputs"].items():
        assert p.file_sha256(tmp_path/"run"/name) == identity["sha256"]


def test_all_unresolved_never_loads_model(tmp_path,monkeypatch):
    def fail(*args,**kwargs):
        raise AssertionError("unsupported inputs must not load weights")
    monkeypatch.setattr(p,"_backend",fail)
    frame,manifest=p.score_pmhc(pl.DataFrame({"peptide":["A"*51,"A"],"hla":["A*02:01",None]}),tmp_path/"out",**KWARGS)
    assert frame["score"].null_count()==2
    assert frame["status"].to_list()==["Unresolved","Unresolved"]
    assert manifest["backend"]["status"]=="not_run"


@pytest.mark.parametrize("hla,expected", [
    ("A*02:01N", "cannot confirm surface restriction"),
    ("A*02:01S", "cannot confirm surface restriction"),
    ("DRA*01:01/DRB1*04:01C", "cannot confirm surface restriction"),
    ("A*02:01Q", "no representative allele is selected"),
    ("A*02:01:01G", "no representative allele is selected"),
    ("A*02:01P", "no representative allele is selected"),
])
def test_profile_expression_and_group_abstention_reasons(tmp_path,monkeypatch,hla,expected):
    def fail(*args,**kwargs):
        raise AssertionError("unsupported HLA must not run inference")
    monkeypatch.setattr(p,"_backend",fail)
    pssm,manifest=p.profile_pmhc(hla,9,tmp_path/"out",**KWARGS)
    assert pssm.is_empty()
    assert manifest["summary"]["status"]=="Unresolved"
    assert expected in manifest["summary"]["reason"]


@pytest.mark.parametrize("dtype", [pl.String, pl.Null])
def test_empty_panel_is_auditable(tmp_path,monkeypatch,dtype):
    panel=pl.DataFrame(schema={"peptide":dtype,"hla":dtype})
    frame,manifest=p.score_pmhc(panel,tmp_path/"out",**KWARGS)
    assert frame.schema==p.RESULT_SCHEMA and not frame.height
    assert manifest["summary"]["input_rows"]==0


@pytest.mark.parametrize("panel", [pl.DataFrame({"peptide":["AXA"],"hla":["A*02:01"]}),
                                    pl.DataFrame({"peptide":["AAA"],"hla":["garbage"]}),
                                    pl.DataFrame({"peptide":[None],"hla":["A*02:01"]}),
                                    pl.DataFrame({"peptide":["AAA"],"hla":["DQA1*01:01/DRB1*07:01"]})])
def test_corrupt_input_fails_before_publish(tmp_path,panel):
    with pytest.raises(InputError):
        p.score_pmhc(panel,tmp_path/"out",**KWARGS)
    assert not (tmp_path/"out").exists()


def test_file_input_snapshot_prevents_changed_input_publish(tmp_path,monkeypatch):
    panel=tmp_path/"input.csv"
    panel.write_text("peptide,hla\nGILGFVFTL,A*02:01\n")
    def mutation(*args,**kwargs):
        result=fake_backend(*args,**kwargs)
        panel.write_text("peptide,hla\nNLVPMVATV,A*02:01\n")
        return result
    monkeypatch.setattr(p,"_backend",mutation)
    with pytest.raises(ValueError,match="Input changed"):
        p.score_pmhc(panel,tmp_path/"out",**KWARGS)
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("corruption",["name","hla","positive","infinite","failed_finite","garbage"])
def test_model_output_contract_rejects_corruption(tmp_path,monkeypatch,corruption):
    def corrupt(input_path,output_path,options,**kwargs):
        result=fake_backend(input_path,output_path,options,**kwargs)
        frame=pl.read_csv(output_path)
        changes={"name":pl.lit("wrong").alias("name"), "hla":pl.lit("HLA-A*01:01").alias("hla"),
                 "positive":pl.lit(0.1).alias(METRIC),"infinite":pl.lit(float("inf")).alias(METRIC),
                 "failed_finite":pl.lit(False).alias("ok"),"garbage":pl.lit("garbage").alias(METRIC)}
        frame.with_columns(changes[corruption]).write_csv(output_path)
        return result
    monkeypatch.setattr(p,"_backend",corrupt)
    with pytest.raises(InputError):
        p.score_pmhc(pl.DataFrame({"peptide":["A"],"hla":["A*02:01"]}),tmp_path/"out",**KWARGS)
    assert not (tmp_path/"out").exists()


def test_profile_uniform_and_zero_probability_floor(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"_backend",fake_backend)
    pssm,manifest=p.profile_pmhc("DRA*01:01/DRB1*04:01",9,tmp_path/"out",**KWARGS)
    assert pssm.height==180
    np.testing.assert_allclose(pssm["log2_odds"].to_numpy(),0,atol=1e-12)
    assert manifest["summary"]["status"]=="Profiled"
    assert manifest["summary"]["floored_entries"]==0
    assert manifest["with_tcr"] is False
    profile=pl.DataFrame({"position":[1],**{aa:[1.0 if aa=="A" else 0.0] for aa in p.AA}})
    derived=p.profile_to_pssm(profile)
    assert derived["hla"].null_count()==20
    assert derived["probability"].sum()==1.0
    assert derived["floor_applied"].sum()==19
    assert np.isfinite(derived["log2_odds"].to_numpy()).all()
    assert derived.filter(pl.col("amino_acid")=="A")["log2_odds"][0]==pytest.approx(np.log2(20))


def test_profile_unresolved_missing_partner_or_lookup(tmp_path,monkeypatch):
    pssm,manifest=p.profile_pmhc("DRB1*04:01",9,tmp_path/"partial",**KWARGS)
    assert not pssm.height and manifest["summary"]["status"]=="Unresolved"
    def absent(input_path,output_path,options,**kwargs):
        return {"status":"Unresolved","reason":"No exact reference sequence"}
    monkeypatch.setattr(p,"_backend",absent)
    pssm,manifest=p.profile_pmhc("A*99:99",9,tmp_path/"missing",**KWARGS)
    assert not pssm.height and manifest["summary"]["reason"]=="No exact reference sequence"


@pytest.mark.parametrize("corruption",["sum","nan","negative","position","string_position","missing_aa"])
def test_pssm_rejects_invalid_profile(corruption):
    profile=pl.DataFrame({"position":[1,2],**{aa:[.05,.05] for aa in p.AA}})
    if corruption=="sum":
        profile=profile.with_columns(pl.lit(.2).alias("A"))
    elif corruption=="nan":
        profile=profile.with_columns(pl.lit(float("nan")).alias("A"))
    elif corruption=="negative":
        profile=profile.with_columns(pl.lit(-.05).alias("A"))
    elif corruption=="position":
        profile=profile.with_columns(pl.lit(1).alias("position"))
    elif corruption=="string_position":
        profile=profile.with_columns(pl.col("position").cast(pl.String))
    else:
        profile=profile.drop("Y")
    with pytest.raises(InputError):
        p.profile_to_pssm(profile)


@pytest.mark.parametrize("option", [{"batch_size":129},{"token_budget":2},{"token_budget":262145},
                                    {"cache_bytes":1073741825},{"device":"tpu"},{"batch_size":True}])
def test_options_fail_before_all_unresolved_short_circuit(tmp_path,option):
    with pytest.raises(InputError):
        p.score_pmhc(pl.DataFrame({"peptide":["A"],"hla":[None]}),tmp_path/"out",**(KWARGS|option))
    assert not (tmp_path/"out").exists()


def test_source_mutation_cannot_publish_false_provenance(tmp_path,monkeypatch):
    values=iter(["before", "after"])
    monkeypatch.setattr(p,"source_digest",lambda:next(values))
    monkeypatch.setattr(p,"_backend",fake_backend)
    with pytest.raises(InputError,match="source changed"):
        p.score_pmhc(pl.DataFrame({"peptide":["A"],"hla":["A*02:01"]}),tmp_path/"out",**KWARGS)
    assert not (tmp_path/"out").exists()
