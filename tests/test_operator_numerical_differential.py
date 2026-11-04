"""Operator-level trace comparisons cannot silently claim causal or native E2 proof."""
from hashlib import sha256
import json

import numpy as np
import pytest

from fpgai_research.experiments.operator_numerical_differential import (
    SCHEMA, analyze_operator_traces, export_operator_differential,
)


def _traces(tmp_path):
    parent=tmp_path/"matched_traces"
    parent.mkdir()
    common={"matmul":np.ones((16,4),dtype=np.float32),
            "relu":np.ones((16,4),dtype=np.float32),
            "softmax":np.full((16,10),0.1,dtype=np.float32)}
    ref={k:v.copy() for k,v in common.items()}
    fpg={k:v.copy() for k,v in common.items()}
    hls={k:v.copy() for k,v in common.items()}
    hls["relu"][4,1]=1.04
    hls["softmax"][4,1]=0.13
    fpg["softmax"][4,1]=0.11
    specs={}
    for who,values in (("reference",ref),("fpgai",fpg),("hls4ml",hls)):
        out=parent/(who+".npz")
        np.savez(out,**values)
        specs[who]={"file":out.name,"sha256":sha256(out.read_bytes()).hexdigest()}
    meta={
        "schema":SCHEMA,
        "model_kind":"mlp",
        "original_onnx_sha256":"a"*64,
        "ordered_sample_cohort_sha256":"b"*64,
        "tensor_contract":"aligned_logical_values",
        "layout_and_shape_alignment_externally_qualified":False,
        "native_execution_independently_attested":False,
        "sample_count":16,
        "diagnostic_abs_tolerance":0.005,
        "ordered_logical_tensors":[
            {"operator":k,"reference":k,"fpgai":k,"hls4ml":k}
            for k in ("matmul","relu","softmax")],
        "traces":specs,
    }
    p=parent/"manifest.json"
    p.write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
    return p


def test_first_threshold_exceedance_detects_selected_operator_not_causality(tmp_path):
    meta=_traces(tmp_path)
    result=analyze_operator_traces(meta)
    assert result["operator_count"]==3
    assert result["sample_count"]==16
    assert result["first_threshold_exceeding_operator_candidate"]=={
        "fpgaI":"softmax","hls4ml":"relu"}
    assert result["operator_level_root_cause_proven"] is False
    assert result["trace_semantic_alignment_verified_by_independent_vendor_receipts"] is False
    assert result["native_e2_qualification_observed"] is False
    assert result["operator_observations"][0]["fpgaI_max_absolute_difference"]==0
    assert result["operator_observations"][0]["hls4ml_max_absolute_difference"]==0


def test_immutable_private_export(tmp_path):
    meta=_traces(tmp_path)
    path=tmp_path/"output"
    result=export_operator_differential(meta,path)
    assert result==json.loads((path/"operator_differential.json").read_text())
    with pytest.raises(FileExistsError):
        export_operator_differential(meta,path)


@pytest.mark.parametrize("change",(
    "tamper_source","missing_operator","shape","order","nonfinite",
    "external_alignment_claim","native_execution_claim","zero_tolerance",
    "bad_layout","symlink","sample_count","duplicate_operator",
))
def test_fail_closed_operator_trace_context_and_source(tmp_path,change):
    path=_traces(tmp_path)
    row=json.loads(path.read_text())
    hls_file=path.parent/"hls4ml.npz"
    if change=="tamper_source":
        with hls_file.open("ab") as f:
            f.write(b"x")
    elif change=="missing_operator":
        row["ordered_logical_tensors"][1]["hls4ml"]="not_a_real_output"
    elif change=="shape":
        with np.load(hls_file) as n:
            x={k:n[k] for k in n.files}
        x["relu"]=np.ones((15,4),dtype=np.float32)
        np.savez(hls_file,**x)
        row["traces"]["hls4ml"]["sha256"]=sha256(hls_file.read_bytes()).hexdigest()
    elif change=="order":
        row["ordered_logical_tensors"].append(row["ordered_logical_tensors"][0])
    elif change=="nonfinite":
        with np.load(hls_file) as n:
            x={k:n[k] for k in n.files}
        x["relu"][1,1]=np.nan
        np.savez(hls_file,**x)
        row["traces"]["hls4ml"]["sha256"]=sha256(hls_file.read_bytes()).hexdigest()
    elif change=="external_alignment_claim":
        row["layout_and_shape_alignment_externally_qualified"]=True
    elif change=="native_execution_claim":
        row["native_execution_independently_attested"]=True
    elif change=="zero_tolerance":
        row["diagnostic_abs_tolerance"]=0
    elif change=="bad_layout":
        row["tensor_contract"]="unknown"
    elif change=="symlink":
        hls_file.unlink()
        elsewhere=tmp_path/"elsewhere.npz"
        np.savez(elsewhere,relu=np.ones((16,4)))
        hls_file.symlink_to(elsewhere)
    elif change=="sample_count":
        row["sample_count"]=17
    elif change=="duplicate_operator":
        row["ordered_logical_tensors"][1]["operator"]="matmul"
    path.write_text(json.dumps(row,sort_keys=True)+"\n")
    with pytest.raises((ValueError,TypeError)):
        analyze_operator_traces(path)
