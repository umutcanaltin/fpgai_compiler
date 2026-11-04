"""Strict native E2/E3 clock/resource gate, no actual vendor calls.

Synthetic summaries here test the parser's rejection/aggregation logic.
Only the real summarize_case + audited_hls_timing path can qualify
original source-attested Vitis evidence outside of these unit tests.
"""
from hashlib import sha256
import json
import pytest

from fpgai_research.experiments.cst_nchw_native_pairs import prepare
from fpgai_research.experiments.cst_nchw_hls_qualification import (
    qualify_case,compare_qualified_pairs,
)


def synthetic_native(case, source_digest, *, evidence="E3-HLS-synthesis-report",
                     csim=True, metrics=None):
    if metrics is None:
        metrics={"latency_best":"5","latency_worst":"10","ii_min":"1",
                 "ii_max":"1","bram_18k":"3","dsp":"4","lut":"8",
                 "ff":"17","uram":"0"}
    return {
        "source_sha256":source_digest,"evidence":evidence,
        "csim_passed":csim,"warnings":[],
        "csim_receipt":{"samples":1,"outputs_per_sample":case["csim_outputs_per_sample"],
                        "failures":0,"max_absolute_error":0.02},
        "metrics":metrics,
        "hls_report_sha256":"f"*64,
    }


def fake_provenance(case_dir):
    import hashlib
    receipt={
        "source_sha256":sha256((case_dir/"model.cpp").read_bytes()).hexdigest(),
        "testbench_sha256":sha256((case_dir/"tb.cpp").read_bytes()).hexdigest(),
        "script_sha256":sha256((case_dir/"run_hls.tcl").read_bytes()).hexdigest(),
        "completed_successfully":True,"inputs_stable_during_execution":True,
    }
    (case_dir/"vitis_invocation.json").write_text(json.dumps(receipt))
    return receipt


def test_no_vendor_metadata_stays_unknown_and_stale_xml_is_not_report(tmp_path):
    root=tmp_path/"study"
    prepared=prepare(output=root)
    row=prepared["cases"][0]
    case=root/row["case"]
    empty={
        "source_sha256":row["source_cpp_sha256"],
        "evidence":"E1-HLS-source","csim_passed":None,
        "warnings":[],"metrics":{},
    }
    result=qualify_case(case,row,summary=empty)
    assert result["status"]=="UNKNOWN_NO_ORIGINAL_VITIS"
    assert result["native_e2_accepted"] is False
    assert result["native_e3_qualified"] is False
    assert result["hls_worst_cycles"] is None
    assert result["physical_clock_ns"] is None
    xml=case/"fpgai_hls_project"/"solution1"/"syn"/"report"
    xml.mkdir(parents=True)
    (xml/"fpgai_model_csynth.xml").write_text("<fake/>")
    assert qualify_case(case,row)["status"]=="UNKNOWN_NO_ORIGINAL_VITIS"


def test_numeric_missing_and_csim_failure_are_not_promoted(tmp_path):
    root=tmp_path/"study"
    prepared=prepare(output=root)
    row=prepared["cases"][0]
    case=root/row["case"]
    cfg=json.loads((case/"manifest.json").read_text())
    fail=synthetic_native(cfg,row["source_cpp_sha256"],csim=False)
    assert qualify_case(case,row,summary=fail)["status"]=="FAILED_ORIGINAL_CSIM"
    incomplete=synthetic_native(cfg,row["source_cpp_sha256"])
    incomplete["csim_receipt"]["samples"]=2
    result=qualify_case(case,row,summary=incomplete)
    assert result["status"]=="REJECTED_NUMERIC_COVERAGE"
    assert result["native_e2_accepted"] is False


def test_native_timing_gate_and_resource_completeness(monkeypatch,tmp_path):
    root=tmp_path/"study"
    prepared=prepare(output=root)
    row=prepared["cases"][0]
    case=root/row["case"]
    cfg=json.loads((case/"manifest.json").read_text())
    native=synthetic_native(cfg,row["source_cpp_sha256"])
    fake_provenance(case)
    rejected={"hls_timing_feasible":False,"hls_estimated_clock_ns":30.0,
              "hls_effective_scheduling_budget_ns":23.36,
              "hls_scheduling_margin_ns":-6.64,"hls_report_sha256":"f"*64}
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_hls_qualification.audited_hls_timing",
        lambda *args,**kwargs: rejected)
    bad=qualify_case(case,row,summary=native)
    assert bad["native_e2_accepted"] is True
    assert bad["native_e3_qualified"] is False
    assert bad["status"]=="FAILED_HLS_EFFECTIVE_CLOCK_BUDGET"
    good={**rejected,"hls_timing_feasible":True,"hls_estimated_clock_ns":16.0,
          "hls_scheduling_margin_ns":7.36}
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_hls_qualification.audited_hls_timing",
        lambda *args,**kwargs: good)
    native["metrics"]["bram_18k"]=None
    incomplete=qualify_case(case,row,summary=native)
    assert incomplete["status"]=="UNKNOWN_INCOMPLETE_HLS_METRICS"
    native["metrics"]["bram_18k"]="3"
    done=qualify_case(case,row,summary=native)
    assert done["native_e3_qualified"] is True
    assert done["status"]=="QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL"
    assert done["estimated_resources"]["bram_18k"]==3
    assert done["hls_worst_cycles"]==10
    assert done["physical_qualified"] is False
    assert done["physical_clock_ns"] is None


def test_hls_pairs_unknown_then_estimated_only_no_physical(tmp_path):
    root=tmp_path/"study"
    prepared=prepare(output=root)
    first,second=prepared["cases"][:2]
    a={"case":first["case"],"source_storage":"real_disjoint",
       "source_cpp_sha256":first["source_cpp_sha256"],
       "graph":first["graph"],"mapping":first["mapping"],
       "requested_clock_ns":32.0,"requested_uncertainty_ns":8.64,
       "native_e2_accepted":False,"native_e3_qualified":False,
       "estimated_resources":{"bram_18k":None},
       "hls_worst_cycles":None}
    b={**a,"case":second["case"],
       "source_storage":"real_shared",
       "source_cpp_sha256":second["source_cpp_sha256"]}
    pending=compare_qualified_pairs([a,b])[0]
    assert pending["status"]=="UNKNOWN_OR_REJECTED_HLS_PAIR"
    assert pending["hls_estimated_bram_difference"] is None
    a={**a,"native_e2_accepted":True,"native_e3_qualified":True,
       "estimated_resources":{"bram_18k":4,"dsp":2,"lut":50,"ff":30},
       "hls_worst_cycles":22}
    b={**b,"native_e2_accepted":True,"native_e3_qualified":True,
       "estimated_resources":{"bram_18k":2,"dsp":2,"lut":60,"ff":35},
       "hls_worst_cycles":20}
    qualified=compare_qualified_pairs([a,b])[0]
    assert qualified["status"]=="PAIRED_HLS_ESTIMATE_ONLY"
    assert qualified["hls_estimated_bram_difference"]==2
    assert qualified["physical_bram_savings"] is None
    assert qualified["achieved_fmax"] is None
    with pytest.raises(ValueError,match="Unmatched"):
        compare_qualified_pairs([a,{**b,"requested_clock_ns":16.0}])
