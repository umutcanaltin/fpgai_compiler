"""R4v original after-Vitis and after-Vivado join: synthetic receipt tests.

These synthetic numbers model the shape of OLD receipts; they are not
substitutes for author original signed reports. No vendor or compiler calls.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from fpgai_research.experiments import (
    cst_paper_r4v_vitis_vivado_joint_compare as sut,
)


def synthetic_receipts():
    source=sut.r4r.SOURCE_REVISION
    fullsha="a"*64
    rowsha="b"*64
    arms=[]
    for arm,sha,cycles,period,bram,dsp,lut,ff in (
            ("full_storage",fullsha,22954,9.857,13,21,5095,2120),
            ("row_storage",rowsha,22602,9.500,13,19,4755,1647)):
        arms.append({
            "arm":arm,
            "E2_qualified":True,"E3_qualified":True,
            "native_invocation_file_present":True,
            "physical_E4":None,
            "source_sha256":sha,
            "HLS_CSynth_report_sha256":"c"*64,
            "HLS_worst_cycles":cycles,
            "HLS_estimated_clock_ns":period,
            "HLS_estimated_resources":{
                "bram_18k":bram,"dsp":dsp,"lut":lut,"ff":ff,"uram":0},
            "CSim_max_absolute_error":0.0001,
        })
    hls={
        "schema":sut.r4l.SCHEMA,
        "status":"PAIRED_QUALIFIED_HLS_ESTIMATES_ONLY",
        "E1_matched_pair_accepted":True,"both_E2_qualified":True,
        "both_E3_qualified":True,
        "E2_qualified_arms":2,"E3_qualified_arms":2,
        "new_vendor_calls_during_read_only_audit":0,
        "original_sources_modified":False,"physical_E4":None,
        "source_revision":source,
        "package_hashes":{"full_storage":{"model.cpp":fullsha},
                          "row_storage":{"model.cpp":rowsha}},
        "observations":arms,
    }
    physical=[]
    for arm,wns,bram,dsp,lut,ff,drc in (
            ("full_storage",-2.170,7,24,1282,883,74),
            ("row_storage",-3.718,4,21,1710,1022,71)):
        physical.append({
            "arm":arm,
            "resources":{"bram":bram,"dsp":dsp,"lut":lut,"ff":ff,"uram":0},
            "routed_timing_met":False,
            "strict_clean_DRC_E4":False,
            "board_qualified":False,
            "blocked_conditions":["NEGATIVE_ROUTED_WNS","DRC_NONCLEAN"],
            "drc_total_violations":drc,
            "routed_wns_ns":wns,
            "source_report_sha256":{k:"f"*64 for k in sut.r4q.REPORTS},
        })
    routed={
        "schema":sut.r4r.SCHEMA,
        "status":"SIGNED_ORIGINAL_ROUTED_OOC_TRADEOFF_TIMING_AND_DRC_NOT_QUALIFIED",
        "source_attested_original_routed_pair":True,
        "original_source_revision":source,
        "strict_clean_DRC_E4":False,"physical_E4":None,
        "board_qualified":False,
        "native_vendor_calls_during_analysis":0,
        "original_source_and_reports_modified":False,
        "original_arms":physical,
        "physical_resource_delta_row_minus_full":{
            "bram":-3,"dsp":-3,"lut":428,"ff":139,"uram":0},
        "routed_WNS_delta_row_minus_full_ns":-1.548,
    }
    return hls,routed


def test_original_stage_matched_derived_latency_is_e3_proxy_only():
    hls,routed=synthetic_receipts()
    score=sut.compare(hls,routed)
    assert score["status"].endswith("NOT_E4")
    assert score["original_source_revision"]==sut.r4r.SOURCE_REVISION
    assert score["native_vitis_vivado_calls_during_analysis"]==0
    assert score["strict_physical_E4"] is None
    assert score["physically_achievable_inference_latency_ns"] is None
    assert score["board_qualified"] is False
    a=score["arms"]
    assert a["full_storage"]["vitis_hls_worst_cycles"]==22954
    assert a["row_storage"]["vitis_hls_worst_cycles"]==22602
    assert a["full_storage"]["vitis_hls_estimated_latency_proxy_ns"]==round(22954*9.857,6)
    assert a["row_storage"]["vitis_hls_estimated_latency_proxy_ns"]==round(22602*9.5,6)
    assert a["full_storage"]["timing_qualified_physical_latency_ns"] is None
    assert a["row_storage"]["timing_qualified_physical_latency_ns"] is None
    d=score["row_minus_full"]
    assert d["vitis_worst_cycles"]==-352
    assert d["vitis_estimated_resources"]["bram_18k"]==0
    assert d["vivado_routed_resources"]["bram"]==-3
    assert d["vivado_routed_resources"]["lut"]==428
    assert d["vivado_routed_wns_ns"]==-1.548
    assert d["vivado_routed_drc_violations"]==-3


@pytest.mark.parametrize("who,operation,expected",[
    ("hls",lambda x:x.update(both_E2_qualified=False),"Vitis"),
    ("hls",lambda x:x["observations"][1].update(E3_qualified=False),"Vitis"),
    ("hls",lambda x:x["observations"][0].update(HLS_estimated_clock_ns=0),"estimated clock"),
    ("hls",lambda x:x["observations"][0].update(CSim_max_absolute_error=None),"numerical"),
    ("hls",lambda x:x["package_hashes"]["row_storage"].update({"model.cpp":"d"*64}),"Vitis"),
    ("routed",lambda x:x.update(original_source_revision="0"*40),"Vivado"),
    ("routed",lambda x:x["original_arms"][0].update(routed_timing_met=True),"routed"),
    ("routed",lambda x:x["original_arms"][1].update(drc_total_violations=0),"routed"),
    ("routed",lambda x:x["original_arms"][1].update(source_report_sha256={} ),"SHA"),
    ("routed",lambda x:x["original_arms"][1]["resources"].update(bram=12),"deltas"),
    ("routed",lambda x:x.update(routed_WNS_delta_row_minus_full_ns=0),"WNS"),
])
def test_tampered_original_stage_provenance_fails_closed(who,operation,expected):
    hls,routed=synthetic_receipts()
    operation(hls if who=="hls" else routed)
    with pytest.raises(ValueError,match=expected):
        sut.compare(hls,routed)


def test_run_reaudits_both_originals_without_vendor_calls(monkeypatch):
    hls,routed=synthetic_receipts()
    seen=[]
    monkeypatch.setattr(sut.r4l,"audit_native",lambda **kw: (
        seen.append(("hls",kw)),hls)[1])
    fake_diag={"original_synthetic":True}
    monkeypatch.setattr(sut.r4q,"inspect",lambda **kw: (
        seen.append(("routed",kw)),fake_diag)[1])
    monkeypatch.setattr(sut.r4r,"summarize",lambda report: (
        seen.append(("signed-score",report)),routed)[1])
    result=sut.run(
        r4p_study="/r4p",r4o_study="/r4o",r4k_study="/r4k",
        original_r4d="/r4d",original_r4a="/r4a")
    assert result["native_vitis_vivado_calls_during_analysis"]==0
    assert [event[0] for event in seen]==["hls","routed","signed-score"]
    assert seen[1][1]["study"]=="/r4p"
    assert seen[2][1] is fake_diag


def test_read_only_fixture_unchanged_after_comparison():
    hls,routed=synthetic_receipts()
    hls0,routed0=deepcopy(hls),deepcopy(routed)
    sut.compare(hls,routed)
    assert (hls,routed)==(hls0,routed0)
