"""R4x valid-design-only latency tests, no native executions."""
from __future__ import annotations
from copy import deepcopy

import pytest

from fpgai_research.experiments import (
    cst_paper_r4x_feasible_only_core_latency as sut,
)


def _fake():
    revision="7"*40
    arms=[
        {"arm":"full_storage","E2_qualified":True,"E3_qualified":True,
         "source_sha256":"a"*64,"native_invocation_file_present":True,
         "HLS_worst_cycles":22954,"physical_E4":None},
        {"arm":"row_storage","E2_qualified":True,"E3_qualified":True,
         "source_sha256":"b"*64,"native_invocation_file_present":True,
         "HLS_worst_cycles":22602,"physical_E4":None},
    ]
    hls={
        "schema":sut.r4l.SCHEMA,
        "source_revision":revision,
        "both_E2_qualified":True,"both_E3_qualified":True,
        "E1_matched_pair_accepted":True,
        "E2_qualified_arms":2,"E3_qualified_arms":2,
        "new_vendor_calls_during_read_only_audit":0,
        "original_sources_modified":False,
        "package_hashes":{
            "full_storage":{"model.cpp":"a"*64},
            "row_storage":{"model.cpp":"b"*64}},
        "observations":arms,
    }
    cases={}
    paired=[]
    for period in sut.r4w.PERIODS_NS:
        for arm in sut.ARMS:
            cases[sut.r4w._case(arm,period)]={
                "arm":arm,"period_ns":period,
                "status":"NONQUALIFIED_TIMING_DRC_OR_RESOURCES",
                "tool_invocation_attested":True,
                "routed_complete":True,
                "timing_met":False,"drc_clean":False,
                "strict_clean_OOC_qualified":False,
                "wns_ns":-1.0,"drc_violations":42,
                "resources":{"bram":7 if arm=="full_storage" else 4,
                             "dsp":24 if arm=="full_storage" else 21,
                             "lut":1282 if arm=="full_storage" else 1710,
                             "ff":883 if arm=="full_storage" else 1022,
                             "uram":0},
                "board_qualified":False,
            }
        paired.append({"clock_period_ns":period,
                       "strict_physical_pair_qualified":False})
    sweep={
        "schema":sut.r4w.SCHEMA,
        "status":"ORIGINAL_NEW_VIVADO_FEASIBILITY_SWEEP_AUDITED_NOT_BOARD",
        "original_R4k_source_revision":revision,
        "original_vendor_evidence_modified":False,
        "actual_new_vivado_calls_during_audit":0,
        "board_qualified":False,"board_inference_latency_ns":None,
        "cases":cases,"paired_periods":paired,"qualified_pair_count":0,
    }
    return hls,sweep


def _mark_pair(sweep,period):
    for arm in sut.ARMS:
        row=sweep["cases"][sut.r4w._case(arm,period)]
        row.update({
            "status":"QUALIFIED_TIMING_CLEAN_DRC_ROUTED_OOC_NOT_BOARD",
            "tool_invocation_attested":True,
            "routed_complete":True,"timing_met":True,
            "drc_clean":True,"strict_clean_OOC_qualified":True,
            "wns_ns":0.4 if arm=="full_storage" else 0.25,
            "drc_violations":0,"board_qualified":False,
        })
    sweep["qualified_pair_count"]+=1
    for row in sweep["paired_periods"]:
        if row["clock_period_ns"]==period:
            row["strict_physical_pair_qualified"]=True


def test_no_valid_clock_no_winner_or_fabricated_fpga_latency():
    hls,sweep=_fake()
    report=sut.qualify(hls,sweep)
    assert report["qualified_paired_clocks"]==0
    assert report["fastest_predeclared_joint_qualified_clock_ns"] is None
    assert report["status"]=="NO_JOINTLY_QUALIFIED_PAIR_NO_HARDWARE_LATENCY_COMPARISON"
    assert report["verified_board_core_latency_ns"] is None
    assert all(c["row_minus_full_core_latency_estimate_ns"] is None
               for c in report["comparisons_including_failures"])


def test_exact_equal_clock_qualified_pair_gets_time_proxy_not_measured():
    hls,sweep=_fake()
    _mark_pair(sweep,20.0)
    report=sut.qualify(hls,sweep)
    assert report["qualified_paired_clocks"]==1
    assert report["fastest_predeclared_joint_qualified_clock_ns"]==20.0
    assert report["status"]=="FEASIBLE_OOC_CORE_LATENCY_PROXIES_ONLY_NOT_MEASURED"
    p=report["comparisons_including_failures"][1]
    assert p["qualified_both_arms"] is True
    lat=p["core_latency_estimate_ns_derived_from_HLS_cycles_at_this_qualified_clock"]
    assert lat["full_storage"]==22954*20
    assert lat["row_storage"]==22602*20
    assert p["row_minus_full_core_latency_estimate_ns"]==-352*20
    assert p["real_measured_core_latency_ns"] is None
    assert p["real_measured_end_to_end_board_latency_ns"] is None
    assert not report["physical_clock_frequency_is_measured"]
    assert all(not c["qualified_both_arms"] for c in
               (report["comparisons_including_failures"][0],
                report["comparisons_including_failures"][2]))


def test_one_arm_timing_passes_is_not_a_valid_paired_comparison():
    hls,sweep=_fake()
    c=sweep["cases"][sut.r4w._case("full_storage",16.0)]
    c.update(status="QUALIFIED_TIMING_CLEAN_DRC_ROUTED_OOC_NOT_BOARD",
             strict_clean_OOC_qualified=True,drc_clean=True,timing_met=True,
             wns_ns=0.2,drc_violations=0)
    result=sut.qualify(hls,sweep)
    assert result["qualified_paired_clocks"]==0
    assert result["comparisons_including_failures"][0][
        "core_latency_estimate_ns_derived_from_HLS_cycles_at_this_qualified_clock"]=={
            "full_storage":None,"row_storage":None}


def test_synthetic_false_valid_flag_or_missing_drc_detected():
    hls,sweep=_fake()
    _mark_pair(sweep,24.0)
    item=sweep["cases"][sut.r4w._case("row_storage",24.0)]
    item["drc_violations"]=11
    with pytest.raises(ValueError,match="strict"):
        sut.qualify(hls,sweep)


def test_source_revision_or_HLS_cycle_data_mismatch_rejected():
    hls,sweep=_fake()
    hls["observations"][1]["HLS_worst_cycles"]=0
    with pytest.raises(ValueError,match="HLS"):
        sut.qualify(hls,sweep)
    hls,sweep=_fake()
    sweep["original_R4k_source_revision"]="a"*40
    with pytest.raises(ValueError,match="source-matched"):
        sut.qualify(hls,sweep)


def test_analysis_does_not_change_signed_original_fixtures():
    hls,sweep=_fake()
    _mark_pair(sweep,16.0)
    before=deepcopy((hls,sweep))
    sut.qualify(hls,sweep)
    assert (hls,sweep)==before


def test_run_uses_original_qualified_E3_and_new_native_audits(monkeypatch):
    hls,sweep=_fake()
    seen=[]
    monkeypatch.setattr(sut.r4l,"audit_native",lambda **kw:(
        seen.append(("hls",kw)),hls)[1])
    monkeypatch.setattr(sut.r4w,"audit_native",lambda **kw:(
        seen.append(("vivado",kw)),sweep)[1])
    out=sut.run(study="/new",r4o_study="/r4o",r4k_study="/r4k",
                original_r4d="/r4d",original_r4a="/r4a")
    assert out["native_vendor_or_compiler_calls_during_analysis"]==0
    assert len(seen)==2
    assert seen[0][1]["study"]=="/r4k"
    assert seen[1][1]["study"]=="/new"
