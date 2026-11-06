"""R4x: compare latency only at STRICT jointly feasible Vivado OOC clocks.

Source parent HLS cycles are SHA-audited originals from R4l.
New Vivado clock sweep is audited via R4w: for each equal-clock pair,
BOTH designs must route, WNS>=0, DRC clean, and have intact source/ROM
and native report receipts. No valid pair => NO latency winner.

cycles * a COMMON timing-qualified constrained period is a *derived core
schedule LATENCY ESTIMATE*, NOT actual measured RTL/board inference time.
True core/board latency and achieved Fmax remain null absent measurements.
"""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

from fpgai_research.experiments import cst_paper_r4w_feasible_clock_sweep as r4w
from fpgai_research.experiments import cst_r4l_schedule_matched_storage_native as r4l

SCHEMA="fpgai-research/R4x-feasible-only-paired-clock-HLS-core-latency-estimate/v1"
ARMS=("full_storage","row_storage")


def qualify(hls: dict, physical: dict) -> dict:
    if (hls.get("schema")!=r4l.SCHEMA
            or hls.get("both_E2_qualified") is not True
            or hls.get("both_E3_qualified") is not True
            or hls.get("E1_matched_pair_accepted") is not True
            or hls.get("E2_qualified_arms")!=2
            or hls.get("E3_qualified_arms")!=2
            or hls.get("new_vendor_calls_during_read_only_audit")!=0
            or hls.get("original_sources_modified") is not False
            or not isinstance(hls.get("source_revision"),str)
            or len(hls["source_revision"])!=40):
        raise ValueError("R4x E2/E3 original HLS data not certified")
    arms=hls.get("observations")
    if (not isinstance(arms,list) or len(arms)!=2
            or [a.get("arm") for a in arms]!=list(ARMS)):
        raise ValueError("R4x matched original HLS arms missing")
    cycles={}
    for a in arms:
        count=a.get("HLS_worst_cycles")
        if (a.get("E2_qualified") is not True
                or a.get("E3_qualified") is not True
                or a.get("native_invocation_file_present") is not True
                or a.get("source_sha256")!=hls.get(
                    "package_hashes",{}).get(a["arm"],{}).get("model.cpp")
                or not isinstance(count,int) or isinstance(count,bool)
                or count<=0
                or a.get("physical_E4") is not None):
            raise ValueError("R4x original source HLS cycles not valid")
        cycles[a["arm"]]=count
    if (physical.get("schema")!=r4w.SCHEMA
            or physical.get("status")!=
                "ORIGINAL_NEW_VIVADO_FEASIBILITY_SWEEP_AUDITED_NOT_BOARD"
            or physical.get("original_R4k_source_revision")
                !=hls["source_revision"]
            or physical.get("original_vendor_evidence_modified") is not False
            or physical.get("actual_new_vivado_calls_during_audit")!=0
            or physical.get("board_qualified") is not False
            or physical.get("board_inference_latency_ns") is not None
            or not isinstance(physical.get("cases"),dict)
            or len(physical["cases"])!=len(r4w.PERIODS_NS)*len(ARMS)):
        raise ValueError("R4x new routed source-matched comparison not attested")
    periods=physical.get("paired_periods")
    if (not isinstance(periods,list)
            or len(periods)!=len(r4w.PERIODS_NS)
            or [p.get("clock_period_ns") for p in periods]
                !=list(r4w.PERIODS_NS)):
        raise ValueError("R4x original prespecified clock sweep incomplete")
    comparisons=[]
    accepted=0
    for p in periods:
        period=p["clock_period_ns"]
        case_full=physical["cases"].get(r4w._case("full_storage",period))
        case_row=physical["cases"].get(r4w._case("row_storage",period))
        if not isinstance(case_full,dict) or not isinstance(case_row,dict):
            raise ValueError("R4x original paired clock cases missing")
        candidate=(case_full,case_row)
        both=all(c.get("strict_clean_OOC_qualified") is True for c in candidate)
        if p.get("strict_physical_pair_qualified") is not both:
            raise ValueError("R4x native clock qualification and paired flag differ")
        # Never accept any one-arm routing/DRC result as a feasible pair.
        if both:
            for a,c in zip(ARMS,candidate):
                if (c.get("arm")!=a or c.get("period_ns")!=period
                        or c.get("tool_invocation_attested") is not True
                        or c.get("routed_complete") is not True
                        or c.get("timing_met") is not True
                        or c.get("drc_clean") is not True
                        or c.get("drc_violations")!=0
                        or c.get("board_qualified") is not False
                        or type(c.get("wns_ns")) not in (float,int)
                        or not isfinite(c["wns_ns"]) or c["wns_ns"]<0
                        or c.get("status")!=
                            "QUALIFIED_TIMING_CLEAN_DRC_ROUTED_OOC_NOT_BOARD"):
                    raise ValueError("R4x qualified clock lacks strict source/timing/DRC gate")
        # All candidate clocks are shown, including both failures or one-arm
        # qualifications. Only a fully paired valid clock gets a core estimate.
        if both:
            accepted+=1
        latency={
            a:round(cycles[a]*period,6) if both else None
            for a in ARMS
        }
        comparisons.append({
            "clock_period_ns":period,
            "qualified_both_arms":both,
            "full_arm_status":case_full.get("status"),
            "row_arm_status":case_row.get("status"),
            "full_arm_routed_wns_ns":case_full.get("wns_ns"),
            "row_arm_routed_wns_ns":case_row.get("wns_ns"),
            "core_latency_estimate_ns_derived_from_HLS_cycles_at_this_qualified_clock":latency,
            "row_minus_full_core_latency_estimate_ns":(
                round(latency["row_storage"]-latency["full_storage"],6)
                if both else None),
            "row_vs_full_relative_latency_reduction_percent":(
                round(100.0*(latency["full_storage"]-latency["row_storage"])/
                      latency["full_storage"],6) if both else None),
            "routed_resources":(
                {a:c["resources"] for a,c in zip(ARMS,candidate)}
                if both else None),
            "real_measured_core_latency_ns":None,
            "real_measured_end_to_end_board_latency_ns":None,
        })
    if accepted!=physical.get("qualified_pair_count"):
        raise ValueError("R4x accepted pair count differs from original native audit")
    fastest=min(
        (c for c in comparisons if c["qualified_both_arms"]),
        key=lambda c:c["clock_period_ns"],default=None)
    return {
        "schema":SCHEMA,
        "status":(
            "FEASIBLE_OOC_CORE_LATENCY_PROXIES_ONLY_NOT_MEASURED"
            if accepted>0 else
            "NO_JOINTLY_QUALIFIED_PAIR_NO_HARDWARE_LATENCY_COMPARISON"),
        "original_source_revision":hls["source_revision"],
        "predeclared_clock_candidates":list(r4w.PERIODS_NS),
        "qualified_paired_clocks":accepted,
        "original_HLS_worst_cycles_by_arm":cycles,
        "comparisons_including_failures":comparisons,
        "fastest_predeclared_joint_qualified_clock_ns":(
            fastest["clock_period_ns"] if fastest else None),
        "source_HLS_II_cycles":None,
        "verified_board_core_latency_ns":None,
        "verified_board_end_to_end_latency_ns":None,
        "estimated_core_latency_only":accepted>0,
        "physical_clock_frequency_is_measured":False,
        "board_qualified":False,
        "native_vendor_or_compiler_calls_during_analysis":0,
        "original_evidence_modified":False,
        "limitations":[
            "Both designs require routed WNS>=0 and ZERO DRC violations at the same clock",
            "Clock search reuses original fixed HLS RTL and cannot correct original RAM-port hazards",
            "cycles times feasible imposed clock is an E3-derived core schedule estimate, NOT measured latency",
            "Real core latency, initiation interval, end-to-end DMA time and board throughput are UNKNOWN",
            "OOC clock feasibility does not establish complete board bitstream validation",
            "Even a qualified clock/slot contrast is one graph, precision, part and placement result",
        ],
    }


def run(*,study:str|Path,r4o_study:str|Path,r4k_study:str|Path,
        original_r4d:str|Path,original_r4a:str|Path)->dict:
    hls=r4l.audit_native(
        study=r4k_study,original_r4d=original_r4d,
        original_r4a=original_r4a)
    physical=r4w.audit_native(
        study=study,r4o_study=r4o_study,r4k_study=r4k_study,
        original_r4d=original_r4d,original_r4a=original_r4a)
    return qualify(hls,physical)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--r4o-study",required=True)
    p.add_argument("--r4k-study",required=True)
    p.add_argument("--original-r4d",required=True)
    p.add_argument("--original-r4a",required=True)
    args=p.parse_args(argv)
    print(json.dumps(run(**vars(args)),indent=2,sort_keys=True,
                     allow_nan=False))


if __name__=="__main__":
    main()
