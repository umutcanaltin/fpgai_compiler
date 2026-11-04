"""Opt-in, original, source-attested Vitis HLS for matched R2h fanout sources.

Pure audit by default; only --execute-native AND
--confirm-isolated-native actually invokes Vitis. Every selected pair
is preflight-verified against its immutable source/Tcl/testbench SHA.
Missing, failed or incomplete vendor results remain UNKNOWN/REJECTED,
never silently scored zero or claimed as routed physical measurements.

Reuses canonical Vitis invocation, numerical E2/report E3 and effective
clock-uncertainty qualification. Does NOT run Vivado or modify RL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from fpgai_research.experiments.cst_r2i_fanout_native_prepare import (
    TOTAL_PAIRS,audit_prepared,SCHEMA as SOURCE_SCHEMA,
)
from fpgai_research.experiments.cst_nchw_hls_qualification import qualify_case
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

SCHEMA="fpgai-research/cst-R2i-original-matched-native-HLS-observation/v1"
RESOURCE_FIELDS=("bram_18k","lut","ff","dsp","uram")


def audit_native(study:str|Path,*,require_current_source:bool=True)->dict:
    """Read-only classification of complete original native case population."""
    root=Path(study)
    prepared=audit_prepared(root,require_current_source=require_current_source)
    source=json.loads((root/"native_source_manifest.json").read_text())
    results=[]
    for row in source["cases"]:
        folder=root/row["case"]
        observation=summarize_case(str(folder))
        if observation["source_sha256"]!=row["source_cpp_sha256"]:
            raise ValueError("Native observed source differs from prepared original")
        status=qualify_case(folder,row,summary=observation)
        accepted=status["native_e3_qualified"] is True
        results.append({
            "case":row["case"],"group":row["group"],
            "graph":row["graph"],"fanout":row["fanout"],
            "mapping":row["mapping"],"source_storage":row["source_storage"],
            "source_cpp_sha256":row["source_cpp_sha256"],
            "source_scratch_unit_slots":
                row["original_source_scratch_unit_slots"],
            "actual_vitis_invocation_receipt_present":
                (folder/"vitis_invocation.json").is_file(),
            "source_attested_evidence_level":observation["evidence"],
            "original_CSim_qualified":status["native_e2_accepted"],
            "original_CSynth_timing_resources_qualified":accepted,
            "qualification_status":status["status"],
            "hls_worst_cycles":status["hls_worst_cycles"] if accepted else None,
            "hls_estimated_clock_ns":
                status["hls_estimated_clock_ns"] if accepted else None,
            "hls_effective_scheduling_budget_ns":
                status["hls_effective_scheduling_budget_ns"] if accepted else None,
            "hls_scheduling_margin_ns":
                status["hls_scheduling_margin_ns"] if accepted else None,
            "hls_estimated_resources":status["estimated_resources"] if accepted else None,
            "hls_report_sha256":status["hls_report_sha256"] if accepted else None,
            "physical_BRAM":None,
            "achieved_routed_Fmax":None,
            "warnings":status["warnings"],
        })
    paired=[]
    for i in range(TOTAL_PAIRS):
        a,b=results[2*i:2*i+2]
        if (a["source_storage"]!="disjoint"
                or b["source_storage"]!="per_element_optimal"
                or a["graph"]!=b["graph"]
                or a["mapping"]!=b["mapping"]
                or a["source_cpp_sha256"]==b["source_cpp_sha256"]):
            raise ValueError("Unmatched native source pair unexpectedly changed")
        both=a["original_CSynth_timing_resources_qualified"] and b[
            "original_CSynth_timing_resources_qualified"]
        estimated_delta={}
        for key in RESOURCE_FIELDS:
            estimated_delta[key]=(
                a["hls_estimated_resources"][key]-b["hls_estimated_resources"][key]
                if both and a["hls_estimated_resources"].get(key) is not None
                and b["hls_estimated_resources"].get(key) is not None
                else None
            )
        paired.append({
            "group":i,"cases":[a["case"],b["case"]],
            "graph":a["graph"],"mapping":a["mapping"],
            "both_original_CSim_qualified":
                a["original_CSim_qualified"] and b["original_CSim_qualified"],
            "both_original_CSynth_timing_resources_qualified":both,
            "HLS_estimated_resource_delta_disjoint_minus_optimized":
                estimated_delta,
            "HLS_estimated_latency_delta_worst_cycles":(
                a["hls_worst_cycles"]-b["hls_worst_cycles"] if both else None),
            "HLS_bram_estimate_reduction_status":(
                "REDUCTION" if estimated_delta["bram_18k"] is not None
                    and estimated_delta["bram_18k"]>0 else
                "TIE" if estimated_delta["bram_18k"]==0 else
                "REGRESSION" if estimated_delta["bram_18k"] is not None else
                "UNKNOWN_NOT_BOTH_QUALIFIED"),
            "physical_resource_measurements":None,
            "physical_qualified":False,
        })
    return {
        "schema":SCHEMA,"original_source_revision":prepared["source_revision"],
        "prespecified_source_cases":len(results),
        "prespecified_matched_pairs":len(paired),
        "original_vendor_invocation_receipts_present":sum(
            r["actual_vitis_invocation_receipt_present"] for r in results),
        "original_CSim_qualified_cases":sum(
            r["original_CSim_qualified"] for r in results),
        "original_CSynth_timing_resource_qualified_cases":sum(
            r["original_CSynth_timing_resources_qualified"] for r in results),
        "original_CSynth_both_qualified_pairs":sum(
            p["both_original_CSynth_timing_resources_qualified"] for p in paired),
        "original_HLS_estimated_BRAM_reductions":sum(
            p["HLS_bram_estimate_reduction_status"]=="REDUCTION" for p in paired),
        "original_HLS_estimated_BRAM_ties":sum(
            p["HLS_bram_estimate_reduction_status"]=="TIE" for p in paired),
        "original_HLS_estimated_BRAM_regressions":sum(
            p["HLS_bram_estimate_reduction_status"]=="REGRESSION" for p in paired),
        "original_HLS_estimated_BRAM_UNKNOWN":sum(
            p["HLS_bram_estimate_reduction_status"]=="UNKNOWN_NOT_BOTH_QUALIFIED"
            for p in paired),
        "per_case_status":results,
        "all_original_matched_pairs":paired,
        "new_vendor_tool_calls_during_readonly_audit":0,
        "physical_BRAM_or_Fmax_established":False,
        "external_compiler_firstness_established":False,
        "limitations":[
            "All cases retained even on failure, unknown CSynth, timeout or missing tool",
            "HLS estimated BRAM is not routed physical BRAM",
            "No achieved Fmax, RAM port feasibility after routing or strict clean-DRC E4",
            "Source scalar scratch reduction does not guarantee HLS resource reduction",
            "Fixed small models may be completely mapped to registers or LUTs",
        ],
    }


def execute(study:str|Path,*,execute_native:bool=False,
            confirm_isolated_native:bool=False,start_pair:int=0,
            max_pairs:int=1,vitis_executable:str="vitis_hls",
            timeout_seconds:int=7200)->dict:
    root=Path(study)
    audit_prepared(root,require_current_source=True)
    if (type(start_pair) is not int or type(max_pairs) is not int
            or not 0<=start_pair<TOTAL_PAIRS
            or not 1<=max_pairs<=TOTAL_PAIRS
            or start_pair+max_pairs>TOTAL_PAIRS):
        raise ValueError("Finite complete native pair range 0..35 required")
    source=json.loads((root/"native_source_manifest.json").read_text())
    selected=source["cases"][2*start_pair:2*(start_pair+max_pairs)]
    if not execute_native:
        return {
            "schema":SCHEMA,"status":"DRY_RUN_ZERO_NATIVE_CALLS",
            "planned_exact_original_cases":[x["case"] for x in selected],
            "requires_explicit_execute_native_and_confirm_isolated_native":True,
            "original_Vitis_calls":0,
        }
    if confirm_isolated_native is not True:
        raise ValueError("Native Vitis requires --confirm-isolated-native")
    if (type(timeout_seconds) is not int or not 1<=timeout_seconds<=86400
            or not isinstance(vitis_executable,str) or not vitis_executable):
        raise ValueError("Finite native Vitis execution required")
    resolved=shutil.which(vitis_executable)
    if resolved is None:
        raise FileNotFoundError("Actual source-pinned native vitis_hls executable missing")
    # Preflight BOTH arms of EVERY selected pair before the first
    # irreversible original Vitis process. Never rerun partial originals.
    for row in selected:
        folder=root/row["case"]
        for name in ("vitis_invocation.json","logs","fpgai_hls_project"):
            path=folder/name
            if path.exists() or path.is_symlink():
                raise FileExistsError(
                    "Refusing to overwrite original native vendor evidence "+
                    row["case"]+"/"+name)
    outcomes=[]
    for entry in selected:
        folder=root/entry["case"]
        original=run_vitis(
            folder,executable=resolved,timeout_seconds=timeout_seconds)
        report=summarize_case(str(folder))
        if report["source_sha256"]!=entry["source_cpp_sha256"]:
            raise ValueError("Native tool changed prepared source input")
        qualified=qualify_case(folder,entry,summary=report)
        outcomes.append({
            "case":entry["case"],"group":entry["group"],
            "source_storage":entry["source_storage"],
            "original_vendor_process_success":original.get("completed_successfully"),
            "original_evidence_level":report.get("evidence"),
            "original_CSim_qualified":qualified["native_e2_accepted"],
            "original_CSynth_qualified":qualified["native_e3_qualified"],
            "original_timing_feasible":qualified["hls_timing_feasible"],
            "qualification_status":qualified["status"],
            "estimated_bram_18k":qualified["estimated_resources"]["bram_18k"]
                if qualified["native_e3_qualified"] else None,
            "HLS_worst_cycles":qualified["hls_worst_cycles"]
                if qualified["native_e3_qualified"] else None,
            "physical_qualified":False,
            "warnings":qualified["warnings"],
        })
    return {
        "schema":SCHEMA,"status":"ORIGINAL_NATIVE_VITIS_EXECUTED_REPORT_MAY_BE_UNKNOWN",
        "source_revision":source["source_revision"],
        "attempted_original_vitis_calls":len(outcomes),
        "native_CSim_qualified_cases":sum(
            x["original_CSim_qualified"] for x in outcomes),
        "native_CSynth_qualified_cases":sum(
            x["original_CSynth_qualified"] for x in outcomes),
        "physical_qualified_cases":0,
        "results":outcomes,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--audit-native",action="store_true")
    p.add_argument("--execute-native",action="store_true")
    p.add_argument("--confirm-isolated-native",action="store_true")
    p.add_argument("--start-pair",type=int,default=0)
    p.add_argument("--max-pairs",type=int,default=1)
    p.add_argument("--vitis-executable",default="vitis_hls")
    p.add_argument("--timeout-seconds",type=int,default=7200)
    args=p.parse_args(argv)
    report=(
        audit_native(args.study)
        if args.audit_native else
        execute(
            args.study,execute_native=args.execute_native,
            confirm_isolated_native=args.confirm_isolated_native,
            start_pair=args.start_pair,max_pairs=args.max_pairs,
            vitis_executable=args.vitis_executable,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
