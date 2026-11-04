"""R2u opt-in actual 32-case Vitis HLS factorial and read-only E3 analysis.

Run a single original case at a time or an explicitly selected range.
No vendor calls without BOTH --execute-native and --confirm-isolated-native.
Never overwrite or re-run an original invoked case. Frozen R2i/R2j/R2l/
R2m vendor outputs and RL code stay untouched.

All reported BRAM/LUT/FF/DSP/cycle numbers are HLS ESTIMATES. Physical
resources remain UNKNOWN until a separate source-attested Vivado E4 run.
Incomplete/cross-arm invalid measurements remain UNKNOWN, not ties.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from fpgai_research.experiments.cst_r2u_hls_mechanism_prepare import (
    ARMS, CASES, FACTOR_NAMES, FACTORS, GRAPHS,
    audit_prepared,
)
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.cst_nchw_hls_qualification import qualify_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

SCHEMA = "fpgai-research/cst-R2u-source-attributed-HLS-factorial-observation/v1"
RESOURCES = ("bram_18k","lut","ff","dsp","uram")


def _contrast(rows:list[dict], factor:str)->list[dict]:
    """Two-arm conditional comparison holding every OTHER factor fixed."""
    if factor not in FACTORS or len(rows)!=CASES:
        raise ValueError("R2u fixed source/factor population required")
    output=[]
    for gi in range(len(GRAPHS)):
        block=rows[gi*len(ARMS):(gi+1)*len(ARMS)]
        for a in block:
            if a[factor]!=FACTORS[factor][0]:
                continue
            other={key:a[key] for key in FACTOR_NAMES if key!=factor}
            candidates=[
                x for x in block
                if x[factor]==FACTORS[factor][1]
                and all(x[key]==value for key,value in other.items())
            ]
            if len(candidates)!=1:
                raise ValueError("R2u matched native contrast missing second arm")
            b=candidates[0]
            qualified=bool(
                a["original_E3_qualified"] and b["original_E3_qualified"]
            )
            data={
                "graph_index":gi,"factor":factor,"controlled_levels":other,
                "arm_a":a["case"],"arm_b":b["case"],
                "matched_original_CSim_qualified":bool(
                    a["original_E2_qualified"] and b["original_E2_qualified"]
                ),
                "matched_original_CSynth_qualified":qualified,
                "status":("MATCHED_HLS_ESTIMATE_ONLY" if qualified
                          else "UNKNOWN_NOT_BOTH_E3_QUALIFIED"),
                "estimated_resource_delta_b_minus_a":{},
                "estimated_latency_worst_cycles_delta_b_minus_a":None,
                "physical_resource_delta":None,
                "achieved_Fmax_difference":None,
            }
            for field in RESOURCES:
                va=a["estimated_resources"][field]
                vb=b["estimated_resources"][field]
                data["estimated_resource_delta_b_minus_a"][field]=(
                    vb-va if qualified and va is not None
                    and vb is not None else None
                )
            if qualified:
                data["estimated_latency_worst_cycles_delta_b_minus_a"]=(
                    b["hls_worst_cycles"]-a["hls_worst_cycles"]
                )
            output.append(data)
    if len(output)!=len(GRAPHS)*len(ARMS)//2:
        raise AssertionError("R2u incomplete matched factor denominator")
    return output


def audit_native(*,study:str|Path,require_current_source:bool=True)->dict:
    root=Path(study)
    provenance=audit_prepared(
        study=root,require_current_source=require_current_source)
    source=json.loads((root/"factorial_source_manifest.json").read_text())
    rows=[]
    for row in source["rows"]:
        folder=root/row["case"]
        measured=summarize_case(str(folder))
        if measured.get("source_sha256")!=row["source_cpp_sha256"]:
            raise ValueError("R2u native report attributed to different source")
        qualification=qualify_case(folder,row,summary=measured)
        is_e3=qualification["native_e3_qualified"] is True
        rows.append({
            **{key:row[key] for key in (
                "case","graph_index","graph","mapping",
                "source_storage","storage","address","binding","padding",
                "source_cpp_sha256","source_logical_scratch_slots",
                "source_declared_scratch_elements"
            )},
            "actual_native_invocation_receipt":bool(
                (folder/"vitis_invocation.json").is_file()
                and not (folder/"vitis_invocation.json").is_symlink()
            ),
            "source_attested_evidence":measured["evidence"],
            "original_E2_qualified":qualification["native_e2_accepted"],
            "original_E3_qualified":is_e3,
            "qualification_status":qualification["status"],
            "hls_worst_cycles":qualification["hls_worst_cycles"]
                if is_e3 else None,
            "hls_scheduling_margin_ns":
                qualification["hls_scheduling_margin_ns"]
                if is_e3 else None,
            "estimated_resources":{
                key:qualification["estimated_resources"][key]
                if is_e3 else None for key in RESOURCES
            },
            "original_HLS_report_sha256":
                qualification["hls_report_sha256"] if is_e3 else None,
            "actual_hardware_BRAM":None,
            "achieved_physical_Fmax":None,
            "warnings":list(qualification["warnings"]),
        })
    contrasts=[]
    for factor in FACTOR_NAMES:
        contrasts.extend(_contrast(rows,factor))
    return {
        "schema":SCHEMA,
        "original_source_revision":provenance["source_revision"],
        "planned_source_cases":CASES,
        "observed_source_cases":len(rows),
        "original_vitis_invocation_receipts_present":sum(
            x["actual_native_invocation_receipt"] for x in rows),
        "native_CSim_E2_qualified_cases":sum(
            bool(x["original_E2_qualified"]) for x in rows),
        "native_CSynth_E3_qualified_cases":sum(
            bool(x["original_E3_qualified"]) for x in rows),
        "prespecified_matched_factor_contrasts":len(contrasts),
        "E3_both_qualified_contrasts":sum(
            c["matched_original_CSynth_qualified"] for c in contrasts),
        "E3_UNKNOWN_or_rejected_contrasts":sum(
            not c["matched_original_CSynth_qualified"] for c in contrasts),
        "cases":rows,"contrasts":contrasts,
        "original_vendor_calls_during_audit":0,
        "physical_BRAM_measurements":None,
        "physical_Fmax_measurements":None,
        "limits":[
            "Vitis CSim and CSynth, not Vivado routed BRAM, timing, or power",
            "Memory directive request is not proof that actual storage was bound as specified",
            "Padding might be optimized away; report cannot claim effective memory size",
            "Unqualified/failed/absent cases remain UNKNOWN, never imputed as tie or zero",
            "No causal attribution to a single factor beyond the exact controlled static source family",
        ],
    }


def execute(*,study:str|Path,start_case:int=0,max_cases:int=1,
            execute_native:bool=False,confirm_isolated_native:bool=False,
            vitis_executable:str="vitis_hls",
            timeout_seconds:int=7200)->dict:
    root=Path(study)
    source=audit_prepared(study=root,require_current_source=True)
    if (type(start_case) is not int or type(max_cases) is not int
            or start_case<0 or not 1<=max_cases<=CASES
            or start_case+max_cases>CASES):
        raise ValueError("R2u original native case indices must be in 0..31")
    source_doc=json.loads((root/"factorial_source_manifest.json").read_text())
    selected=source_doc["rows"][start_case:start_case+max_cases]
    if not execute_native:
        return {
            "schema":SCHEMA,
            "status":"DRY_RUN_NO_NATIVE_CALLS",
            "cases":[row["case"] for row in selected],
            "requires_explicit_confirm_isolated_native":True,
            "original_Vitis_calls":0,
        }
    if confirm_isolated_native is not True:
        raise ValueError("R2u actual Vitis execution needs explicit confirmation")
    if (type(timeout_seconds) is not int or not 1<=timeout_seconds<=86400):
        raise ValueError("R2u finite native case timeout required")
    executable=shutil.which(vitis_executable)
    if executable is None:
        raise FileNotFoundError("R2u actual Vitis HLS executable unavailable")
    # Check ALL selected cases before executing the first; never overwrite.
    for row in selected:
        folder=root/row["case"]
        for name in ("vitis_invocation.json","logs","fpgai_hls_project"):
            artifact=folder/name
            if artifact.exists() or artifact.is_symlink():
                raise FileExistsError(
                    "R2u original HLS result already exists: "+str(artifact))
    outcomes=[]
    for row in selected:
        folder=root/row["case"]
        receipt=run_vitis(
            folder,executable=executable,timeout_seconds=timeout_seconds)
        measured=summarize_case(str(folder))
        qualified=qualify_case(folder,row,summary=measured)
        outcomes.append({
            "case":row["case"],
            "original_vitis_completed":receipt.get("completed_successfully"),
            "source_evidence":measured["evidence"],
            "E2_CSim_qualified":qualified["native_e2_accepted"],
            "E3_HLS_estimates_qualified":qualified["native_e3_qualified"],
            "qualification_status":qualified["status"],
            "estimated_bram_18k":(
                qualified["estimated_resources"]["bram_18k"]
                if qualified["native_e3_qualified"] else None
            ),
            "physical_BRAM":None,
        })
    return {
        "schema":SCHEMA,
        "original_source_revision":source["source_revision"],
        "status":"ORIGINAL_NATIVE_EXECUTED_OUTCOMES_MAY_BE_UNKNOWN",
        "attempted_original_vitis_calls":len(outcomes),
        "results":outcomes,
        "physical_qualified_cases":0,
    }


def main(argv:list[str]|None=None)->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--audit-native",action="store_true")
    p.add_argument("--execute-native",action="store_true")
    p.add_argument("--confirm-isolated-native",action="store_true")
    p.add_argument("--start-case",type=int,default=0)
    p.add_argument("--max-cases",type=int,default=1)
    p.add_argument("--vitis-executable",default="vitis_hls")
    p.add_argument("--timeout-seconds",type=int,default=7200)
    args=p.parse_args(argv)
    report=(audit_native(study=args.study) if args.audit_native
            else execute(
                study=args.study,start_case=args.start_case,
                max_cases=args.max_cases,execute_native=args.execute_native,
                confirm_isolated_native=args.confirm_isolated_native,
                vitis_executable=args.vitis_executable,
                timeout_seconds=args.timeout_seconds))
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
