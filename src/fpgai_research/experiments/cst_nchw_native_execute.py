"""Explicit foreground Vitis HLS execution for matched NCHW scratch cases.

No Vitis invocation on module import, default CLI invocation, tests,
preparation, dry-run or read-only audit. The native executor requires
BOTH --execute-native and --confirm-isolated-native, validates immutable
source/Tcl/testbench/case manifests, and bounds exact matched pairs.
Tool return code never substitutes for independent source-attributed
CSim/CSynth results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from fpgai_research.experiments.cst_nchw_native_pairs import (
    audit_prepared,SCHEMA as SOURCE_SCHEMA,
)
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis
from fpgai_research.experiments.cst_nchw_hls_qualification import (
    qualify_case,compare_qualified_pairs,
)

SCHEMA="fpgai-research/cst-NCHW-source-paired-native-Vitis-audit/v1"


def audit_native(study:str|Path,*,require_current_source:bool=True):
    """Read-only: missing actual native evidence stays explicitly NOT_RUN."""
    root=Path(study)
    attested=audit_prepared(root,require_current_source=require_current_source)
    manifest=json.loads((root/"native_source_manifest.json").read_text())
    rows=[]
    full_qualification=[]
    for item in manifest["cases"]:
        case=root/item["case"]
        source=summarize_case(str(case))
        if source["source_sha256"]!=item["source_cpp_sha256"]:
            raise ValueError("Native source differs from prepared original source")
        qualification=qualify_case(case,item,summary=source)
        full_qualification.append(qualification)
        native=(case/"vitis_invocation.json").is_file()
        has_original_hls=qualification["native_e3_qualified"]
        rows.append({
            "case":item["case"],"source_storage":item["source_storage"],
            "mapping":item["mapping"],"graph":item["graph"],
            "source_cpp_sha256":item["source_cpp_sha256"],
            "native_invocation_present":native,
            "attested_evidence_level":source["evidence"],
            "csim_passed":source["csim_passed"],
            "reported_hls_worst_cycles":qualification["hls_worst_cycles"]
                if has_original_hls else None,
            "reported_hls_bram_18k":qualification["estimated_resources"]["bram_18k"]
                if has_original_hls else None,
            "observed_numerical_and_hls_report":has_original_hls,
            "native_qualification_status":qualification["status"],
            "original_csim_qualified":qualification["native_e2_accepted"],
            "original_hls_effective_clock_feasible":qualification["hls_timing_feasible"],
            "original_hls_estimated_clock_ns":qualification["hls_estimated_clock_ns"],
            "original_hls_effective_budget_ns":qualification["hls_effective_scheduling_budget_ns"],
            "original_hls_timing_margin_ns":qualification["hls_scheduling_margin_ns"],
            "estimated_hls_resources":qualification["estimated_resources"]
                if has_original_hls else None,
            "warnings":qualification["warnings"],
            "achieved_fpga_clock_ns":None,
            "qualified_physical_BRAM_savings":None,
        })
    # No selection bias: all six prespecified source pairs are retained.
    # A missing/failed CSim, incomplete CSynth XML or ineffective HLS
    # clock yields explicit UNKNOWN/REJECTED, never zero resource savings.
    qualified_pairs=compare_qualified_pairs(full_qualification)
    pairs=[]
    for entry in qualified_pairs:
        a,b=rows[2*entry["group"]:2*entry["group"]+2]
        paired=entry["hls_both_qualified"]
        pairs.append({
            "group":entry["group"],"cases":entry["cases"],
            "both_original_CSim_CSynth_attested":paired,
            "both_original_CSim_qualified":entry["csim_both_pass"],
            "original_matched_worst_cycles":(
                [a["reported_hls_worst_cycles"],b["reported_hls_worst_cycles"]]
                if paired else None),
            "original_matched_bram_18k":(
                [a["reported_hls_bram_18k"],b["reported_hls_bram_18k"]]
                if paired else None),
            "estimated_HLS_bram_difference":entry["hls_estimated_bram_difference"],
            "qualified_HLS_estimate_only":paired,
            "qualification_status":entry["status"],
            "HLS_estimate_is_not_routed_timing":True,
            "physical_QoR_qualified":False,
        })
    return {
        "schema":SCHEMA,"source_revision":attested["source_revision"],
        "cases":len(rows),"matched_pairs":len(pairs),
        "native_invocation_receipts_present":sum(
            r["native_invocation_present"] for r in rows),
        "original_CSim_CSynth_attested_cases":sum(
            r["observed_numerical_and_hls_report"] for r in rows),
        "original_matched_report_pairs":sum(
            p["both_original_CSim_CSynth_attested"] for p in pairs),
        "cases_audited":rows,"pairs_audited":pairs,
        "physical_qualified_cases":0,
        "notes":[
            "All native metrics are null without original source-attributed vendor files",
            "An HLS target/latency estimate does NOT prove timing closure or routed Fmax",
            "Each HLS pair additionally requires clock-uncertainty-adjusted timing and complete native resources",
            "Original matched CSim/CSynth pairs are not physical BRAM savings evidence",
        ],
    }


def execute(study:str|Path,*,execute_native:bool=False,
            confirm_isolated_native:bool=False,start_pair:int=0,
            max_pairs:int=1,vitis_executable:str="vitis_hls",
            timeout_seconds:int=7200):
    """Bounded actual Vitis execution, NEVER from normal paper/CPU CI."""
    root=Path(study)
    audit_prepared(root,require_current_source=True)
    if (type(start_pair) is not int or type(max_pairs) is not int
            or not 0<=start_pair<6 or not 1<=max_pairs<=6
            or start_pair+max_pairs>6):
        raise ValueError("Select one to six complete matched native HLS pairs")
    rows=json.loads((root/"native_source_manifest.json").read_text())["cases"]
    chosen=rows[2*start_pair:2*(start_pair+max_pairs)]
    if not execute_native:
        return {"schema":SCHEMA,"status":"DRY_RUN_NO_VENDOR_CALLS",
                "planned_source_cases":[r["case"] for r in chosen],
                "requires_execute_native_and_confirm_isolated_native":True,
                "original_vitis_calls":0}
    if type(confirm_isolated_native) is not bool or not confirm_isolated_native:
        raise ValueError("Original vendor execution requires --confirm-isolated-native")
    if (type(timeout_seconds) is not int or not 1<=timeout_seconds<=86400
            or not isinstance(vitis_executable,str) or not vitis_executable.strip()):
        raise ValueError("Finite per-case native timeout and executable required")
    resolved=shutil.which(vitis_executable)
    if not resolved:
        raise FileNotFoundError("Original Vitis HLS executable unavailable")
    # Preflight EVERY chosen case before launching first vendor process.
    # Never silently overwrite original vendor evidence or rerun a case.
    for row in chosen:
        folder=root/row["case"]
        if (any((folder/name).exists() or (folder/name).is_symlink()
                for name in ("vitis_invocation.json","logs","fpgai_hls_project"))):
            raise FileExistsError("Refusing to overwrite existing native evidence "+row["case"])
    outcomes=[]
    for row in chosen:
        folder=root/row["case"]
        tool=run_vitis(folder,executable=resolved,timeout_seconds=timeout_seconds)
        measured=summarize_case(str(folder))
        if measured["source_sha256"]!=row["source_cpp_sha256"]:
            raise ValueError("Original Vitis execution changed source identity")
        qualified=qualify_case(folder,row,summary=measured)
        outcomes.append({
            "case":row["case"],
            "vendor_process_success":tool.get("completed_successfully"),
            "source_attested_evidence":measured.get("evidence"),
            "native_qualification_status":qualified["status"],
            "original_csim_qualified":qualified["native_e2_accepted"],
            "original_hls_timing_feasible":qualified["hls_timing_feasible"],
            "original_csim_csynth_qualified":qualified["native_e3_qualified"],
            "hls_worst_cycles":qualified["hls_worst_cycles"]
                if qualified["native_e3_qualified"] else None,
            "hls_estimated_bram_18k":qualified["estimated_resources"]["bram_18k"]
                if qualified["native_e3_qualified"] else None,
            "hls_timing_margin_ns":qualified["hls_scheduling_margin_ns"],
            "physical_qor_qualified":False,
            "warnings":qualified["warnings"],
        })
    return {
        "schema":SCHEMA,"status":"FOREGROUND_ORIGINAL_VITIS_EXECUTED",
        "source_revision":json.loads(
            (root/"native_source_manifest.json").read_text())["source_revision"],
        "attempted_original_vitis_calls":len(outcomes),
        "source_attributed_csim_qualified_cases":sum(
            r["original_csim_qualified"] for r in outcomes),
        "native_timing_resource_qualified_cases":sum(
            r["original_csim_csynth_qualified"] for r in outcomes),
        "hardware_implemented_cases":0,
        "results":outcomes,
        "note":"Native tool execution is not the same as E2/E3/physical qualification",
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
    if args.audit_native and args.execute_native:
        p.error("Audit-only and native execution are mutually exclusive")
    result=(audit_native(args.study) if args.audit_native else
            execute(args.study,execute_native=args.execute_native,
                    confirm_isolated_native=args.confirm_isolated_native,
                    start_pair=args.start_pair,max_pairs=args.max_pairs,
                    vitis_executable=args.vitis_executable,
                    timeout_seconds=args.timeout_seconds))
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
