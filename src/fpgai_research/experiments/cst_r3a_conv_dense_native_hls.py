"""R3a: opt-in original Vitis CSim+CSynth for R2z full CNN→Dense networks.

Reuses canonical R2z six source packages (3 mixed networks × scalar vs
per-operator independent Phi), source-pinned CSim/Tcl and the same
vendor-receipt and effective-clock qualification as older R2u.

Dry-run and read-only audit launch ZERO native calls. Native execution
requires BOTH --execute-native and --confirm-isolated-native; attempted
original source cases can NEVER be overwritten. No physical FPGA QoR.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from fpgai_research.experiments.cst_r2z_conv_dense_network_campaign import (
    SOURCE_CASES, HOST_INPUTS_PER_CASE, CLOCK_NS, UNCERTAINTY_NS,
    SCHEMA as R2Z_SCHEMA, audit as audit_source,
)
from fpgai_research.experiments.cst_nchw_hls_qualification import qualify_case
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

SCHEMA="fpgai-research/R3a-original-CNN-Dense-E2-E3-native-HLS/v1"
PAIR_SCHEMA="fpgai-research/R3a-CNN-Dense-controlled-native-HLS-factor-pairs/v1"
METRICS=("bram_18k","lut","ff","dsp","uram")
PAIRS=((0,1),(2,3),(4,5))


def _source_cases(root:Path)->tuple[dict,list[dict]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R3a original R2z source folder unavailable or symlinked")
    source=audit_source(study=root)
    if (source.get("original_source_cases")!=SOURCE_CASES
            or source.get("original_host_GCC_qualified_input_receipts")!=
                    SOURCE_CASES*HOST_INPUTS_PER_CASE
            or source.get("original_source_packages_SHA_attested")!=SOURCE_CASES):
        raise ValueError("R3a source-prepared CNN complete 6/12 denominator not attested")
    path=root/"cases.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("R3a original source case manifest unavailable")
    cases=json.loads(path.read_text())
    if len(cases)!=SOURCE_CASES:
        raise ValueError("R3a expected six immutable CNN/Dense cases")
    if [r.get("case") for r in cases]!=[
            f"case_{i:03d}" for i in range(SOURCE_CASES)]:
        raise ValueError("R3a original source case order changed")
    for a,b in PAIRS:
        left,right=cases[a],cases[b]
        if (left.get("network")!=right.get("network")
                or left.get("independent_phi") is not False
                or right.get("independent_phi") is not True
                or left.get("functional_sha256")!=right.get("functional_sha256")
                or left.get("original_input_batch_sha256")!=
                        right.get("original_input_batch_sha256")
                or left.get("prepared_vitis_testbench_sha256")!=
                        right.get("prepared_vitis_testbench_sha256")
                or left.get("prepared_vitis_tcl_sha256")!=
                        right.get("prepared_vitis_tcl_sha256")
                or left.get("source_cpp_sha256")==right.get("source_cpp_sha256")):
            raise ValueError("R3a factorial scalar/independent Phi controls do not match")
    return source,cases


def audit(*,study:str|Path)->dict:
    """Read only all six original tool-source and report SHA receipts."""
    root=Path(study)
    verified,cases=_source_cases(root)
    results=[]
    for row in cases:
        case=root/row["case"]
        # This is the existing fail-closed native E2/E3 auditor:
        # exact original source/Tcl/TB receipt, fresh CSim and CSynth,
        # numeric test count/tolerance, requested-vs-XML effective
        # clock budget and all mandatory HLS used resource fields.
        expected={
            "case":row["case"],
            "source_cpp_sha256":row["source_cpp_sha256"],
            "source_storage":"full-materialized-tensors",
            "graph":row["network"],
            "mapping":("independent-conv-and-dense-Phi"
                       if row["independent_phi"] else "scalar"),
        }
        original=summarize_case(str(case))
        qualified=qualify_case(case,expected,summary=original)
        receipt=case/"vitis_invocation.json"
        if receipt.is_symlink():
            raise ValueError("R3a original native receipt may not be symlinked")
        receipt_present=receipt.is_file()
        results.append({
            "case":row["case"],"network":row["network"],
            "independent_phi":row["independent_phi"],
            "functional_sha256":row["functional_sha256"],
            "source_cpp_sha256":row["source_cpp_sha256"],
            "original_vitis_invocation_present":receipt_present,
            "native_CSim_E2_accepted":qualified["native_e2_accepted"],
            "native_CSynth_E3_qualified":qualified["native_e3_qualified"],
            "qualification_status":qualified["status"],
            "source_attested_evidence":qualified["evidence_level"],
            "hls_worst_cycles":(
                qualified["hls_worst_cycles"]
                if qualified["native_e3_qualified"] else None),
            "hls_scheduling_margin_ns":(
                qualified["hls_scheduling_margin_ns"]
                if qualified["native_e3_qualified"] else None),
            "hls_estimated_clock_ns":(
                qualified["hls_estimated_clock_ns"]
                if qualified["native_e3_qualified"] else None),
            "HLS_estimated_resources":({
                key:qualified["estimated_resources"][key] for key in METRICS}
                if qualified["native_e3_qualified"] else
                {key:None for key in METRICS}),
            "original_HLS_report_sha256":(
                qualified["hls_report_sha256"]
                if qualified["native_e3_qualified"] else None),
            # Diagnostic tier only: original source/CSynth-attested timing may
            # be observed even when the *effective* clock budget rejects E3.
            # Never expose rejected resource/cycle figures as qualified QoR.
            "attested_HLS_timing_diagnostic": ({
                "estimated_clock_ns":qualified.get("hls_estimated_clock_ns"),
                "effective_budget_ns":qualified.get("hls_effective_scheduling_budget_ns"),
                "margin_ns":qualified.get("hls_scheduling_margin_ns"),
                "feasible":qualified.get("hls_timing_feasible"),
                "original_HLS_report_sha256":qualified.get("hls_report_sha256"),
            } if (receipt_present
                  and qualified.get("hls_timing_feasible") is not None
                  and qualified.get("hls_report_sha256")
                  and qualified.get("evidence_level") == "E3-HLS-synthesis-report")
              else None),
            "native_warnings":qualified["warnings"],
            "physical_BRAM":None,"routed_Fmax":None,
        })
    return {
        "schema":SCHEMA,
        "source_revision":verified["source_revision"],
        "original_source_cases":SOURCE_CASES,
        "original_GCC_source_inputs_qualified":
            verified["original_host_GCC_qualified_input_receipts"],
        "original_vitis_invocation_receipts":sum(
            row["original_vitis_invocation_present"] for row in results),
        "native_CSim_E2_qualified_cases":sum(
            row["native_CSim_E2_accepted"] for row in results),
        "native_CSynth_E3_qualified_cases":sum(
            row["native_CSynth_E3_qualified"] for row in results),
        "cases":results,
        "new_original_Vitis_calls_during_audit":0,
        "physical_BRAM":None,"routed_Fmax":None,
        "limits":[
            "HLS CSynth resources are ESTIMATES, not physical Vivado BRAM",
            "One deterministic small CNN per family, two source Phi arms",
            "R2z source input and all tested stage elements passed GCC, NOT bit-exact CSim",
            "Strict original CSim numeric and effective-clock gates are reused unchanged",
            "Failed and unattempted E2/E3 evidence remains UNKNOWN, never imputed as ties",
        ],
    }


def contrasts(*,study:str|Path)->dict:
    """Three prespecified same-network, source-controlled Phi contrasts."""
    reviewed=audit(study=study)
    rows=reviewed["cases"]
    matched=[]
    for a,b in PAIRS:
        left,right=rows[a],rows[b]
        if (left["network"]!=right["network"]
                or left["functional_sha256"]!=right["functional_sha256"]
                or left["independent_phi"] is not False
                or right["independent_phi"] is not True):
            raise ValueError("R3a native output pairing broken")
        both=bool(left["native_CSynth_E3_qualified"]
                  and right["native_CSynth_E3_qualified"])
        diff=lambda key:(right["HLS_estimated_resources"][key]-
                         left["HLS_estimated_resources"][key]) if both else None
        matched.append({
            "network":left["network"],"control":left["case"],
            "independent_Phi":right["case"],"both_E2":bool(
                left["native_CSim_E2_accepted"]
                and right["native_CSim_E2_accepted"]),
            "both_E3":both,
            "HLS_estimated_BRAM18k_delta":diff("bram_18k"),
            "HLS_estimated_LUT_delta":diff("lut"),
            "HLS_estimated_FF_delta":diff("ff"),
            "HLS_estimated_DSP_delta":diff("dsp"),
            "HLS_worst_latency_cycles_delta":(
                right["hls_worst_cycles"]-left["hls_worst_cycles"]
                if both else None),
            "physical_BRAM_delta":None,
            # A legitimately attempted/rejected arm is not an unknown
            # unattempted arm; keep both categories and numeric deltas null.
            "status":("MATCHED_HLS_E3_ESTIMATES_ONLY" if both
                      else ("REJECTED_ORIGINAL_E3_ARM" if any(
                          x["original_vitis_invocation_present"]
                          and not x["native_CSynth_E3_qualified"]
                          for x in (left,right))
                            else "UNKNOWN_NOT_BOTH_ORIGINAL_E3")),
        })
    return {
        "schema":PAIR_SCHEMA,
        "source_revision":reviewed["source_revision"],
        "prespecified_original_neural_pairs":len(PAIRS),
        "E3_both_qualified_pairs":sum(x["both_E3"] for x in matched),
        "E3_unknown_or_rejected_pairs":sum(
            not x["both_E3"] for x in matched),
        "pairs":matched,"cases":rows,
        "physical_QoR":None,
        "new_original_Vitis_calls_during_analysis":0,
        "limits":reviewed["limits"],
    }


def execute(*,study:str|Path,start_case:int=0,max_cases:int=1,
            execute_native:bool=False,confirm_isolated_native:bool=False,
            vitis_executable:str="vitis_hls",
            timeout_seconds:int=7200)->dict:
    root=Path(study)
    _verified,cases=_source_cases(root)
    if (type(start_case) is not int or type(max_cases) is not int
            or start_case < 0 or not 1 <= max_cases <= SOURCE_CASES
            or start_case+max_cases>SOURCE_CASES):
        raise ValueError("R3a original native case selection must be within 0..5")
    selected=cases[start_case:start_case+max_cases]
    # Audit the original source AND selected package state before any native
    # run to catch a partial/stale original native attempt.
    _=audit(study=root)
    if not execute_native:
        return {
            "schema":SCHEMA,"status":"DRY_RUN_NO_VITIS_CALLS",
            "selected_original_cases":[row["case"] for row in selected],
            "original_vitis_calls":0,
        }
    if confirm_isolated_native is not True:
        raise ValueError("R3a source-pinned original native execution needs explicit confirmation")
    if (type(timeout_seconds) is not int or
            not 1 <= timeout_seconds <= 86400):
        raise ValueError("R3a bounded original native time budget required")
    executable=shutil.which(vitis_executable)
    if executable is None:
        raise FileNotFoundError("R3a actual original Vitis HLS executable unavailable")
    for row in selected:
        case=root/row["case"]
        for name in ("vitis_invocation.json","logs","fpgai_hls_project"):
            artifact=case/name
            if artifact.exists() or artifact.is_symlink():
                raise FileExistsError(
                    "R3a original native case already attempted: "+str(artifact))
    output=[]
    for row in selected:
        case=root/row["case"]
        receipt=run_vitis(
            case,executable=executable,timeout_seconds=timeout_seconds)
        # DO NOT report success or zero resources merely from exit code.
        measured={r["case"]:r for r in audit(study=root)["cases"]}[row["case"]]
        output.append({
            "case":row["case"],
            "vendor_process_completed":receipt.get("completed_successfully"),
            "original_CSim_E2_accepted":measured["native_CSim_E2_accepted"],
            "original_CSynth_E3_qualified":measured["native_CSynth_E3_qualified"],
            "status":measured["qualification_status"],
            "original_HLS_estimated_BRAM18k":(
                measured["HLS_estimated_resources"]["bram_18k"]
                if measured["native_CSynth_E3_qualified"] else None),
        })
    return {
        "schema":SCHEMA,
        "status":"ORIGINAL_NATIVE_ATTEMPTED_UNQUALIFIED_CASES_RETAINED",
        "new_original_Vitis_calls":len(output),
        "cases":output,"physical_QoR":None,
    }


def main(argv:list[str]|None=None)->None:
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit",action="store_true")
    mode.add_argument("--contrasts",action="store_true")
    mode.add_argument("--execute",action="store_true")
    p.add_argument("--study",required=True)
    p.add_argument("--start-case",type=int,default=0)
    p.add_argument("--max-cases",type=int,default=1)
    p.add_argument("--execute-native",action="store_true")
    p.add_argument("--confirm-isolated-native",action="store_true")
    p.add_argument("--vitis-executable",default="vitis_hls")
    p.add_argument("--timeout-seconds",type=int,default=7200)
    args=p.parse_args(argv)
    if args.audit:
        result=audit(study=args.study)
    elif args.contrasts:
        result=contrasts(study=args.study)
    else:
        result=execute(
            study=args.study,start_case=args.start_case,max_cases=args.max_cases,
            execute_native=args.execute_native,
            confirm_isolated_native=args.confirm_isolated_native,
            vitis_executable=args.vitis_executable,
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
