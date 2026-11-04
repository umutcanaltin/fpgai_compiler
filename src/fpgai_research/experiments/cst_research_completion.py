"""One-command, source-attested NCHW physical-study continuation and paper gate.

Audits all 12 prespecified cases and six source-matched storage pairs.
Default is READ ONLY. Vendor tools are never invoked without two explicit
foreground opt-in flags. A bad/unknown first case blocks automatic
continuation; the exact original source-pinned files are NEVER rerun,
modified or cleaned. Data from warning-only DRC is descriptive, NOT E4.

The publication readiness check is deliberately narrower than a paper
submission gate: novelty and independently executed prior-art artifacts
are NOT established by any Vivado outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fpgai_research.experiments.cst_nchw_physical import (
    audit_routed_study, export_routed_evidence, run_routed_study,
)

SCHEMA="fpgai-research/cst-source-pinned-physical-completion/v1"
_ACCEPTED=frozenset((
    "QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4",
    "OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS",
))
_PREPARED="PREPARED_NOT_RUN"


def _snapshot(audit:dict)->dict:
    cases=audit.get("cases")
    if (audit.get("prespecified_cases")!=12 or not isinstance(cases,list)
            or len(cases)!=12 or
            any(c.get("case")!=f"case_{i:03d}"
                for i,c in enumerate(cases))):
        raise ValueError("The original physical population must be exactly 12 cases")
    if any(c.get("source_storage")!=("real_disjoint" if i%2==0 else "real_shared")
           for i,c in enumerate(cases)):
        raise ValueError("Physical source storage pair order must be preserved")
    for i in range(0,12,2):
        a,b=cases[i:i+2]
        if (a.get("graph")!=b.get("graph")
                or a.get("mapping")!=b.get("mapping")
                or a.get("requested_clock_ns")!=b.get("requested_clock_ns")):
            raise ValueError("The source-matched physical pair contract changed")
    statuses=[r["status"] for r in cases]
    finished=sum(s in _ACCEPTED for s in statuses)
    prepared=sum(s==_PREPARED for s in statuses)
    missing=sum(not r.get("original_hls_qualified",False) for r in cases)
    return {
        "schema":SCHEMA,
        "prespecified_cases":12,"prespecified_pairs":6,
        "strict_e4_cases":sum(r.get("routed_qualified") is True for r in cases),
        "warning_observation_cases":sum(
            r.get("routed_warning_observation") is True for r in cases),
        "valid_routed_observation_cases":sum(
            r.get("routed_observation_valid") is True for r in cases),
        "completed_classified_cases":finished,
        "prepared_cases":prepared,
        "blocked_original_hls_cases":missing,
        "unqualified_attempted_or_blocked_cases":12-finished-prepared,
        "original_vivado_receipts":audit.get("vivado_invocation_receipts",0),
        "cases":[{"case":c["case"],"status":c["status"],
                  "strict_e4":c.get("routed_qualified") is True,
                  "routed_observation_valid":
                      c.get("routed_observation_valid") is True}
                 for c in cases],
        "paper_novelty_established":False,
        "external_compiler_parity_established":False,
        "original_vendor_calls_this_command":0,
        "publication_ready":False,
    }


def completion_plan(study:str|Path)->dict:
    """Purely read-only, no native tools and no file writes."""
    report=_snapshot(audit_routed_study(study))
    cases=report["cases"]
    first=next((c["case"] for c in cases if c["status"]==_PREPARED),None)
    blocker=next((c for c in cases
                  if c["status"] not in _ACCEPTED|{_PREPARED}),None)
    report["next_prepared_case"]=first
    report["first_blocking_case"]=blocker
    report["physical_study_complete"]=all(c["status"] in _ACCEPTED for c in cases)
    report["allowed_to_execute_one_original_case"]=(
        first is not None and blocker is None
        and report["blocked_original_hls_cases"]==0)
    report["evidence_limitations"]=[
        "Strict E4 still requires zero DRC violations",
        "Warning-allowed observations are exploratory and pilot-informed",
        "No native FPGA bitstream, board measurement, power or achieved Fmax",
        "All six matched pairs require two original source-attested cases",
        "Prior-art artifact parity and novel Phi-certificate discrimination remain independent",
    ]
    return report


def continue_physical(*,study:str|Path,max_new_calls:int=1,
                      confirm_isolated_native:bool=False,
                      timeout_seconds:int=14400,
                      vivado_executable:str="vivado")->dict:
    """Execute one original case at a time, stop on any unqualified result.

    The run is synchronous. Its effects are ONLY the original Vivado run
    handled by run_routed_study, never repair or overwrite a source case.
    """
    if type(max_new_calls) is not int or not 1<=max_new_calls<=12:
        raise ValueError("Bounded original Vivado calls must be 1..12")
    if confirm_isolated_native is not True:
        raise ValueError("Foreground native execution needs explicit confirmation")
    plan=completion_plan(study)
    if plan["blocked_original_hls_cases"]:
        raise ValueError("Incomplete original HLS population; no selective physical run")
    if plan["first_blocking_case"] is not None:
        raise ValueError("Original failed/unsupported case must be investigated, never rerun")
    if plan["physical_study_complete"]:
        return {**plan,"execution_status":"ALL_CASES_ALREADY_CLASSIFIED"}
    ran=[]
    for index in range(12):
        if len(ran)>=max_new_calls:
            break
        report=audit_routed_study(study)
        row=report["cases"][index]
        if row["status"] in _ACCEPTED:
            continue
        if row["status"]!=_PREPARED:
            break
        native=run_routed_study(
            study,execute_native=True,confirm_isolated_native=True,
            start_case=index,max_cases=1,
            vivado_executable=vivado_executable,
            timeout_seconds=timeout_seconds)
        if native["new_vivado_calls"]!=1 or len(native["cases"])!=1:
            raise AssertionError("Unexpected original Vivado invocation count")
        observed=audit_routed_study(study)["cases"][index]
        ran.append({"case":row["case"],"status":observed["status"],
                    "routed_qualified":observed["routed_qualified"],
                    "routed_observation_valid":
                        observed["routed_observation_valid"]})
        if observed["status"] not in _ACCEPTED:
            break
    result=completion_plan(study)
    result["original_vendor_calls_this_command"]=len(ran)
    result["executed_cases"]=ran
    result["execution_status"]=(
        "STOPPED_ON_UNQUALIFIED_CASE" if ran and
        ran[-1]["status"] not in _ACCEPTED else
        "COMPLETE" if result["physical_study_complete"] else
        "BOUNDED_ORIGINAL_CALL_BUDGET_REACHED")
    return result


def export_paper(*,study:str|Path,output:str|Path)->dict:
    """Create-only original audit export; retains missing cases as NULL."""
    summary=export_routed_evidence(study=study,output=output)
    plan=completion_plan(study)
    return {"schema":SCHEMA,"physical_export":summary,
            "strict_e4_pairs":summary["routed_qualified_pairs"],
            "exploratory_routed_pairs":summary["routed_observation_valid_pairs"],
            "publication_ready":False,
            "novelty_and_prior_art_baselines":"NOT_ESTABLISHED",
            "original_vendor_calls_this_command":0,
            "input_study":str(study),
            "output":str(output),
            "prespecified_cases":plan["prespecified_cases"]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--execute-native",action="store_true")
    mode.add_argument("--export-output")
    p.add_argument("--confirm-isolated-native",action="store_true")
    p.add_argument("--max-new-calls",type=int,default=1)
    p.add_argument("--timeout-seconds",type=int,default=14400)
    p.add_argument("--vivado-executable",default="vivado")
    a=p.parse_args(argv)
    if a.execute_native:
        r=continue_physical(
            study=a.study,max_new_calls=a.max_new_calls,
            confirm_isolated_native=a.confirm_isolated_native,
            timeout_seconds=a.timeout_seconds,
            vivado_executable=a.vivado_executable)
    elif a.export_output:
        if a.confirm_isolated_native:
            p.error("Audit/export must not request native execution confirmation")
        r=export_paper(study=a.study,output=a.export_output)
    else:
        if a.confirm_isolated_native:
            p.error("Read-only plan does not accept native execution confirmation")
        r=completion_plan(a.study)
    print(json.dumps(r,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
