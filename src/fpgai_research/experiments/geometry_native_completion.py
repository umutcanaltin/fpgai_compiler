"""Execute one bounded seven-axis geometry phase, reusing qualified E4 runs.

This is an opt-in native workstation driver. The existing comprehensive
campaign runner remains the ONLY source/CSim/HLS/RTL/Vivado qualification
authority. No parallel vendor sessions, no auto-retry of failed cases, no
imputed numerical effects, no overwriting previous campaign journals.

Use --phase main first, then --phase interactions. A separate fresh
journal directory is required for each phase (and each attempted run).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fpgai_research.experiments.comprehensive_native_campaign import (
    execute_comprehensive_suite,
)
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.suite_native_batch import _qualify

SCHEMA="fpgai-research/geometry-native-phased-completion/v1"
STUDY_PAIRS={
    "medium":("probe_matmul_16","probe_conv_12"),
    "large":("probe_matmul_32","probe_conv_16"),
}
WINDOWS={
    "main":((0,18),(42,18)),
    "interactions":((18,24),(60,24)),
}
PILOT_GATE_CASES=("case_0000","case_0013","case_0014")


def _inventory(root):
    manifest=root/"suite_results.json"
    if root.is_symlink() or not root.is_dir() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Expected regular, prepared geometry suite")
    document=json.loads(manifest.read_text())
    studies=document.get("studies")
    if not isinstance(studies,list) or len(studies)!=2:
        raise ValueError("Exactly two prepared operator studies required")
    names=tuple(x.get("id") for x in studies)
    if names not in STUDY_PAIRS.values():
        raise ValueError("Expected isolated same-size MatMul/Conv geometry studies")
    if any(x.get("status")!="E1-PREPARED" for x in studies):
        raise ValueError("Both studies must be source-prepared")
    for name in names:
        path=root/name/"campaign.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing prepared study architecture inventory")
        campaign=json.loads(path.read_text())
        cases=campaign.get("cases")
        if (campaign.get("schema")!="fpgai-research/paper-campaign/v1"
                or not isinstance(cases,list) or len(cases)!=42
                or any(row.get("case")!=f"case_{i:04d}"
                       or row.get("status")!="READY"
                       for i,row in enumerate(cases))):
            raise ValueError("Expected 42 frozen READY cases per operator")
    size=next(size for size,pair in STUDY_PAIRS.items() if pair==names)
    return size,names


def _qualification_counts(root,names,case_ids):
    result={}
    for name in names:
        approved=[]
        missing=[]
        for ident in case_ids:
            outcome=summarize_case(str(root/name/ident))
            if not _qualify(outcome):
                approved.append(ident)
            else:
                missing.append({
                    "case":ident,
                    "evidence":outcome.get("evidence"),
                    "blockers":_qualify(outcome),
                })
        result[name]={"qualified":len(approved),
                      "required":len(case_ids),
                      "missing":missing}
    return result


def plan_phased_geometry(directory, *, phase):
    if phase not in WINDOWS:
        raise ValueError("Unknown bounded campaign phase")
    root=Path(directory)
    size,names=_inventory(root)
    required=(PILOT_GATE_CASES if phase=="main" else
              tuple(f"case_{n:04d}" for n in range(18)))
    checks=_qualification_counts(root,names,required)
    ready=all(x["qualified"]==x["required"] for x in checks.values())
    return {
        "schema":SCHEMA,
        "size":size,
        "phase":phase,
        "directory":str(root),
        "source_windows":[{
            "study":name,"start":start,"max_cases":limit,
            "expected_prepared_cases":limit,
        } for name,(start,limit) in zip(names,WINDOWS[phase])],
        "prerequisite_native_qualification":checks,
        "ready_to_execute":ready,
        "native_tools_executed":False,
        "existing_E4_cases_will_be_reaudited_and_skipped":True,
    }


def execute_phased_geometry(directory, *, phase, journal_root, execute=False):
    plan=plan_phased_geometry(directory,phase=phase)
    if not execute:
        return plan
    if not plan["ready_to_execute"]:
        raise ValueError("Native prerequisite gate not E4-qualified; do not execute phase")
    root=Path(journal_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Use a NEW journal directory; never overwrite native evidence")
    root.parent.mkdir(parents=True,exist_ok=True)
    root.mkdir()
    stages=[]
    for window in plan["source_windows"]:
        label=window["study"]
        journal=root/(label+"_"+phase+".json")
        result=execute_comprehensive_suite(
            directory,execute=True,continue_on_error=False,
            start=window["start"],max_cases=window["max_cases"],
            journal=journal)
        passed=result.get("every_selected_case_routed_qualified") is True
        stages.append({
            "study":label,
            "native_journal":str(journal),
            "qualified":result.get("native_routed_qualified_selected"),
            "selected":result.get("selected_legal_cases"),
            "passed":passed,
            "new_native_cases":sum(
                x.get("vendor_tools_executed_for_case") is True
                for x in result.get("cases",[])),
            "reused_qualified_cases":sum(
                x.get("action")=="ALREADY_QUALIFIED"
                for x in result.get("cases",[])),
        })
        if not passed:
            break
    summary={
        **plan,
        "execution_explicitly_requested":True,
        "native_tools_executed":any(x["new_native_cases"]>0 for x in stages),
        "phase_complete":len(stages)==2 and all(x["passed"] for x in stages),
        "stages":stages,
        "on_incomplete_phase":"Inspect immutable journal and case native logs; do not proceed to interaction phase.",
    }
    with (root/"summary.json").open("x",encoding="utf-8") as f:
        json.dump(summary,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",required=True)
    parser.add_argument("--phase",choices=tuple(WINDOWS),required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--journal-root",help="New output journal directory (required for --execute)")
    args=parser.parse_args(argv)
    if args.execute and not args.journal_root:
        parser.error("--execute requires --journal-root")
    result=execute_phased_geometry(
        args.directory,phase=args.phase,
        journal_root=args.journal_root,execute=args.execute)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return int(args.execute and not result["phase_complete"])


if __name__=="__main__":
    raise SystemExit(main())
