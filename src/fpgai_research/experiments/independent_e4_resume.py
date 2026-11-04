"""Fail-closed, resumable explicit native E4 campaign for the PRIVATE E1 workspace.

Preparation/dry run is read-only. --execute is user opt-in to call licensed
native AMD tools locally; previously completed or failed invocations NEVER
rerun automatically. The publication E3 replay is not a valid target.
"""
from __future__ import annotations

import json
from pathlib import Path

from fpgai_research.experiments.execution import run_campaign
from fpgai_research.experiments.independent_e4_audit import (
    MODELS,AXES,summarize_independent_e4,
)
from fpgai_research.experiments.isolated_hls_replay import audit_isolated_hls_replay
from fpgai_research.experiments.results import summarize_case

SCHEMA="fpgai-research/explicit-resume-matched-e4/v1"
# Answer the first architectural falsification first: same fresh RTL in E3.
ORDER=(
    ("mnist_cnn","case_0006"),
    ("mnist_cnn","case_0004"),
    ("mnist_cnn","case_0005"),
    ("mnist_mlp","case_0000"),
    ("mnist_mlp","case_0006"),
    ("mnist_mlp","case_0004"),
    ("mnist_mlp","case_0005"),
    ("mnist_cnn","case_0000"),
)


def _ready_indices(root:Path)->dict:
    indexes={}
    for model in MODELS:
        campaign=root/model/"campaign.json"
        if not campaign.is_file() or campaign.is_symlink():
            raise ValueError("Missing/linked original independent E1 campaign")
        data=json.loads(campaign.read_text(encoding="utf-8"))
        if data.get("schema")!="fpgai-research/paper-campaign/v1":
            raise ValueError("Unexpected independent model campaign")
        ready=[x for x in data.get("cases",[]) if x.get("status")=="READY"]
        if len(ready)!=4 or {x.get("case") for x in ready}!=set(AXES):
            raise ValueError("Expected exactly the four pinned READY cases")
        for i,item in enumerate(ready):
            if item.get("dimension")!=AXES[item["case"]]:
                raise ValueError("READY axis differs from pinned E3 replay")
            case_dir=root/model/item["case"]
            staged_tcl=case_dir/"vivado"/"run_vivado.tcl"
            if (staged_tcl.is_symlink() or not staged_tcl.is_file()
                    or case_dir.is_symlink()):
                raise ValueError("Not a separately staged E4 workspace: missing canonical OOC Tcl")
            indexes[(model,item["case"])]=(i,item.get("cpp_sha256"))
    return indexes


def _case_stage(root:Path,model:str,case:str,summary:dict)->tuple[str,str]:
    folder=root/model/case
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Missing/linked private case")
    vivado_receipt=folder/"vivado"/"vivado_invocation.json"
    vitis_receipt=folder/"vitis_invocation.json"
    if vivado_receipt.is_symlink() or vitis_receipt.is_symlink():
        return "BLOCKED_MANUAL_REVIEW","Linked native invocation receipt"
    if summary["qualified_e4"]:
        return "QUALIFIED_SKIP","Already source-attested and physically qualified"
    if vivado_receipt.exists():
        return "BLOCKED_MANUAL_REVIEW","Prior Vivado invocation requires manual review; never overwrite"
    if summary["qualified_e3"]:
        # An unfinished physical implementation must never be overwritten.
        vivado=folder/"vivado"
        if vivado.is_symlink() or not (vivado/"run_vivado.tcl").is_file():
            return "BLOCKED_MANUAL_REVIEW","Missing or linked canonical Vivado Tcl"
        if any(p.name!="run_vivado.tcl" for p in vivado.iterdir()):
            return "BLOCKED_MANUAL_REVIEW","Vivado output already exists; preserve for diagnosis"
        return "VIVADO","Attested E3/CSim ready for first routed OOC implementation"
    if vitis_receipt.exists():
        return "BLOCKED_MANUAL_REVIEW","Prior Vitis invocation lacks qualified E3; never overwrite"
    if summary["evidence"]!="E1-HLS-source":
        return "BLOCKED_MANUAL_REVIEW","Unqualified numerical/native intermediate evidence"
    if ((folder/"fpgai_hls_project").exists()
            or (folder/"logs").exists()
            or (folder/"vitis_invocation.json").is_symlink()):
        return "BLOCKED_MANUAL_REVIEW","Previous HLS output without receipt; preserve for diagnosis"
    return "VITIS","Fresh E1 source requires fixed-point CSim and native HLS synthesis"


def plan_resume_matched_e4(directory)->dict:
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Independent native E1 workspace is required")
    # Explicit E1-only origin marker in the frozen private replay receipt.
    receipt=root/"replay_provenance.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("Missing independent E1 workspace source receipt")
    identity=json.loads(receipt.read_text(encoding="utf-8"))
    if (identity.get("schema")!="fpgai-research/isolated-native-hls-replay/v1"
            or identity.get("private_inputs") is not True):
        raise ValueError("Not an isolated private experiment replay")
    audit=audit_isolated_hls_replay(root)
    indices=_ready_indices(root)
    summaries={}
    for r in audit.get("cases",[]):
        key=(r.get("model"),r.get("case"))
        if key in summaries or key not in indices:
            raise ValueError("Unexpected duplicated source audit case")
        summary=summarize_case(str(root/key[0]/key[1]))
        if (summary["source_sha256"]!=indices[key][1] or
                summary["source_sha256"]!=r.get("source_sha256")):
            raise ValueError("Original E1 source or manifest changed")
        summaries[key]=summary
    verified=summarize_independent_e4(root,audit,summaries)
    public={(x["model"],x["case"]):x for x in verified["cases"]}
    plans=[]
    for model,case in ORDER:
        state,reason=_case_stage(root,model,case,public[(model,case)])
        plans.append({"model":model,"case":case,"axis":AXES[case],
                      "ready_index":indices[(model,case)][0],
                      "next_action":state,"reason":reason,
                      "e3_qualified":public[(model,case)]["qualified_e3"],
                      "e4_qualified":public[(model,case)]["qualified_e4"]})
    return {
        "schema":SCHEMA,"cases":plans,"case_count":8,
        "e3_qualified_case_count":verified["e3_qualified_case_count"],
        "e4_qualified_case_count":verified["e4_qualified_case_count"],
        "blocked_case_count":sum(x["next_action"]=="BLOCKED_MANUAL_REVIEW" for x in plans),
        "pending_vitis_case_count":sum(x["next_action"]=="VITIS" for x in plans),
        "pending_vivado_case_count":sum(x["next_action"]=="VIVADO" for x in plans),
        "native_tools_executed":False,"original_publication_e3_modified":False,
        "private_paths_or_hashes_exported":False,
        "scope_note":(
            "Never rerun native invocations with existing receipts. A completed "
            "tool does not alone qualify E2/E3/E4; stop on rejected evidence. "
            "All requested work is in a distinct private source-identical "
            "E1 campaign, not the publication E3 replay."
        ),
    }


def resume_matched_e4(directory,*,execute=False,max_cases=8)->dict:
    """Run at most max_cases pending native cases, each E2/E3 before E4.

    No native vendor tools run unless --execute is explicitly passed. Even
    then, all prior attestations are reaudited before EACH action.
    """
    if type(execute) is not bool or type(max_cases) is not int or not 1<=max_cases<=8:
        raise ValueError("Explicit opt-in and max_cases between 1 and 8 required")
    root=Path(directory)
    plan=plan_resume_matched_e4(root)
    if not execute:
        return {**plan,"executed":False,"executions":[]}
    completed=[]
    stop_reason=None
    visited=set()
    for row in plan["cases"]:
        key=(row["model"],row["case"])
        if row["next_action"]=="QUALIFIED_SKIP":
            continue
        if row["next_action"]=="BLOCKED_MANUAL_REVIEW":
            stop_reason="Manual native provenance review required: "+row["model"]+"/"+row["case"]
            break
        if len(visited)>=max_cases:
            break
        visited.add(key)
        # Replan after each irreversible execution; never use stale audit.
        for _ in range(2):
            current=plan_resume_matched_e4(root)
            active=next(c for c in current["cases"] if
                        (c["model"],c["case"])==key)
            stage=active["next_action"]
            if stage=="QUALIFIED_SKIP":
                break
            if stage not in ("VITIS","VIVADO"):
                stop_reason="Native source evidence blocked: "+row["model"]+"/"+row["case"]
                break
            receipt=run_campaign(
                root/row["model"],stages=(stage.lower(),),execute=True,
                first=active["ready_index"],limit=1,
                continue_on_error=False,require_csim_pass=stage=="VIVADO")
            statuses=[state["status"] for outcome in receipt["cases"]
                      for state in outcome["stages"].values()]
            completed.append({"model":row["model"],"case":row["case"],
                              "stage":stage,"statuses":statuses})
            if statuses!=["COMPLETED"]:
                stop_reason="Native tool failed or blocked; preserve invocation receipt"
                break
            # A successful process exit is not sufficient. The next
            # preflight either promotes to E3/Vivado/E4 or blocks.
            after=plan_resume_matched_e4(root)
            promoted=next(c for c in after["cases"] if
                          (c["model"],c["case"])==key)
            if stage=="VITIS" and promoted["next_action"]!="VIVADO":
                stop_reason="Completed Vitis lacks attested E3/CSim; stop"
                break
            if stage=="VIVADO":
                if promoted["next_action"]!="QUALIFIED_SKIP":
                    stop_reason="Vivado report failed physical E4 gate; stop"
                break
        if stop_reason:
            break
    final=plan_resume_matched_e4(root)
    return {**final,"executed":True,"native_tools_executed":bool(completed),
            "executions":completed,"stop_reason":stop_reason}
