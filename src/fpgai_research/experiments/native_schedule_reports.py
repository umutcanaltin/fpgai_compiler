"""Read-only inventory of independently source-attested native Vitis HLS text reports.

An actual per-loop schedule claim requires an inspected, source-bound
vendor report and a validated parser. This module makes no per-loop
latency, bottleneck, memory-port or achieved-II assertion.

Current source-attestation can only be earned by the explicit native
runner fingerprinting *fresh* csynth_rpt during a Vitis execution.
Older, present but unhashed reports remain unverified and cannot be
retrospectively promoted to native schedule evidence.
"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.experiments.results import _attested, _native_tool_identity

SCHEMA="fpgai-research/native-hls-schedule-report-inventory/v1"
REPORT_REL=Path("fpgai_hls_project/solution1/syn/report/fpgai_model_csynth.rpt")
REPORT_LIMIT=12*1024*1024
COLUMNS=("model","case","axis","case_evidence","status",
         "report_sha256","report_size_bytes","source_sha256",
         "report_still_needs_loop_parser")


def inspect_native_schedule_case(case_dir:Path, source_sha256:str,
                                 evidence:str)->dict:
    case=Path(case_dir)
    if case.is_symlink() or not case.is_dir():
        raise ValueError("A normal prepared case directory is required")
    if (not isinstance(source_sha256,str) or len(source_sha256)!=64
            or any(ch not in "0123456789abcdef" for ch in source_sha256)):
        raise ValueError("Expected a saved verified lower-case HLS SHA-256")
    if evidence not in ("E1-HLS-source","E2-Vitis-C-simulation",
                        "E3-HLS-synthesis-report","E4-Vivado-implementation"):
        raise ValueError("Unsupported source evidence level")
    source=case/"model.cpp"
    if source.is_symlink() or not source.is_file():
        raise ValueError("Prepared HLS source is missing or linked")
    if sha256(source.read_bytes()).hexdigest()!=source_sha256:
        raise ValueError("Report case does not match saved IR HLS source hash")
    rpt=case/REPORT_REL
    base={
        "case_evidence":evidence,"source_sha256":source_sha256,
        "report_path":str(REPORT_REL),"report_sha256":None,
        "report_size_bytes":None,"parsed_loop_ii":None,
        "per_loop_schedule_attested":False,
        "native_tools_executed":False,
        "report_still_needs_loop_parser":True,
    }
    # Path components must not resolve via symlinks outside the case
    # even if the report does not exist yet.
    cur=case
    for part in REPORT_REL.parts:
        cur=cur/part
        if cur.is_symlink():
            return {**base,"status":"UNSAFE_SYMLINKED_REPORT"}
    if not rpt.is_file():
        return {**base,"status":"REPORT_NOT_PRESENT"}
    size=rpt.stat().st_size
    if size<=0 or size>REPORT_LIMIT:
        return {**base,"status":"REPORT_EMPTY_OR_TOO_LARGE","report_size_bytes":size}
    digest=sha256(rpt.read_bytes()).hexdigest()
    base.update(report_sha256=digest,report_size_bytes=size)
    # This is not a second discovery mechanism: use exactly the attestation
    # requirements that already gate E2/E3 in results.summarize_case.
    inv=case/"vitis_invocation.json"
    if inv.is_symlink() or not inv.is_file():
        return {**base,"status":"PRESENT_NOT_ATTESTED"}
    tcl=case/"run_hls.tcl"
    tb=case/"tb.cpp"
    if tcl.is_symlink() or tb.is_symlink():
        return {**base,"status":"PRESENT_NOT_ATTESTED"}
    receipt=_attested(inv,"vitis_hls",tcl,source,testbench=tb)
    if receipt is None or not _native_tool_identity(receipt,"vitis_hls"):
        return {**base,"status":"PRESENT_NOT_ATTESTED"}
    matched=(receipt.get("fresh_report_sha256") or {}).get("csynth_rpt")
    if matched!=digest:
        return {**base,"status":(
            "HASH_MISMATCH" if matched is not None else
            "PRESENT_NOT_FRESHLY_ATTESTED")}
    if evidence not in ("E3-HLS-synthesis-report",
                        "E4-Vivado-implementation"):
        # Source-attributed vendor text cannot promote an invalid
        # synthesis summary to E3 or to loop-level results.
        return {**base,"status":"NATIVE_TEXT_WITHOUT_E3_QUALIFICATION"}
    return {**base,"status":"ATTESTED_RAW_REPORT_AVAILABLE",
            "per_loop_schedule_attested":False}


def analyze_native_schedule_inventory(root:Path,study:dict)->dict:
    root=Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("A regular research suite root is required")
    if study.get("schema")!="fpgai-research/scientific-ir-mechanism-study/v1":
        raise ValueError("Expected verified IR science report")
    rows=study.get("cases")
    if not isinstance(rows,list) or len(rows)!=study.get("case_count"):
        raise ValueError("Source study case inventory is inconsistent")
    identities=set()
    entries=[]
    for record in rows:
        model,case=record.get("model"),record.get("case")
        if (not isinstance(model,str) or not isinstance(case,str)
                or not model or not case
                or not model.replace("_","").isalnum()
                or not case.replace("_","").isalnum()):
            raise ValueError("Invalid model or case path")
        key=(model,case)
        if key in identities:
            raise ValueError("Duplicate model/case")
        identities.add(key)
        model_path=root/model
        if model_path.is_symlink() or not model_path.is_dir():
            raise ValueError("A normal research model directory is required")
        path=model_path/case
        item=inspect_native_schedule_case(
            path,record["source_sha256"],record["evidence"])
        entries.append({
            "model":model,"case":case,"axis":record.get("axis"),
            **item})
    tally={}
    for item in entries:
        key=item["status"]
        tally[key]=tally.get(key,0)+1
    return {
        "schema":SCHEMA,
        "source_suite_sha256":study.get("source_suite_sha256"),
        "case_count":len(entries),
        "attested_raw_reports":tally.get("ATTESTED_RAW_REPORT_AVAILABLE",0),
        "per_loop_schedule_evidence_available":False,
        "native_tools_executed":False,
        "weight_values_exported":False,
        "status_counts":tally,"cases":entries,
        "limitations":(
            "A source-bound, fresh Vitis HLS human-readable synthesis report "
            "is raw evidence only. This implementation does not parse or "
            "attribute per-loop II/latency/scheduling, memory dependencies "
            "or bottlenecks. A present but unauthenticated report is never "
            "native schedule evidence; only future explicit runs can add "
            "missing fresh-report hashes. Existing paper E3/E4 measurements "
            "and timing acceptance are unchanged."
        )
    }


def export_native_schedule_inventory(root:Path,study:dict,stage:Path)->dict:
    result=analyze_native_schedule_inventory(root,study)
    dest=Path(stage)
    (dest/"native_schedule_inventory.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with (dest/"native_schedule_inventory.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS)
        writer.writeheader()
        for item in result["cases"]:
            writer.writerow({
                k:"TBD" if item.get(k) is None else item[k] for k in COLUMNS
            })
    lines=[
        "# Source attribution of Vitis human-readable synthesis reports","",
        "**This is a report availability and provenance inventory, not a "
        "loop-II analysis.** The existing HLS XML offers a top-level "
        "latency/interval summary, not a validated loop schedule parser.","",
        "| Model | Case | Axis | Source-attestation status | Report SHA-256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in result["cases"]:
        lines.append("| "+" | ".join(
            str(item[k]) if item.get(k) is not None else "TBD"
            for k in ("model","case","axis","status","report_sha256")
        )+" |")
    lines.extend(["","## Evidence required next","",
        "Native Vitis must generate a fresh human-readable "
        "`fpgai_model_csynth.rpt` and the source-attested invocation "
        "receipt must include its digest under `fresh_report_sha256.csynth_rpt`. "
        "A report that already exists but is not in a valid receipt "
        "cannot be retrospectively attributed.","",
        "Even a matching raw report requires a separately tested, "
        "version-aware loop-level parser and unambiguous binding to "
        "the selected Architectural IR operator before a causal "
        "scheduling explanation can be published.","",
        result["limitations"],""])
    (dest/"native_schedule_inventory.md").write_text("\n".join(lines))
    return result
