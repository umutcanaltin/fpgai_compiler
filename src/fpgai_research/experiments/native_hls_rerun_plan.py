"""Plan bounded, explicitly opt-in Vitis replays from pinned research evidence.

No native invocation, no mutation of the prepared suite and no confidence
upgrade. The planner derives ready-case indices from each campaign.json;
case_0004 is not necessarily READY index 4 if earlier cases are blocked.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
from tempfile import TemporaryDirectory

from fpgai_research.experiments.loop_attribution_gaps import SCHEMA as TRIAGE_SCHEMA
from fpgai_research.experiments.native_schedule_reports import SCHEMA as INVENTORY_SCHEMA

SCHEMA = "fpgai-research/bounded-native-hls-rerun-plan/v1"
FOCUS_AXES = ("baseline", "parallelism", "pes", "simd")
_ALLOWED_MODELS = frozenset(("mnist_mlp", "mnist_cnn"))
_ID = re.compile(r"[a-z][a-z0-9_]{0,40}\Z")
_CASE = re.compile(r"case_[0-9]{4,8}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*1024*1024:
        raise ValueError("Expected regular, bounded pinned JSON")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _identity(model: str, case: str) -> tuple[str,str]:
    if (not isinstance(model,str) or not _ID.fullmatch(model)
            or not isinstance(case,str) or not _CASE.fullmatch(case)):
        raise ValueError("Unsafe model or case identity")
    return model,case


def _windows(indices: list[int]) -> list[tuple[int,int]]:
    """Emit only contiguous READY slices; never run unrequested cases."""
    if not indices:
        return []
    windows=[]
    first=last=indices[0]
    for index in indices[1:]:
        if index==last+1:
            last=index
        else:
            windows.append((first,last-first+1))
            first=last=index
    windows.append((first,last-first+1))
    return windows


def plan_hls_reruns(suite_directory, inventory: dict, triage: dict) -> dict:
    root=Path(suite_directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Prepared suite must be a regular directory")
    suite_file=root/"suite_results.json"
    suite=_read_json(suite_file)
    suite_digest=sha256(suite_file.read_bytes()).hexdigest()
    if (suite.get("schema")!="fpgai-research/full-paper-suite/v1"
            or inventory.get("schema")!=INVENTORY_SCHEMA
            or triage.get("schema")!=TRIAGE_SCHEMA
            or inventory.get("source_suite_sha256")!=suite_digest
            or triage.get("source_suite_sha256")!=suite_digest):
        raise ValueError("Rerun inputs do not match the pinned prepared suite")
    if (triage.get("operator_schedule_attribution_verified") is not False
            or triage.get("native_tools_executed") is not False
            or triage.get("existing_evidence_upgraded") is not False):
        raise ValueError("Input triage must be diagnostic and non-attesting")
    studies=suite.get("studies")
    if not isinstance(studies,list) or not studies:
        raise ValueError("No prepared study inventory")
    native=inventory.get("cases")
    comparisons=triage.get("comparisons")
    if (not isinstance(native,list) or len(native)!=inventory.get("case_count")
            or not isinstance(comparisons,list)
            or len(comparisons)!=triage.get("verified_question_count")):
        raise ValueError("Inconsistent report or comparison inventory")
    by_case={}
    for record in native:
        key=_identity(record.get("model"),record.get("case"))
        if key in by_case:
            raise ValueError("Duplicate native report case")
        by_case[key]=record
    required={}
    unique_comparisons=set()
    for row in comparisons:
        key=_identity(row.get("model"),row.get("case"))
        if key in unique_comparisons or key not in by_case:
            raise ValueError("Duplicate or foreign triage comparison")
        unique_comparisons.add(key)
        if row.get("axis")!=by_case[key].get("axis"):
            raise ValueError("Triage axis conflicts with pinned native report")
        if row.get("axis") in FOCUS_AXES[1:]:
            required.setdefault(key[0],set()).add(key[1])
    if not required:
        raise ValueError("No priority architectural contrasts in pinned triage")
    plans=[]
    seen_models=set()
    total_selected=0
    total_commands=0
    for study in studies:
        model=study.get("id")
        if not isinstance(model,str) or not _ID.fullmatch(model) or model in seen_models:
            raise ValueError("Duplicate or unsafe prepared model")
        seen_models.add(model)
        if model not in required:
            continue
        if model not in _ALLOWED_MODELS or study.get("status")!="E1-PREPARED":
            raise ValueError("Priority model is not a prepared MLP/CNN study")
        directory=root/model
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Unsafe model directory")
        campaign_file=directory/"campaign.json"
        campaign=_read_json(campaign_file)
        if campaign.get("schema")!="fpgai-research/paper-campaign/v1":
            raise ValueError("Invalid prepared campaign schema")
        declared=campaign.get("cases")
        if not isinstance(declared,list) or not declared:
            raise ValueError("Empty campaign case list")
        ready=[]
        declared_cases=set()
        for entry in declared:
            if not isinstance(entry,dict):
                raise ValueError("Malformed campaign entry")
            case=entry.get("case")
            key=_identity(model,case)
            if case in declared_cases:
                raise ValueError("Duplicate prepared case")
            declared_cases.add(case)
            state=entry.get("status")
            if state=="READY":
                ready.append(entry)
            elif state!="BLOCKED":
                raise ValueError("Unknown case state")
        requested={"case_0000",*required[model]}
        if not requested.issubset(declared_cases):
            raise ValueError("Focused baseline/variant case missing from campaign")
        ready_index={entry["case"]:i for i,entry in enumerate(ready)}
        missing_ready=sorted(requested-set(ready_index))
        if missing_ready:
            raise ValueError("Focused case blocked: "+", ".join(missing_ready))
        selected=[]
        for case in sorted(requested,key=lambda name:ready_index[name]):
            key=(model,case)
            if key not in by_case:
                raise ValueError("No native inventory for focused case")
            row=by_case[key]
            campaign_entry=ready[ready_index[case]]
            if row.get("axis")!=campaign_entry.get("dimension"):
                raise ValueError("Campaign axis and scientific inventory differ")
            expected=campaign_entry.get("cpp_sha256")
            source_hash=row.get("source_sha256")
            if (not isinstance(expected,str) or not _SHA.fullmatch(expected)
                    or source_hash!=expected):
                raise ValueError("Campaign HLS SHA does not match report inventory")
            folder=directory/case
            cpp=folder/"model.cpp"
            if (folder.is_symlink() or not folder.is_dir() or
                    cpp.is_symlink() or not cpp.is_file() or
                    sha256(cpp.read_bytes()).hexdigest()!=expected):
                raise ValueError("Prepared source hash mismatch or unsafe case")
            selected.append({
                "case":case, "axis":row["axis"],
                "ready_index":ready_index[case],
                "report_status_before_rerun":row["status"],
                "source_sha256":expected,
            })
        chosen=[row["ready_index"] for row in selected]
        steps=[]
        for first,limit in _windows(chosen):
            cases=[row["case"] for row in selected
                   if first<=row["ready_index"]<first+limit]
            command=" ".join(shlex.quote(part) for part in [
                "fpgai-research-tools","run-campaign",
                "--directory",str(directory),"--stage","vitis",
                "--first",str(first),"--limit",str(limit)])
            steps.append({
                "first_ready_index":first,"limit":limit,
                "selected_cases":cases,"dry_run_command":command,
                "explicit_execution_addition":"--execute",
            })
        total_selected+=len(selected)
        total_commands+=len(steps)
        plans.append({
            "model":model,
            "campaign_sha256":sha256(campaign_file.read_bytes()).hexdigest(),
            "selected_cases":selected,"windows":steps,
        })
    if not set(required).issubset(seen_models):
        raise ValueError("Triage priority model not declared in suite")
    return {
        "schema":SCHEMA,"source_suite_sha256":suite_digest,
        "comparison_gap_count":len(comparisons),
        "distinct_priority_case_count":total_selected,
        "priority_model_count":len(plans),
        "dry_run_command_count":total_commands,
        "focus_axes":list(FOCUS_AXES),
        "models":plans,
        "tools_executed":False,"execution_authorized":False,
        "native_schedule_attested":False,
        "caveat":(
            "20 scientific comparison gaps are not 20 separate vendor invocations. "
            "Commands shown are safe defaults WITHOUT --execute. Opt-in --execute "
            "mutates generated native outputs in the prepared suite; preserve "
            "archived receipts/source and inspect the dry-run selection first. "
            "A Vitis run does not guarantee fresh report digest, fixed-point "
            "CSim acceptance, usable per-loop II or any causal conclusion."
        ),
    }


def export_hls_rerun_plan(suite_directory, inventory_path, triage_path, output) -> dict:
    plan=plan_hls_reruns(suite_directory,_read_json(Path(inventory_path)),
                        _read_json(Path(triage_path)))
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Rerun plan is immutable; use a new output directory")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-native-rerun-plan-",dir=dest.parent) as temp:
        stage=Path(temp)
        (stage/"native_hls_rerun_plan.json").write_text(
            json.dumps(plan,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        lines=[
            "# Bounded Vitis HLS report-attestation rerun plan","",
            f"Source-verified research comparison gaps: {plan['comparison_gap_count']}. "
            f"Unique priority cases: {plan['distinct_priority_case_count']}.","",
            "**These commands are DRY RUNS. No vendor tools are executed.**","",
        ]
        for model in plan["models"]:
            lines.extend([f"## {model['model']}","",
                          "| Case | Architecture axis | READY index | Prior report status |",
                          "| --- | --- | ---: | --- |"])
            for case in model["selected_cases"]:
                lines.append("| "+" | ".join(str(case[k]) for k in (
                    "case","axis","ready_index","report_status_before_rerun"))+" |")
            lines.extend(["","### Review exact bounded dry-run windows",""])
            for step in model["windows"]:
                lines.extend([
                    "Selected: "+", ".join(step["selected_cases"]), "",
                    "```bash",step["dry_run_command"],"```","",
                ])
        lines.extend([
            "## Required execution and evidence boundaries","",
            "Only after reviewing source, campaign indices and existing receipt "
            "preservation, an operator may explicitly append `--execute` to "
            "a chosen dry-run command. Never insert it automatically.","",
            "After execution, independently regenerate source/receipt-bound native "
            "inventories and rerun the redacted grammar diagnostics against those "
            "new inventories. Passing an invocation is not E2/E3 and an unavailable "
            "achieved field is not a measured II.","",plan["caveat"],"",
        ])
        (stage/"native_hls_rerun_plan.md").write_text("\n".join(lines),
                                                        encoding="utf-8")
        stage.rename(dest)
    return plan
