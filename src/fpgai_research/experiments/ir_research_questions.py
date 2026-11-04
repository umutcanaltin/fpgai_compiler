"""Falsifiable scientific questions from source-attested Model IR/HLS evidence.

This is a *research planning* export, never a schedule parser or optimizer.
It preserves negative results, incomplete native evidence and provenance
blockers instead of asserting why a PE/SIMD choice behaved as observed.
"""
from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path

SCHEDULE_SCHEMA="fpgai-research/ir-to-hls-schedule-evidence/v1"
INVENTORY_SCHEMA="fpgai-research/native-hls-schedule-report-inventory/v1"
SCHEMA="fpgai-research/ir-optimization-research-questions/v1"
COLUMNS=("model","case","axis","observation","latency_reduction_pct",
         "changed_compute_operators","baseline_raw_report_status",
         "variant_raw_report_status","schedule_evidence_gap",
         "cause_established","testable_question")


def _identity(rows):
    if not isinstance(rows,list) or not rows:
        raise ValueError("Expected nonempty cases")
    mapped={}
    for item in rows:
        if not isinstance(item,dict):
            raise ValueError("Invalid research evidence record")
        key=(item.get("model"),item.get("case"))
        if not all(isinstance(k,str) and k for k in key) or key in mapped:
            raise ValueError("Missing or duplicate model/case identity")
        mapped[key]=item
    return mapped


def _number(value):
    return type(value) in (int,float) and isfinite(value)


def _status(base,variant):
    statuses=(base["status"],variant["status"])
    if all(s=="ATTESTED_RAW_REPORT_AVAILABLE" for s in statuses):
        return "BOTH_RAW_REPORTS_ATTESTED_BUT_LOOP_PARSER_MISSING"
    if "UNSAFE_SYMLINKED_REPORT" in statuses or "HASH_MISMATCH" in statuses:
        return "REPORT_INTEGRITY_OR_PROVENANCE_FAILURE"
    if "REPORT_NOT_PRESENT" in statuses:
        return "NATIVE_TEXT_REPORT_ABSENT"
    if any(s in ("PRESENT_NOT_ATTESTED","PRESENT_NOT_FRESHLY_ATTESTED",
                 "NATIVE_TEXT_WITHOUT_E3_QUALIFICATION") for s in statuses):
        return "NATIVE_TEXT_REPORT_PRESENT_BUT_UNATTESTED"
    return "NO_VERIFIED_PER_LOOP_SCHEDULE"


def _question(row):
    axis=row.get("axis")
    if axis=="simd":
        return ("Does increased reduction-lane SIMD change the actual "
                "operator-local achieved loop II, reduction schedule, "
                "memory-port pressure or dependence chain?")
    if axis in ("parallelism","pes"):
        return ("Do replicated outputs versus PE-level lanes produce "
                "different loop schedules or memory access constraints "
                "despite their nominal P×PE capacity?")
    if axis=="serialization":
        return ("Do additional temporal folds change the achieved reduction "
                "schedule and source-attested HLS latency?")
    if axis=="tiling":
        return ("Do the effective tiling/tail decisions change operator "
                "access locality or achieved schedule?")
    if axis=="pipelining":
        return ("Does the requested operator II correspond to an actual "
                "achieved per-loop II in the native scheduler?")
    if axis=="precision":
        return ("Does changed numeric width affect the scheduled datapath "
                "while satisfying the fixed-point acceptance gate?")
    return ("Which verified Lowered Model IR mechanisms explain "
            "the measured operator scheduling response?")


def _same_nominal_mapping(a,b):
    """Require actual same-compute-operator P×PE capacity, not axis labels."""
    def index(row):
        opmap={}
        for op in row["changed_compute_operators"]:
            name=op["operator"]
            p,pe=op.get("effective_replicas"),op.get("effective_pe")
            lane,fold=op.get("effective_simd_lanes"),op.get("effective_fold_factor")
            if (name in opmap or any(type(v) is not int or v<1
                                     for v in (p,pe,lane,fold))):
                return None
            opmap[name]=(p*pe,lane,fold)
        return opmap or None
    left,right=index(a),index(b)
    return left is not None and right is not None and left==right


def analyze_ir_research_questions(schedule,inventory):
    """Join immutable exports only when source identity and inventory agree."""
    if (schedule.get("schema")!=SCHEDULE_SCHEMA or
            inventory.get("schema")!=INVENTORY_SCHEMA):
        raise ValueError("Need verified IR-to-HLS and native report inventories")
    root_sha=schedule.get("source_suite_sha256")
    if (not isinstance(root_sha,str) or len(root_sha)!=64 or
            inventory.get("source_suite_sha256")!=root_sha):
        raise ValueError("Two research exports must match one pinned suite")
    sr=_identity(schedule.get("cases"))
    nr=_identity(inventory.get("cases"))
    if sr.keys()!=nr.keys():
        raise ValueError("Schedule and raw-report case inventories disagree")
    if (len(sr)!=schedule.get("case_count") or
            len(nr)!=inventory.get("case_count")):
        raise ValueError("Declared study size disagrees with case inventory")
    for key,case in sr.items():
        report=nr[key]
        if (case.get("hls_source_sha256")!=report.get("source_sha256") or
                case.get("hls_evidence")!=report.get("case_evidence")):
            raise ValueError("Native report and source-verified IR provenance disagree")
    grouped={}
    for (model,case),item in sr.items():
        grouped.setdefault(model,[]).append(item)
    questions=[]
    for model,records in grouped.items():
        base=[row for row in records if row.get("axis")=="baseline"
              and row.get("case")=="case_0000"]
        if len(base)!=1:
            raise ValueError("Every model requires its pinned baseline")
        baseline=base[0]
        base_report=nr[(model,baseline["case"])]
        for row in records:
            if row is baseline:
                continue
            key=(model,row["case"])
            variant_report=nr[key]
            changed=row.get("changed_compute_operators")
            if not isinstance(changed,list):
                raise ValueError("Missing IR change evidence")
            reduction=row.get("qualified_latency_reduction_pct")
            delta=row.get("qualified_latency_delta_cycles")
            qualified=row.get("outcome")=="QUALIFIED"
            if qualified:
                if (baseline.get("outcome")!="QUALIFIED"
                        or not _number(reduction) or not _number(delta)
                        or not _number(row.get("reported_hls_latency_cycles"))):
                    raise ValueError("Qualified scientific claim lacks accepted native metrics")
                observation=("HLS_ESTIMATED_LATENCY_DECREASED" if delta<0 else
                             "HLS_ESTIMATED_LATENCY_INCREASED" if delta>0 else
                             "HLS_ESTIMATED_LATENCY_UNCHANGED")
            else:
                if reduction is not None or delta is not None:
                    raise ValueError("Unqualified case has a claimed accepted delta")
                observation="NO_QUALIFIED_HLS_CONTRAST"
            questions.append({
                "model":model,"case":row["case"],"axis":row.get("axis"),
                "baseline_case":baseline["case"],
                "observation":observation,
                "latency_reduction_pct":reduction,
                "qualified_latency_delta_cycles":delta,
                "baseline_latency_cycles":baseline.get("reported_hls_latency_cycles"),
                "variant_latency_cycles":row.get("reported_hls_latency_cycles"),
                "changed_compute_operators":[op["operator"] for op in changed],
                "source_changed":row.get("source_changed"),
                "baseline_raw_report_status":base_report["status"],
                "variant_raw_report_status":variant_report["status"],
                "schedule_evidence_gap":_status(base_report,variant_report),
                "cause_established":False,
                "testable_question":_question(row),
                "evidence_requirement":(
                    "Source-attributed, version-validated per-loop achieved II/"
                    "latency and operator binding for both baseline/variant; "
                    "inspect dependencies and memory ports before causal claims."
                ),
            })
    # Within each model, PE-vs-replication equality is a *question*.
    contrasts=[]
    for model,records in grouped.items():
        relevant={r["axis"]:r for r in questions if r["model"]==model}
        if "parallelism" in relevant and "pes" in relevant:
            a,b=relevant["parallelism"],relevant["pes"]
            if (a["observation"]!="NO_QUALIFIED_HLS_CONTRAST"
                    and b["observation"]!="NO_QUALIFIED_HLS_CONTRAST"
                    and _same_nominal_mapping(
                        sr[(model,a["case"])],sr[(model,b["case"])])
                    and a["variant_latency_cycles"]==b["variant_latency_cycles"]):
                contrasts.append({
                    "type":"WITHIN_MODEL_EQUAL_PE_AND_REPLICA_LATENCY_ESTIMATE",
                    "model":model,
                    "case_a":a["case"],"case_b":b["case"],
                    "equal_hls_estimated_cycles":a["variant_latency_cycles"],
                    "cause_established":False,
                    "question":("Do the two distinct IR mapping choices compile "
                                "to a similar native operator schedule, or do "
                                "other stages mask a difference? Equal top-level "
                                "latency does not imply equal RTL."),
                })
    simd=[r for r in questions if r["axis"]=="simd"
          and r["observation"]!="NO_QUALIFIED_HLS_CONTRAST"]
    for i,a in enumerate(simd):
        for b in simd[i+1:]:
            if (a["model"]!=b["model"] and a["observation"]!=b["observation"]):
                contrasts.append({
                    "type":"CROSS_MODEL_SIMD_RESPONSE_DIFFERS",
                    "model_a":a["model"],"model_b":b["model"],
                    "case_a":a["case"],"case_b":b["case"],
                    "reduction_pct_a":a["latency_reduction_pct"],
                    "reduction_pct_b":b["latency_reduction_pct"],
                    "cause_established":False,
                    "question":("Why do these models respond differently to "
                                "SIMD? Each percent change is relative only "
                                "to its own model baseline, not a direct "
                                "cross-model architecture ranking."),
                })
    return {
        "schema":SCHEMA,"source_suite_sha256":root_sha,
        "case_count":len(sr),"research_question_count":len(questions),
        "contrasts":contrasts,"questions":questions,
        "native_tools_executed":False,"weight_values_exported":False,
        "per_loop_causal_explanation_verified":False,
        "limitations":(
            "This is a pre-registered question inventory based on verified "
            "Model IR changes, qualified within-model HLS estimates and raw "
            "native report *availability*. It contains no parsed per-loop "
            "schedule. Unattested report bytes and E1/E2 cases do not justify "
            "physical speedup or causal claims. No cross-model latency or "
            "resource rankings and no board-runtime measurements."
        ),
    }


def export_ir_research_questions(schedule,inventory,stage:Path):
    result=analyze_ir_research_questions(schedule,inventory)
    dest=Path(stage)
    (dest/"ir_research_questions.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    with (dest/"ir_research_questions.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS)
        writer.writeheader()
        for row in result["questions"]:
            writer.writerow({key:("TBD" if row.get(key) is None else
                                  ";".join(row[key]) if isinstance(row[key],list) else row[key])
                             for key in COLUMNS})
    lines=[
        "# Testable IR-to-native scheduling questions","",
        "Each result is a **within-model** HLS estimate, never board timing. "
        "Causal synthesis explanations require separately source-attested, "
        "operator-bound per-loop reports.","",
        "| Model | Case | Axis | Native observation | Relative HLS latency change | Raw-report gap |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in result["questions"]:
        values=(row["model"],row["case"],row["axis"],row["observation"],
                None if row["latency_reduction_pct"] is None else
                f'{row["latency_reduction_pct"]:+.3f}% reduction',
                row["schedule_evidence_gap"])
        lines.append("| "+" | ".join("TBD" if v is None else str(v)
                                     for v in values)+" |")
    lines.extend(["","## Falsifiable contrasts",""])
    for item in result["contrasts"]:
        lines.extend(["### "+item["type"],"",
                      item["question"],""])
    lines.extend(["## Evidence still required","",result["limitations"],""])
    (dest/"ir_research_questions.md").write_text("\n".join(lines))
    return result
