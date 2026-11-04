"""IR-to-HLS scheduling evidence gap study, derived from attested suite receipts.

This does not parse non-attributed schedule snippets or pretend the
Vitis top-level call interval is a requested per-loop pipeline II.
Experimental observations are estimates; mapping causality and actual
RTL/memory bottlenecks remain unverified without loop-level reports.
"""
from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path

SCHEMA = "fpgai-research/ir-to-hls-schedule-evidence/v1"
SOURCE_SCHEMA = "fpgai-research/scientific-ir-mechanism-study/v1"
COLUMNS = (
    "model", "case", "axis", "outcome", "hls_evidence",
    "changed_compute_operators", "source_changed",
    "reported_hls_latency_cycles", "reported_hls_top_interval_cycles",
    "requested_pipeline_ii_by_operator", "attested_loop_ii_by_operator",
    "qualified_latency_delta_cycles", "qualified_latency_reduction_pct",
    "observed_latency_direction", "explanation_status",
)


def _number(value):
    return type(value) in (int,float) and isfinite(value)


def _compute_operators(row):
    operators=row.get("operators")
    if not isinstance(operators,list):
        raise ValueError("Missing verified IR operator inventory")
    seen=set()
    result=[]
    for op in operators:
        name=op.get("operator")
        if not isinstance(name,str) or not name or name in seen:
            raise ValueError("Invalid or duplicated IR operator")
        seen.add(name)
        if op.get("kind") in ("MatMul","Gemm","Conv"):
            result.append(op)
    return result


def analyze_ir_hls_schedule(study:dict)->dict:
    """Correlate per-operator IR changes with available E3/E4 summary data.

    Does not infer *why* a reported top-level latency changed. That
    requires attested per-loop schedules, which this evidence lacks.
    """
    if study.get("schema")!=SOURCE_SCHEMA:
        raise ValueError("Expected a source-verified scientific IR study")
    entries=study.get("cases")
    models=study.get("models")
    if (not isinstance(entries,list) or not entries
            or not isinstance(models,list) or not models
            or len(entries)!=study.get("case_count")):
        raise ValueError("Inconsistent source IR case inventory")
    grouped={}
    keys=set()
    for row in entries:
        if not isinstance(row,dict):
            raise ValueError("Malformed IR case")
        key=(row.get("model"),row.get("case"))
        if (not all(isinstance(v,str) and v for v in key)
                or key in keys):
            raise ValueError("Missing or duplicated model/case")
        keys.add(key)
        grouped.setdefault(key[0],[]).append(row)
    if set(grouped)!={m.get("model") for m in models}:
        raise ValueError("Model inventory disagrees with scientific IR data")

    output=[]
    for model,group in grouped.items():
        baselines=[x for x in group if x.get("axis")=="baseline"
                   and x["case"]=="case_0000"]
        if len(baselines)!=1:
            raise ValueError("Each model needs one pinned baseline")
        baseline=baselines[0]
        base_latency=baseline.get("hls_latency_cycles")
        for case in group:
            compute=_compute_operators(case)
            changed=[{
                "operator":op["operator"], "kind":op["kind"],
                "changed_effective_mapping":op["changed_effective_fields"],
                "changed_lowering_mechanisms":op["changed_lowering_fields"],
                "effective_pe":op["lowered_mechanisms"].get("pe_count"),
                "effective_replicas":op["lowered_mechanisms"].get("replica_count"),
                "effective_simd_lanes":op["lowered_mechanisms"].get("simd_lanes"),
                "effective_fold_factor":op["lowered_mechanisms"].get("fold_factor"),
            } for op in compute if op.get("changed_effective_fields")
                or op.get("changed_lowering_fields")]
            requested_ii={op["operator"]:op["lowered_mechanisms"].get("pipeline_ii")
                          for op in compute}
            evidence=case.get("evidence")
            attested=evidence in ("E3-HLS-synthesis-report",
                                  "E4-Vivado-implementation")
            latency=case.get("hls_latency_cycles") if attested else None
            interval=case.get("hls_top_interval_cycles") if attested else None
            if latency is not None and (not _number(latency) or latency<0):
                raise ValueError("Malformed attested HLS latency")
            if interval is not None and (not _number(interval) or interval<0):
                raise ValueError("Malformed attested HLS top-level interval")
            qualified=case.get("outcome")=="QUALIFIED"
            if qualified and (evidence!="E4-Vivado-implementation"
                              or case.get("numerical_pass") is not True
                              or not _number(latency)
                              or baseline.get("outcome")!="QUALIFIED"
                              or baseline.get("evidence")!="E4-Vivado-implementation"
                              or not _number(base_latency) or base_latency<=0):
                raise ValueError("Qualified E4 contrast lacks matching accepted HLS baseline")
            delta=case.get("hls_latency_delta_cycles") if qualified else None
            if qualified:
                if not _number(delta) or abs((latency-base_latency)-delta)>1e-6:
                    raise ValueError("Qualified latency delta does not match source-attested baseline")
                direction=("DECREASED" if delta<0 else "INCREASED" if delta>0
                           else "UNCHANGED")
                reduction=round(100*(base_latency-latency)/base_latency,6)
            else:
                direction=None
                reduction=None
            if not changed:
                hypothesis="NO_CHANGED_COMPUTE_MAPPING"
            elif qualified and direction=="UNCHANGED":
                hypothesis="IR_CHANGED_WITH_UNCHANGED_TOP_LEVEL_HLS_LATENCY"
            elif qualified:
                hypothesis="IR_CHANGED_WITH_HLS_LATENCY_OBSERVATION"
            else:
                hypothesis="IR_CHANGED_NATIVE_QUALIFICATION_PENDING_OR_REJECTED"
            output.append({
                "model":model,"case":case["case"],"axis":case.get("axis"),
                "outcome":case.get("outcome"),"hls_evidence":evidence,
                "functional_sha256":case.get("functional_sha256"),
                "hls_source_sha256":case.get("source_sha256"),
                "source_changed":case.get("hls_source_changed"),
                "changed_compute_operators":changed,
                "reported_hls_latency_cycles":latency,
                "reported_hls_top_interval_cycles":interval,
                "requested_pipeline_ii_by_operator":requested_ii,
                # This suite has source-attested synthesis *summary* XML,
                # not per-loop timing/II binding. Never compare the two.
                "attested_loop_ii_by_operator":None,
                "qualified_latency_delta_cycles":delta,
                "qualified_latency_reduction_pct":reduction,
                "observed_latency_direction":direction,
                "explanation_status":hypothesis,
                "requires_per_loop_schedule_evidence":bool(changed),
                "bottleneck_cause_verified":False,
            })
    if (sum(row["hls_evidence"]=="E4-Vivado-implementation" for row in output)
            !=study.get("e4_routed_cases") or
            sum(row["outcome"]=="QUALIFIED" for row in output)
            !=study.get("qualified_cases")):
        raise ValueError("E4 or qualified totals differ from source study")
    return {
        "schema":SCHEMA,
        "source_suite_sha256":study.get("source_suite_sha256"),
        "model_count":len(models),"case_count":len(output),
        "e4_routed_cases":study["e4_routed_cases"],
        "qualified_cases":study["qualified_cases"],
        "native_tools_executed":False,"weight_values_exported":False,
        "per_loop_schedule_evidence_available":False,
        "cases_requiring_per_loop_schedule":sum(
            row["requires_per_loop_schedule_evidence"] for row in output),
        "cases":output,
        "limitations":(
            "Native HLS latency and top-level call interval are source-attested "
            "synthesis estimates, not measured inference runtime. An operator "
            "requested pipeline II is not comparable to top-level call interval. "
            "Per-loop schedule, achieved loop II, memory port contention, "
            "critical operation/dependency chain and RTL occupancy are not "
            "established by the available summary data. Same-model correlation "
            "of changed IR with estimates is not a verified causal explanation."
        ),
    }


def export_ir_hls_schedule(study:dict,stage:Path)->dict:
    result=analyze_ir_hls_schedule(study)
    stage=Path(stage)
    (stage/"ir_hls_schedule.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    with (stage/"ir_hls_schedule.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=COLUMNS)
        writer.writeheader()
        for row in result["cases"]:
            writer.writerow({
                k:("TBD" if row.get(k) is None else
                   json.dumps(row[k],sort_keys=True,allow_nan=False)
                   if isinstance(row[k],(list,dict)) else row[k])
                for k in COLUMNS
            })
    lines=[
        "# IR-to-HLS scheduling evidence and unresolved mechanisms","",
        "Within-model comparisons of source-attested IR mapping and HLS "
        "summary estimates. **No per-loop schedule evidence is available "
        "in this export.** Top-level call interval is not operator loop II.","",
        "| Model | Case | Axis | IR compute mapping changed | HLS latency | HLS top interval | Qualified Δ cycles | Estimated latency reduction | Explanation |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["cases"]:
        values=[
            row["model"],row["case"],row["axis"],
            "yes" if row["changed_compute_operators"] else "no",
            row["reported_hls_latency_cycles"],
            row["reported_hls_top_interval_cycles"],
            row["qualified_latency_delta_cycles"],
            (f'{row["qualified_latency_reduction_pct"]:.3f}%'
             if row["qualified_latency_reduction_pct"] is not None else None),
            row["explanation_status"]
        ]
        lines.append("| "+" | ".join(
            "TBD" if value is None else str(value) for value in values)+" |")
    lines.extend(["","## Missing proof obligations","",
        "To explain PE/SIMD/coarse-parallelism differences, obtain source-"
        "attributed native **per-loop** scheduling evidence for the actual "
        "compute operator, including achieved loop II, latency, pipelining "
        "reason, memory access conflicts and operator dependencies. Do not "
        "interpret a source-level UNROLL or a requested pipeline II as "
        "measured acceleration.","",result["limitations"],""])
    (stage/"ir_hls_schedule.md").write_text("\n".join(lines))
    return result
