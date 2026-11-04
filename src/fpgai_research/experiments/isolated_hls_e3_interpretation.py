"""Compare source digests and Pareto relationships in attested E3 estimates.

Input is the already exported, weight-free isolated_hls_e3_ablation.json.
No private compiler source, raw vendor reports, native tools, or prior E4
measurements are read. Equal summary estimates DO NOT imply same circuit.
"""
from __future__ import annotations

import csv
import json
from math import isclose
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.isolated_hls_e3_ablation import SCHEMA as E3_SCHEMA
from fpgai_research.experiments.native_hls_rerun_plan import _read_json

SCHEMA="fpgai-research/isolated-e3-estimate-interpretation/v1"
_METRICS=("lut","ff","dsp","bram_18k","uram")
_DIGEST=re.compile(r"[0-9a-f]{64}\Z")
_ID=re.compile(r"[a-z][a-z0-9_]{0,40}\Z")
_CASE=re.compile(r"case_[0-9]{4,8}\Z")
_COLUMNS=("model","case","axis","source_vs_baseline",
          "equal_cpp_source_cases","pareto_status","dominated_by","dominates","equal_estimate_cases",
          "estimated_latency_cycles","operator_loop_ii_verified")


def _integer(x, *, positive=False):
    return type(x) is int and x >= (1 if positive else 0)


def _delta_record(row, baseline):
    if not _integer(row.get("estimated_latency_cycles"),positive=True):
        raise ValueError("Eligible E3 latency estimate missing or invalid")
    b=baseline["estimated_latency_cycles"]
    if not _integer(b,positive=True):
        raise ValueError("Eligible E3 baseline latency invalid")
    if (row.get("baseline_latency_cycles")!=b
            or row.get("estimated_latency_delta_cycles")!=row["estimated_latency_cycles"]-b
            or type(row.get("estimated_latency_reduction_pct")) not in (float,int)
            or not isclose(
                row["estimated_latency_reduction_pct"],
                round(100*(b-row["estimated_latency_cycles"])/b,6),
                abs_tol=1e-6,rel_tol=0)):
        raise ValueError("Tampered or inconsistent baseline-relative E3 latency")
    resource=row.get("estimated_hls_resource_counts")
    baseline_resource=baseline.get("estimated_hls_resource_counts")
    if not isinstance(resource,dict) or not isinstance(baseline_resource,dict):
        raise ValueError("Missing resource measurement structure")
    values=[row["estimated_latency_cycles"]]
    for key in _METRICS:
        item=resource.get(key)
        base=baseline_resource.get(key)
        if not isinstance(item,dict) or not isinstance(base,dict):
            raise ValueError("Missing HLS resource")
        observed=item.get("case_hls_estimated")
        prior=item.get("baseline_hls_estimated")
        initial=base.get("case_hls_estimated")
        difference=item.get("estimated_delta")
        if (prior!=initial or (initial is not None and not _integer(initial))
                or (observed is not None and not _integer(observed))
                or difference!=(observed-initial
                  if observed is not None and initial is not None else None)):
            raise ValueError("Tampered or inconsistent E3 resource delta")
        values.append(observed)
    return tuple(values) if all(v is not None for v in values) else None


def _dominates(a:tuple,b:tuple)->bool:
    """Strict minimization across HLS latency and five resource estimates."""
    return all(x<=y for x,y in zip(a,b)) and any(x<y for x,y in zip(a,b))


def interpret_isolated_hls_e3(document:dict)->dict:
    if (document.get("schema")!=E3_SCHEMA
            or document.get("native_tools_executed") is not False
            or document.get("operator_schedule_attribution_verified") is not False
            or document.get("e4_physical_comparison_performed") is not False
            or document.get("private_tensor_values_exported") is not False):
        raise ValueError("Expected only non-attributing E3-only scientific comparison")
    saved=document.get("comparison_rows")
    if (not isinstance(saved,list) or not saved
            or len(saved)!=document.get("case_count")):
        raise ValueError("Invalid E3 comparison inventory")
    grouped={}
    unique=set()
    eligible_count=blocked_count=variant_count=0
    for row in saved:
        if not isinstance(row,dict):
            raise ValueError("Invalid E3 row")
        model,case,axis=row.get("model"),row.get("case"),row.get("axis")
        if (not isinstance(model,str) or not _ID.fullmatch(model)
                or not isinstance(case,str) or not _CASE.fullmatch(case)
                or not isinstance(axis,str) or not _ID.fullmatch(axis)
                or (model,case) in unique):
            raise ValueError("Missing, unsafe or duplicated study identity")
        unique.add((model,case))
        if axis!="baseline":
            variant_count+=1
        if (not isinstance(row.get("source_sha256"),str)
                or not _DIGEST.fullmatch(row["source_sha256"])
                or not isinstance(row.get("functional_sha256"),str)
                or not _DIGEST.fullmatch(row["functional_sha256"])
                or row.get("operator_loop_ii_verified") is not False
                or row.get("measured_postroute_qor") is not False):
            raise ValueError("Unverified HLS source or invalid evidence boundary")
        status=row.get("comparison_status")
        if status=="QUALIFIED_E3_HLS_ESTIMATE":
            eligible_count+=1
            if (row.get("blocker") is not None
                    or row.get("evidence")!="E3-HLS-synthesis-report"
                    or row.get("numerical_csim_passed") is not True
                    or row.get("fresh_report_attested") is not True):
                raise ValueError("Invalid E3-qualification provenance")
        elif status=="BLOCKED":
            blocked_count+=1
            if not isinstance(row.get("blocker"),str) or not row["blocker"]:
                raise ValueError("Blocked E3 comparison must have an explicit blocker")
        else:
            raise ValueError("Unknown E3 comparison status")
        grouped.setdefault(model,[]).append(row)
    if (variant_count!=document.get("variant_count")
            or eligible_count!=document.get("qualified_e3_hls_estimate_case_count")
            or blocked_count!=document.get("blocked_case_count")):
        raise ValueError("E3 evidence totals disagree with the case inventory")
    interpretations=[]
    ties=0
    dominated=0
    incomplete=0
    identical_sources=0
    source_equal_pairs=0
    for model,cases in grouped.items():
        baselines=[row for row in cases if row["case"]=="case_0000"
                   and row["axis"]=="baseline"]
        if len(baselines)!=1 or sum(r["axis"]=="baseline" for r in cases)!=1:
            raise ValueError("Missing or ambiguous canonical baseline")
        baseline=baselines[0]
        if baseline["comparison_status"]=="BLOCKED" and any(
                r["comparison_status"]=="QUALIFIED_E3_HLS_ESTIMATE" for r in cases):
            raise ValueError("Variant E3 qualified without an eligible baseline")
        vectors={}
        for row in cases:
            if row["comparison_status"]!="QUALIFIED_E3_HLS_ESTIMATE":
                for col in ("estimated_latency_cycles","estimated_latency_delta_cycles",
                            "estimated_latency_reduction_pct","baseline_latency_cycles"):
                    if row.get(col) is not None:
                        raise ValueError("Blocked E3 row contains baseline-derived measurements")
                continue
            if row["functional_sha256"]!=baseline["functional_sha256"]:
                raise ValueError("E3 variant functional fingerprint changed")
            vectors[row["case"]]=_delta_record(row,baseline)
        for row in cases:
            case=row["case"]
            qualified=row["comparison_status"]=="QUALIFIED_E3_HLS_ESTIMATE"
            if not qualified:
                pareto="BLOCKED_E3";same=[];by=[];beats=[]
                source_twins=[]
                source_relation="NOT_E3_COMPARABLE"
            else:
                source_relation=("BASELINE" if row["axis"]=="baseline"
                     else "SAME_CPP_SHA256" if
                     row["source_sha256"]==baseline["source_sha256"]
                     else "DIFFERENT_CPP_SHA256")
                if source_relation=="SAME_CPP_SHA256":
                    identical_sources+=1
                # Compare source hashes against EVERY eligible peer, not only
                # the baseline. PE and parallelism can both differ from the
                # baseline while still emitting byte-identical HLS C++.
                source_twins=sorted(
                    other["case"] for other in cases
                    if other["case"]!=case
                    and other["comparison_status"]=="QUALIFIED_E3_HLS_ESTIMATE"
                    and other["source_sha256"]==row["source_sha256"]
                )
                source_equal_pairs+=sum(peer>case for peer in source_twins)
                own=vectors[case]
                if own is None:
                    pareto="INCOMPLETE_HLS_ESTIMATE_VECTOR"
                    incomplete+=1
                    same=[];by=[];beats=[]
                else:
                    peers={k:v for k,v in vectors.items() if k!=case and v is not None}
                    same=sorted(k for k,v in peers.items() if v==own)
                    by=sorted(k for k,v in peers.items() if _dominates(v,own))
                    beats=sorted(k for k,v in peers.items() if _dominates(own,v))
                    pareto="DOMINATED_HLS_ESTIMATE" if by else "NONDOMINATED_AMONG_COMPLETE_VECTORS"
                    if by:
                        dominated+=1
                    ties+=sum(k>case for k in same)
            interpretations.append({
                "model":model,"case":case,"axis":row["axis"],
                "source_vs_baseline":source_relation,
                "equal_cpp_source_cases":source_twins,
                "pareto_status":pareto,
                "dominated_by":by,"dominates":beats,
                "equal_estimate_cases":same,
                "estimated_latency_cycles":row.get("estimated_latency_cycles") if qualified else None,
                "operator_loop_ii_verified":False,
                "causal_scheduling_explanation_verified":False,
                "e4_physical_qor_verified":False,
            })
    return {
        "schema":SCHEMA,
        "source_suite_sha256":document.get("source_suite_sha256"),
        "case_count":len(interpretations),
        "source_identical_to_baseline_variant_count":identical_sources,
        "equal_cpp_source_pair_count":source_equal_pairs,
        "equal_hls_estimate_pair_count":ties,
        "dominated_hls_estimate_case_count":dominated,
        "incomplete_hls_estimate_vector_count":incomplete,
        "cases":interpretations,
        "native_tools_executed":False,
        "operator_schedule_attribution_verified":False,
        "e4_physical_comparison_performed":False,
        "private_tensor_values_exported":False,
        "limitations":(
            "Six-dimensional minimization uses only E3 HLS latency plus "
            "LUT/FF/DSP/BRAM_18K/URAM synthesis estimates on matching models; "
            "higher clock frequency, actual routed physical QoR, power, and "
            "task accuracy are NOT optimized. Equal summary estimates do not "
            "establish equal RTL or operator mapping. A source SHA equality "
            "shows emitted C++ byte identity, not equal tool outcomes. "
            "Non-dominated means among cases with complete vectors only; "
            "Pareto estimates are not physical design/PPA conclusions, and "
            "no achieved loop II or bottleneck causality is inferred."
        )
    }


def export_isolated_hls_e3_interpretation(comparison_path,output)->dict:
    result=interpret_isolated_hls_e3(_read_json(Path(comparison_path)))
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Interpretation export is immutable; pick new destination")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-e3-interpret-",dir=target.parent) as tmp:
        stage=Path(tmp)
        (stage/"e3_estimate_interpretation.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        with (stage/"e3_estimate_interpretation.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=_COLUMNS)
            w.writeheader()
            for row in result["cases"]:
                w.writerow({k:json.dumps(row[k]) if k in (
                    "dominated_by","dominates","equal_estimate_cases",
                    "equal_cpp_source_cases") else row[k]
                    for k in _COLUMNS})
        lines=[
            "# Source identity and HLS-estimate Pareto diagnosis","",
            "**E3 HLS estimates only. Not routed E4, operator achieved II or "
            "verified source-level cause.**","",
            f"Cases: {result['case_count']}; dominated: "
            f"{result['dominated_hls_estimate_case_count']}; tied pairs: "
            f"{result['equal_hls_estimate_pair_count']}; source-identical variants "
            f"(against baseline): {result['source_identical_to_baseline_variant_count']}; "
            f"all source-identical pairs: {result['equal_cpp_source_pair_count']}.","",
            "| Model | Case | Axis | Source vs baseline | Same C++ source as | HLS-estimate Pareto | Dominated by | Equal estimates as |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in result["cases"]:
            lines.append("| "+" | ".join((
                row["model"],row["case"],row["axis"],
                row["source_vs_baseline"],
                ", ".join(row["equal_cpp_source_cases"]) or "—",
                row["pareto_status"],
                ", ".join(row["dominated_by"]) or "—",
                ", ".join(row["equal_estimate_cases"]) or "—",
            ))+" |")
        lines.extend(["","## Evidence boundary","",result["limitations"],""])
        (stage/"e3_estimate_interpretation.md").write_text(
            "\n".join(lines),encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Interpretation destination appeared")
        stage.rename(target)
    return result
