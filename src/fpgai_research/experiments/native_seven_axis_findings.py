"""Descriptive, reproducible findings from the source-audited native effects JSON.

This layer ONLY reads a prior native-seven-axis effects export. It does not
reaudit FPGA evidence, run vendor tools, or establish statistical significance.
Rankings are within one operator and one target, never across model families.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_seven_axis_effects import (
    SCHEMA as INPUT_SCHEMA, OUTPUTS,
)

SCHEMA="fpgai-research/native-seven-axis-descriptive-findings/v1"
COSTS=("hls_estimated_cycles","postroute_lut","postroute_dsp")


def _number(value):
    return (type(value) in (int,float) and isfinite(value))


def _pareto(cases):
    """Nondominated estimates for the same fixed operator/clock/model only."""
    eligible=[]
    for row in cases:
        metrics=row.get("qualified_metrics")
        if not isinstance(metrics,dict):
            continue
        if all(_number(metrics.get(m)) and metrics[m]>=0 for m in COSTS):
            eligible.append((row["case"],tuple(metrics[m] for m in COSTS)))
    frontier=[]
    for case,values in eligible:
        dominated=any(
            other!=case
            and all(x<=y for x,y in zip(other_values,values))
            and any(x<y for x,y in zip(other_values,values))
            for other,other_values in eligible
        )
        if not dominated:
            frontier.append({"case":case,"costs":dict(zip(COSTS,values))})
    return sorted(frontier,key=lambda x:(x["costs"][COSTS[0]],x["case"]))


def _case_breakdown(study):
    categories={
        "invalid_hls_xml":[],
        "numerical_acceptance_not_verified":[],
        "source_attested_implementation_missing":[],
        "postroute_timing_not_verified":[],
    }
    axis={}
    # Mutually exclusive *diagnostic buckets* are useful for a case-count
    # table. Priority is explicit; the bucket is NOT a proven root cause.
    exclusive={
        "numerical_acceptance_not_verified":[],
        "invalid_hls_xml":[],
        "source_attested_implementation_missing":[],
        "postroute_timing_not_verified":[],
        "other_unqualified":[],
    }
    for item in study["cases"]:
        name=item["case"]
        axis_name=item["axis"]
        x=axis.setdefault(axis_name,{"eligible":0,"qualified":0})
        x["eligible"]+=1
        if item["qualification"]=="QUALIFIED_ROUTED":
            x["qualified"]+=1
            continue
        blockers=item.get("blockers") or []
        warnings=item.get("source_warnings") or []
        if any("HLS XML measurement rejected" in v for v in warnings):
            categories["invalid_hls_xml"].append(name)
        if "Fixed-point CSim not numerically accepted" in blockers:
            categories["numerical_acceptance_not_verified"].append(name)
        if "Missing source/RTL-attested E4 Vivado implementation" in blockers:
            categories["source_attested_implementation_missing"].append(name)
        if "Post-route timing closure not verified" in blockers:
            categories["postroute_timing_not_verified"].append(name)
        for key in exclusive:
            if key=="other_unqualified" or name in categories[key]:
                exclusive[key].append(name)
                break
    return {
        "case_ids_by_diagnostic":categories,
        "diagnostic_counts_overlap":True,
        "case_ids_by_exclusive_triage_priority":exclusive,
        "exclusive_triage_is_not_causal":True,
        "case_status_by_axis":axis,
    }


def analyze_findings(effects):
    if (not isinstance(effects,dict) or effects.get("schema")!=INPUT_SCHEMA
            or effects.get("vendor_tools_executed_by_analysis") is not False
            or not isinstance(effects.get("studies"),list)
            or not effects["studies"]):
        raise ValueError("Expected original source-audited native effects export")
    studies=[]
    total={"eligible":0,"qualified":0,"single_factor":0,"pairwise":0,
           "infeasible":0}
    for item in effects["studies"]:
        if item.get("status")=="BLOCKED_SOURCE_MODEL":
            studies.append({"study":item["study"],"status":"BLOCKED_SOURCE_MODEL"})
            continue
        name=item.get("study")
        cases=item.get("cases")
        if not isinstance(name,str) or not isinstance(cases,list) or not cases:
            raise ValueError("Invalid operator-local effects inventory")
        names=[r.get("case") for r in cases]
        if any(not isinstance(n,str) for n in names) or len(names)!=len(set(names)):
            raise ValueError("Duplicate or invalid case identity in audited export")
        qualified=[r for r in cases if r.get("qualification")=="QUALIFIED_ROUTED"]
        if (any(r.get("qualification") not in ("QUALIFIED_ROUTED","UNQUALIFIED")
                for r in cases)
                or any(not isinstance(r.get("qualified_metrics"),dict) for r in qualified)
                or any(r.get("qualified_metrics") is not None
                       for r in cases if r.get("qualification")=="UNQUALIFIED")
                or len(cases)!=item.get("eligible_case_count")
                or len(qualified)!=item.get("routed_qualified_case_count")):
            raise ValueError("Per-case qualification disagrees with audited counts")
        baseline=next((r for r in cases if r.get("axis")=="baseline"),None)
        if baseline is None:
            raise ValueError("Operator-local baseline missing")
        baseline_metrics=baseline.get("qualified_metrics")
        main=[]
        for contrast in item.get("single_factor_contrasts",[]):
            if contrast.get("qualified") is not True:
                continue
            case=contrast.get("case")
            if case not in names:
                raise ValueError("Missing intervention case")
            delta=contrast.get("delta_b_minus_a")
            if not isinstance(delta,dict):
                raise ValueError("Qualified main effect has no numerical differences")
            comparison={
                "case":case,"axis":contrast["axis"],
                "requested_value":contrast["requested_value"],
                "delta_b_minus_a":delta,
                # Source: full_pipeline.py propagates precision to the
                # model-wide default; emitter.py currently rejects mixed
                # per-operator ap_fixed types. Do not falsely attribute
                # precision-only effects to the nominated operator.
                "intervention_scope":(
                    "model_global_homogeneous_precision"
                    if contrast["axis"]=="precision_bits"
                    else "selected_operator_mapping"
                ),
            }
            if isinstance(baseline_metrics,dict):
                percentages={}
                for metric in OUTPUTS:
                    initial=baseline_metrics.get(metric)
                    change=delta.get(metric)
                    percentages[metric]=(round(100*change/initial,6)
                        if _number(initial) and initial>0 and _number(change)
                        else None)
                comparison["percent_change_from_baseline"]=percentages
            main.append(comparison)
        pair=[]
        for interaction in item.get("pairwise_or_higher_order_contrasts",[]):
            if interaction.get("status")!="QUALIFIED_FACTORIAL_CONTRAST":
                continue
            values=interaction.get("pairwise_second_difference")
            if not isinstance(values,dict):
                raise ValueError("Qualified pairwise contrast has no numerical differences")
            pair.append({"case":interaction["case"],"axes":interaction["axes"],
                         "second_difference":values})
        if (len(main)!=item.get("qualified_single_factor_count")
                or len(pair)!=item.get("qualified_pairwise_count")):
            raise ValueError("Qualified contrast counts disagree with original export")
        # These are descriptive contrasts, not independent runs or p-values.
        top_changes={}
        for metric in OUTPUTS:
            numeric=[x for x in main if _number(x["delta_b_minus_a"].get(metric))]
            is_higher_better=(metric=="postroute_wns_ns")
            numeric.sort(
                key=lambda x:(-x["delta_b_minus_a"][metric]
                              if is_higher_better else x["delta_b_minus_a"][metric],
                              x["case"]))
            top_changes[metric]=numeric[0] if numeric else None
        breakdown=_case_breakdown(item)
        frontier=_pareto(cases)
        summary={
            "study":name,"status":item["status"],
            "eligible_case_count":len(cases),
            "routed_qualified_case_count":len(qualified),
            "infeasible_architecture_count":len(item.get("infeasible_architectures",[])),
            "baseline_case":baseline["case"],
            "baseline_qualified":baseline_metrics is not None,
            "precision_axis_is_model_global":True,
            "operator_local_precision_claim_supported":False,
            "best_single_factor_by_metric":top_changes,
            "qualified_single_factor_contrasts":main,
            "qualified_pairwise_second_differences":pair,
            "pareto_frontier_minimizing":list(COSTS),
            "pareto_frontier":frontier,
            **breakdown,
        }
        studies.append(summary)
        total["eligible"]+=len(cases)
        total["qualified"]+=len(qualified)
        total["single_factor"]+=len(main)
        total["pairwise"]+=len(pair)
        total["infeasible"]+=summary["infeasible_architecture_count"]
    return {
        "schema":SCHEMA,"totals":total,"studies":studies,
        "tool_execution":False,"source_evidence_reaudited_by_this_report":False,
        "metrics_are":("HLS estimated cycles, routed resource counts and "
                       "postroute timing slack; no board inference latency or power"),
        "scientific_limits":(
            "Descriptive results for the original four operator-local studies. "
            "Individual comparisons reuse common baselines, are not independent "
            "statistical replicates and do not establish hls4ml superiority. "
            "Missing estimates are not zero and unqualified architectures "
            "cannot enter Pareto or numerical effect rankings."),
        "direction_note":(
            "Smaller cycles/resources and larger timing slack are favorable. "
            "Interaction second differences measure non-additivity, not "
            "intrinsically beneficial or harmful effects."),
    }


def export_findings(effects_file,output):
    source=Path(effects_file)
    dest=Path(output)
    if (source.is_symlink() or not source.is_file()
            or source.stat().st_size>32*1024*1024):
        raise ValueError("Expected regular bounded source-audited effects.json")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Use a fresh findings output directory")
    original=source.read_bytes()
    report=analyze_findings(json.loads(original))
    report["source_effects_json_sha256"]=sha256(original).hexdigest()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".native-findings-") as temp:
        root=Path(temp)/"findings"
        root.mkdir()
        (root/"findings.json").write_text(
            json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        with (root/"qualified_main_effects.csv").open("w",encoding="utf-8",newline="") as f:
            columns=["study","case","axis","requested_value",*OUTPUTS]
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for study in report["studies"]:
                for change in study.get("qualified_single_factor_contrasts",[]):
                    writer.writerow({
                        "study":study["study"],"case":change["case"],
                        "axis":change["axis"],
                        "requested_value":json.dumps(change["requested_value"]),
                        **{k:change["delta_b_minus_a"].get(k) for k in OUTPUTS},
                    })
        with (root/"qualified_pairwise_interactions.csv").open(
                "w",encoding="utf-8",newline="") as f:
            columns=["study","case","axes",*OUTPUTS]
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for study in report["studies"]:
                for change in study.get("qualified_pairwise_second_differences",[]):
                    writer.writerow({
                        "study":study["study"],"case":change["case"],
                        "axes":",".join(change["axes"]),
                        **{k:change["second_difference"].get(k) for k in OUTPUTS},
                    })
        lines=[
            "# Operator-local native architecture findings",
            "",
            "These are source-audited results *summarized*, not newly measured "
            "or independently reaudited here. No hls4ml comparison or FPGA "
            "board-latency claim.",
            "",
            "| Operator | Qualified / eligible | Main contrasts | Pairwise contrasts | Pareto designs |",
            "|---|---:|---:|---:|---:|",
        ]
        for study in report["studies"]:
            if study["status"]=="BLOCKED_SOURCE_MODEL":
                continue
            lines.append(
                f"| {study['study']} | {study['routed_qualified_case_count']} / "
                f"{study['eligible_case_count']} | "
                f"{len(study['qualified_single_factor_contrasts'])} | "
                f"{len(study['qualified_pairwise_second_differences'])} | "
                f"{len(study['pareto_frontier'])} |")
        for study in report["studies"]:
            if study["status"]=="BLOCKED_SOURCE_MODEL":
                continue
            lines.extend(["",f"### {study['study']}",""])
            for metric,label in (
                ("hls_estimated_cycles","HLS estimated cycles"),
                ("postroute_lut","Routed LUT"),
                ("postroute_dsp","Routed DSP"),
                ("postroute_wns_ns","Post-route slack (ns)"),
            ):
                top=study["best_single_factor_by_metric"][metric]
                if top is None:
                    lines.append(f"- Best single-factor {label}: unavailable")
                    continue
                delta=top["delta_b_minus_a"][metric]
                pct=top.get("percent_change_from_baseline",{}).get(metric)
                lines.append(
                    f"- Best single-factor {label}: {top['axis']} "
                    f"({top['case']}), signed difference {delta:+g}"
                    + (f" ({pct:+g}% of baseline)" if pct is not None else "")
                )
            interactions=[
                x for x in study["qualified_pairwise_second_differences"]
                if _number(x["second_difference"].get("hls_estimated_cycles"))
            ]
            if interactions:
                strongest=max(interactions,key=lambda x:abs(
                    x["second_difference"]["hls_estimated_cycles"]))
                lines.append(
                    "- Largest absolute pairwise non-additivity in HLS "
                    f"estimated cycles: {', '.join(strongest['axes'])} "
                    f"({strongest['case']}), second difference "
                    f"{strongest['second_difference']['hls_estimated_cycles']:+g}; "
                    "this is not necessarily an improvement."
                )
            lines.append(
                "- Nondominated designs (minimize estimated cycles, routed "
                "LUT and DSP): "
                + (", ".join(x["case"] for x in study["pareto_frontier"])
                   or "none with all three objective measurements")
            )
            labels=study["case_ids_by_exclusive_triage_priority"]
            lines.append(
                "- Unqualified cases by mutually exclusive first-applicable "
                "diagnostic (not proven causes): "
                + ", ".join(f"{k}={len(v)}" for k,v in labels.items())
            )
        lines.extend([
            "",
            "**Precision-scope qualification:** The current HLS emitter "
            "requires one homogeneous fixed-point type for every operator. "
            "Changing precision is therefore a MODEL-GLOBAL perturbation, "
            "even in an operator-named study. Repeated precision-only "
            "conditions across operators of the same trained network "
            "must not be interpreted as independent operator-local effects.",
            "",
            "HLS estimated cycles and routed resources are not FPGA board "
            "execution or power measurements.",
            "Positive interaction second differences do not automatically "
            "mean improvement; these are non-additivity measurements.",
            "Numerical acceptance blockers, invalid HLS XML and route "
            "blockers overlap. The exclusive priority table is for "
            "counting only, not causal attribution.",
            "",
        ])
        (root/"summary.md").write_text("\n".join(lines),encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Findings destination appeared during export")
        root.rename(dest)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects",required=True,
                        help="Existing source-audited effects.json")
    parser.add_argument("--output",required=True,
                        help="Fresh directory for descriptive analysis only")
    args=parser.parse_args(argv)
    result=export_findings(args.effects,args.output)
    print(json.dumps(result["totals"],sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
