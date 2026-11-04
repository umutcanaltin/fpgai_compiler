"""Complete, no-imputation seven-axis effect matrix from qualified native receipts.

Include *all* requested main effects, source-blocked knobs, and paired /
higher-order requests. Matched numerical differences exist ONLY for genuine
source- and route-qualified cases; a zero observed difference is a valid tie,
not proof of equivalent RTL or ineffective compiler transformations.
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
    SCHEMA as EFFECTS_SCHEMA,AXES,OUTPUTS,
)

SCHEMA="fpgai-research/complete-seven-axis-size-effect-matrix/v1"
PHYSICAL=("postroute_lut","postroute_ff","postroute_dsp",
          "postroute_bram","postroute_uram","postroute_wns_ns")


def _valid_number(value):
    return type(value) in (int,float) and isfinite(value)


def _changes(request,baseline):
    if (not isinstance(request,dict) or not isinstance(baseline,dict)
            or set(request)!=set(baseline)):
        raise ValueError("Source request not comparable to baseline")
    return [k for k in AXES if request[k]!=baseline[k]]


def _scope(axis):
    return ("model_global_homogeneous_precision" if axis=="precision_bits"
            else "selected_operator_mapping")


def _cost_tie(a,b):
    """Equal reported physical observables, NOT netlist/functional equivalence."""
    if not isinstance(a,dict) or not isinstance(b,dict):
        return None
    if not all(_valid_number(a.get(k)) and _valid_number(b.get(k))
               for k in PHYSICAL):
        return None
    return all(a[k]==b[k] for k in PHYSICAL)


def analyze_axis_matrix(effects,geometry_receipt=None):
    if (not isinstance(effects,dict) or effects.get("schema")!=EFFECTS_SCHEMA
            or effects.get("vendor_tools_executed_by_analysis") is not False
            or not isinstance(effects.get("studies"),list)):
        raise ValueError("Require source-audited seven-axis effects input")
    geometry={}
    if geometry_receipt is not None:
        if (not isinstance(geometry_receipt,dict) or
                geometry_receipt.get("schema")!=
                    "fpgai-research/geometry-controlled-seven-axis-probes/v1"
                or geometry_receipt.get(
                    "weights_are_deterministic_synthetic_not_trained") is not True):
            raise ValueError("Geometry must be verified synthetic source receipt")
        geometry={x["id"]:{"size":x["size"],"kind":x["kind"],
                           "geometry":x["geometry"]}
                  for x in geometry_receipt["studies"]}
        present={x["study"] for x in effects["studies"]}
        if set(geometry)!=present:
            raise ValueError("Geometry receipt and audited effect studies disagree")
    main=[]
    pair=[]
    high_order=[]
    axis_coverage=[]
    study_totals=[]
    for study in effects["studies"]:
        if study.get("status")=="BLOCKED_SOURCE_MODEL":
            raise ValueError("Cannot claim complete effects for blocked source model")
        name=study.get("study")
        cases=study.get("cases",[])
        if not isinstance(name,str) or not isinstance(cases,list):
            raise ValueError("Malformed study in source-audited effects")
        indexed={x["case"]:x for x in cases}
        if len(indexed)!=len(cases):
            raise ValueError("Duplicate case identity")
        bases=[x for x in cases if x.get("axis")=="baseline"]
        if len(bases)!=1:
            raise ValueError("One exact baseline required per operator")
        baseline=bases[0]
        defaults=baseline["requested"]
        bmetrics=baseline.get("qualified_metrics")
        one={x["case"]:x for x in study["single_factor_contrasts"]}
        pairs={x["case"]:x for x in study["pairwise_or_higher_order_contrasts"]}
        blocked=study.get("infeasible_architectures",[])
        all_rows=[(x,False) for x in cases]+[(x,True) for x in blocked]
        for item,is_blocked in all_rows:
            cid=item["case"]
            request=item.get("requested")
            changed=_changes(request,defaults)
            axis=item.get("axis")
            if axis=="baseline":
                if changed:
                    raise ValueError("Malformed experimental baseline")
                continue
            common={"study":name,"case":cid,"axis":axis,
                    "source_blocked":is_blocked,
                    "requested_value":json.dumps(
                        request,sort_keys=True,separators=(",",":")),
                    "size_category":geometry.get(name,{}).get("size"),
                    "operator_kind":geometry.get(name,{}).get("kind"),
                    "dimension_changed_count":len(changed)}
            if axis in AXES:
                if changed!=[axis]:
                    raise ValueError("Malformed single-axis contrast")
                effect=one.get(cid)
                if not is_blocked and effect is None:
                    raise ValueError("Missing original single-factor observation")
                metrics=None if is_blocked else item.get("qualified_metrics")
                qualified=(not is_blocked and effect.get("qualified") is True
                           and isinstance(metrics,dict)
                           and isinstance(bmetrics,dict))
                if not is_blocked and qualified!=(item["qualification"]==
                    "QUALIFIED_ROUTED" and isinstance(bmetrics,dict)):
                    raise ValueError("Main-effect qualification inconsistent")
                source_same=None if is_blocked else (
                    baseline.get("source_sha256")==item.get("source_sha256"))
                record={
                    **common,"knob":axis,
                    "knob_level":json.dumps(request[axis],sort_keys=True),
                    "intervention_scope":_scope(axis),
                    "status":("BLOCKED_SOURCE" if is_blocked else
                              "QUALIFIED_COMPARISON" if qualified else
                              "UNQUALIFIED_COMPARISON"),
                    "same_generated_cpp_sha256_as_baseline":source_same,
                    "physical_observables_equal_to_baseline":(
                        _cost_tie(bmetrics,metrics) if qualified else None),
                    "delta":effect.get("delta_b_minus_a") if qualified else None,
                    "blockers":item.get("blockers",[]) if not is_blocked else
                        [item.get("reason")],
                }
                if qualified and not isinstance(record["delta"],dict):
                    raise ValueError("Qualified effect missing differences")
                main.append(record)
            elif axis=="interaction":
                if len(changed)<2:
                    raise ValueError("Interaction needs multiple changed knobs")
                effect=pairs.get(cid)
                if not is_blocked and effect is None:
                    raise ValueError("Missing original interaction observation")
                status=("BLOCKED_SOURCE" if is_blocked else effect.get("status"))
                record={
                    **common,"knobs":changed,
                    "status":status,
                    "contains_model_global_precision":("precision_bits" in changed),
                    "second_difference":(
                        effect.get("pairwise_second_difference")
                        if not is_blocked else None),
                }
                if len(changed)==2:
                    pair.append(record)
                else:
                    high_order.append(record)
            else:
                raise ValueError("Unknown non-baseline intervention axis")
        for knob in AXES:
            rows=[r for r in main if r["study"]==name and r["knob"]==knob]
            axis_coverage.append({
                "study":name,"knob":knob,"intervention_scope":_scope(knob),
                "declared_levels":len(rows),
                "source_blocked":sum(x["status"]=="BLOCKED_SOURCE" for x in rows),
                "qualified_comparisons":sum(x["status"]=="QUALIFIED_COMPARISON" for x in rows),
                "physical_observational_ties":sum(
                    x["physical_observables_equal_to_baseline"] is True
                    for x in rows),
            })
        study_totals.append({
            "study":name,"size_category":geometry.get(name,{}).get("size"),
            "kind":geometry.get(name,{}).get("kind"),
            "qualified_routed_case_count":study["routed_qualified_case_count"],
            "source_blocked_case_count":len(blocked),
            "main_comparisons_qualified":sum(x["study"]==name and
                x["status"]=="QUALIFIED_COMPARISON" for x in main),
            "pairwise_comparisons_qualified":sum(x["study"]==name and
                x["status"]=="QUALIFIED_FACTORIAL_CONTRAST" for x in pair),
        })
    return {
        "schema":SCHEMA,"source_evidence_reaudited_by_this_report":False,
        "vendor_tools_executed":False,
        "source_effects_schema":EFFECTS_SCHEMA,
        "source_geometry_synthetic":geometry_receipt is not None,
        "studies":study_totals,"axis_coverage":axis_coverage,
        "main_effects":main,"pairwise_effects":pair,
        "higher_order_requests":high_order,
        "scientific_limitations":(
            "Requested knob levels and geometry do not guarantee a hardware "
            "change or an achievable II. Only qualified comparisons have "
            "numeric effects; unqualified/blocked comparisons remain null. "
            "Equal routed observables do not prove identical hardware. "
            "Global homogeneous precision is not an operator-local knob. "
            "No board-measured inference time, power, or hls4ml superiority."),
    }



def _signed_effect(value):
    """Preserve genuine zero, and never turn missing evidence into zero."""
    if value is None:
        return "—"
    if not _valid_number(value):
        raise ValueError("Nonfinite native QoR contrast cannot be published")
    if value == 0:
        return "0"
    return f"{value:+g}"


def _write_numerical_markdown(stage, report):
    """Complete numeric tables, not just winner-selection summaries."""
    metrics=("hls_estimated_cycles","postroute_lut","postroute_ff",
             "postroute_dsp","postroute_bram","postroute_uram",
             "postroute_wns_ns")
    headers=("Δ HLS cycles","Δ LUT","Δ FF","Δ DSP","Δ BRAM",
             "Δ URAM","Δ WNS (ns)")
    lines=[
        "# All main effects: numeric routed observations by knob level","",
        "Each signed value = intervention minus matched baseline within "
        "the SAME source-generator revision, model, target and clock. "
        "HLS cycles are estimated, not board latency. WNS is a slack "
        "difference in nanoseconds, not a percentage. "
        "Missing values are shown as —, NEVER zero.","",
        "Precision changes the homogeneous model-global numerical format; "
        "all other knobs are selected-operator mapping settings. "
        "Physical observables alone do not establish identical RTL.","",
    ]
    for item in report["studies"]:
        name=item["study"]
        lines.extend([
            "## "+name,"",
            "| Case | Knob | Requested level | Qualification | "+
            " | ".join(headers)+" |",
            "|---|---|---|---|"+"|".join("---:" for _ in metrics)+"|",
        ])
        for row in report["main_effects"]:
            if row["study"]!=name:
                continue
            delta=row["delta"] or {}
            values=" | ".join(_signed_effect(delta.get(m)) for m in metrics)
            level=str(row["knob_level"]).replace("|","\\|")
            lines.append(
                f"| {row['case']} | {row['knob']} | {level} | "
                f"{row['status']} | {values} |")
        lines.extend(["",""])
    lines.extend(["**Interpretation:** Zero means an observed equality "
        "of the reported quantity, not proof the requested knob was "
        "optimized out. Resource and slack changes can be tradeoffs.",""])
    (stage/"main_effects.md").write_text("\n".join(lines),encoding="utf-8")

    cols=("hls_estimated_cycles","postroute_lut",
          "postroute_dsp","postroute_wns_ns")
    pairlines=[
        "# All pairwise interactions: qualified second differences","",
        "For each qualified factorial quadruple, "
        "Δ₂ = Y(AB) − Y(A) − Y(B) + Y(baseline). "
        "A missing quadruple has NO numerical second difference. "
        "These are not optimization gain rankings.","",
    ]
    for item in report["studies"]:
        name=item["study"]
        pairlines.extend([
            "## "+name,"",
            "| Case | Axes | Qualification | Δ₂ HLS cycles | "+
            "Δ₂ LUT | Δ₂ DSP | Δ₂ WNS (ns) |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for row in report["pairwise_effects"]:
            if row["study"]!=name:
                continue
            delta=row["second_difference"] or {}
            pairlines.append(
                f"| {row['case']} | {' + '.join(row['knobs'])} | "
                f"{row['status']} | "+
                " | ".join(_signed_effect(delta.get(key)) for key in cols)+
                " |")
        pairlines.extend(["",""])
    pairlines.extend(["Higher-order requests are not included in this "
                     "pairwise table; see complete_knob_effects.json.",""])
    (stage/"pairwise_effects.md").write_text(
        "\n".join(pairlines),encoding="utf-8")


def export_axis_matrix(effects_path,destination,*,geometry_receipt=None):
    source=Path(effects_path)
    output=Path(destination)
    if source.is_symlink() or not source.is_file() or not 0<source.stat().st_size<=32*1024*1024:
        raise ValueError("Expected regular bounded audited effects file")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use fresh complete-effects output directory")
    geometry=None
    if geometry_receipt is not None:
        receipt=Path(geometry_receipt)
        if receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size>4*1024*1024:
            raise ValueError("Missing regular synthetic geometry receipt")
        geometry=json.loads(receipt.read_text())
    data=source.read_bytes()
    report=analyze_axis_matrix(json.loads(data),geometry)
    report["source_effects_sha256"]=sha256(data).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output.parent,prefix=".complete-knob-") as t:
        stage=Path(t)/"matrix"
        stage.mkdir()
        (stage/"complete_knob_effects.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
        def rows_csv(filename,rows,cols):
            with (stage/filename).open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=cols)
                writer.writeheader()
                for item in rows:
                    writer.writerow({k:json.dumps(item.get(k),sort_keys=True,
                                    allow_nan=False) if isinstance(item.get(k),(dict,list))
                                    else item.get(k) for k in cols})
        summary_fields=["study","knob","intervention_scope","declared_levels",
                        "source_blocked","qualified_comparisons",
                        "physical_observational_ties"]
        rows_csv("seven_axis_coverage.csv",report["axis_coverage"],summary_fields)
        main_fields=["study","case","size_category","operator_kind","knob",
                     "knob_level","intervention_scope","status",
                     "same_generated_cpp_sha256_as_baseline",
                     "physical_observables_equal_to_baseline",
                     *OUTPUTS]
        pairs_fields=["study","case","size_category","operator_kind","knobs",
                      "status","contains_model_global_precision",*OUTPUTS]
        rows_csv("all_main_effects.csv",[
            {**row,**(row["delta"] or {})} for row in report["main_effects"]],
            main_fields)
        rows_csv("all_pairwise_interactions.csv",[
            {**row,**(row["second_difference"] or {})}
            for row in report["pairwise_effects"]],pairs_fields)
        lines=["# All seven knobs: qualified, blocked and absent effects","",
            "These are source-audited existing results, summarized only; "
            "blank numerical effects are deliberately missing, not zero.","",
            "| Study | Knob | Levels requested | Qualified effects | Source blocked | Physical observables tied |",
            "|---|---|---:|---:|---:|---:|"]
        for r in report["axis_coverage"]:
            lines.append(f"| {r['study']} | {r['knob']} | "
                         f"{r['declared_levels']} | "
                         f"{r['qualified_comparisons']} | "
                         f"{r['source_blocked']} | "
                         f"{r['physical_observational_ties']} |")
        lines.extend(["",
            "## Interaction coverage","",
            "| Study | Declared pairwise | Qualified pairwise | Source-blocked pairs |",
            "|---|---:|---:|---:|"])
        for item in report["studies"]:
            related=[row for row in report["pairwise_effects"]
                     if row["study"]==item["study"]]
            lines.append(f"| {item['study']} | {len(related)} | "
                         f"{item['pairwise_comparisons_qualified']} | "
                         f"{sum(x['status']=='BLOCKED_SOURCE' for x in related)} |")
        lines.extend(["",
            "## Complete numerical effects","",
            "Every requested knob level, including zero observed differences "
            "and unqualified cases, appears in main_effects.md. "
            "Every requested pair, with signed second differences only for "
            "qualified factorial quadruples, appears in pairwise_effects.md. "
            "Both tables also have CSV equivalents.","",
            "Model-global precision is not an independently replicated "
            "operator-local effect; equality of post-route metric values "
            "does not imply bit-identical netlists.",
            "No unqualified point is assigned a fabricated improvement.",
            ""])
        _write_numerical_markdown(stage,report)
        (stage/"summary.md").write_text("\n".join(lines))
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output appeared during immutable write")
        stage.rename(output)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--effects",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--geometry-receipt",
                   help="Optional synthetic geometry source_receipt.json")
    args=p.parse_args(argv)
    result=export_axis_matrix(args.effects,args.output,
                              geometry_receipt=args.geometry_receipt)
    print(json.dumps({
        "study_count":len(result["studies"]),
        "main_effects_requested":len(result["main_effects"]),
        "main_effects_qualified":sum(r["status"]=="QUALIFIED_COMPARISON"
                                    for r in result["main_effects"]),
        "pairwise_effects_requested":len(result["pairwise_effects"]),
        "pairwise_effects_qualified":sum(r["status"]=="QUALIFIED_FACTORIAL_CONTRAST"
                                        for r in result["pairwise_effects"]),
        "vendor_tools_executed":False,
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
