"""Reaudit and quantify operator-local FPGA architecture interventions.

Differences are observable within one fixed functional model (synthetic or
trained) and physical target, not proof of general superiority. Pairwise interactions use the
standard factorial second difference Y(ab)-Y(a)-Y(b)+Y(baseline).
Only independently qualified routed circuits enter numeric effects.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.suite_native_batch import _qualify

SCHEMA="fpgai-research/native-seven-axis-controlled-effects/v1"
OUTPUTS=("hls_estimated_cycles","postroute_lut","postroute_ff",
         "postroute_dsp","postroute_bram","postroute_uram","postroute_wns_ns")
AXES=("precision_bits","parallelism","pes","simd","serialization","tile","pipeline_ii")


def _value(v):
    if type(v) not in (int,float) or not isfinite(v):
        return None
    return float(v)


def _requested_signature(request):
    if not isinstance(request,dict) or set(request) != {
            "precision_bits","integer_bits","parallelism","pes","simd",
            "serialization","tile","pipeline_ii"}:
        raise ValueError("Missing full architecture request")
    return json.dumps(request,sort_keys=True,separators=(",",":"),allow_nan=False)


def _row(case, observation):
    hls=observation.get("metrics") or {}
    physical=observation.get("vivado_metrics") or {}
    util=physical.get("utilization") or {}
    timing=physical.get("timing") or {}
    qualified=not _qualify(observation)
    metric={
        "hls_estimated_cycles":_value(
            int(hls["latency_best"])
            if isinstance(hls.get("latency_best"),str)
               and hls["latency_best"].isdigit() else hls.get("latency_best")),
        "postroute_lut":_value(util.get("lut")),
        "postroute_ff":_value(util.get("ff")),
        "postroute_dsp":_value(util.get("dsp")),
        "postroute_bram":_value(util.get("bram")),
        "postroute_uram":_value(util.get("uram")),
        "postroute_wns_ns":_value(timing.get("wns_ns")),
    }
    # Never report a deltas when ANY required physical/numeric proof fails.
    return {"case":case["case"],"axis":case["dimension"],
            "requested":case["requested"],"source_sha256":observation["source_sha256"],
            "native_evidence":observation.get("evidence"),
            "source_warnings":observation.get("warnings",[]),
            "qualification":"QUALIFIED_ROUTED" if qualified else "UNQUALIFIED",
            "blockers":_qualify(observation),
            "observed":metric,
            "qualified_metrics":metric if qualified else None}


def _delta(a,b):
    """Difference b - a for every metric, or missing if not qualified."""
    if a is None or b is None:
        return None
    return {key:round(b[key]-a[key],9)
            if a.get(key) is not None and b.get(key) is not None else None
            for key in OUTPUTS}


def calculate_study_effects(rows):
    if not isinstance(rows,list) or not rows:
        raise ValueError("At least one source-matched architecture case needed")
    base=[r for r in rows if r["axis"]=="baseline"]
    if len(base)!=1:
        raise ValueError("Exactly one same-network baseline required")
    baseline=base[0]
    sigs=set()
    for row in rows:
        s=_requested_signature(row["requested"])
        if s in sigs:
            raise ValueError("Repeated architecture setting contaminates the experiment")
        sigs.add(s)
    defaults=baseline["requested"]
    index={_requested_signature(row["requested"]):row for row in rows}
    main=[]
    interactions=[]
    for row in rows:
        if row["case"]==baseline["case"]:
            continue
        variations={key:row["requested"][key] for key in AXES
                    if row["requested"][key]!=defaults[key]}
        if row["axis"] in AXES:
            if len(variations)!=1 or row["axis"] not in variations:
                raise ValueError("Misidentified single-factor architecture contrast")
            main.append({
                "case":row["case"],"axis":row["axis"],
                "requested_value":variations[row["axis"]],
                "baseline_case":baseline["case"],
                "qualified":bool(row["qualified_metrics"] and baseline["qualified_metrics"]),
                "delta_b_minus_a":_delta(baseline["qualified_metrics"],row["qualified_metrics"]),
            })
            continue
        if row["axis"]!="interaction" or len(variations)<2:
            raise ValueError("Malformed joint architecture intervention")
        if len(variations)!=2:
            interactions.append({
                "case":row["case"],"axes":list(variations),
                "status":"HIGHER_ORDER_NOT_IDENTIFIED_BY_PAIRWISE_EFFECT",
                "pairwise_second_difference":None,
            })
            continue
        first,second=variations
        first_request={**defaults,first:variations[first]}
        second_request={**defaults,second:variations[second]}
        left=index.get(_requested_signature(first_request))
        right=index.get(_requested_signature(second_request))
        quad=(baseline,left,right,row)
        qualified=all(point is not None and point["qualified_metrics"] is not None
                      for point in quad)
        second_diff=None
        if qualified:
            m=[point["qualified_metrics"] for point in quad]
            second_diff={
                metric:round(m[3][metric]-m[1][metric]-m[2][metric]+m[0][metric],9)
                if all(x.get(metric) is not None for x in m) else None
                for metric in OUTPUTS
            }
        interactions.append({
            "case":row["case"],"axes":[first,second],
            "matched_single_case_ids":[left["case"] if left else None,
                                       right["case"] if right else None],
            "status":"QUALIFIED_FACTORIAL_CONTRAST" if qualified else "MISSING_NATIVE_QUADRUPLE",
            "pairwise_second_difference":second_diff,
        })
    return {
        "baseline_case":baseline["case"],
        "baseline_routed_qualified":baseline["qualified_metrics"] is not None,
        "single_factor_contrasts":main,
        "pairwise_or_higher_order_contrasts":interactions,
        "qualified_single_factor_count":sum(x["qualified"] for x in main),
        "qualified_pairwise_count":sum(
            x["status"]=="QUALIFIED_FACTORIAL_CONTRAST" for x in interactions),
    }


def analyze_native_seven_axes(directory):
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing prepared native research suite")
    audit=audit_suite(root,require_host=True,require_all_axes=False)
    manifest=root/"suite_results.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Missing immutable source-suite declaration")
    declared=json.loads(manifest.read_text())
    studies=[]
    for item in declared["studies"]:
        if item.get("status")!="E1-PREPARED":
            studies.append({"study":item["id"],"status":"BLOCKED_SOURCE_MODEL",
                            "reason":item.get("reason")})
            continue
        name=item["id"]
        path=root/name/"campaign.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing immutable operator campaign")
        campaign=json.loads(path.read_text())
        records=[]
        excluded=[]
        for entry in campaign["cases"]:
            if entry["status"]=="BLOCKED":
                excluded.append({"case":entry["case"],"axis":entry["dimension"],
                                 "requested":entry["requested"],
                                 "reason":entry.get("reason")})
                continue
            case=root/name/entry["case"]
            observation=summarize_case(str(case))
            if observation.get("source_sha256")!=entry.get("cpp_sha256"):
                raise ValueError("Native observation source differs from architecture contract")
            records.append(_row(entry,observation))
        if not records:
            raise ValueError("No eligible circuit for "+name)
        effects=calculate_study_effects(records)
        studies.append({
            "study":name,
            "status":"QUALIFIED_WITHIN_MODEL" if all(
                x["qualified_metrics"] for x in records) and not excluded
                else "PARTIAL_OR_FEASIBILITY_EXCLUSIONS",
            "campaign_sha256":sha256(path.read_bytes()).hexdigest(),
            "requested_case_count":len(campaign["cases"]),
            "eligible_case_count":len(records),
            "routed_qualified_case_count":sum(
                row["qualified_metrics"] is not None for row in records),
            "infeasible_architectures":excluded,
            "cases":records,
            **effects
        })
    return {
        "schema":SCHEMA,
        "source_suite_sha256":sha256(manifest.read_bytes()).hexdigest(),
        "source_audit_schema":audit["schema"],
        "vendor_tools_executed_by_analysis":False,
        "studies":studies,
        "qualification_warning":"A numeric contrast requires matching source-attested fixed-point C simulation, HLS synthesis and post-route timing/DRC; unqualified data remain null.",
        "interpretation":"HLS cycles are tool estimates, not measured inference time. Joint second differences are finite-sample within-model interactions, not cross-compiler superiority or hardware measurements taken here.",
    }


def export_native_seven_axes(directory,output):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Scientific effects output must be a fresh directory")
    summary=analyze_native_seven_axes(directory)
    rows=[]
    for model in summary["studies"]:
        for case in model.get("cases",[]):
            line={"study":model["study"],"case":case["case"],
                  "axis":case["axis"],"qualification":case["qualification"],
                  "source_sha256":case["source_sha256"]}
            line.update({k:case["observed"].get(k) for k in OUTPUTS})
            rows.append(line)
    root.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".native-effects-",dir=root.parent) as tmp:
        stage=Path(tmp)/"effects"
        stage.mkdir()
        (stage/"effects.json").write_text(json.dumps(summary,sort_keys=True,
                    indent=2,allow_nan=False)+"\n")
        with (stage/"case_observations.csv").open("w",newline="",encoding="utf-8") as f:
            cols=["study","case","axis","qualification","source_sha256",*OUTPUTS]
            writer=csv.DictWriter(f,fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        if root.exists() or root.is_symlink():
            raise FileExistsError("Effects output appeared during export")
        stage.rename(root)
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args(argv)
    result=export_native_seven_axes(args.directory,args.output)
    print(json.dumps({"schema":result["schema"],
       "study_count":len(result["studies"]),
       "qualified_main_effects":sum(s.get("qualified_single_factor_count",0)
                                    for s in result["studies"]),
       "qualified_pairwise_effects":sum(s.get("qualified_pairwise_count",0)
                                        for s in result["studies"]),
       "tools_executed":False},sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
