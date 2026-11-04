"""Read-only relative resource-expansion advisory from source-audited E4 QoR.

Thresholds are descriptive *relative to an independently qualified
same-study baseline*. They are not FPGA device limits, route feasibility
predictions, HLS scheduling proof, or grounds for changing preregistered
experiment outcomes. Never run licensed tools in this module.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_seven_axis_effects import (
    SCHEMA as EFFECTS_SCHEMA, OUTPUTS,
)

SCHEMA="fpgai-research/source-audited-relative-resource-advisory/v1"
RESOURCES=("postroute_lut","postroute_ff","postroute_dsp",
           "postroute_bram","postroute_uram")


def _finite(v):
    return type(v) in (int,float) and isfinite(v)


def analyze_resource_advisory(effects,*,study,ratio=8.0):
    if (not isinstance(effects,dict) or effects.get("schema")!=EFFECTS_SCHEMA
            or effects.get("vendor_tools_executed_by_analysis") is not False
            or not isinstance(effects.get("studies"),list)):
        raise ValueError("Expected source-audited native effects.json")
    if not (_finite(ratio) and ratio>1):
        raise ValueError("Relative resource threshold must exceed one")
    matching=[x for x in effects["studies"]
              if isinstance(x,dict) and x.get("study")==study]
    if len(matching)!=1:
        raise ValueError("Expected exactly one named prepared operator study")
    entries=matching[0].get("cases")
    if not isinstance(entries,list) or not entries:
        raise ValueError("Missing native case inventory")
    cases=[c.get("case") for c in entries if isinstance(c,dict)]
    if len(cases)!=len(entries) or len(set(cases))!=len(cases):
        raise ValueError("Duplicate/invalid native case identities")
    baselines=[c for c in entries if c.get("axis")=="baseline"]
    if len(baselines)!=1:
        raise ValueError("Expected one independently qualified baseline")
    def vector(c):
        if c.get("qualification")!="QUALIFIED_ROUTED":
            return None
        values=c.get("qualified_metrics")
        if not isinstance(values,dict) or set(values)!=set(OUTPUTS):
            raise ValueError("Missing complete qualified native measurements")
        if any(not _finite(values[k]) for k in OUTPUTS):
            raise ValueError("Nonfinite qualified native measurements")
        if (not isinstance(c.get("source_sha256"),str) or
                len(c["source_sha256"])!=64):
            raise ValueError("Missing source identity on native observation")
        return values
    base=vector(baselines[0])
    if base is None:
        raise ValueError("Baseline must be source-qualified E4")
    flagged=[]
    qualified=0
    skipped=[]
    for c in entries:
        measured=vector(c)
        if measured is None:
            skipped.append(c["case"])
            continue
        qualified+=1
        if c["axis"]=="baseline":
            continue
        alerts=[]
        for name in RESOURCES:
            x0,x1=base[name],measured[name]
            if x0<0 or x1<0:
                raise ValueError("Resources cannot be negative")
            if x0==0:
                if x1>0:
                    alerts.append({
                        "resource":name,"baseline_zero":True,
                        "absolute_routed":x1,"ratio_to_baseline":None,
                        "reason":"positive_usage_from_zero_baseline",
                    })
                continue
            r=x1/x0
            if r>=ratio:
                alerts.append({
                    "resource":name,"baseline_zero":False,
                    "absolute_routed":x1,"ratio_to_baseline":round(r,6),
                    "reason":"large_relative_increase",
                })
        if alerts:
            flagged.append({
                "case":c["case"],"axis":c["axis"],
                "source_sha256":c["source_sha256"],
                "absolute_routed_qor":dict(measured),
                "baseline_qor":dict(base),
                "alerts":alerts,
                "hls_estimated_cycle_delta":round(
                    measured["hls_estimated_cycles"]-
                    base["hls_estimated_cycles"],9),
                "postroute_wns_ns":measured["postroute_wns_ns"],
            })
    return {
        "schema":SCHEMA,
        "study":study,
        "threshold_ratio":ratio,
        "source_evidence_reaudited_by_analysis":False,
        "vendor_tools_executed":False,
        "device_capacity_checked":False,
        "total_case_count":len(entries),
        "e4_qualified_case_count":qualified,
        "unqualified_cases":skipped,
        "flagged_cases":flagged,
        "limitations":(
            "The advisory compares observed post-route resource usage against "
            "a same-study E4-qualified baseline. Thresholds are operator-local "
            "descriptive heuristics, NOT target FPGA capacity checks. "
            "A zero baseline is never divided by zero. Source/RTL attestation "
            "was performed upstream, not anew here. This report cannot "
            "predict future geometry QoR, prove achieved II, or replace "
            "actual Vitis/Vivado and numerical checks."),
    }


def export_resource_advisory(effects_path,output,*,study,ratio=8.0):
    src=Path(effects_path)
    target=Path(output)
    if (src.is_symlink() or not src.is_file()
            or not 0<src.stat().st_size<=32*1024*1024):
        raise ValueError("Require bounded source-audited effects.json")
    if target.exists() or target.is_symlink():
        raise FileExistsError("Relative resource output must be fresh")
    raw=src.read_bytes()
    result=analyze_resource_advisory(json.loads(raw),study=study,ratio=ratio)
    result["source_effects_sha256"]=sha256(raw).hexdigest()
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=target.parent,prefix=".resource-advisory-") as d:
        folder=Path(d)/"report"
        folder.mkdir()
        (folder/"advisory.json").write_text(json.dumps(
            result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=[
            "# Routed resource expansion advisory: "+study,"",
            "This is a read-only descriptive review of E4-qualified "
            "native results, NOT a device resource feasibility test.",
            "",
            f"Threshold: at least {ratio:g}× the corresponding same-study "
            "baseline post-route LUT/FF/DSP/BRAM/URAM count; "
            "positive resources from a zero baseline are flagged separately.",
            "",
            f"Qualified: {result['e4_qualified_case_count']} / "
            f"{result['total_case_count']}; flagged: "
            f"{len(result['flagged_cases'])}.","",
            "| Case | Resource | Ratio vs baseline | Absolute routed | Δ HLS cycles | WNS ns |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for case in result["flagged_cases"]:
            for alert in case["alerts"]:
                ratio_string=("zero baseline" if alert["baseline_zero"]
                              else f"{alert['ratio_to_baseline']:g}×")
                lines.append(
                    f"| {case['case']} | {alert['resource']} | "
                    f"{ratio_string} | {alert['absolute_routed']:g} | "
                    f"{case['hls_estimated_cycle_delta']:+g} | "
                    f"{case['postroute_wns_ns']:g} |")
        if not result["flagged_cases"]:
            lines.append("| None observed | — | — | — | — | — |")
        lines.extend(["","This summary contains no synthesis prediction "
                      "or acceptance/rejection recommendation. Inspect "
                      "Vitis schedule, RTL source, routed utilization, "
                      "constraints and original logs before explaining "
                      "large architectural differences.",""])
        (folder/"summary.md").write_text("\n".join(lines))
        if target.exists() or target.is_symlink():
            raise FileExistsError("Output created by another process")
        folder.rename(target)
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--effects",required=True)
    ap.add_argument("--study",required=True)
    ap.add_argument("--ratio",type=float,default=8.0)
    ap.add_argument("--output",required=True)
    args=ap.parse_args(argv)
    report=export_resource_advisory(args.effects,args.output,
                                    study=args.study,ratio=args.ratio)
    print(json.dumps({
        "schema":SCHEMA,"study":report["study"],
        "qualified_cases":report["e4_qualified_case_count"],
        "flagged_cases":len(report["flagged_cases"]),
        "vendor_tools_executed":False,
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
