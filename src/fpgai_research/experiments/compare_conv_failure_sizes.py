"""Read-only, source-audited comparison of medium/large Conv case_0035.

Preserve each native CSim/HLS/Vivado failure without recomputing
unqualified factorial effects or rerunning vendor tools.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.geometry_native_failure_forensics import inspect_failure

SCHEMA="fpgai-research/cross-size-conv-negative-e4/v1"
CASE="case_0035"


def analyze(medium,large):
    expected=(("medium","probe_conv_12",medium),
              ("large","probe_conv_16",large))
    result={}
    for size,name,r in expected:
        if (r.get("schema")!="fpgai-research/geometry-native-failure-forensics/v1"
                or r.get("study")!=name or r.get("case")!=CASE
                or r.get("vendor_tools_executed") is not False):
            raise ValueError("Expected original source-audited Conv geometry case")
        if (r.get("csim_passed") is not True
                or r.get("hls_synthesis_qualified") is not True
                or r.get("e4_routed_qualified") is not False):
            raise ValueError("Expected E2/E3-accepted, E4-unqualified case")
        req=r.get("requested_architecture")
        if not isinstance(req,dict):
            raise ValueError("Missing requested native architecture")
        source=r.get("source_sha256")
        if (not isinstance(source,str) or len(source)!=64 or
                any(char not in "0123456789abcdef" for char in source)):
            raise ValueError("Missing source identity")
        timing=r.get("postroute_timing")
        if not isinstance(timing,dict) or type(timing.get("wns_ns")) not in (int,float):
            raise ValueError("Missing postroute timing")
        from math import isfinite
        if not isfinite(timing["wns_ns"]):
            raise ValueError("Nonfinite routed WNS")
        if not r.get("physical_blockers"):
            raise ValueError("Original E4 blockers must be explicit")
        result[size]={
            "study":name,"case":CASE,"source_sha256":source,
            "requested_architecture":req,
            "original_hls_metrics":r.get("hls_metrics"),
            "routed_wns_ns":timing["wns_ns"],
            "routed_timing_met":timing.get("timing_met"),
            "routed_utilization":r.get("postroute_utilization"),
            "drc":r.get("postroute_drc"),
            "e4_blockers":r["physical_blockers"],
            "warnings":r.get("source_warnings",[]),
            "e4_qualified":False,
            "suite_sha256":r.get("source_suite_sha256"),
            "campaign_sha256":r.get("campaign_sha256"),
        }
    comparable_keys=("precision_bits","integer_bits","parallelism","pes",
                     "simd","serialization","pipeline_ii")
    if any(result["medium"]["requested_architecture"].get(k)!=
           result["large"]["requested_architecture"].get(k)
           for k in comparable_keys):
        raise ValueError("Non-geometry architecture requests do not match")
    return {
        "schema":SCHEMA,"vendor_tools_executed":False,
        "same_non_geometry_knobs":True,"observations":result,
        "limitations":(
            "These are source-bound, read-only physical observations, "
            "NOT qualified E4 effects; no factorial second differences, "
            "speedup ratios, root-cause equivalence or board claims."),
    }


def export(medium_dir,large_dir,output):
    if Path(medium_dir).resolve()==Path(large_dir).resolve():
        raise ValueError("Independent prepared sizes required")
    dst=Path(output)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError("Choose a new immutable diagnosis directory")
    rec=analyze(
        inspect_failure(medium_dir,study="probe_conv_12",case=CASE),
        inspect_failure(large_dir,study="probe_conv_16",case=CASE))
    dst.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dst.parent,prefix=".conv35-") as work:
        stage=Path(work)/"report"
        stage.mkdir()
        (stage/"comparison.json").write_text(
            json.dumps(rec,sort_keys=True,indent=2,allow_nan=False)+"\n")
        lines=[
            "# Conv SIMD–pipeline interaction failure at two sizes","",
            "Source-reattested original E4 failures. Never treat either "
            "absolute QoR report as qualified or infer a root cause.","",
            "| Size | WNS ns | Timing met | LUT | FF | DSP | DRC warnings | E4 blockers |",
            "|---|---:|---|---:|---:|---:|---:|---|"]
        for size,row in rec["observations"].items():
            util=row["routed_utilization"] or {}
            drc=row["drc"] or {}
            lines.append(
                f"| {size} | {row['routed_wns_ns']:+g} | "
                f"{row['routed_timing_met']} | {util.get('lut','—')} | "
                f"{util.get('ff','—')} | {util.get('dsp','—')} | "
                f"{drc.get('violations','—')} | "
                +"; ".join(row["e4_blockers"]).replace("|","/")+" |")
        lines.extend(["",
            "Each value is a separate original native measurement. "
            "Negative/unqualified E4 is preserved and omitted from "
            "factorial comparisons.",""])
        (stage/"summary.md").write_text("\n".join(lines))
        if dst.exists() or dst.is_symlink():
            raise FileExistsError("Output appeared during read-only export")
        stage.rename(dst)
    return rec


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--medium-directory",required=True)
    p.add_argument("--large-directory",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    rec=export(args.medium_directory,args.large_directory,args.output)
    print(json.dumps({
        "schema":SCHEMA,"vendor_tools_executed":False,
        "observed_wns_ns":{k:v["routed_wns_ns"]
                           for k,v in rec["observations"].items()},
    },sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
