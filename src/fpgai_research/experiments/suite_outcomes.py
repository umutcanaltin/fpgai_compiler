"""Paper-safe outcome analysis from an attested within-model comparison snapshot.

E4 routing is not proof of numerical acceptance or a passing timing
constraint. Failed-but-measured designs retain their *observed* metrics
without being promoted to qualified architectural deltas.
"""
from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path

SCHEMA = "fpgai-research/suite-architecture-outcomes/v1"
SOURCE_SCHEMA = "fpgai-research/full-paper-suite-comparison/v1"
COLUMNS = (
    "model", "case", "axis", "precision_bits", "evidence",
    "outcome", "qualified", "blocker", "numerical_pass",
    "observed_hls_latency_cycles", "baseline_hls_latency_cycles",
    "qualified_latency_delta_cycles", "parallelism_latency_observation",
    "observed_lut", "baseline_lut", "qualified_lut_delta",
    "observed_wns_ns", "postroute_timing_met", "timing_deficit_ns",
)


def _finite(value):
    return type(value) in (int,float) and isfinite(value)


def summarize_architecture_outcomes(comparison: dict) -> dict:
    """Classify original cases, never assigning a pass to incomplete E1/E3.

    Qualified deltas are copied from the source comparator, which already
    checks the same-model E4/numerical/physical identity. No new
    measurement or unsupported cycle-to-throughput inference is made.
    """
    if comparison.get("schema") != SOURCE_SCHEMA:
        raise ValueError("A collected source-attested comparison is required")
    entries=comparison.get("rows")
    if not isinstance(entries,list) or not entries or len(entries)!=comparison.get("total_cases"):
        raise ValueError("Inconsistent source architecture case inventory")
    out=[]
    identities=set()
    for item in entries:
        if not isinstance(item,dict):
            raise ValueError("Malformed compared architecture case")
        key=(item.get("model"),item.get("case"))
        if (not all(isinstance(v,str) and v for v in key)
                or key in identities):
            raise ValueError("Missing or duplicate architecture case identity")
        identities.add(key)
        evidence=item.get("evidence")
        qualified=item.get("comparison_status")=="QUALIFIED"
        if qualified and (evidence!="E4-Vivado-implementation"
                          or item.get("numerical_pass") is not True
                          or item.get("postroute_timing_met") is not True
                          or not _finite(item.get("observed_wns_ns"))
                          or item["observed_wns_ns"] < 0):
            raise ValueError("Qualified outcome contradicts native physical/numerical evidence")
        if evidence=="E4-Vivado-implementation":
            if item.get("numerical_pass") is not True:
                status="E4_NUMERICAL_REJECTED"
            elif item.get("postroute_timing_met") is not True:
                status="E4_TIMING_REJECTED"
            elif qualified:
                status="E4_QUALIFIED"
            else:
                status="E4_OTHER_REJECTED"
        elif evidence in ("E3-HLS-synthesis-report","E2-Vitis-C-simulation"):
            status="NATIVE_NOT_ROUTED"
        else:
            status="SOURCE_ONLY"
        wns=item.get("observed_wns_ns") if evidence=="E4-Vivado-implementation" else None
        if wns is not None and not _finite(wns):
            raise ValueError("Invalid post-route WNS")
        deficit=round(-wns,9) if wns is not None and wns<0 else None
        latency_delta=item.get("delta_hls_latency_cycles") if qualified else None
        lut_delta=item.get("delta_lut") if qualified else None
        if qualified and (not _finite(latency_delta) or not _finite(lut_delta)):
            raise ValueError("Qualified case is missing source-attested latency/LUT deltas")
        observation=None
        if item.get("axis")=="parallelism" and qualified:
            observation=("HLS_ESTIMATE_INCREASED" if latency_delta>0 else
                         "HLS_ESTIMATE_DECREASED" if latency_delta<0 else
                         "HLS_ESTIMATE_UNCHANGED")
        out.append({
            "model":key[0], "case":key[1],"axis":item.get("axis"),
            "precision_bits":item.get("precision_bits"),
            "evidence":evidence,"outcome":status,"qualified":qualified,
            "blocker":item.get("blocker"),
            "numerical_pass":item.get("numerical_pass"),
            "observed_hls_latency_cycles":item.get("observed_hls_latency_cycles"),
            "baseline_hls_latency_cycles":item.get("baseline_hls_latency_cycles")
                if qualified else None,
            "qualified_latency_delta_cycles":latency_delta,
            "parallelism_latency_observation":observation,
            "observed_lut":item.get("observed_lut") if evidence=="E4-Vivado-implementation" else None,
            "baseline_lut":item.get("baseline_lut") if qualified else None,
            "qualified_lut_delta":lut_delta,
            "observed_wns_ns":wns,
            "postroute_timing_met":item.get("postroute_timing_met")
                if evidence=="E4-Vivado-implementation" else None,
            "timing_deficit_ns":deficit,
        })
    tallies={}
    for item in out:
        tallies[item["outcome"]]=tallies.get(item["outcome"],0)+1
    e4=sum(x["evidence"]=="E4-Vivado-implementation" for x in out)
    eligible=sum(x["qualified"] for x in out)
    if (e4!=comparison.get("e4_routed_cases") or
            eligible!=comparison.get("qualified_cases")):
        raise ValueError("Native/qualified case totals disagree with source comparator")
    return {
        "schema":SCHEMA,"source_schema":SOURCE_SCHEMA,
        "source_suite_sha256":comparison.get("source_suite_sha256"),
        "source_table_sha256":comparison.get("source_table_sha256"),
        "case_count":len(out),"e4_routed_cases":e4,
        "qualified_cases":eligible,"outcome_counts":tallies,"rows":out,
        "limitations":(
            "A routed E4 design may fail numerical acceptance or the target "
            "clock; its physical measurements are reported but no qualified "
            "comparison delta is assigned. HLS latency is an estimate, "
            "not measured inference runtime. Parallelism intent does not "
            "guarantee cycle reduction. Out-of-context routing does not "
            "establish board readiness or full-dataset accuracy."
        ),
    }


def write_architecture_outcomes(comparison:dict, destination:Path) -> dict:
    """Write new derived files in the caller's uncommitted staging directory."""
    result=summarize_architecture_outcomes(comparison)
    destination=Path(destination)
    (destination/"architecture_outcomes.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with (destination/"architecture_outcomes.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({
            k:("TBD" if row.get(k) is None else row[k]) for k in COLUMNS
        } for row in result["rows"])
    md=[
        "# Source-attributed architecture outcomes",
        "",
        "| Model | Case | Axis | Status | HLS latency | Δ cycles (qualified) | LUT | WNS (ns) | Timing deficit (ns) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        fields=("model","case","axis","outcome","observed_hls_latency_cycles",
                "qualified_latency_delta_cycles","observed_lut","observed_wns_ns",
                "timing_deficit_ns")
        values=["TBD" if row.get(k) is None else str(row[k]) for k in fields]
        md.append("| "+" | ".join(values)+" |")
    md.extend(["", "## Interpretation", "",
        "Only E4_QUALIFIED cases have passing numerical evidence and the "
        "required timing/DRC evidence. Other E4 rows retain observed "
        "resource and timing numbers but **must not** be treated as "
        "qualified design points.", "",
        "A negative routed WNS is a measured timing failure; the deficit "
        "is the magnitude of this negative slack, not a predicted clock "
        "period or a recommendation to relax the clock.", "",
        "Requested spatial parallelism can increase HLS latency estimates. "
        "A source-level UNROLL and a request for more replicas do not "
        "guarantee a faster end-to-end implementation.", "",
        result["limitations"], ""])
    (destination/"architecture_outcomes.md").write_text("\n".join(md))
    return result
