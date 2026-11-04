"""Read-only observability diagnostics for source-audited seven-axis QoR.

A physical-observable tie is not circuit equivalence. Correlated
parallelism/PE controls are not independently identifiable from a small
single workload. Pairwise second differences require qualified factorial
quadruples and are never inferred from nominal source requests.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_axis_matrix import SCHEMA as MATRIX_SCHEMA
from fpgai_research.experiments.native_seven_axis_effects import OUTPUTS

SCHEMA="fpgai-research/seven-axis-observational-identifiability/v1"
KEY_METRICS=("hls_estimated_cycles","postroute_lut","postroute_dsp",
             "postroute_wns_ns")


def _valid(value):
    return type(value) in (int,float) and isfinite(value)


def _full(vector):
    return (isinstance(vector,dict)
            and all(_valid(vector.get(k)) for k in OUTPUTS))


def _same(a,b):
    return _full(a) and _full(b) and all(a[k]==b[k] for k in OUTPUTS)


def _format_number(value):
    if value is None:
        return "—"
    if not _valid(value):
        raise ValueError("Cannot format nonfinite native observation")
    return "0" if value==0 else f"{value:+g}"


def analyze_observability(matrix):
    """Inspect qualified observations without requalifying or inferring QoR."""
    if (not isinstance(matrix,dict) or matrix.get("schema")!=MATRIX_SCHEMA
            or matrix.get("vendor_tools_executed") is not False
            or not isinstance(matrix.get("main_effects"),list)
            or not isinstance(matrix.get("pairwise_effects"),list)
            or not isinstance(matrix.get("studies"),list)):
        raise ValueError("Expected source-audited exported axis matrix")
    studies=[]
    for meta in matrix["studies"]:
        name=meta["study"]
        main=[r for r in matrix["main_effects"] if r["study"]==name]
        pairs=[r for r in matrix["pairwise_effects"] if r["study"]==name]
        qualified=[r for r in main if r["status"]=="QUALIFIED_COMPARISON"]
        for row in main:
            if row["status"]=="QUALIFIED_COMPARISON" and not _full(row.get("delta")):
                raise ValueError("Missing required metrics in qualified main effect")
            if row["status"]!="QUALIFIED_COMPARISON" and row.get("delta") is not None:
                raise ValueError("Unqualified main effect contains numeric claims")
        zeros=[{
            "case":r["case"],"knob":r["knob"],"level":r["knob_level"],
            "source_identical_to_baseline":
                r.get("same_generated_cpp_sha256_as_baseline"),
        } for r in qualified if all(r["delta"][key]==0 for key in OUTPUTS)]
        alias=[]
        for i,first in enumerate(qualified):
            for second in qualified[i+1:]:
                if first["knob"]==second["knob"] or not _same(
                        first["delta"],second["delta"]):
                    continue
                alias.append({
                    "left_case":first["case"],"left_knob":first["knob"],
                    "left_level":first["knob_level"],
                    "right_case":second["case"],"right_knob":second["knob"],
                    "right_level":second["knob_level"],
                    "source_comparison_performed":False,
                    "reported_observables_equal":True,
                })
        qualified_pair=[]
        for row in pairs:
            delta=row.get("second_difference")
            if row["status"]=="QUALIFIED_FACTORIAL_CONTRAST":
                if not _full(delta):
                    raise ValueError("Qualified factorial contrast is missing metrics")
                qualified_pair.append({
                    "case":row["case"],"knobs":row["knobs"],
                    "second_difference":delta,
                    "nonzero_metrics":[k for k in OUTPUTS if delta[k]!=0],
                })
            elif delta is not None:
                raise ValueError("Unqualified interaction has numeric effects")
        prominent={}
        for metric in KEY_METRICS:
            ranked=sorted(qualified_pair,key=lambda x:(
                -abs(x["second_difference"][metric]),x["case"]))
            prominent[metric]=[{
                "case":x["case"],"knobs":x["knobs"],
                "signed_second_difference":x["second_difference"][metric],
            } for x in ranked[:5] if x["second_difference"][metric]!=0]
        studies.append({
            "study":name,
            "qualified_main":len(qualified),"requested_main":len(main),
            "zero_effects_across_all_reported_observables":zeros,
            "cross_knob_observational_aliases":alias,
            "requested_pairwise":len(pairs),
            "qualified_pairwise":len(qualified_pair),
            "nonzero_pairwise_by_metric":{
                metric:sum(x["second_difference"][metric]!=0
                           for x in qualified_pair)
                for metric in OUTPUTS},
            "largest_magnitude_pairwise_by_metric":prominent,
        })
    return {
        "schema":SCHEMA,
        "vendor_tools_executed":False,
        "source_matrix_verified_against_native_receipts_here":False,
        "studies":studies,
        "limitations":(
            "All results are exact numerical observations from an existing "
            "source-audited, native-qualified matrix. Identical reported "
            "metrics do not imply identical RTL/netlists or knob ineffectiveness. "
            "Parallelism/PE observational aliases on one geometry do not prove "
            "global redundancy. Positive/negative second differences are "
            "interactions, not automatically optimization gains. No board "
            "latency/power or statistical confidence across mapping seeds."),
    }


def export_observability(matrix_path,output):
    source=Path(matrix_path)
    dest=Path(output)
    if source.is_symlink() or not source.is_file() or not 0<source.stat().st_size<=32*1024*1024:
        raise ValueError("Expected bounded regular complete knob matrix JSON")
    if dest.is_symlink() or dest.exists():
        raise FileExistsError("Output requires a new immutable directory")
    raw=source.read_bytes()
    report=analyze_observability(json.loads(raw))
    report["source_matrix_sha256"]=sha256(raw).hexdigest()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".knob-observability-") as work:
        stage=Path(work)/"result"
        stage.mkdir()
        (stage/"observability.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=["# Seven-knob architecture observability and pairwise interactions","",
               "Only existing, source-audited native observations are used. "
               "An observational tie is not a proof of identical netlists, "
               "and a zero interaction is not proof of universal separability.",""]
        alias_rows=[]
        for group in report["studies"]:
            lines.extend(["## "+group["study"],"",
                f"Qualified main effects: {group['qualified_main']} / "
                f"{group['requested_main']}. Qualified pairwise factorial "
                f"contrasts: {group['qualified_pairwise']} / "
                f"{group['requested_pairwise']}.","",
                "### Main effects with zero changes across all seven metrics",""])
            if group["zero_effects_across_all_reported_observables"]:
                for row in group["zero_effects_across_all_reported_observables"]:
                    lines.append(f"- {row['case']}: {row['knob']} = "
                                 f"{row['level']} (source identical to "
                                 f"baseline: {row['source_identical_to_baseline']})")
            else:
                lines.append("None qualified, or none with complete seven-metric ties.")
            lines.extend(["","### Cross-knob observational aliases",""])
            if group["cross_knob_observational_aliases"]:
                for row in group["cross_knob_observational_aliases"]:
                    alias_rows.append({"study":group["study"],**row})
                    lines.append(
                        f"- {row['left_knob']}={row['left_level']} "
                        f"({row['left_case']}) and "
                        f"{row['right_knob']}={row['right_level']} "
                        f"({row['right_case']}): equal seven-metric deltas")
            else:
                lines.append("No fully qualified cross-knob seven-metric ties.")
            lines.extend(["","### Count of qualified nonzero pairwise second differences","",
                "| Observable | Nonzero / qualified pairwise |",
                "|---|---:|"])
            for metric in OUTPUTS:
                lines.append(f"| {metric} | "
                             f"{group['nonzero_pairwise_by_metric'][metric]} / "
                             f"{group['qualified_pairwise']} |")
            lines.extend(["","### Largest absolute pairwise interactions by observable","",
                "Values are signed second differences, never raw gains.",""])
            for metric in KEY_METRICS:
                candidates=group["largest_magnitude_pairwise_by_metric"][metric]
                lines.append("**"+metric+"**")
                if not candidates:
                    lines.append("No qualified nonzero interaction for this metric.")
                for row in candidates:
                    lines.append(
                        f"- {row['case']}: {' + '.join(row['knobs'])}, "
                        f"{_format_number(row['signed_second_difference'])}")
                lines.append("")
        lines.extend(["","These observations concern out-of-context "
            "post-route FPGA module resources/timing and Vitis HLS estimates. "
            "One model/geometry cannot independently establish architecture "
            "control effectiveness across different workloads.",""])
        (stage/"summary.md").write_text("\n".join(lines))
        with (stage/"cross_knob_observational_aliases.csv").open(
                "w",newline="",encoding="utf-8") as stream:
            cols=("study","left_case","left_knob","left_level",
                  "right_case","right_knob","right_level",
                  "source_comparison_performed","reported_observables_equal")
            writer=csv.DictWriter(stream,fieldnames=cols)
            writer.writeheader()
            writer.writerows(alias_rows)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Output directory appeared during export")
        stage.rename(dest)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix",required=True,
                        help="Existing complete_knob_effects.json")
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    report=export_observability(args.matrix,args.output)
    print(json.dumps({
        "study_count":len(report["studies"]),
        "qualified_pairwise":sum(x["qualified_pairwise"] for x in report["studies"]),
        "cross_knob_observational_aliases":sum(
            len(x["cross_knob_observational_aliases"]) for x in report["studies"]),
        "vendor_tools_executed":False,
    },sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
