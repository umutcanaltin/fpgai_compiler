"""Descriptive, no-imputation small-to-medium FPGA effect transportability.

Each effect remains a within-operator, within-geometry, source/route-
qualified contrast. Compare small and medium ONLY within the same operator.
Do not normalize estimated cycles across geometries, pool MatMul/Conv,
or interpret two geometry points as statistical generalization.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_axis_matrix import SCHEMA as MATRIX_SCHEMA
from fpgai_research.experiments.native_seven_axis_effects import OUTPUTS

SCHEMA="fpgai-research/small-medium-operator-effect-transport/v1"
STUDIES={
    "MatMul":("probe_matmul_08","probe_matmul_16"),
    "Conv":("probe_conv_08","probe_conv_12"),
}
COUNT_MAIN=17
COUNT_PAIR=21


def _finite(n):
    return type(n) in (int,float) and isfinite(n)


def _load_study(matrix,*,name,kind,size):
    if (not isinstance(matrix,dict) or matrix.get("schema")!=MATRIX_SCHEMA
            or matrix.get("source_geometry_synthetic") is not True
            or matrix.get("vendor_tools_executed") is not False):
        raise ValueError("Expected a source-audited synthetic effect matrix")
    studies=matrix.get("studies")
    if not isinstance(studies,list):
        raise ValueError("Missing source study inventory")
    entries=[s for s in studies if isinstance(s,dict) and s.get("study")==name]
    if len(entries)!=1:
        raise ValueError("Exactly one selected operator/geometry study required")
    meta=entries[0]
    if meta.get("kind")!=kind or meta.get("size_category")!=size:
        raise ValueError("Selected study has incorrect geometry size or operator")
    all_rows={}
    for table,key,count in (("main_effects","main",COUNT_MAIN),
                            ("pairwise_effects","pair",COUNT_PAIR),
                            ("higher_order_requests","higher",3)):
        content=matrix.get(table)
        if not isinstance(content,list):
            raise ValueError("Incomplete scientific matrix table")
        selected=[x for x in content if isinstance(x,dict) and x.get("study")==name]
        if len(selected)!=count or len({x.get("case") for x in selected})!=count:
            raise ValueError("Selected source has malformed 42-case inventory")
        all_rows[key]=selected
    if (meta.get("main_comparisons_qualified")!=
            sum(x.get("status")=="QUALIFIED_COMPARISON"
                for x in all_rows["main"])
            or meta.get("pairwise_comparisons_qualified")!=
            sum(x.get("status")=="QUALIFIED_FACTORIAL_CONTRAST"
                for x in all_rows["pair"])):
        raise ValueError("Claimed effects qualification differs from row evidence")
    # A failed E4 case is not silently converted to zero.
    for table,metric_key,qualified_state in (
        ("main","delta","QUALIFIED_COMPARISON"),
        ("pair","second_difference","QUALIFIED_FACTORIAL_CONTRAST"),
    ):
        for row in all_rows[table]:
            effect=row.get(metric_key)
            if row.get("status")==qualified_state:
                if (not isinstance(effect,dict) or set(effect)!=set(OUTPUTS)
                        or any(not _finite(effect[k]) for k in OUTPUTS)):
                    raise ValueError("Qualified row lacks finite native metrics")
            elif effect is not None:
                raise ValueError("Unqualified row must not contain numerical effect")
    return meta,all_rows


def _direction(n):
    return "zero" if n==0 else "positive" if n>0 else "negative"


def _compare(a,b,*,kind):
    field="delta" if kind=="main" else "second_difference"
    status=("QUALIFIED_COMPARISON" if kind=="main"
            else "QUALIFIED_FACTORIAL_CONTRAST")
    v1=a[field] if a.get("status")==status else None
    v2=b[field] if b.get("status")==status else None
    if kind=="main":
        matched=(a["knob"]==b["knob"] and
                 a["knob_level"]==b["knob_level"])
        identity={"knob":a["knob"],
                  "small_level":a["knob_level"],
                  "medium_level":b["knob_level"]}
        if a["knob"]!=b["knob"]:
            raise ValueError("Main axis changed across sizes")
    else:
        if a["knobs"]!=b["knobs"]:
            raise ValueError("Pairwise axis set changed across sizes")
        left=json.loads(a["requested_value"])
        right=json.loads(b["requested_value"])
        matched=all(left[k]==right[k] for k in a["knobs"])
        identity={"knobs":a["knobs"],
                  "small_joint_levels":{k:left[k] for k in a["knobs"]},
                  "medium_joint_levels":{k:right[k] for k in a["knobs"]}}
    eligible=matched and v1 is not None and v2 is not None
    signs={
        key:("both_zero" if v1[key]==v2[key]==0
             else "zero_vs_nonzero" if v1[key]==0 or v2[key]==0
             else "same_sign" if _direction(v1[key])==_direction(v2[key])
             else "opposite_sign")
        for key in OUTPUTS
    } if eligible else None
    return {
        **identity,
        "small_case":a["case"],
        "medium_case":b["case"],
        "levels_identical":matched,
        "small_status":a["status"],
        "medium_status":b["status"],
        "small_within_baseline_effect":v1,
        "medium_within_baseline_effect":v2,
        "sign_patterns_if_matched_and_qualified":signs,
        "not_comparable_reason":(
            None if eligible else
            "requested_levels_differ" if not matched else
            "unqualified_native_effect"),
    }


def analyze_size_transport(*,small_matmul,small_conv,medium):
    output=[]
    for kind in ("MatMul","Conv"):
        sm_name,med_name=STUDIES[kind]
        old=small_matmul if kind=="MatMul" else small_conv
        small_meta,small_rows=_load_study(
            old,name=sm_name,kind=kind,size="small")
        medium_meta,med_rows=_load_study(
            medium,name=med_name,kind=kind,size="medium")
        aligned={}
        for key in ("main","pair"):
            left=sorted(small_rows[key],key=lambda x:x["case"])
            right=sorted(med_rows[key],key=lambda x:x["case"])
            if any(a["case"]!=b["case"] for a,b in zip(left,right)):
                raise ValueError("Geometry campaign case order changed")
            aligned[key]=[_compare(a,b,kind=key)
                          for a,b in zip(left,right)]
        output.append({
            "operator":kind,
            "small_study":sm_name,
            "medium_study":med_name,
            "small_qualified_routed_cases":
                small_meta["qualified_routed_case_count"],
            "medium_qualified_routed_cases":
                medium_meta["qualified_routed_case_count"],
            "main_effects":aligned["main"],
            "pairwise_effects":aligned["pair"],
            "matched_and_qualified_main":sum(
                r["sign_patterns_if_matched_and_qualified"] is not None
                for r in aligned["main"]),
            "matched_and_qualified_pairs":sum(
                r["sign_patterns_if_matched_and_qualified"] is not None
                for r in aligned["pair"]),
        })
    return {
        "schema":SCHEMA,
        "vendor_tools_executed":False,
        "source_version_reattested_by_this_analysis":False,
        "comparison_scope":"within_each_operator_across_synthetic_geometry",
        "operator_comparisons":output,
        "limitation":(
            "Each main/pair effect is calculated within its own fixed "
            "operator geometry/model baseline. Difference in signs or "
            "magnitudes across geometry does not isolate geometry as "
            "a causal variable. Only identical requested knob levels "
            "and two qualified source-linked E4 effects receive sign "
            "comparisons. Different tiling shapes are excluded. "
            "An unqualified medium case is preserved as missing; "
            "no reclassification, normalization or pooling is allowed. "
            "This report does not reverify exact HLS compiler revision; "
            "use source-attested archived case receipts for that claim. "
            "No board latency or power, no statistical replication and "
            "no hls4ml superiority are established."),
    }


def _fmt(e,key):
    if e is None:
        return "—"
    value=e[key]
    return "0" if value==0 else f"{value:+g}"


def export_size_transport(*,small_matmul,small_conv,medium,output):
    sources={
        "small_matmul":Path(small_matmul),
        "small_conv":Path(small_conv),
        "medium":Path(medium),
    }
    raw={}
    for key,path in sources.items():
        if (path.is_symlink() or not path.is_file()
                or not 0<path.stat().st_size<=32*1024*1024):
            raise ValueError("Expected existing bounded audited effect matrix")
        raw[key]=path.read_bytes()
    report=analyze_size_transport(**{
        key:json.loads(val) for key,val in raw.items()})
    report["input_matrix_sha256"]={
        key:sha256(value).hexdigest() for key,value in raw.items()}
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Export to a fresh immutable scientific report directory")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".size-transport-") as temp:
        stage=Path(temp)/"report"
        stage.mkdir()
        (stage/"transport.json").write_text(json.dumps(
            report,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=[
            "# Small-to-medium operator-specific architecture effects","",
            "The two operands sizes use separate native baselines. "
            "Only matching requested knob levels and independent E4-qualified "
            "within-size effects receive sign-comparison interpretations. "
            "Estimated cycles are NOT on-device latency.","",
        ]
        for study in report["operator_comparisons"]:
            lines.extend([
                "## "+study["operator"],"",
                f"Routed-qualified: small {study['small_qualified_routed_cases']}/42; "
                f"medium {study['medium_qualified_routed_cases']}/42.","",
                f"Comparable: {study['matched_and_qualified_main']}/17 main levels; "
                f"{study['matched_and_qualified_pairs']}/21 pair interactions.","",
                "### Main-effect contrasts","",
                "| Case | Knob | Same level? | Small Δcycles / ΔLUT | "
                "Medium Δcycles / ΔLUT | Medium status |",
                "|---|---|---|---:|---:|---|",
            ])
            for row in study["main_effects"]:
                a=row["small_within_baseline_effect"]
                b=row["medium_within_baseline_effect"]
                lines.append(
                    f"| {row['small_case']} | {row['knob']} | "
                    f"{row['levels_identical']} | "
                    f"{_fmt(a,'hls_estimated_cycles')} / {_fmt(a,'postroute_lut')} | "
                    f"{_fmt(b,'hls_estimated_cycles')} / {_fmt(b,'postroute_lut')} | "
                    f"{row['medium_status']} |")
            lines.extend([
                "","### Factorial second differences","",
                "| Case | Pair | Same joint levels? | "
                "Small Δ₂cycles / Δ₂LUT | Medium Δ₂cycles / Δ₂LUT | "
                "Medium status |",
                "|---|---|---|---:|---:|---|",
            ])
            for row in study["pairwise_effects"]:
                a=row["small_within_baseline_effect"]
                b=row["medium_within_baseline_effect"]
                lines.append(
                    f"| {row['small_case']} | {' + '.join(row['knobs'])} | "
                    f"{row['levels_identical']} | "
                    f"{_fmt(a,'hls_estimated_cycles')} / {_fmt(a,'postroute_lut')} | "
                    f"{_fmt(b,'hls_estimated_cycles')} / {_fmt(b,'postroute_lut')} | "
                    f"{row['medium_status']} |")
            lines.append("")
        lines.extend([
            "A difference in requested tiling coordinates is not a "
            "matched-level comparison. A missing E4 factor remains missing "
            "rather than zero. One geometry pair does not establish "
            "statistical generalization, resource scaling law or an "
            "FPGAI-vs-hls4ml performance result.",""])
        (stage/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Destination appeared during immutable export")
        stage.rename(dest)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small-matmul",required=True)
    p.add_argument("--small-conv",required=True)
    p.add_argument("--medium",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    report=export_size_transport(
        small_matmul=args.small_matmul,
        small_conv=args.small_conv,
        medium=args.medium,
        output=args.output)
    print(json.dumps({
        "schema":SCHEMA,
        "vendor_tools_executed":False,
        "operators":[{
            "operator":x["operator"],
            "small_routed":x["small_qualified_routed_cases"],
            "medium_routed":x["medium_qualified_routed_cases"],
            "comparable_main":x["matched_and_qualified_main"],
            "comparable_pairs":x["matched_and_qualified_pairs"],
        } for x in report["operator_comparisons"]],
    },sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
