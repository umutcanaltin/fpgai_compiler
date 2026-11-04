"""Read-only descriptive synthesis of separately audited MatMul/Conv pilots.

Do NOT pool QoR, normalize cross-operator cycles, call two source releases
replicates, or claim comparative compiler superiority. Each numeric
contrast is retained only within its own operator and source version.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_axis_matrix import SCHEMA as MATRIX_SCHEMA
from fpgai_research.experiments.native_seven_axis_effects import AXES,OUTPUTS

SCHEMA="fpgai-research/two-operator-descriptive-sensitivity/v1"
DISPLAY=("hls_estimated_cycles","postroute_lut","postroute_dsp","postroute_wns_ns")


def _finite(v):
    return type(v) in (int,float) and isfinite(v)


def _vector(row,kind):
    if kind=="main":
        qualified=row.get("status")=="QUALIFIED_COMPARISON"
        value=row.get("delta")
    else:
        qualified=row.get("status")=="QUALIFIED_FACTORIAL_CONTRAST"
        value=row.get("second_difference")
    if not qualified:
        if value is not None:
            raise ValueError("Unqualified contrast contains numerical effect")
        return None
    if not isinstance(value,dict) or set(value)!=set(OUTPUTS):
        raise ValueError("Qualified native effect lacks complete metric contract")
    if any(not _finite(value[k]) for k in OUTPUTS):
        raise ValueError("Qualified native effect contains missing/nonfinite metric")
    return {k:value[k] for k in OUTPUTS}


def _study(matrix,expected_kind):
    if (not isinstance(matrix,dict) or matrix.get("schema")!=MATRIX_SCHEMA
            or matrix.get("vendor_tools_executed") is not False
            or matrix.get("source_geometry_synthetic") is not True
            or not isinstance(matrix.get("studies"),list)
            or not matrix["studies"]
            or not all(isinstance(matrix.get(k),list) for k in (
                "main_effects","pairwise_effects","higher_order_requests"))):
        raise ValueError("Require source-audited synthetic axis matrix")
    # The original MatMul pilot was exported from the two-study suite
    # *before* the Conv native campaign was run. It legitimately includes
    # an unqualified Conv study. Select and independently gate the target
    # operator rather than requiring unrelated studies to be complete.
    expected_name={"MatMul":"probe_matmul_08","Conv":"probe_conv_08"}.get(
        expected_kind)
    if expected_name is None:
        raise ValueError("Unsupported operator selection")
    names=[x.get("study") for x in matrix["studies"]
           if isinstance(x,dict)]
    if len(names)!=len(matrix["studies"]) or len(names)!=len(set(names)):
        raise ValueError("Malformed or duplicate source study identities")
    if names.count(expected_name)!=1:
        raise ValueError("Expected exactly one named operator study")
    meta=next(x for x in matrix["studies"] if x["study"]==expected_name)
    if (meta.get("kind")!=expected_kind or meta.get("size_category")!="small"):
        raise ValueError("Expected named small-geometry operator study")
    name=expected_name
    source_rows=(matrix["main_effects"]+matrix["pairwise_effects"]+
                 matrix["higher_order_requests"])
    if any(not isinstance(row,dict) or row.get("study") not in names
           for row in source_rows):
        raise ValueError("Unrecognized source study on axis-matrix row")
    main=[row for row in matrix["main_effects"] if row["study"]==name]
    pairs=[row for row in matrix["pairwise_effects"] if row["study"]==name]
    higher=[row for row in matrix["higher_order_requests"]
            if row["study"]==name]
    if len(main)!=17 or len(pairs)!=21 or len(higher)!=3:
        raise ValueError("Incomplete selected seven-axis pilot design declaration")
    if (len({row.get("case") for row in main})!=17 or
            len({row.get("case") for row in pairs})!=21):
        raise ValueError("Duplicate selected architecture case identity")
    if (meta.get("qualified_routed_case_count")!=42 or
            meta.get("main_comparisons_qualified")!=17 or
            meta.get("pairwise_comparisons_qualified")!=21):
        raise ValueError("Two-operator synthesis requires completed E4 pilots")
    # Make new report fail closed on stale/tampered count claims.
    for row in main:
        _vector(row,"main")
        if row["status"]!="QUALIFIED_COMPARISON":
            raise ValueError("Incomplete main-effect qualification")
    for row in pairs:
        _vector(row,"pair")
        if row["status"]!="QUALIFIED_FACTORIAL_CONTRAST":
            raise ValueError("Incomplete factorial qualification")
    return main,pairs


def _direction(v):
    return "+" if v>0 else "-" if v<0 else "0"


def _pattern(a,b):
    if a=="0" and b=="0":
        return "both_zero"
    if "0" in (a,b):
        return "zero_vs_nonzero"
    return "same_sign" if a==b else "opposite_sign"


def _record(left,right,*,request_equal,kind,labels):
    lm=_vector(left,kind)
    rm=_vector(right,kind)
    if lm is None or rm is None:
        raise ValueError("Complete pilots cannot contain missing numerical effects")
    # Preserve original two within-operator observations. Never compute
    # pooled/averaged or cross-operator cycle/resource differences.
    signs=None
    if request_equal:
        signs={metric:{
            "matmul_direction":_direction(lm[metric]),
            "conv_direction":_direction(rm[metric]),
            "pattern":_pattern(_direction(lm[metric]),_direction(rm[metric])),
        } for metric in OUTPUTS}
    return {
        **labels,
        "requested_levels_matched":request_equal,
        "comparison_scope":("matched_requested_level_descriptive_only"
                            if request_equal else
                            "different_requested_levels_no_direction_comparison"),
        "matmul":{ "case":left["case"],"within_baseline_delta":lm},
        "conv":{ "case":right["case"],"within_baseline_delta":rm},
        "direction_patterns":signs,
    }


def analyze_two_operator_pilots(matmul,conv):
    mmain,mpairs=_study(matmul,"MatMul")
    cmain,cpairs=_study(conv,"Conv")
    bm=defaultdict(list)
    bc=defaultdict(list)
    for row in mmain: bm[row["knob"]].append(row)
    for row in cmain: bc[row["knob"]].append(row)
    aligned=[]
    for axis in AXES:
        ml=sorted(bm[axis],key=lambda x:x["case"])
        cl=sorted(bc[axis],key=lambda x:x["case"])
        if not ml or len(ml)!=len(cl):
            raise ValueError("Unmatched main-effect level inventory")
        for l,r in zip(ml,cl):
            same=l["knob_level"]==r["knob_level"]
            aligned.append(_record(l,r,request_equal=same,kind="main",
                                   labels={"knob":axis,
                                           "matmul_level":l["knob_level"],
                                           "conv_level":r["knob_level"]}))
    def index_pairs(rows):
        dest={}
        for row in rows:
            key=tuple(sorted(row["knobs"]))
            if len(key)!=2 or key in dest:
                raise ValueError("Duplicate or malformed factorial axis pair")
            dest[key]=row
        return dest
    m=index_pairs(mpairs)
    c=index_pairs(cpairs)
    if set(m)!=set(c):
        raise ValueError("Different factorial pair inventories")
    aligned_pairs=[]
    for pair in sorted(m):
        l,r=m[pair],c[pair]
        a=json.loads(l["requested_value"])
        b=json.loads(r["requested_value"])
        same=all(a.get(axis)==b.get(axis) for axis in pair)
        aligned_pairs.append(_record(
            l,r,request_equal=same,kind="pair",
            labels={"axes":list(pair),
                    "matmul_joint_levels":{axis:a.get(axis) for axis in pair},
                    "conv_joint_levels":{axis:b.get(axis) for axis in pair}}))
    def count(rows,metric,pattern):
        return sum(r["direction_patterns"] is not None
                   and r["direction_patterns"][metric]["pattern"]==pattern
                   for r in rows)
    summary={}
    for metric in DISPLAY:
        summary[metric]={
            "main_same_sign":count(aligned,metric,"same_sign"),
            "main_opposite_sign":count(aligned,metric,"opposite_sign"),
            "main_both_zero":count(aligned,metric,"both_zero"),
            "main_zero_vs_nonzero":count(aligned,metric,"zero_vs_nonzero"),
            "pair_same_sign":count(aligned_pairs,metric,"same_sign"),
            "pair_opposite_sign":count(aligned_pairs,metric,"opposite_sign"),
            "pair_both_zero":count(aligned_pairs,metric,"both_zero"),
            "pair_zero_vs_nonzero":count(aligned_pairs,metric,"zero_vs_nonzero"),
        }
    return {
        "schema":SCHEMA,
        "vendor_tools_executed":False,
        "combined_or_pooled_qor":False,
        "independently_native_qualified_designs":84,
        "expected_source_codegen_versions_not_reverified_here":{
            "matmul":"static-trip-v2",
            "conv":"static-conv-trip-v3"},
        "source_version_attested_by_this_analysis":False,
        "operator_specific_synthetic_models_not_trained":True,
        "main_effects":aligned,
        "pairwise_effects":aligned_pairs,
        "direction_counts":summary,
        "matched_main_levels":sum(x["requested_levels_matched"] for x in aligned),
        "matched_pair_levels":sum(x["requested_levels_matched"] for x in aligned_pairs),
        "limitations":(
            "Each side is one separately source-audited within-operator "
            "contrast. This report cannot independently verify FPGA part, "
            "clock, compiler version, or vendor receipts from the already "
            "exported matrices. Model operators, shapes, baseline cycles, "
            "and Conv/MatMul codegen releases differ. A common sign is not "
            "a replication or generality result; no cross-operator numeric "
            "deltas, percentages, averaged metrics, or causal claims are "
            "computed. Requested tiling level geometries may differ. "
            "No repeated place/route seeds, board measurements, trained "
            "network accuracy, or hls4ml comparison are implied."),
    }


def _fmt(v):
    return "0" if v==0 else f"{v:+g}"


def export_two_operator_pilots(matmul_path,conv_path,output):
    files=[Path(matmul_path),Path(conv_path)]
    if files[0].resolve()==files[1].resolve():
        raise ValueError("Independent MatMul and Conv matrix files required")
    raw=[]
    for source in files:
        if (source.is_symlink() or not source.is_file()
                or not 0<source.stat().st_size<=32*1024*1024):
            raise ValueError("Expected bounded regular native matrix JSON")
        raw.append(source.read_bytes())
    report=analyze_two_operator_pilots(*[
        json.loads(data) for data in raw])
    report["source_matrix_sha256"]={
        "matmul":sha256(raw[0]).hexdigest(),
        "conv":sha256(raw[1]).hexdigest()}
    target=Path(output)
    if target.is_symlink() or target.exists():
        raise FileExistsError("Write to a new descriptive-report directory")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=target.parent,prefix=".two-operator-") as work:
        stage=Path(work)/"report"
        stage.mkdir()
        (stage/"comparison.json").write_text(json.dumps(
            report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        lines=["# Two operator-specific synthetic seven-axis studies","",
            "Expected source releases: MatMul static-trip-v2, Conv "
            "static-conv-trip-v3. The input matrices were independently "
            "source-audited, but this descriptive exporter does not "
            "reverify codegen versions. Each study contains 42 E4-qualified "
            "designs under its own baseline.","",
            "This report is not a head-to-head performance ranking. "
            "Two within-study deltas are shown side by side. "
            "Different requested levels are explicitly marked "
            "incomparable; zero does not imply equivalent RTL.","",
            "## Main effects","",
            "| Knob | MatMul level | Conv level | MatMul Δcycles / ΔLUT | "
            "Conv Δcycles / ΔLUT | Matched level? |",
            "|---|---|---|---:|---:|---|"]
        for row in report["main_effects"]:
            l=row["matmul"]["within_baseline_delta"]
            r=row["conv"]["within_baseline_delta"]
            lines.append("| "+str(row["knob"])+" | "+str(row["matmul_level"])+
                         " | "+str(row["conv_level"])+" | "+
                         _fmt(l["hls_estimated_cycles"])+" / "+_fmt(l["postroute_lut"])+
                         " | "+_fmt(r["hls_estimated_cycles"])+" / "+_fmt(r["postroute_lut"])+
                         " | "+("yes" if row["requested_levels_matched"] else "no")+" |")
        lines.extend(["","## Factorial second differences","",
            "Δ₂=Y(AB)-Y(A)-Y(B)+Y(baseline) separately in each operator.","",
            "| Pair | MatMul Δ₂cycles / Δ₂LUT | Conv Δ₂cycles / Δ₂LUT | "
            "Matched joint levels? |",
            "|---|---:|---:|---|"])
        for row in report["pairwise_effects"]:
            l=row["matmul"]["within_baseline_delta"]
            r=row["conv"]["within_baseline_delta"]
            lines.append("| "+" + ".join(row["axes"])+" | "+
                         _fmt(l["hls_estimated_cycles"])+" / "+_fmt(l["postroute_lut"])+
                         " | "+_fmt(r["hls_estimated_cycles"])+" / "+_fmt(r["postroute_lut"])+
                         " | "+("yes" if row["requested_levels_matched"] else "no")+" |")
        lines.extend(["","## Descriptive effect-direction counts","",
            "Directions are compared only when requested levels match. "
            "Different levels are excluded, not counted as disagreement.","",
            "| Metric | Main same / opposite / zero-vs-nonzero | "
            "Pair same / opposite / zero-vs-nonzero |",
            "|---|---|---|"])
        for k,counts in report["direction_counts"].items():
            lines.append(f"| {k} | {counts['main_same_sign']} / "
                         f"{counts['main_opposite_sign']} / "
                         f"{counts['main_zero_vs_nonzero']} | "
                         f"{counts['pair_same_sign']} / "
                         f"{counts['pair_opposite_sign']} / "
                         f"{counts['pair_zero_vs_nonzero']} |")
        lines.extend(["","No statistical/generalization conclusions: "
            "one small synthetic geometry per operator and different "
            "codegen revisions. Compare resource and latency magnitudes "
            "only within the same operator baseline.",""])
        (stage/"summary.md").write_text("\n".join(lines))
        if target.exists() or target.is_symlink():
            raise FileExistsError("Output appeared during immutable write")
        stage.rename(target)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matmul",required=True)
    p.add_argument("--conv",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    report=export_two_operator_pilots(args.matmul,args.conv,args.output)
    print(json.dumps({
        "schema":SCHEMA,
        "native_qualified_designs_from_input_receipts":
            report["independently_native_qualified_designs"],
        "matched_main_levels":report["matched_main_levels"],
        "matched_pair_levels":report["matched_pair_levels"],
        "vendor_tools_executed":False,
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
