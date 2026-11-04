"""Source-audited descriptive three-geometry FPGA architecture sensitivity.

Only within-operator, within-size, E4-qualified deltas can enter signed
comparisons. Requested changes to tile shapes across geometries exclude
matched-level claims. No vendor execution, interpolation, cycle scaling
law, pooling of operators, or hls4ml superiority claims.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_seven_axis_effects import OUTPUTS
from fpgai_research.experiments.size_effect_transport import _load_study

SCHEMA="fpgai-research/three-size-operator-effect-transport/v1"
OPERATORS={
    "MatMul":("probe_matmul_08","probe_matmul_16","probe_matmul_32"),
    "Conv":("probe_conv_08","probe_conv_12","probe_conv_16"),
}
SIZES=("small","medium","large")


def _qual(row,kind):
    key,accepted=(("delta","QUALIFIED_COMPARISON") if kind=="main"
                  else ("second_difference","QUALIFIED_FACTORIAL_CONTRAST"))
    return row[key] if row["status"]==accepted else None


def _label(value):
    return "0" if value==0 else f"{value:+g}"


def _pattern(nums):
    if all(v==0 for v in nums):
        return "all_zero"
    if any(v==0 for v in nums):
        return "zero_and_nonzero"
    signs=set(v>0 for v in nums)
    return "same_sign" if len(signs)==1 else "mixed_signs"


def _one_row(three,kind):
    if len(three)!=3 or len({x["case"] for x in three})!=1:
        raise ValueError("Misaligned operator case identities")
    if kind=="main":
        knobs={x["knob"] for x in three}
        if len(knobs)!=1:
            raise ValueError("Single-axis identity changed across sizes")
        levels={size:x["knob_level"]
                for size,x in zip(SIZES,three)}
        matched=len(set(levels.values()))==1
        identification={"knob":three[0]["knob"],"requested_levels":levels}
    else:
        axes=[tuple(x["knobs"]) for x in three]
        if axes[0]!=axes[1] or axes[0]!=axes[2]:
            raise ValueError("Factorial pair identity changed across sizes")
        requests={size:json.loads(x["requested_value"])
                  for size,x in zip(SIZES,three)}
        levels={size:{k:request[k] for k in axes[0]}
                for size,request in requests.items()}
        matched=all(levels[size]==levels["small"] for size in SIZES[1:])
        identification={"knobs":list(axes[0]),"requested_levels":levels}
    effects={size:_qual(x,kind) for size,x in zip(SIZES,three)}
    qualified={size:effect is not None for size,effect in effects.items()}
    comparable=matched and all(qualified.values())
    sign_patterns={
        name:_pattern([effects[size][name] for size in SIZES])
        for name in OUTPUTS
    } if comparable else None
    return {
        **identification,
        "case":three[0]["case"],
        "matched_requested_levels":matched,
        "native_E4_qualified_effect":qualified,
        "source_audited_status":{
            size:x["status"] for size,x in zip(SIZES,three)},
        "within_size_effects":effects,
        "three_size_direction_patterns":sign_patterns,
        "not_comparable_reason":(
            None if comparable else
            "different_requested_knob_levels" if not matched else
            "missing_source_qualified_E4_effect"),
    }


def analyze_three_size(*,small_matmul,small_conv,medium,large):
    results=[]
    for operator,names in OPERATORS.items():
        original=(small_matmul if operator=="MatMul" else small_conv,
                  medium,large)
        inputs=[
            _load_study(matrix,name=name,kind=operator,size=size)
            for size,matrix,name in zip(SIZES,original,names)
        ]
        cases={}
        for kind in ("main","pair"):
            sorted_rows=[sorted(study[kind],key=lambda x:x["case"])
                         for _,study in inputs]
            cases[kind]=[
                _one_row(values,kind)
                for values in zip(*sorted_rows)
            ]
        per_size={
            size:{
                "study":meta["study"],
                "qualified_routed_cases":meta["qualified_routed_case_count"],
                "qualified_main_effects":meta["main_comparisons_qualified"],
                "qualified_pairwise_effects":meta["pairwise_comparisons_qualified"],
            } for size,(meta,_) in zip(SIZES,inputs)
        }
        results.append({
            "operator":operator,"geometry_evidence":per_size,
            "main_effects":cases["main"],
            "pairwise_effects":cases["pair"],
            "matched_and_qualified_main":sum(
                x["three_size_direction_patterns"] is not None
                for x in cases["main"]),
            "matched_and_qualified_pairs":sum(
                x["three_size_direction_patterns"] is not None
                for x in cases["pair"]),
            "main_direction_counts":{
                metric:{category:sum(
                    x["three_size_direction_patterns"] is not None and
                    x["three_size_direction_patterns"][metric]==category
                    for x in cases["main"])
                    for category in ("same_sign","mixed_signs",
                                     "all_zero","zero_and_nonzero")}
                for metric in OUTPUTS
            },
            "pair_direction_counts":{
                metric:{category:sum(
                    x["three_size_direction_patterns"] is not None and
                    x["three_size_direction_patterns"][metric]==category
                    for x in cases["pair"])
                    for category in ("same_sign","mixed_signs",
                                     "all_zero","zero_and_nonzero")}
                for metric in OUTPUTS
            },
        })
    return {
        "schema":SCHEMA,
        "vendor_tools_executed":False,
        "source_generator_versions_independently_reattested":False,
        "comparison_scope":"one_operator_three_synthetic_geometry_sizes",
        "operator_reports":results,
        "limitations":(
            "The three sizes represent separate synthetic fixed functional "
            "operator models, each with its own E4-qualified baseline. "
            "No model-independent scaling law, statistical reproducibility, "
            "achieved per-loop II, throughput, power, board latency, "
            "or cross-compiler superiority is established. "
            "Comparison of directions is descriptive, only for "
            "identical requested knob values and independently qualified "
            "E4 effects at all three sizes. Tile shapes differ by size "
            "and are excluded. Failed E4 observations remain null."),
    }


def export_three_size(*,small_matmul,small_conv,medium,large,output):
    paths={"small_matmul":Path(small_matmul),
           "small_conv":Path(small_conv),
           "medium":Path(medium),
           "large":Path(large)}
    inputs={}
    hashes={}
    for key,path in paths.items():
        if (path.is_symlink() or not path.is_file()
                or not 0<path.stat().st_size<=32*1024*1024):
            raise ValueError("Expected immutable bounded source-audited matrix")
        raw=path.read_bytes()
        hashes[key]=sha256(raw).hexdigest()
        inputs[key]=json.loads(raw)
    result=analyze_three_size(**inputs)
    result["input_matrix_sha256"]=hashes
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Use a new report directory; preserve existing evidence")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".three-geometry-") as d:
        stage=Path(d)/"report"
        stage.mkdir()
        (stage/"three_size.json").write_text(json.dumps(
            result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=["# Three-size operator-local FPGA native effects","",
               "Within each operator, compare independent E4-qualified "
               "effects against that geometry's OWN baseline. "
               "HLS estimated cycles are not on-board latency.",""]
        for operator in result["operator_reports"]:
            lines.extend(["## "+operator["operator"],"",
                "Qualified routed cases: "+", ".join(
                    f"{size} {record['qualified_routed_cases']}/42"
                    for size,record in operator["geometry_evidence"].items()),
                "",
                f"Matched E4-qualified settings across all three sizes: "
                f"{operator['matched_and_qualified_main']}/17 main effects; "
                f"{operator['matched_and_qualified_pairs']}/21 pairwise effects.",
                "","### Main architecture effects","",
                "| Case | Knob | Same requested level? | Small Δcycles / ΔLUT | "
                "Medium Δcycles / ΔLUT | Large Δcycles / ΔLUT | Sign comparability |",
                "|---|---|---|---:|---:|---:|---|"])
            for kind in ("main_effects","pairwise_effects"):
                if kind=="pairwise_effects":
                    lines.extend(["","### Pairwise factorial second differences","",
                      "| Case | Knobs | Same requested joint levels? | Small Δ₂cycles / Δ₂LUT | "
                      "Medium Δ₂cycles / Δ₂LUT | Large Δ₂cycles / Δ₂LUT | "
                      "Sign comparability |",
                      "|---|---|---|---:|---:|---:|---|"])
                for x in operator[kind]:
                    def fmt(size):
                        e=x["within_size_effects"][size]
                        return ("— / —" if e is None else
                                _label(e["hls_estimated_cycles"])+" / "+
                                _label(e["postroute_lut"]))
                    knob=(x["knob"] if kind=="main_effects"
                          else " + ".join(x["knobs"]))
                    lines.append(
                        f"| {x['case']} | {knob} | "
                        f"{x['matched_requested_levels']} | "
                        f"{fmt('small')} | {fmt('medium')} | {fmt('large')} | "
                        f"{x['not_comparable_reason'] or 'qualified'} |")
            lines.append("")
            lines.append("Direction patterns (only matched, E4-qualified):")
            lines.extend(["",
                "| Metric | Main same / mixed / all-zero / zero-mixed | "
                "Pair same / mixed / all-zero / zero-mixed |",
                "|---|---:|---:|"])
            for metric in ("hls_estimated_cycles","postroute_lut",
                           "postroute_dsp","postroute_wns_ns"):
                def counts(field):
                    obj=operator[field][metric]
                    return " / ".join(str(obj[k]) for k in (
                        "same_sign","mixed_signs","all_zero","zero_and_nonzero"))
                lines.append(f"| {metric} | {counts('main_direction_counts')} "
                             f"| {counts('pair_direction_counts')} |")
            lines.append("")
        lines.extend([
            "Missing effects are —, not zero. Different requested tiling "
            "coordinates are noncomparable across sizes. Direction "
            "similarity is descriptive, not a verified resource "
            "scaling law, physical speedup, or hls4ml comparison.",""])
        (stage/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Output appeared during immutable export")
        stage.rename(dest)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small-matmul",required=True)
    p.add_argument("--small-conv",required=True)
    p.add_argument("--medium",required=True)
    p.add_argument("--large",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    result=export_three_size(**vars(args))
    print(json.dumps({
        "schema":SCHEMA,"vendor_tools_executed":False,
        "operators":[{
            "operator":row["operator"],
            "qualified_routed_by_size":{
                size:x["qualified_routed_cases"]
                for size,x in row["geometry_evidence"].items()},
            "matched_qualified_main":row["matched_and_qualified_main"],
            "matched_qualified_pairs":row["matched_and_qualified_pairs"],
        } for row in result["operator_reports"]],
    },sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
