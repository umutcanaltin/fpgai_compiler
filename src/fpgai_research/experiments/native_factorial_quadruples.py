"""Explain native E4 factorial effects with their four actual circuit outcomes.

Input must be a pre-existing native-seven-axis source-audited effects.json,
not an exported table of unverified claims. This read-only analysis does
not execute vendor tools or infer any missing numerical observations.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_seven_axis_effects import (
    SCHEMA as EFFECTS_SCHEMA, AXES, OUTPUTS, _requested_signature,
)

SCHEMA = "fpgai-research/native-qualified-factorial-quadruple-audit/v1"


def _vector(case):
    if case.get("qualification") != "QUALIFIED_ROUTED":
        raise ValueError("Factorial corner lacks E4 routed qualification")
    values = case.get("qualified_metrics")
    if not isinstance(values, dict) or set(values) != set(OUTPUTS):
        raise ValueError("Factorial corner lacks complete qualified QoR vector")
    if any(type(values[k]) not in (int, float) or not isfinite(values[k])
           for k in OUTPUTS):
        raise ValueError("Factorial corner contains nonfinite or missing QoR")
    source = case.get("source_sha256")
    if (not isinstance(source, str) or len(source) != 64 or
            any(letter not in "0123456789abcdef" for letter in source)):
        raise ValueError("Factorial corner lacks valid source identity")
    return values


def analyze_factorial_quadruples(effects, *, study, cases=None):
    if (not isinstance(effects, dict) or effects.get("schema") != EFFECTS_SCHEMA
            or effects.get("vendor_tools_executed_by_analysis") is not False
            or not isinstance(effects.get("studies"), list)):
        raise ValueError("Expected immutable source-audited native effects")
    candidates = [x for x in effects["studies"]
                  if isinstance(x, dict) and x.get("study") == study]
    if len(candidates) != 1:
        raise ValueError("Exactly one matching operator study is required")
    data = candidates[0]
    if data.get("routed_qualified_case_count") != 42:
        raise ValueError("Full routed pilot is required for audited interactions")
    raw = data.get("cases")
    if not isinstance(raw, list) or len(raw) != 42:
        raise ValueError("Expected original complete 42-case pilot")
    names = [x.get("case") for x in raw]
    if len(set(names)) != 42:
        raise ValueError("Duplicate case IDs in native source inventory")
    base = [x for x in raw if x.get("axis") == "baseline"]
    if len(base) != 1:
        raise ValueError("Expected one qualified native baseline")
    baseline = base[0]
    defaults = baseline.get("requested")
    by_request = {_requested_signature(x["requested"]): x for x in raw}
    if len(by_request) != 42:
        raise ValueError("Repeated architecture requests")
    if data.get("qualified_single_factor_count") != 17:
        raise ValueError("Missing qualified single-factor comparisons")
    if data.get("qualified_pairwise_count") != 21:
        raise ValueError("Missing qualified two-factor comparisons")
    published = {x["case"]:x for x in data.get(
        "pairwise_or_higher_order_contrasts", [])
        if x.get("status") == "QUALIFIED_FACTORIAL_CONTRAST"}
    if len(published) != 21:
        raise ValueError("Expected 21 distinct published native factorial effects")
    selected = set(cases) if cases is not None else set(published)
    if not selected or not selected <= set(published):
        raise ValueError("Requested case is absent or unqualified")
    rows = []
    for joint in raw:
        if joint["case"] not in selected:
            continue
        if joint.get("axis") != "interaction":
            raise ValueError("Selected factorial case is not a joint request")
        changes = {k:joint["requested"][k] for k in AXES
                   if joint["requested"][k] != defaults[k]}
        if len(changes) != 2:
            raise ValueError("Factorial case must change exactly two knobs")
        left_axis,right_axis = tuple(changes)
        lreq = {**defaults,left_axis:changes[left_axis]}
        rreq = {**defaults,right_axis:changes[right_axis]}
        left = by_request.get(_requested_signature(lreq))
        right = by_request.get(_requested_signature(rreq))
        if left is None or right is None:
            raise ValueError("Missing exact matching single-factor corner")
        if left["axis"] != left_axis or right["axis"] != right_axis:
            raise ValueError("Single-factor corner has incorrect axis")
        native = published[joint["case"]]
        if set(native["axes"]) != set(changes):
            raise ValueError("Published effect axes differ from source requests")
        if set(native.get("matched_single_case_ids",[])) != {
                left["case"],right["case"]}:
            raise ValueError("Published single-factor corners do not match")
        corner = (baseline,left,right,joint)
        vals = [_vector(x) for x in corner]
        calculated = {
            key:round(vals[3][key]-vals[1][key]-vals[2][key]+vals[0][key],9)
            for key in OUTPUTS
        }
        claimed = native.get("pairwise_second_difference")
        if not isinstance(claimed,dict) or set(claimed)!=set(OUTPUTS):
            raise ValueError("Missing published second-difference vector")
        if any(type(claimed[k]) not in (int,float) or
               not isfinite(claimed[k]) or
               abs(calculated[k]-claimed[k])>1e-8 for k in OUTPUTS):
            raise ValueError("Native second difference does not equal its four corners")
        rows.append({
            "case":joint["case"],
            "axes":list(changes),
            "requested_changed_levels":changes,
            "corner_cases":{
                "baseline":baseline["case"],
                "single_a":left["case"],
                "single_b":right["case"],
                "joint":joint["case"],
            },
            "source_attested_corners":[{
                "role":role,"case":part["case"],
                "source_sha256":part["source_sha256"],
                "native_evidence":part.get("native_evidence"),
                "absolute_qor":dict(values),
            } for role,part,values in zip(
                ("baseline","single_a","single_b","joint"),corner,vals)],
            "second_difference":calculated,
            "joint_delta_vs_baseline":{
                k:round(vals[3][k]-vals[0][k],9) for k in OUTPUTS
            },
            "factorial_difference_recomputed_from_four_E4_corners":True,
            "outlier_is_causal_mechanism_proven":False,
        })
    if len(rows)!=len(selected):
        raise ValueError("Qualified source inventory did not cover selected cases")
    ranked = {
        key:[
            {"case":x["case"],"axes":x["axes"],
             "second_difference":x["second_difference"][key],
             "joint_absolute_qor":next(c for c in x["source_attested_corners"]
                                      if c["role"]=="joint")["absolute_qor"][key]}
            for x in sorted(rows,key=lambda x:(
                -abs(x["second_difference"][key]),x["case"]))[:5]
        ] for key in OUTPUTS
    }
    return {
        "schema":SCHEMA,
        "study":study,
        "vendor_tools_executed":False,
        "effect_source_reaudit_executed_by_this_report":False,
        "corners_source_attested_by_input_effects":True,
        "selected_pairwise_cases":len(rows),
        "selected_case_ids":sorted(selected),
        "quadruples":rows,
        "top_absolute_second_differences":ranked,
        "limits":(
            "This is deterministic algebra over a prior source-audited "
            "native effects file. Large second differences may arise "
            "from one expensive joint design, nonlinear resource mapping, "
            "or synthesis/reporting behavior. This report does NOT "
            "establish a root cause, achieved loop II, absolute on-board "
            "latency, power, netlist identity, or statistical stability."),
    }


def export_factorial_quadruples(effects_path,output,*,study,cases=None):
    src=Path(effects_path)
    dest=Path(output)
    if (src.is_symlink() or not src.is_file()
            or not 0<src.stat().st_size<=32*1024*1024):
        raise ValueError("Require bounded existing audited effects.json")
    if dest.is_symlink() or dest.exists():
        raise FileExistsError("Use a new immutable factorial audit directory")
    raw=src.read_bytes()
    report=analyze_factorial_quadruples(
        json.loads(raw),study=study,cases=cases)
    report["input_effects_sha256"]=sha256(raw).hexdigest()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".quadruples-") as temporary:
        folder=Path(temporary)/"study"
        folder.mkdir()
        (folder/"quadruples.json").write_text(json.dumps(
            report,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=[
            "# Native E4 factorial-corner diagnostics: "+study,
            "",
            "Read-only reconstruction from original qualified absolute "
            "metrics and generated-source SHA-256; no new vendor execution.",
            "",
            "Second difference = joint − single A − single B + baseline. "
            "It is NOT the total absolute LUT/cycle cost of the joint design.",
            "",
            "| Pair case | Axes | Δ₂ cycles | Δ₂ LUT | Δ₂ DSP | Δ₂ WNS ns | Joint absolute LUT |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for item in report["quadruples"]:
            m=item["second_difference"]
            joint=next(c for c in item["source_attested_corners"]
                       if c["role"]=="joint")
            lines.append(
                "| "+item["case"]+" | "+" + ".join(item["axes"])+" | "+
                f"{m['hls_estimated_cycles']:+g} | {m['postroute_lut']:+g} | "+
                f"{m['postroute_dsp']:+g} | {m['postroute_wns_ns']:+g} | "+
                f"{joint['absolute_qor']['postroute_lut']:g} |")
        lines.extend(["","## Largest absolute second differences",""])
        for metric in ("postroute_lut","hls_estimated_cycles","postroute_wns_ns"):
            lines.extend(["### "+metric,""])
            for item in report["top_absolute_second_differences"][metric]:
                lines.append(
                    "- "+item["case"]+" ("+" + ".join(item["axes"])+
                    f"): Δ₂ {item['second_difference']:+g}, "
                    f"joint absolute {item['joint_absolute_qor']:g}")
            lines.append("")
        lines.extend(["","## Four absolute E4 corner measurements",""])
        for item in report["quadruples"]:
            lines.extend(["### "+item["case"]+" ("+" + ".join(item["axes"])+")",
                          "",
                          "| Corner | Case | HLS estimated cycles | LUT | FF | DSP | WNS ns | Source SHA-256 |",
                          "|---|---|---:|---:|---:|---:|---:|---|"])
            for corner in item["source_attested_corners"]:
                m=corner["absolute_qor"]
                lines.append(
                    f"| {corner['role']} | {corner['case']} | "
                    f"{m['hls_estimated_cycles']:g} | {m['postroute_lut']:g} | "+
                    f"{m['postroute_ff']:g} | {m['postroute_dsp']:g} | "
                    f"{m['postroute_wns_ns']:g} | {corner['source_sha256']} |")
            lines.append("")
        lines.extend(["These findings are source-linked vendor estimates/"
                      "post-route module observations, not board benchmarks. "
                      "Large interactions require scheduler, synthesis, "
                      "utilization and timing log inspection before "
                      "mechanism attribution.",""])
        (folder/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Output directory appeared during export")
        folder.rename(dest)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects",required=True)
    parser.add_argument("--study",required=True,
                        choices=("probe_matmul_08","probe_conv_08"))
    parser.add_argument("--case",action="append",dest="cases",
                        help="Optional specific pairwise case, repeatable")
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    report=export_factorial_quadruples(
        args.effects,args.output,study=args.study,cases=args.cases)
    print(json.dumps({
        "schema":SCHEMA,"study":report["study"],
        "audited_pairwise_quadruples":report["selected_pairwise_cases"],
        "vendor_tools_executed":False,
    },sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
