"""Attested cross-stage observational equivalence classes for FPGAI paper.

Four preregistered within-model pairs are compared across emitted HLS C++,
complete fresh RTL file bytes, six-objective Vitis E3 estimates and routed
Vivado E4 WNS+five utilization counts. No formal equivalence, runtime,
causal mapping conclusion or universal search-space inference is implied.
"""
from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.isolated_hls_e3_ablation import collect_isolated_hls_e3_ablation
from fpgai_research.experiments.isolated_hls_replay import audit_isolated_hls_replay
from fpgai_research.experiments.isolated_hls_e3_rtl_inventory import _case_rtl_inventory
from fpgai_research.experiments.isolated_e3_design_quotient import (
    SCHEMA as QUOTIENT_SCHEMA, _estimate_vector,
    analyze_e3_design_space_quotient,
)
from fpgai_research.experiments.independent_e4_audit import collect_independent_e4
from fpgai_research.experiments.independent_e4_pairs import PAIR_SPECS

SCHEMA="fpgai-research/cross-stage-observational-pairs/v1"
_E3_SCHEMA="fpgai-research/isolated-native-hls-e3-ablation/v1"
_E4_SCHEMA="fpgai-research/independent-matched-e4-audit/v1"
_MODELS=("mnist_cnn","mnist_mlp")


def _indexed(rows, key):
    if not isinstance(rows,list):
        raise ValueError("Expected complete case/pair inventory")
    result={}
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError("Non-object case record")
        k=key(row)
        if k in result:
            raise ValueError("Duplicate case/pair")
        result[k]=row
    return result


def analyze_cross_stage_pairs(e3:dict, quotient:dict, e4:dict,
                              *, independent_source_hashes:dict)->dict:
    """All eight E1/E3/E4 cases and all four physical pairs must be qualified."""
    if (e3.get("schema")!=_E3_SCHEMA
            or quotient.get("schema")!=QUOTIENT_SCHEMA
            or e4.get("schema")!=_E4_SCHEMA
            or e3.get("source_suite_sha256")!=quotient.get("source_suite_sha256")
            or e3.get("source_suite_sha256")!=e4.get("source_suite_sha256")
            or e3.get("native_tools_executed") is not False
            or quotient.get("native_tools_executed") is not False
            or e4.get("vendor_tools_executed_by_audit") is not False
            or e4.get("e4_qualified_case_count")!=8
            or e4.get("qualified_matched_e4_pair_count")!=4
            or e4.get("on_device_latency_verified") is not False
            or quotient.get("formal_rtl_equivalence_verified") is not False):
        raise ValueError("Different, incomplete or unqualified E3/E4 studies")
    original=_indexed(e3.get("comparison_rows"),lambda r:(r.get("model"),r.get("case")))
    physical=_indexed(e4.get("cases"),lambda r:(r.get("model"),r.get("case")))
    classes=_indexed([
        {"model":group.get("model"),**case}
        for group in quotient.get("models",[])
        for case in group.get("cases",[])
    ],lambda r:(r.get("model"),r.get("case")))
    expected={(m,case) for m in _MODELS for case in (
        "case_0000","case_0004","case_0005","case_0006")}
    if (set(original)!=expected or set(physical)!=expected
            or set(classes)!=expected or set(independent_source_hashes)!=expected
            or e3.get("case_count")!=8 or quotient.get("case_count")!=8):
        raise ValueError("Source/architecture/physical case inventories do not align")
    for key in expected:
        a=original[key]; b=physical[key]; c=classes[key]
        if (a.get("comparison_status")!="QUALIFIED_E3_HLS_ESTIMATE"
                or a.get("numerical_csim_passed") is not True
                or a.get("fresh_report_attested") is not True
                or b.get("qualified_e3") is not True
                or b.get("qualified_e4") is not True
                or b.get("qualification_status")!="E4_QUALIFIED_MATCHED_OOC"
                or b.get("numerical_csim_passed") is not True
                or a.get("axis")!=b.get("axis")
                or a.get("axis")!=c.get("axis")
                or a.get("source_sha256")!=independent_source_hashes[key]):
            raise ValueError("Source or scientific qualification disagreement")
        _estimate_vector(a)
        for col in ("generated_cpp_class","generated_verilog_file_byte_class",
                    "hls_six_objective_estimate_class"):
            if not isinstance(c.get(col),str) or len(c[col]) not in (2,3):
                raise ValueError("Missing complete design class labels")
    pairs=(e4.get("paired_physical_comparisons") or {}).get("pairs")
    paired=_indexed(pairs,lambda r:r.get("pair_id"))
    if set(paired)!={spec[3] for spec in PAIR_SPECS}:
        raise ValueError("Incomplete preregistered physical contrasts")
    published=[]
    for model,case_a,case_b,name in PAIR_SPECS:
        a=classes[(model,case_a)]
        b=classes[(model,case_b)]
        oa=original[(model,case_a)]
        ob=original[(model,case_b)]
        row=paired[name]
        if (row.get("model")!=model or row.get("case_a")!=case_a
                or row.get("case_b")!=case_b
                or row.get("axis_a")!=a["axis"]
                or row.get("axis_b")!=b["axis"]
                or row.get("status")!="QUALIFIED_MATCHED_PHYSICAL_COMPARISON"
                or row.get("both_e4_qualified") is not True
                or row.get("matched_physical_context_verified") is not True):
            raise ValueError("Physical pair not source/context-qualified")
        cpp_equal=a["generated_cpp_class"]==b["generated_cpp_class"]
        rtl_equal=a["generated_verilog_file_byte_class"]==b["generated_verilog_file_byte_class"]
        e3_equal=_estimate_vector(oa)==_estimate_vector(ob)
        if e3_equal != (
            a["hls_six_objective_estimate_class"]==b["hls_six_objective_estimate_class"]):
            raise ValueError("E3 design class disagrees with actual HLS estimates")
        wns=row.get("postroute_wns_delta_b_minus_a_ns")
        resources=row.get("postroute_utilization_delta_b_minus_a")
        if (type(wns) not in (int,float) or not isfinite(wns)
                or not isinstance(resources,dict)
                or any(type(resources.get(k)) not in (int,float)
                       or not isfinite(resources[k]) for k in
                       ("lut","ff","dsp","bram","uram"))):
            raise ValueError("Partial or nonfinite physical E4 deltas")
        physical_tie=wns==0 and all(resources[k]==0 for k in (
            "lut","ff","dsp","bram","uram"))
        published.append({
            "pair_id":name,"model":model,"axis_a":a["axis"],"axis_b":b["axis"],
            "source_cpp_file_bytes_equal":cpp_equal,
            "fresh_generated_rtl_file_set_bytes_equal":rtl_equal,
            "hls_six_objective_e3_estimates_equal":e3_equal,
            "routed_e4_wns_and_five_resources_equal":physical_tie,
            "e3_estimate_tie_hides_e4_qor_difference":e3_equal and not physical_tie,
            "e3_latency_estimate_a_cycles":oa["estimated_latency_cycles"],
            "e3_latency_estimate_b_cycles":ob["estimated_latency_cycles"],
            "e4_postroute_wns_delta_b_minus_a_ns":wns,
            "e4_postroute_lut_delta_b_minus_a":resources["lut"],
            "e4_postroute_dsp_delta_b_minus_a":resources["dsp"],
            "formal_circuit_equivalence_verified":False,
            "measured_inference_latency_verified":False,
        })
    return {
        "schema":SCHEMA,
        "case_count":8,"preregistered_pair_count":len(published),
        "e3_tied_pair_count":sum(p["hls_six_objective_e3_estimates_equal"] for p in published),
        "e3_tie_with_e4_difference_count":sum(
            p["e3_estimate_tie_hides_e4_qor_difference"] for p in published),
        "physical_qor_tie_pair_count":sum(
            p["routed_e4_wns_and_five_resources_equal"] for p in published),
        "pairs":published,
        "native_vendor_tools_executed_by_analysis":False,
        "formal_logic_equivalence_claimed":False,
        "measured_board_inference_latency_claimed":False,
        "private_case_hashes_or_paths_exported":False,
        "weights_exported":False,
        "limitations":(
            "The four contrasts are preregistered observational pairs, not "
            "independent random draws. Equal RTL bytes do not prove equality of "
            "post-route netlists; equal WNS and resources do not prove formal "
            "functional equivalence. Source-attested HLS E3 cycles are "
            "synthesis estimates, not measured application latency. "
            "Different routed QoR following an E3 tie is evidence of "
            "imperfect estimate discrimination, not proof of causal "
            "effect from a particular IR axis. Only four architecture "
            "settings per model were studied."
        ),
    }


def cross_stage_latex(result:dict)->str:
    """No user strings or private model/tool hashes are interpolated into TeX."""
    if (result.get("schema")!=SCHEMA
            or result.get("case_count")!=8
            or result.get("preregistered_pair_count")!=4
            or result.get("private_case_hashes_or_paths_exported") is not False
            or result.get("formal_logic_equivalence_claimed") is not False
            or result.get("measured_board_inference_latency_claimed") is not False):
        raise ValueError("Unqualified observational comparison")
    expected={spec[3] for spec in PAIR_SPECS}
    pairs=_indexed(result.get("pairs"),lambda r:r.get("pair_id"))
    if set(pairs)!=expected:
        raise ValueError("Missing preregistered cross-stage pairs")
    names={"mnist_cnn":"CNN","mnist_mlp":"MLP"}
    axes={"baseline":"baseline","parallelism":"replicas","pes":"PEs","simd":"SIMD"}
    lines=[
        "% Reaudited source-to-physical observational identities; no board latency.",
        r"\begin{table}[t]",r"\centering\scriptsize",
        r"\resizebox{\textwidth}{!}{%",r"\begin{tabular}{@{}lllllll@{}}",
        r"\toprule",
        r"Model & Pair (A/B) & C++ tie & RTL tie & E3 tie & E4 QoR tie & E3 tie/E4 difference \\",
        r"\midrule",
    ]
    for model,ca,cb,ident in PAIR_SPECS:
        row=pairs[ident]
        if (row.get("model")!=model or row.get("axis_a") not in axes
                or row.get("axis_b") not in axes):
            raise ValueError("Unsafe or mismatched pair identity")
        fields=(
            row.get("source_cpp_file_bytes_equal"),
            row.get("fresh_generated_rtl_file_set_bytes_equal"),
            row.get("hls_six_objective_e3_estimates_equal"),
            row.get("routed_e4_wns_and_five_resources_equal"),
            row.get("e3_estimate_tie_hides_e4_qor_difference"),
        )
        if any(type(v) is not bool for v in fields):
            raise ValueError("Partial empirical identity classifications")
        values=("Yes" if flag else "No" for flag in fields)
        lines.append(" & ".join((names[model],
            axes[row["axis_a"]]+"/"+axes[row["axis_b"]],*values))+r" \\")
    lines.extend([
        r"\bottomrule",r"\end{tabular}}",
        r"\caption{Four prespecified, source-aligned architecture pairs "
        r"across HLS C++ bytes, fresh RTL file-set bytes, Vitis E3 six-objective "
        r"estimates and independently routed Vivado E4 WNS and five utilization "
        r"counts. A tie is equality of the stated observations only, never "
        r"functional equivalence or measured inference latency.}",
        r"\label{tab:fpgai-cross-stage-pairs}",r"\end{table}","",
    ])
    return "\n".join(lines)


def collect_cross_stage_pairs(original_e3_directory,independent_e4_directory):
    """Private-source real audit: no snapshots trusted and no tool execution."""
    from fpgai_research.experiments.isolated_hls_replay import audit_isolated_hls_replay
    root=Path(original_e3_directory); physical=Path(independent_e4_directory)
    original=collect_isolated_hls_e3_ablation(root)
    native_e4=collect_independent_e4(physical)
    e3_audit=audit_isolated_hls_replay(root)
    physical_audit=audit_isolated_hls_replay(physical)
    inventoried={}
    physical_hashes={}
    for row in e3_audit["cases"]:
        key=(row["model"],row["case"])
        if key in inventoried:
            raise ValueError("Duplicated original HLS replay")
        inventoried[key]=_case_rtl_inventory(root/key[0]/key[1],row)
    for row in physical_audit["cases"]:
        key=(row["model"],row["case"])
        if key in physical_hashes:
            raise ValueError("Duplicated independent physical case")
        physical_hashes[key]=row["source_sha256"]
    quotient=analyze_e3_design_space_quotient(original,e3_audit,inventoried)
    return analyze_cross_stage_pairs(
        original,quotient,native_e4,independent_source_hashes=physical_hashes)


def export_cross_stage_pairs(original_e3_directory,independent_e4_directory,
                             output):
    result=collect_cross_stage_pairs(original_e3_directory,independent_e4_directory)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Cross-stage scientific result must be immutable")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-cross-stage-",dir=target.parent) as tmp:
        stage=Path(tmp)
        (stage/"cross_stage_pairs.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        (stage/"cross_stage_pairs.tex").write_text(cross_stage_latex(result))
        lines=[
            "# Source-to-physical observational design identity","",
            "Four prespecified architecture contrasts from two model workloads.",
            "HLS cycles are E3 estimates; physical resources and WNS are "
            "source-attested out-of-context E4 observations.","",
            "| Model | A/B | C++ tie | RTL-byte tie | E3-estimate tie | E4-QoR tie | Hidden by E3 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for pair in result["pairs"]:
            lines.append("| "+" | ".join((
                pair["model"],pair["axis_a"]+"/"+pair["axis_b"],
                *(("yes" if pair[field] else "no") for field in (
                    "source_cpp_file_bytes_equal",
                    "fresh_generated_rtl_file_set_bytes_equal",
                    "hls_six_objective_e3_estimates_equal",
                    "routed_e4_wns_and_five_resources_equal",
                    "e3_estimate_tie_hides_e4_qor_difference",
                )),
            ))+" |")
        lines.extend(["","## Interpretation boundary","",result["limitations"],""])
        (stage/"cross_stage_pairs.md").write_text("\n".join(lines))
        if target.exists() or target.is_symlink():
            raise FileExistsError("Cross-stage output appeared during export")
        stage.rename(target)
    return result
