"""Publication-bound, source-attested E3 to Verilog evidence chain.

No native tool execution, retrospective E4 claims, private tensors, source
code, Verilog filenames, RTL digests, or unverified scheduling metrics.
Scientific status is descriptive: identical output files are not a proof
of circuit equivalence, and equal estimates are not physical QoR.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.isolated_hls_e3_ablation import (
    collect_isolated_hls_e3_ablation,
)
from fpgai_research.experiments.isolated_hls_e3_interpretation import (
    interpret_isolated_hls_e3,
)
from fpgai_research.experiments.isolated_hls_e3_source_diagnostics import (
    diagnose_equal_e3_sources,
)
from fpgai_research.experiments.isolated_hls_e3_rtl_inventory import (
    inspect_tied_e3_rtl,
)
from fpgai_research.experiments.isolated_hls_replay import (
    audit_isolated_hls_replay,
)
from fpgai_research.experiments.native_hls_rerun_plan import _read_json

SCHEMA="fpgai-research/paper-e3-source-to-rtl-evidence/v1"
_RESOURCE_FIELDS=("lut","ff","dsp","bram_18k","uram")
_CASE_FIELDS=("model","case","axis","qualification","csim_passed",
              "hls_estimated_latency_cycles","hls_estimated_top_call_interval",
              "hls_estimated_latency_reduction_pct","hls_estimated_lut",
              "hls_estimated_ff","hls_estimated_dsp","hls_estimated_bram_18k",
              "hls_estimated_uram","hls_estimate_pareto_status",
              "source_vs_baseline","operator_schedule_attribution_verified",
              "routed_e4_evidence_verified")
_PAIR_FIELDS=("model","case_a","case_b","axis_a","axis_b",
              "different_generated_cpp","equal_hls_estimate_vector",
              "rtl_evidence_class","verified_rtl_byte_equality",
              "common_rtl_file_count","different_rtl_byte_file_count",
              "different_rtl_token_file_count","lexical_status_unclassified_file_count",
              "affected_operator_count","changed_lowered_mechanism_keys",
              "static_simd_upper_bound_equal","static_spatial_upper_bound_equal",
              "source_level_hypothesis_only","operator_schedule_attribution_verified",
              "routed_e4_evidence_verified")

def _indexed(rows, *, model_key="model", a_key="case_a", b_key="case_b"):
    found={}
    for row in rows:
        key=(row[model_key],*sorted((row[a_key],row[b_key])))
        if key in found:
            raise ValueError("Duplicated source/RTL pair identity")
        found[key]=row
    return found


def _rtl_evidence_label(pair:dict)->str:
    status=pair["rtl_comparison_status"]
    if status=="IDENTICAL_GENERATED_RTL_FILE_BYTES":
        if pair.get("rtl_file_set_byte_identical") is not True:
            raise ValueError("Contradictory RTL byte-identity evidence")
        return "BYTE_IDENTICAL_FRESH_GENERATED_RTL"
    if status=="INCONCLUSIVE_RTL_PROVENANCE_OR_AVAILABILITY":
        if pair.get("rtl_file_set_byte_identical") is not None:
            raise ValueError("Inconclusive RTL cannot claim equality")
        return "RTL_EVIDENCE_INCONCLUSIVE"
    if status!="DIFFERENT_GENERATED_RTL_FILE_BYTES" or pair.get("rtl_file_set_byte_identical") is not False:
        raise ValueError("Unrecognized or contradictory RTL evidence")
    delta=pair.get("rtl_file_difference_counts")
    if not isinstance(delta,dict):
        raise ValueError("Fresh, differing RTL requires a file-difference inventory")
    only_a=delta.get("only_in_case_a_file_count")
    only_b=delta.get("only_in_case_b_file_count")
    changed=delta.get("changed_byte_file_count")
    equal=delta.get("changed_files_lexically_equal_count")
    different=delta.get("changed_files_lexically_different_count")
    unknown=delta.get("changed_files_lexically_unclassified_count")
    if (any(type(x) is not int or x<0 for x in (only_a,only_b,changed,equal,different,unknown))
            or equal+different+unknown!=changed
            or changed+only_a+only_b==0):
        raise ValueError("Inconsistent byte/lexical difference counts")
    if only_a or only_b:
        return "GENERATED_RTL_FILE_SET_MEMBERSHIP_DIFFERS"
    if different:
        return "DIFFERENT_NONCOMMENT_RTL_TOKENS"
    if unknown:
        return "DIFFERENT_RTL_BYTES_LEXICAL_STATUS_INCOMPLETE"
    if equal==changed:
        return "RTL_BYTES_DIFFER_ONLY_IN_COMMENTS_OR_SPACING"
    raise ValueError("Unclassifiable RTL byte difference")


def build_paper_e3_evidence(comparison:dict,source_diagnostics:dict,rtl_diagnostics:dict)->dict:
    """Join only mutually attested E3 case/pair inventories.

    Export calls independently recalculate all three source-backed inputs.
    This pure aggregation API also checks paired identities and evidence
    boundaries, and is intentionally useful for synthetic regression tests.
    """
    interpreted=interpret_isolated_hls_e3(comparison)
    if any(flag is not False for flag in (
            source_diagnostics.get("native_tools_executed"),
            source_diagnostics.get("operator_schedule_attribution_verified"),
            source_diagnostics.get("weight_values_exported"),
            source_diagnostics.get("private_source_exported"),
            source_diagnostics.get("operator_names_exported"),
            rtl_diagnostics.get("native_tools_executed"),
            rtl_diagnostics.get("operator_schedule_attribution_verified"),
            rtl_diagnostics.get("logic_equivalence_verified"),
            rtl_diagnostics.get("e4_physical_comparison_performed"),
            rtl_diagnostics.get("private_rtl_exported"),
            rtl_diagnostics.get("rtl_filenames_exported"),
            rtl_diagnostics.get("rtl_digests_exported"),
            rtl_diagnostics.get("weight_values_exported"))):
        raise ValueError("Missing research provenance boundaries")
    source_pairs=_indexed(source_diagnostics["pairs"])
    rtl_pairs=_indexed(rtl_diagnostics["pairs"])
    if (set(source_pairs)!=set(rtl_pairs)
            or len(source_pairs)!=interpreted["equal_hls_estimate_pair_count"]
            or len(source_pairs)!=source_diagnostics.get("equal_e3_estimate_pair_count")
            or len(rtl_pairs)!=rtl_diagnostics.get("equal_e3_estimate_pair_count")
            or source_diagnostics.get("source_suite_sha256")!=comparison.get("source_suite_sha256")
            or rtl_diagnostics.get("source_suite_sha256")!=comparison.get("source_suite_sha256")):
        raise ValueError("Cross-study E3 comparison/RTL/source identity mismatch")
    rows_by_key={(r["model"],r["case"]):r for r in comparison["comparison_rows"]}
    interpreted_by_key={(r["model"],r["case"]):r for r in interpreted["cases"]}
    if len(rows_by_key)!=len(interpreted_by_key):
        raise ValueError("Case identities cannot be duplicated")
    cases=[]
    for key in sorted(rows_by_key):
        row=rows_by_key[key]
        extra=interpreted_by_key[key]
        eligible=row["comparison_status"]=="QUALIFIED_E3_HLS_ESTIMATE"
        resources=row["estimated_hls_resource_counts"] if eligible else {}
        cases.append({
            "model":row["model"],"case":row["case"],"axis":row["axis"],
            "qualification":row["comparison_status"],
            "csim_passed":row["numerical_csim_passed"] is True,
            "hls_estimated_latency_cycles":row["estimated_latency_cycles"] if eligible else None,
            "hls_estimated_top_call_interval":row["estimated_top_call_interval_min"] if eligible else None,
            "hls_estimated_latency_reduction_pct":row["estimated_latency_reduction_pct"] if eligible else None,
            **{"hls_estimated_"+kind:resources[kind]["case_hls_estimated"]
               if eligible else None for kind in _RESOURCE_FIELDS},
            "hls_estimate_pareto_status":extra["pareto_status"],
            "source_vs_baseline":extra["source_vs_baseline"],
            "operator_schedule_attribution_verified":False,
            "routed_e4_evidence_verified":False,
        })
    joined=[]
    for key in sorted(source_pairs):
        src=source_pairs[key]
        rtl=rtl_pairs[key]
        model,a,b=key
        if (src["case_a"]!=a or src["case_b"]!=b
                or rtl["case_a"]!=a or rtl["case_b"]!=b
                or src["source_cpp_sha256_equal"] is not False
                or src["e3_hls_estimate_vector_equal"] is not True
                or rtl.get("logic_equivalence_verified") is not False
                or rtl.get("routed_e4_evidence_verified") is not False
                or rtl.get("operator_loop_ii_verified") is not False
                or src.get("causal_mechanism_verified") is not False
                or src.get("operator_loop_ii_verified") is not False
                or rtl["hls_estimate_vector_equal"] is not True
                or rtl["source_cpp_identical"] is not False
                or any(rows_by_key[(model,k)]["comparison_status"]!="QUALIFIED_E3_HLS_ESTIMATE"
                       for k in (a,b))
                or b not in interpreted_by_key[(model,a)]["equal_estimate_cases"]):
            raise ValueError("Source/RTL pair contradicts E3-qualified equality")
        geometry=src["changed_operator_mechanisms"]
        if (not isinstance(geometry,list) or
                len(geometry)!=src["changed_operator_count"] or
                not geometry):
            raise ValueError("Paired E3 tie lacks aligned operator mechanism change")
        lowers=set()
        simd_upper_equal=None
        spatial_upper_equal=None
        for op in geometry:
            changed=op["changed_lowered_mechanism_keys"]
            if not isinstance(changed,list) or not all(isinstance(x,str) for x in changed):
                raise ValueError("Invalid mechanism whitelist values")
            if not set(changed)<={"tile_shape","replicas","pe_count","simd_lanes",
                                 "fold_factor","pipeline_ii","tail_policy","numerical_type"}:
                raise ValueError("Unknown mechanism field must never be exported")
            lowers.update(changed)
            g=op.get("static_workload_geometry") or {}
            if g.get("geometry_status")=="SUPPORTED_STATIC_EMITTER_COORDINATES":
                simd_upper_equal=(g["simd_lanes_static_upper_bound_equal"]
                                  if simd_upper_equal is None else
                                  simd_upper_equal and g["simd_lanes_static_upper_bound_equal"])
                spatial_upper_equal=(g["spatial_slots_static_upper_bound_equal"]
                                     if spatial_upper_equal is None else
                                     spatial_upper_equal and g["spatial_slots_static_upper_bound_equal"])
        delta=rtl.get("rtl_file_difference_counts")
        if delta is not None and not isinstance(delta,dict):
            raise ValueError("Invalid RTL difference inventory")
        label=_rtl_evidence_label(rtl)
        if label=="BYTE_IDENTICAL_FRESH_GENERATED_RTL" and delta is not None:
            if (delta.get("changed_byte_file_count")!=0
                    or delta.get("only_in_case_a_file_count")!=0
                    or delta.get("only_in_case_b_file_count")!=0):
                raise ValueError("Byte-equal RTL cannot contain different files")
        joined.append({
            "model":model,"case_a":a,"case_b":b,
            "axis_a":src["axis_a"],"axis_b":src["axis_b"],
            "different_generated_cpp":True,"equal_hls_estimate_vector":True,
            "rtl_evidence_class":label,
            "verified_rtl_byte_equality":rtl["rtl_file_set_byte_identical"],
            "common_rtl_file_count":delta.get("common_file_count") if delta else None,
            "different_rtl_byte_file_count":delta.get("changed_byte_file_count") if delta else None,
            "different_rtl_token_file_count":delta.get("changed_files_lexically_different_count") if delta else None,
            "lexical_status_unclassified_file_count":
                delta.get("changed_files_lexically_unclassified_count") if delta else None,
            "affected_operator_count":len(geometry),
            "changed_lowered_mechanism_keys":sorted(lowers),
            "static_simd_upper_bound_equal":simd_upper_equal,
            "static_spatial_upper_bound_equal":spatial_upper_equal,
            "source_level_hypothesis_only":True,
            "operator_schedule_attribution_verified":False,
            "routed_e4_evidence_verified":False,
        })
    from collections import Counter
    counts=Counter(pair["rtl_evidence_class"] for pair in joined)
    return {
        "schema":SCHEMA,
        "source_suite_sha256":comparison["source_suite_sha256"],
        "case_count":len(cases),
        "qualified_e3_case_count":sum(r["qualification"]=="QUALIFIED_E3_HLS_ESTIMATE"
                                     for r in cases),
        "dominated_hls_estimate_case_count":interpreted["dominated_hls_estimate_case_count"],
        "equal_hls_estimate_pair_count":len(joined),
        "rtl_evidence_class_counts":dict(sorted(counts.items())),
        "cases":cases,"pairs":joined,
        "native_tools_executed":False,
        "operator_schedule_attribution_verified":False,
        "routed_e4_evidence_verified":False,
        "logic_equivalence_verified":False,
        "weight_values_exported":False,
        "private_rtl_exported":False,
        "private_source_exported":False,
        "limitations":(
            "E2 numerical acceptance covers the prepared CSim sample set; "
            "E3 latency, call interval and resource values are HLS synthesis "
            "ESTIMATES, not achieved operator loop II, implemented utilization "
            "or physical power/timing. E1 requested/effective/lowered "
            "architecture changes and source-level static geometry only "
            "support hypotheses. Byte-identical generated Verilog is not "
            "formal equivalence; different Verilog tokens do not establish "
            "behavioral inequivalence. No routed E4, board deployment, full "
            "test-set accuracy or generalization beyond these cases."
        ),
    }



# Intentional, closed-vocabulary LaTeX rendering: no raw JSON labels,
# paths, source fragments, weights, operator names, or RTL identifiers.
_MODEL_LATEX={"mnist_mlp":"MLP","mnist_cnn":"CNN"}
_AXIS_LATEX={"baseline":"baseline","parallelism":"replicas",
             "pes":"PEs","simd":"SIMD","precision":"precision",
             "serialization":"reuse","tile":"tile","pipeline_ii":"pipeline"}
_RTL_LATEX={
    "BYTE_IDENTICAL_FRESH_GENERATED_RTL":"byte-identical",
    "DIFFERENT_NONCOMMENT_RTL_TOKENS":"different tokens",
    "RTL_BYTES_DIFFER_ONLY_IN_COMMENTS_OR_SPACING":"comments/spacing only",
    "DIFFERENT_RTL_BYTES_LEXICAL_STATUS_INCOMPLETE":"unclassified text",
    "GENERATED_RTL_FILE_SET_MEMBERSHIP_DIFFERS":"different file sets",
    "RTL_EVIDENCE_INCONCLUSIVE":"inconclusive",
}
_MECHANISM_LATEX={
    "replicas":"replicas","pe_count":"PE count","simd_lanes":"SIMD width",
    "tile_shape":"tile shape","fold_factor":"fold factor",
    "pipeline_ii":"pipeline II","tail_policy":"tail policy",
    "numerical_type":"number format",
}


def paper_e3_latex(result:dict)->str:
    """Typeset an already-evidence-checked result using fixed safe labels."""
    if (result.get("schema")!=SCHEMA or
            result.get("native_tools_executed") is not False or
            result.get("routed_e4_evidence_verified") is not False or
            result.get("operator_schedule_attribution_verified") is not False or
            result.get("weight_values_exported") is not False or
            result.get("private_rtl_exported") is not False or
            result.get("private_source_exported") is not False):
        raise ValueError("Unqualified E3 data is not permitted in manuscript tables")
    lines=[
        "% Autogenerated from source-reaudited, private native Vitis E3 evidence.",
        "% This is NOT a routed E4, bitwise equivalence or achieved loop-II table.",
        r"\begin{table}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrrl@{}}",
        r"\toprule",
        r"Model & Axis & Cycles & $\Delta L$ (\%) & LUT & FF & DSP & BRAM & URAM & Pareto \\",
        r"\midrule",
    ]
    if not isinstance(result.get("cases"),list):
        raise ValueError("Missing paper E3 case inventory")
    for row in result["cases"]:
        model=_MODEL_LATEX.get(row.get("model"))
        axis=_AXIS_LATEX.get(row.get("axis"))
        if model is None or axis is None:
            raise ValueError("Unknown E3 model or architecture axis")
        pareto=row.get("hls_estimate_pareto_status")
        if pareto=="NONDOMINATED_AMONG_COMPLETE_VECTORS":
            short="ND"
        elif pareto=="DOMINATED_HLS_ESTIMATE":
            short="D"
        else:
            raise ValueError("Only complete, qualified E3 vectors may be typeset")
        if row.get("qualification")!="QUALIFIED_E3_HLS_ESTIMATE" or row.get("csim_passed") is not True:
            raise ValueError("Unqualified E3 numerical evidence cannot enter final table")
        keys=["hls_estimated_latency_cycles","hls_estimated_lut",
              "hls_estimated_ff","hls_estimated_dsp",
              "hls_estimated_bram_18k","hls_estimated_uram"]
        metrics=[row.get(k) for k in keys]
        pct=row.get("hls_estimated_latency_reduction_pct")
        if any(type(v) is not int or v<0 for v in metrics) or type(pct) not in (int,float):
            raise ValueError("Unqualified numerical HLS estimates")
        if metrics[0]<=0 or not (-10000<=pct<=100):
            raise ValueError("Unexpected estimate magnitude")
        fields=[model,axis,str(metrics[0]),f"{pct:+.2f}",
                *[str(v) for v in metrics[1:]],short]
        lines.append(" & ".join(fields)+r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Qualified within-model native E3 HLS synthesis estimates only. "
        r"$\Delta L$ is estimated latency reduction from each model baseline. "
        r"ND/D denote nondominated/dominated among complete six-objective "
        r"(cycles, LUT, FF, DSP, BRAM, URAM) estimate vectors; "
        r"neither implies routed E4 timing or achieved per-operator II.}",
        r"\label{tab:fpgai-native-e3-estimates}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llllr@{}}",
        r"\toprule",
        r"Model & Pair & Changed E1 mechanisms & Fresh Verilog & Token-different files \\",
        r"\midrule",
    ])
    for pair in result.get("pairs",[]):
        model=_MODEL_LATEX.get(pair.get("model"))
        aa=_AXIS_LATEX.get(pair.get("axis_a"))
        bb=_AXIS_LATEX.get(pair.get("axis_b"))
        status=_RTL_LATEX.get(pair.get("rtl_evidence_class"))
        mechanisms=pair.get("changed_lowered_mechanism_keys")
        if (not all((model,aa,bb,status)) or
                not isinstance(mechanisms,list) or
                not all(k in _MECHANISM_LATEX for k in mechanisms)):
            raise ValueError("Unsupported private or unknown manuscript field")
        count=pair.get("different_rtl_token_file_count")
        if count is not None and (type(count) is not int or count<0):
            raise ValueError("Invalid RTL lexical difference count")
        lines.append(" & ".join((
            model,aa+" / "+bb,
            ", ".join(_MECHANISM_LATEX[k] for k in mechanisms) or "none",
            status,str(count) if count is not None else "TBD",
        ))+r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{E3-equal-estimate architecture pairs with differing generated "
        r"HLS C++ sources. Fresh generated Verilog equality is byte-level, "
        r"not formal logic equivalence. Token-different files reflect textual "
        r"differences beyond comments/whitespace, not necessarily changed "
        r"hardware behavior. No routed E4 evidence is inferred.}",
        r"\label{tab:fpgai-native-e3-rtl-convergence}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def export_paper_e3_evidence(directory,comparison_path,output)->dict:
    """Recompute all raw-source-attested evidence; never trust stale exports."""
    root=Path(directory)
    saved=_read_json(Path(comparison_path))
    native=collect_isolated_hls_e3_ablation(root)
    if saved!=native:
        raise ValueError("E3 comparison snapshot differs from reaudited native replay")
    audited=audit_isolated_hls_replay(root)
    source=diagnose_equal_e3_sources(root,native,audited)
    rtl=inspect_tied_e3_rtl(root,native,audited)
    result=build_paper_e3_evidence(native,source,rtl)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable paper E3 evidence output already exists")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-paper-e3-",dir=target.parent) as tmp:
        stage=Path(tmp)
        (stage/"paper_e3_evidence.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        for filename,fields,rows in (
                ("paper_e3_cases.csv",_CASE_FIELDS,result["cases"]),
                ("paper_e3_pairs.csv",_PAIR_FIELDS,result["pairs"])):
            with (stage/filename).open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        k:",".join(row[k]) if isinstance(row.get(k),list)
                        else "TBD" if row.get(k) is None else row[k] for k in fields
                    })
        lines=[
            "# Evidence-bounded MNIST architecture ablations (E1→E3→generated RTL)","",
            "**All latency and resource measurements below are E3 HLS estimates, "
            "not routed E4 physical results.**","",
            f"Cases: {result['case_count']}; qualified E3: "
            f"{result['qualified_e3_case_count']}; E3-tied pairs: "
            f"{result['equal_hls_estimate_pair_count']}; dominated HLS "
            f"estimate configurations: {result['dominated_hls_estimate_case_count']}.","",
            "## Source-to-RTL convergence in E3-tied architectures","",
            "| Model | Cases | Architecture mechanism keys changed | RTL evidence | "
            "RTL files with different tokens |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for pair in result["pairs"]:
            count=pair["different_rtl_token_file_count"]
            lines.append("| "+" | ".join((
                pair["model"],pair["case_a"]+" / "+pair["case_b"],
                ", ".join(pair["changed_lowered_mechanism_keys"]) or "—",
                pair["rtl_evidence_class"],
                str(count) if count is not None else "TBD",
            ))+" |")
        lines.extend(["",
            "## Boundaries on hypotheses, not HLS schedule assertions","",
            "Each row below is an architectural geometry observation from E1 "
            "source and the HLS emitter, not achieved physical lane utilization.",
            "",
            "| Model | Cases | Static SIMD lane upper bounds equal | "
            "Static spatial slot upper bounds equal |",
            "| --- | --- | --- | --- |",
        ])
        for pair in result["pairs"]:
            lines.append("| "+" | ".join((
                pair["model"],pair["case_a"]+" / "+pair["case_b"],
                str(pair["static_simd_upper_bound_equal"])
                    if pair["static_simd_upper_bound_equal"] is not None else "TBD",
                str(pair["static_spatial_upper_bound_equal"])
                    if pair["static_spatial_upper_bound_equal"] is not None else "TBD",
            ))+" |")
        lines.extend(["","## Within-model, source-attested E3 estimates","",
            "| Model | Case | Axis | Best latency (cycles) | "
            "Latency reduction vs baseline (%) | HLS LUT | FF | DSP | "
            "BRAM_18K | URAM | Estimated Pareto |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in result["cases"]:
            fields=(row["model"],row["case"],row["axis"],
                    row["hls_estimated_latency_cycles"],
                    row["hls_estimated_latency_reduction_pct"],
                    *[row["hls_estimated_"+k] for k in _RESOURCE_FIELDS],
                    row["hls_estimate_pareto_status"])
            lines.append("| "+" | ".join(str(v) if v is not None else "TBD" for v in fields)+" |")
        lines.extend(["","## Evidence and interpretation limits","",result["limitations"],""])
        (stage/"paper_e3_evidence.md").write_text("\n".join(lines),encoding="utf-8")
        (stage/"paper_e3_tables.tex").write_text(
            paper_e3_latex(result),encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Paper evidence output appeared during export")
        stage.rename(target)
    return result
