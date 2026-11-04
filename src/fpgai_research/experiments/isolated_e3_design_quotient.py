"""Source-attested architecture design-space quotient across E1/E3/Verilog.

An architecture request is not a unique synthesized circuit. This experiment
counts distinct within-model outputs at three *observational* levels:
emitted HLS C++ source, fresh complete Verilog file-byte inventories, and
a six-objective native E3 estimate vector (latency/LUT/FF/DSP/BRAM/URAM).
It never claims a logic-equivalence relation, achieved loop II or E4 QoR.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.isolated_hls_e3_ablation import collect_isolated_hls_e3_ablation
from fpgai_research.experiments.isolated_hls_replay import audit_isolated_hls_replay
from fpgai_research.experiments.isolated_hls_e3_rtl_inventory import _case_rtl_inventory
from fpgai_research.experiments.native_hls_rerun_plan import _read_json

SCHEMA="fpgai-research/source-to-rtl-design-space-quotient/v1"
_MODELS=("mnist_cnn","mnist_mlp")
_AXES=("baseline","parallelism","pes","simd")
_RESOURCES=("lut","ff","dsp","bram_18k","uram")
_SHA=re.compile(r"[a-f0-9]{64}\Z")
_FILENAME=re.compile(r"[A-Za-z0-9_.-]+\.(?:v|sv)\Z")


def _estimate_vector(row:dict)->tuple[int,...]:
    if row.get("comparison_status")!="QUALIFIED_E3_HLS_ESTIMATE":
        raise ValueError("Design-space quotient needs qualified E3 measurements")
    latency=row.get("estimated_latency_cycles")
    resource=row.get("estimated_hls_resource_counts")
    if not isinstance(resource,dict):
        raise ValueError("Missing E3 resource vector")
    values=[latency]
    for name in _RESOURCES:
        field=resource.get(name)
        if not isinstance(field,dict):
            raise ValueError("Missing E3 resource objective")
        values.append(field.get("case_hls_estimated"))
    if any(type(v) is not int or v<0 for v in values) or latency<=0:
        raise ValueError("Partial or invalid E3 objective vector")
    return tuple(values)


def _rtl_fingerprint(inventory:dict)->tuple[tuple[str,str],...]:
    if inventory.get("status")!="FRESH_SOURCE_ATTESTED_RTL":
        raise ValueError("Incomplete, stale or unaudited RTL cannot establish identity")
    sources=inventory.get("_digest_map")
    if (not isinstance(sources,dict) or not sources
            or inventory.get("file_count")!=len(sources)
            or len(sources)>4096):
        raise ValueError("Missing or incomplete fresh RTL inventory")
    if not all(isinstance(k,str) and _FILENAME.fullmatch(k)
               and isinstance(v,str) and _SHA.fullmatch(v)
               for k,v in sources.items()):
        raise ValueError("Unsafe private RTL digest map")
    # Used as a private comparison key ONLY. Neither names nor hashes export.
    return tuple(sorted(sources.items()))


def _classes(rows, key):
    mapping={}
    groups={}
    for row in sorted(rows,key=lambda r:r["case"]):
        value=key(row)
        if value not in mapping:
            mapping[value]=len(mapping)+1
        groups[row["case"]]=mapping[value]
    return groups,len(mapping)


def analyze_e3_design_space_quotient(comparison:dict,audit:dict,rtl_inventories:dict)->dict:
    """Pure, fail-closed comparison over independently attested case inputs."""
    if (comparison.get("schema")!="fpgai-research/isolated-native-hls-e3-ablation/v1"
            or audit.get("schema")!="fpgai-research/isolated-native-hls-replay-audit/v1"
            or comparison.get("source_suite_sha256")!=audit.get("source_suite_sha256")
            or comparison.get("native_tools_executed") is not False
            or comparison.get("operator_schedule_attribution_verified") is not False
            or comparison.get("e4_physical_comparison_performed") is not False
            or audit.get("native_tools_executed") is not False
            or audit.get("per_loop_schedule_evidence_available") is not False):
        raise ValueError("Comparison/audit provenance or evidence level invalid")
    rows=comparison.get("comparison_rows")
    cases=audit.get("cases")
    if (not isinstance(rows,list) or not isinstance(cases,list)
            or len(rows)!=comparison.get("case_count")
            or len(cases)!=audit.get("case_count")
            or len(rows)!=len(_MODELS)*len(_AXES)):
        raise ValueError("Expected complete eight-case matched MLP/CNN E3 inventory")
    by_key={}
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError("Invalid E3 row")
        key=(row.get("model"),row.get("case"))
        if key in by_key or key[0] not in _MODELS:
            raise ValueError("Duplicate or unexpected E3 case")
        if (not isinstance(key[1],str) or not re.fullmatch(r"case_[0-9]{4,8}",key[1])
                or row.get("axis") not in _AXES
                or row.get("numerical_csim_passed") is not True
                or row.get("fresh_report_attested") is not True
                or row.get("evidence")!="E3-HLS-synthesis-report"
                or row.get("operator_loop_ii_verified") is not False
                or row.get("measured_postroute_qor") is not False
                or not isinstance(row.get("source_sha256"),str)
                or not _SHA.fullmatch(row["source_sha256"])):
            raise ValueError("Unqualified or invalid E3 case")
        _estimate_vector(row)
        by_key[key]=row
    audited={}
    for item in cases:
        if not isinstance(item,dict):
            raise ValueError("Invalid native audit case")
        key=(item.get("model"),item.get("case"))
        if key in audited:
            raise ValueError("Duplicate native case")
        audited[key]=item
    if set(by_key)!=set(audited) or set(by_key)!=set(rtl_inventories):
        raise ValueError("Native cases do not align with the E3/RTL inventory")
    for key,row in by_key.items():
        a=audited[key]
        if (a.get("source_sha256")!=row["source_sha256"]
                or a.get("axis")!=row.get("axis")
                or a.get("evidence")!=row.get("evidence")
                or a.get("fixed_point_csim_passed") is not True
                or a.get("fresh_hls_report_attested") is not True
                or a.get("operator_loop_ii_verified") is not False):
            raise ValueError("Source/audit attestation disagrees")
    output=[]
    for model in _MODELS:
        subset=sorted((r for (m,_),r in by_key.items() if m==model),
                      key=lambda r:r["case"])
        if len(subset)!=len(_AXES) or {r["axis"] for r in subset}!=set(_AXES):
            raise ValueError("Missing or duplicated architecture axis")
        if not (subset[0]["case"]=="case_0000" and subset[0]["axis"]=="baseline"):
            raise ValueError("Canonical baseline unavailable")
        rtlkeys={r["case"]:_rtl_fingerprint(rtl_inventories[(model,r["case"])])
                 for r in subset}
        cpp,unique_cpp=_classes(subset,lambda r:r["source_sha256"])
        rtl,unique_rtl=_classes(subset,lambda r:rtlkeys[r["case"]])
        estimate,unique_e3=_classes(subset,_estimate_vector)
        cases_public=[{
            "case":r["case"],"axis":r["axis"],
            "generated_cpp_class":"C"+str(cpp[r["case"]]),
            "generated_verilog_file_byte_class":"R"+str(rtl[r["case"]]),
            "hls_six_objective_estimate_class":"H"+str(estimate[r["case"]]),
            "estimated_hls_latency_cycles":r["estimated_latency_cycles"],
            "operator_loop_ii_verified":False,
            "routed_e4_verified":False,
        } for r in subset]
        pairs=[]
        for a,b in combinations(subset,2):
            namea,nameb=a["case"],b["case"]
            samecpp=cpp[namea]==cpp[nameb]
            samertl=rtl[namea]==rtl[nameb]
            samee3=estimate[namea]==estimate[nameb]
            pairs.append({
                "case_a":namea,"case_b":nameb,
                "axis_a":a["axis"],"axis_b":b["axis"],
                "generated_cpp_bytes_equal":samecpp,
                "fresh_verilog_file_bytes_equal":samertl,
                "hls_six_objective_estimates_equal":samee3,
                "different_cpp_same_rtl":not samecpp and samertl,
                "different_rtl_same_estimates":not samertl and samee3,
                "logic_equivalence_verified":False,
                "routed_e4_verified":False,
            })
        output.append({
            "model":model,
            "architecture_cases":len(subset),
            "distinct_generated_cpp_file_classes":unique_cpp,
            "distinct_generated_rtl_file_set_classes":unique_rtl,
            "distinct_hls_six_objective_estimate_classes":unique_e3,
            "case_comparison_count":len(pairs),
            "different_cpp_same_rtl_pair_count":sum(p["different_cpp_same_rtl"] for p in pairs),
            "different_rtl_same_e3_estimate_pair_count":
                sum(p["different_rtl_same_estimates"] for p in pairs),
            "cases":cases_public,"pairs":pairs,
        })
    return {
        "schema":SCHEMA,
        "source_suite_sha256":comparison["source_suite_sha256"],
        "model_count":len(output),
        "case_count":len(rows),
        "within_model_pair_count":sum(v["case_comparison_count"] for v in output),
        "models":output,
        "native_tools_executed":False,
        "routed_e4_comparison_performed":False,
        "operator_schedule_attribution_verified":False,
        "formal_rtl_equivalence_verified":False,
        "private_sources_exported":False,
        "private_rtl_filenames_or_hashes_exported":False,
        "weight_values_exported":False,
        "limitations":(
            "The quotient is observational, not formal circuit equivalence. "
            "Distinct RTL file byte sets are not necessarily functionally "
            "or physically distinct circuits. Coincident E3 six-objective "
            "estimates do not imply identical RTL or P&R results. No "
            "operator achieved II, routed E4, timing, power, board latency "
            "or full MNIST test-set accuracy is established. "
            "Only four matched architecture points per model were tested; "
            "no statistical/generalization claim about the full seven-axis "
            "architecture space is supported."
        ),
    }


def render_design_quotient_latex(result:dict)->str:
    """Privacy-safe, deterministic observational class table for publication."""
    if (result.get("schema")!=SCHEMA
            or result.get("native_tools_executed") is not False
            or result.get("routed_e4_comparison_performed") is not False
            or result.get("private_rtl_filenames_or_hashes_exported") is not False
            or result.get("weight_values_exported") is not False
            or result.get("formal_rtl_equivalence_verified") is not False):
        raise ValueError("Design-class publication requires qualified E3-only evidence")
    # Render no dynamic source names, RTL filenames, hashes, or weights.
    # The only model tokens are closed vocabulary; all other cells are
    # nonnegative integer group cardinalities.
    latex=[
        "% Source-reaudited native E3: design identity at observable levels only.",
        "% Byte identity is NOT formal logic or post-route equivalence.",
        r"\begin{table}[t]",r"\centering",r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Model & Requests & C++ classes & Verilog-byte classes & E3-estimate classes \\",
        r"\midrule",
    ]
    for item in result["models"]:
        name={"mnist_cnn":"CNN","mnist_mlp":"MLP"}.get(item["model"])
        if name is None:
            raise ValueError("Unsafe scientific model label")
        cells=(item["architecture_cases"],
               item["distinct_generated_cpp_file_classes"],
               item["distinct_generated_rtl_file_set_classes"],
               item["distinct_hls_six_objective_estimate_classes"])
        if any(type(c) is not int or not 1<=c<=4 for c in cells):
            raise ValueError("Unqualified design-space class cardinality")
        latex.append(" & ".join((name,*(str(c) for c in cells)))+r" \\")
    latex.extend([
        r"\bottomrule",r"\end{tabular}",
        r"\caption{Observed diversity of four source-attested architecture "
        r"settings per trained MNIST model. Verilog classes require exact "
        r"byte equality of complete fresh RTL file sets, not functional "
        r"or physical equivalence. E3 classes use the six HLS estimates "
        r"(cycles, LUT, FF, DSP, BRAM, URAM); these are not routed E4 "
        r"quality-of-result measurements.}",
        r"\label{tab:fpgai-e3-design-quotient}",
        r"\end{table}","",
    ])
    return "\n".join(latex)


def export_e3_design_space_quotient(directory,comparison_path,output)->dict:
    """Never synthesize, mutate native case outputs or trust a stale snapshot."""
    root=Path(directory)
    doc=_read_json(Path(comparison_path))
    if doc!=collect_isolated_hls_e3_ablation(root):
        raise ValueError("Saved comparison diverges from current source-attested E3")
    audit=audit_isolated_hls_replay(root)
    inventories={}
    for record in audit["cases"]:
        key=(record["model"],record["case"])
        if key in inventories:
            raise ValueError("Duplicate private case identity")
        inventories[key]=_case_rtl_inventory(root/key[0]/key[1],record)
    result=analyze_e3_design_space_quotient(doc,audit,inventories)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Do not overwrite original native/derived findings")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-design-quotient-",dir=target.parent) as td:
        stage=Path(td)
        (stage/"design_space_quotient.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        lines=[
            "# Architecture-space observability from source to native E3","",
            "**Four matched architecture choices per model; only E3 synthesis "
            "estimates and fresh generated RTL bytes, not physical E4.**","",
            "| Model | Choices | Distinct C++ | Distinct generated RTL byte sets | "
            "Distinct E3 six-objective vectors |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for item in result["models"]:
            lines.append("| "+" | ".join(str(x) for x in (
                item["model"],item["architecture_cases"],
                item["distinct_generated_cpp_file_classes"],
                item["distinct_generated_rtl_file_set_classes"],
                item["distinct_hls_six_objective_estimate_classes"]
            ))+" |")
        lines+=["","## Within-model design identities","",
            "Class labels C/R/H are local to one model; they are not "
            "cross-model hashes, physical equivalence, or hardware IDs.","",
            "| Model | Case | Axis | C++ class | RTL byte class | E3 estimate class |",
            "| --- | --- | --- | --- | --- | --- |"]
        for item in result["models"]:
            for row in item["cases"]:
                lines.append("| "+" | ".join(str(x) for x in (
                    item["model"],row["case"],row["axis"],
                    row["generated_cpp_class"],
                    row["generated_verilog_file_byte_class"],
                    row["hls_six_objective_estimate_class"]
                ))+" |")
        lines+=["","## Experimental predictions worth falsifying with routed E4","",
            "1. CNN baseline/SIMD and MLP replicas/PEs have byte-identical "
            "fresh RTL. Under matched OOC Vivado constraints, any "
            "difference in physical QoR must be investigated as "
            "implementation/environment variability, not attributed "
            "to source-level PE/SIMD directly.","",
            "2. CNN replicas/PEs have token-different RTL but identical "
            "E3 latency and resource estimates. Matched routed "
            "comparison can test whether physical implementations "
            "remain distinct or converge; either outcome is publishable "
            "when source- and constraint-attested.","",
            "3. MLP SIMD has a lower E3 latency estimate. A routed "
            "study cannot establish application throughput or "
            "real runtime by itself; report timing closure and "
            "separate achieved latency evidence when available.","",
            "These are pre-registered hypotheses only: **not run**.",
            "Do not compare unmatched implementation modes, numerical "
            "failures, different clocks/parts or unqualified reports.","",
            "## Boundaries","",result["limitations"],""]
        (stage/"design_space_quotient.md").write_text("\n".join(lines))
        (stage/"design_space_quotient.tex").write_text(
            render_design_quotient_latex(result),encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Destination appeared during export")
        stage.rename(target)
    return result
