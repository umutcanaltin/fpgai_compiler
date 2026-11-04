"""Evidence-bounded audit of a separately staged native E4 research replay.

Read-only: no Vitis/Vivado execution, receipt copying or manipulation of
generated sources. Only source-bound, tool-attributed results are considered.
Published output contains no RTL filenames, hashes, private paths or weights.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.isolated_hls_replay import audit_isolated_hls_replay
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.suite_native_batch import _qualify, _qualify_hls
from fpgai_research.experiments.independent_e4_pairs import (
    PAIR_SPECS, compare_qualified_e4_pairs,
)
from fpgai_research.experiments.independent_e4_latex import render_native_e4_latex

SCHEMA="fpgai-research/independent-matched-e4-audit/v1"
MODELS=("mnist_cnn","mnist_mlp")
AXES={"case_0000":"baseline","case_0004":"parallelism",
      "case_0005":"pes","case_0006":"simd"}
_HEX=re.compile(r"[0-9a-f]{64}\Z")


def summarize_independent_e4(root,audit,summaries)->dict:
    """Pure comparison; never substitute tool success for E4 qualification."""
    if (audit.get("schema")!="fpgai-research/isolated-native-hls-replay-audit/v1"
            or audit.get("native_tools_executed") is not False
            or audit.get("per_loop_schedule_evidence_available") is not False
            or not isinstance(audit.get("source_suite_sha256"),str)
            or not _HEX.fullmatch(audit["source_suite_sha256"])):
        raise ValueError("Untrusted source replay audit")
    rows=audit.get("cases")
    expected={(m,c) for m in MODELS for c in AXES}
    if (not isinstance(rows,list) or len(rows)!=8
            or not isinstance(summaries,dict) or set(summaries)!=expected):
        raise ValueError("Expected all eight matched model/architecture cases")
    attested={}
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError("Invalid source provenance")
        key=(row.get("model"),row.get("case"))
        if (key not in expected or key in attested
                or row.get("axis")!=AXES[key[1]]
                or row.get("operator_loop_ii_verified") is not False):
            raise ValueError("Duplicated or malformed native case")
        attested[key]=row
    if set(attested)!=expected:
        raise ValueError("Source replay inventory incomplete")
    public=[]
    for model in MODELS:
        for case,axis in AXES.items():
            key=(model,case)
            summary=summaries[key]
            record=attested[key]
            if (not isinstance(summary,dict)
                    or summary.get("source_sha256")!=record.get("source_sha256")
                    or record.get("evidence")!=summary.get("evidence")
                    or record.get("fixed_point_csim_passed")!=summary.get("csim_passed")
                    or not isinstance(summary.get("source_sha256"),str)
                    or not _HEX.fullmatch(summary["source_sha256"])):
                raise ValueError("Case report and source audit disagree")
            hls_blockers=_qualify_hls(summary)
            e4_blockers=_qualify(summary)
            if hls_blockers:
                e4_blockers=[*e4_blockers,
                             "Source-attested E3 and fixed-point numerical gates are incomplete"]
            physical=summary.get("vivado_metrics") or {}
            e4_attested=summary["evidence"]=="E4-Vivado-implementation"
            if not e4_attested:
                physical={}
            if not e4_blockers:
                status="E4_QUALIFIED_MATCHED_OOC"
            elif e4_attested:
                status="E4_ATTESTED_BUT_QUALIFICATION_BLOCKED"
            elif not hls_blockers:
                status="E3_QUALIFIED_E4_NOT_ATTESTED"
            else:
                status="E3_OR_E2_NOT_QUALIFIED"
            timing=physical.get("timing") or {}
            util=physical.get("utilization") or {}
            drc=physical.get("drc") or {}
            # All exported numerical fields are from source-attested
            # summarize_case; no RTL source, path, report SHA or raw warnings.
            public.append({
                "model":model,"case":case,"axis":axis,
                "evidence":summary["evidence"],
                "numerical_csim_passed":summary.get("csim_passed") is True,
                "qualified_e3":not hls_blockers,
                "source_attested_routed_e4":e4_attested,
                "qualified_e4":not e4_blockers,
                "qualification_status":status,
                "postroute_wns_ns":timing.get("wns_ns"),
                "postroute_timing_met":timing.get("timing_met"),
                "routed_nets_complete":physical.get("routed_nets_complete"),
                "implementation_mode":summary.get("vivado_implementation_mode") if e4_attested else None,
                "postroute_utilization":{
                    x:util.get(x) for x in ("lut","ff","dsp","bram","uram")
                } if e4_attested else None,
                "drc_summary_verified":drc.get("summary_verified"),
                "drc_critical_warnings":drc.get("critical_warnings"),
                "drc_unconstrained_io":drc.get("unconstrained_io"),
                "e3_blockers":hls_blockers,
                "e4_blockers":e4_blockers,
                "operator_loop_ii_verified":False,
                "board_runtime_measured":False,
            })
    return {
        "schema":SCHEMA,"source_suite_sha256":audit["source_suite_sha256"],
        "case_count":len(public),
        "e3_qualified_case_count":sum(row["qualified_e3"] for row in public),
        "routed_e4_attested_case_count":sum(row["source_attested_routed_e4"] for row in public),
        "e4_qualified_case_count":sum(row["qualified_e4"] for row in public),
        "cases":public,
        "vendor_tools_executed_by_audit":False,
        "operator_schedule_attribution_verified":False,
        "on_device_latency_verified":False,
        "original_publication_e3_untouched":True,
        "private_paths_or_digests_exported":False,
        "private_weights_exported":False,
        "limitations":(
            "E4 requires source- and RTL-attested Vivado reports, complete routing, "
            "nonnegative WNS, recognized postroute resources, source-attested "
            "fixed-point CSim, OOC mode and DRC critical/unconstrained-I/O gates. "
            "Passing vendor exit alone is insufficient. A single qualified E4 "
            "case is not a matched cross-architecture comparison or measured "
            "FPGA-board inference latency."
        ),
    }


def _private_matched_context(case:Path, summary:dict)->tuple|None:
    """Read already source-attested input/receipt context; never publish it."""
    paths=(case/"manifest.json",case/"vitis_invocation.json",
           case/"vivado"/"vivado_invocation.json",
           case/"run_hls.tcl",case/"vivado"/"run_vivado.tcl")
    if any(p.is_symlink() or not p.is_file() for p in paths):
        return None
    try:
        manifest=json.loads(paths[0].read_text(encoding="utf-8"))
        vitis=json.loads(paths[1].read_text(encoding="utf-8"))
        vivado=json.loads(paths[2].read_text(encoding="utf-8"))
    except (OSError,ValueError):
        return None
    model_digest=manifest.get("functional_sha256")
    versions=(vitis.get("tool_version_output"),vivado.get("tool_version_output"))
    clock=summary.get("clock_ns")
    tolerance=summary.get("numerical_tolerance")
    bits=summary.get("precision_bits")
    integer=summary.get("integer_bits")
    part=summary.get("target_part")
    if (not isinstance(model_digest,str) or not _HEX.fullmatch(model_digest)
            or manifest.get("cpp_sha256")!=summary.get("source_sha256")
            or manifest.get("part")!=part
            or manifest.get("clock_ns")!=clock
            or not isinstance(part,str) or not part
            or type(clock) not in (int,float)
            or type(tolerance) not in (int,float)
            or type(bits) is not int or type(integer) is not int
            or clock<=0 or bits<=0 or not 0<integer<=bits
            or not all(isinstance(version,str) and version for version in versions)):
        return None
    return (model_digest,part,clock,bits,integer,tolerance,
            versions[0],versions[1],summary.get("vivado_implementation_mode"),
            sha256(paths[3].read_bytes()).hexdigest(),
            sha256(paths[4].read_bytes()).hexdigest())


def collect_independent_e4(directory)->dict:
    """Recompute private native E4 and context-qualified pair evidence, no writes."""
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing independent native workspace")
    audit=audit_isolated_hls_replay(root)
    summaries={}
    for record in audit["cases"]:
        key=(record["model"],record["case"])
        if key in summaries:
            raise ValueError("Duplicated private native case")
        case=root/key[0]/key[1]
        if case.is_symlink() or not case.is_dir():
            raise ValueError("Unsafe independent experiment case directory")
        summaries[key]=summarize_case(str(case))
    result=summarize_independent_e4(root,audit,summaries)
    indexed={(row["model"],row["case"]):row for row in result["cases"]}
    private_context={}
    for model,a,b,_ in PAIR_SPECS:
        for case in (a,b):
            key=(model,case)
            if indexed[key]["qualified_e4"] and key not in private_context:
                private_context[key]=_private_matched_context(
                    root/model/case,summaries[key])
    comparisons=compare_qualified_e4_pairs(
        result["cases"],contexts=private_context)
    result["paired_physical_comparisons"]=comparisons
    result["qualified_matched_e4_pair_count"]=comparisons["qualified_matched_e4_pair_count"]
    result["planned_e4_pair_count"]=comparisons["planned_within_model_pair_count"]
    return result


def audit_independent_e4(directory,output)->dict:
    result=collect_independent_e4(directory)
    comparisons=result["paired_physical_comparisons"]
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Physical audit export must be a fresh directory")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-independent-e4-audit-",dir=target.parent) as td:
        stage=Path(td)
        (stage/"independent_e4_audit.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        (stage/"independent_e4_tables.tex").write_text(
            render_native_e4_latex(result),encoding="utf-8")
        lines=[
            "# Independent source-attested matched E4 audit","",
            "Only native reports bound to the exact Vitis/Vivado invocations "
            "are eligible; this command does NOT execute either tool.","",
            "Qualified E3: "+str(result["e3_qualified_case_count"])+"/8; "
            "attested routed E4: "+str(result["routed_e4_attested_case_count"])+"/8; "
            "physically qualified E4: "+str(result["e4_qualified_case_count"])+"/8.","",
            "| Model | Case | Axis | E3 | E4 provenance | E4 qualification | WNS (ns) |",
            "| --- | --- | --- | --- | --- | --- | ---: |",
        ]
        for row in result["cases"]:
            lines.append("| "+" | ".join((
                row["model"],row["case"],row["axis"],
                "yes" if row["qualified_e3"] else "no",
                "yes" if row["source_attested_routed_e4"] else "no",
                row["qualification_status"],
                str(row["postroute_wns_ns"]) if row["postroute_wns_ns"] is not None else "TBD",
            ))+" |")
        lines.extend(["","## Preregistered matched physical comparisons","",
            "Compare routed OOC resources and WNS only when BOTH source-attested "
            "E4 cases qualify and the private model, part, clock, numeric "
            "and tool-version contexts match. No board-latency measurement.","",
            "| Model | Contrast | Status | WNS B-A (ns) | LUT B-A | DSP B-A |",
            "| --- | --- | --- | ---: | ---: | ---: |"])
        for pair in comparisons["pairs"]:
            delta=pair["postroute_utilization_delta_b_minus_a"] or {}
            lines.append("| "+" | ".join((
                pair["model"],pair["axis_a"]+" / "+pair["axis_b"],
                pair["status"],
                str(pair["postroute_wns_delta_b_minus_a_ns"])
                    if pair["postroute_wns_delta_b_minus_a_ns"] is not None else "TBD",
                str(delta["lut"]) if "lut" in delta else "TBD",
                str(delta["dsp"]) if "dsp" in delta else "TBD",
            ))+" |")
        lines.extend(["","## Evidence limitations","",result["limitations"],"",
                      comparisons["limitations"],""])
        (stage/"independent_e4_audit.md").write_text("\n".join(lines))
        if target.exists() or target.is_symlink():
            raise FileExistsError("Physical evidence output appeared")
        stage.rename(target)
    return result
