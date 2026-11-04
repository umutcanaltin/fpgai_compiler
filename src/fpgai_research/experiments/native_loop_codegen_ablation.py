"""Read-only source-attested comparison of legacy and static-trip loop codegen.

This is a mechanism/qualification comparison of *compiler revisions*, not
a seven-axis single-factor effect. It never reuses old native measurements
as evidence for new source and never invokes licensed vendor tools.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.codegen.hls.emitter import (
    LOOP_CODEGEN_LEGACY, LOOP_CODEGEN_STATIC, LOOP_CODEGEN_CONV_STATIC,
)
from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.suite_native_batch import (
    _qualify, _qualify_hls,
)
from fpgai_research.model_ir.artifacts import read_ir_artifact
from fpgai_research.experiments.native_seven_axis_effects import OUTPUTS, _value

SCHEMA="fpgai-research/native-loop-codegen-revision-comparison/v1"
CASES=("case_0000","case_0013","case_0014")
STUDY_VERSIONS={
    "probe_matmul_08":(LOOP_CODEGEN_LEGACY,LOOP_CODEGEN_STATIC),
    "probe_conv_08":(LOOP_CODEGEN_STATIC,LOOP_CODEGEN_CONV_STATIC),
}


def _case(root,study,ident):
    case=Path(root)/study/ident
    if case.is_symlink() or not case.is_dir():
        raise ValueError("Expected real prepared native case directory")
    campaign=json.loads((case.parent/"campaign.json").read_text())
    if campaign.get("schema")!="fpgai-research/paper-campaign/v1":
        raise ValueError("Wrong native campaign schema")
    rows=[c for c in campaign["cases"] if c.get("case")==ident]
    if len(rows)!=1 or rows[0].get("status")!="READY":
        raise ValueError("Expected one READY case")
    declared=rows[0]
    observation=summarize_case(str(case))
    if observation["source_sha256"]!=declared["cpp_sha256"]:
        raise ValueError("Original source receipt disagrees with campaign")
    artifact=read_ir_artifact(case/"progressive_ir.json")
    manifest=artifact["stages"]["hls"]["manifest"]
    if manifest["cpp_sha256"]!=observation["source_sha256"]:
        raise ValueError("IR artifact/HLS source identity disagreement")
    mechanism=manifest.get("loop_codegen_version",LOOP_CODEGEN_LEGACY)
    physical=observation.get("vivado_metrics") or {}
    util=physical.get("utilization") or {}
    timing=physical.get("timing") or {}
    hls=observation.get("metrics") or {}
    qualified=not _qualify(observation)
    cost=None
    if qualified:
        cost={
            "hls_estimated_cycles":_value(int(hls["latency_best"])),
            "postroute_lut":_value(util.get("lut")),
            "postroute_ff":_value(util.get("ff")),
            "postroute_dsp":_value(util.get("dsp")),
            "postroute_bram":_value(util.get("bram")),
            "postroute_uram":_value(util.get("uram")),
            "postroute_wns_ns":_value(timing.get("wns_ns")),
        }
        if any(x is None for x in cost.values()):
            raise ValueError("E4 qualified case has missing routed comparison metric")
    return {
        "case":ident,"axis":declared["dimension"],
        "requested":declared["requested"],
        "functional_sha256":artifact["functional_sha256"],
        "loop_codegen_version":mechanism,
        "cpp_sha256":observation["source_sha256"],
        "part":observation["target_part"],
        "clock_ns":observation["clock_ns"],
        "numerical_tolerance":observation["numerical_tolerance"],
        "csim_passed":observation.get("csim_passed"),
        "hls_source_attested_qualified":not _qualify_hls(observation),
        "routed_qualified":qualified,
        "native_evidence":observation.get("evidence"),
        "blockers":_qualify(observation),
        "source_warnings":observation.get("warnings",[]),
        "routed_qualified_cost":cost,
    }


def analyze_revision_pairs(pairs, *, study="probe_matmul_08"):
    if study not in STUDY_VERSIONS:
        raise ValueError("Unsupported controlled loop-codegen study")
    old_version,new_version=STUDY_VERSIONS[study]
    if not isinstance(pairs,list) or len(pairs)!=3:
        raise ValueError("Baseline and exactly two tiled cases required")
    case_ids=set()
    out=[]
    for pair in pairs:
        before,after=pair["before"],pair["after"]
        ident=before.get("case")
        if ident!=after.get("case") or ident not in CASES or ident in case_ids:
            raise ValueError("Case identity mismatch")
        case_ids.add(ident)
        if (before.get("loop_codegen_version")!=old_version or
                after.get("loop_codegen_version")!=new_version):
            raise ValueError("Expected documented source-version transition")
        same=("requested","axis","functional_sha256",
              "part","clock_ns","numerical_tolerance")
        if any(before.get(k)!=after.get(k) for k in same):
            raise ValueError("Codegen revisions differ in model, target or requested knobs")
        if before.get("cpp_sha256")==after.get("cpp_sha256"):
            raise ValueError("Expected distinct generated C++ source digests")
        out.append({
            "case":ident,"axis":before["axis"],
            "architecture_unchanged":True,"functional_model_unchanged":True,
            "generated_hls_source_changed":True,
            "source_sha256_before":before["cpp_sha256"],
            "source_sha256_after":after["cpp_sha256"],
            "before":{k:before[k] for k in (
                "native_evidence","csim_passed",
                "hls_source_attested_qualified","routed_qualified",
                "routed_qualified_cost","blockers","source_warnings")},
            "after":{k:after[k] for k in (
                "native_evidence","csim_passed",
                "hls_source_attested_qualified","routed_qualified",
                "routed_qualified_cost","blockers","source_warnings")},
        })
    if case_ids!=set(CASES):
        raise ValueError("Missing or duplicate required baseline/tiled case")
    baseline=next(x for x in out if x["case"]=="case_0000")
    if not baseline["before"]["routed_qualified"] or not baseline["after"]["routed_qualified"]:
        raise ValueError("Both compiler revisions need independently qualified baseline")
    for row in out:
        if row["axis"]=="baseline":
            continue
        if row["axis"]!="tile":
            raise ValueError("Expected exact tile-only cases")
    return {
        "schema":SCHEMA,
        "study":study,
        "source_codegen_versions":{"before":old_version,"after":new_version},
        "compiler_revision_effect_not_knob_main_effect":True,
        "old_source_reused_as_new_hardware_evidence":False,
        "vendor_tools_executed":False,
        "cases":out,
        "before_routed_qualified":sum(x["before"]["routed_qualified"] for x in out),
        "after_routed_qualified":sum(x["after"]["routed_qualified"] for x in out),
        "tiling_recovered":sum(
            x["axis"]=="tile" and not x["before"]["routed_qualified"]
            and x["after"]["routed_qualified"] for x in out),
        "scientific_limitations":(
            "A versioned HLS source-generation intervention restores observed "
            "qualification in these cases; it does not prove HLS scheduler "
            "root cause, generality across shapes, independent replication, "
            "or superior hardware performance. Each revision is separately "
            "measured; no measurements are carried between revisions."),
    }


def export_revision_comparison(before,after,output,*,study="probe_matmul_08"):
    if study not in STUDY_VERSIONS:
        raise ValueError("This comparison supports only registered operator pilots")
    roots=(Path(before),Path(after))
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Expected original prepared native research suite")
        audit=audit_suite(root,require_host=True,require_all_axes=False)
        if not audit.get("all_prepared_models_have_legal_executable_subsets"):
            raise ValueError("Suite source IR cannot be reaudited")
    if roots[0].resolve()==roots[1].resolve():
        raise ValueError("Revisions must use independent prepared directories")
    pairs=[{"before":_case(roots[0],study,i),
            "after":_case(roots[1],study,i)} for i in CASES]
    report=analyze_revision_pairs(pairs,study=study)
    report["prepared_source_suite_sha256"]=[
        sha256((r/"suite_results.json").read_bytes()).hexdigest()
        for r in roots
    ]
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Comparison output must be a new directory")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=target.parent,prefix=".codegen-ablation-") as t:
        stage=Path(t)/"study"
        stage.mkdir()
        (stage/"comparison.json").write_text(json.dumps(
            report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        lines=[
            "# Matched "+study+" HLS loop-codegen revision ablation","",
            "Read-only, independently source-attested evidence: E2/E3/E4 "
            "are not inferred from tool return codes.","",
            "| Case | Axis | Before evidence | After evidence | Before E4 | After E4 |",
            "|---|---|---|---|---|---|",
        ]
        for row in report["cases"]:
            a,b=row["before"],row["after"]
            lines.append(
                f"| {row['case']} | {row['axis']} | "
                f"{a['native_evidence']} | {b['native_evidence']} | "
                f"{a['routed_qualified']} | {b['routed_qualified']} |")
        lines.extend(["","## "+report["source_codegen_versions"]["after"]+": routed-qualified numeric effects","",
            "Only compare each new-source tile to the **same new-source baseline**. "
            "Do not compute an old-source tiling QoR improvement when HLS evidence is missing.","",
            "| Case | HLS cycles Δ | Routed LUT Δ | Routed FF Δ | Routed DSP Δ | Routed slack Δ (ns) |",
            "|---|---:|---:|---:|---:|---:|"])
        baseline=next(x for x in report["cases"] if x["axis"]=="baseline")
        base=baseline["after"]["routed_qualified_cost"]
        for row in report["cases"]:
            if row["axis"]=="baseline":
                continue
            m=row["after"]["routed_qualified_cost"]
            deltas={k:(round(m[k]-base[k],9)
                       if m is not None and base is not None else None)
                    for k in OUTPUTS}
            cols=("hls_estimated_cycles","postroute_lut","postroute_ff",
                  "postroute_dsp","postroute_wns_ns")
            lines.append("| "+row["case"]+" | "+" | ".join(
                "unqualified" if deltas[k] is None else f"{deltas[k]:+g}"
                for k in cols)+" |")
        lines.extend(["",
            "These are postroute module resource and timing observations, "
            "not FPGA board latency or power. The codegen change is a "
            "separate experimental intervention, not a tile-size effect."])
        (stage/"summary.md").write_text("\n".join(lines)+"\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Comparison output appeared during export")
        stage.rename(target)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",default="probe_matmul_08",choices=tuple(STUDY_VERSIONS))
    p.add_argument("--before",required=True)
    p.add_argument("--after",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    r=export_revision_comparison(args.before,args.after,args.output,study=args.study)
    print(json.dumps({"schema":r["schema"],"study":r["study"],
                      "old_routed_qualified":r["before_routed_qualified"],
                      "new_routed_qualified":r["after_routed_qualified"],
                      "tiling_recovered":r["tiling_recovered"],
                      "vendor_tools_executed":False},indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
