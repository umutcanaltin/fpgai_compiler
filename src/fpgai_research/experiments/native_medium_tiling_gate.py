"""Source-attested six-case medium geometry gate: MatMul16 and Conv12.

READ ONLY. This bounded report audits three native cases per operator
(baseline + two tiling variants) and computes within-operator routed
deltas only if the original E4 vendor evidence is independently valid.
It never triggers HLS/Vivado, extrapolates other sizes, or imputes QoR.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.codegen.hls.emitter import (
    LOOP_CODEGEN_STATIC, LOOP_CODEGEN_CONV_STATIC,
)
from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.native_loop_codegen_ablation import _case
from fpgai_research.experiments.native_seven_axis_effects import OUTPUTS

SCHEMA="fpgai-research/medium-native-tiling-gate/v1"
STUDY_VERSIONS={
    "probe_matmul_16":LOOP_CODEGEN_STATIC,
    "probe_conv_12":LOOP_CODEGEN_CONV_STATIC,
}
CASES=("case_0000","case_0013","case_0014")


def _finite(v):
    return type(v) in (int,float) and isfinite(v)


def _qualified(row):
    if (row.get("routed_qualified") is not True or
            row.get("hls_source_attested_qualified") is not True or
            row.get("csim_passed") is not True):
        return False
    values=row.get("routed_qualified_cost")
    if not isinstance(values,dict) or set(values)!=set(OUTPUTS):
        raise ValueError("Routed E4 case lacks the full seven observable contract")
    if any(not _finite(values.get(key)) for key in OUTPUTS):
        raise ValueError("Routed E4 case has missing/nonfinite observables")
    if values["postroute_wns_ns"] < 0:
        raise ValueError("Routed E4 case lacks timing closure")
    return True


def analyze_medium_tiling_gate(studies):
    if not isinstance(studies,dict) or set(studies)!=set(STUDY_VERSIONS):
        raise ValueError("Require both registered medium operator studies")
    results=[]
    for name,version in STUDY_VERSIONS.items():
        supplied=studies[name]
        if (not isinstance(supplied,dict) or set(supplied)!=set(CASES)):
            raise ValueError("Need exact baseline and two tiled medium cases")
        base=supplied["case_0000"]
        if base.get("axis")!="baseline" or not _qualified(base):
            raise ValueError("Each medium study requires an E4-qualified baseline")
        request=base.get("requested")
        if not isinstance(request,dict) or not isinstance(request.get("tile"),list):
            raise ValueError("Malformed baseline architecture request")
        target=base.get("part")
        clock=base.get("clock_ns")
        numerical_tolerance=base.get("numerical_tolerance")
        functional=base.get("functional_sha256")
        if (not isinstance(functional,str) or len(functional)!=64
                or not isinstance(target,str) or not target
                or not _finite(clock) or clock<=0
                or not _finite(numerical_tolerance) or numerical_tolerance<=0):
            raise ValueError("Incomplete original model, clock or numeric identity")
        base_cpp=base.get("cpp_sha256")
        if not isinstance(base_cpp,str) or len(base_cpp)!=64:
            raise ValueError("Missing source SHA for E4 baseline")
        entries=[]
        for ident in CASES:
            row=supplied[ident]
            if row.get("case")!=ident:
                raise ValueError("Case identity does not match requested case")
            if row.get("loop_codegen_version")!=version:
                raise ValueError("Mixed or unexpected native source-codegen version")
            if any(row.get(field)!=base.get(field) for field in (
                    "functional_sha256","part","clock_ns","numerical_tolerance")):
                raise ValueError("Medium tile case differs in model, target or clock")
            changed=row.get("requested")
            if not isinstance(changed,dict) or set(changed)!=set(request):
                raise ValueError("Inconsistent tiled architecture request")
            deltas={key for key in changed if changed[key]!=request[key]}
            if ident=="case_0000":
                if deltas:
                    raise ValueError("Baseline changed its architecture")
            elif (row.get("axis")!="tile" or deltas!={"tile"} or
                    row.get("cpp_sha256")==base_cpp):
                raise ValueError("Tile-only case changes more than tiling or not source-distinct")
            q=_qualified(row)
            values=row.get("routed_qualified_cost") if q else None
            if not q and row.get("routed_qualified_cost") is not None:
                raise ValueError("Unqualified case cannot provide routed-qualified QoR")
            costs={key:round(values[key]-base["routed_qualified_cost"][key],9)
                   for key in OUTPUTS} if q else None
            ratios={key:(round(values[key]/base["routed_qualified_cost"][key],6)
                         if base["routed_qualified_cost"][key]>0 else None)
                    for key in ("postroute_lut","postroute_ff","postroute_dsp")
                   } if q else None
            entries.append({
                "case":ident,"axis":row["axis"],"source_sha256":row["cpp_sha256"],
                "requested_tile":changed["tile"],
                "routed_qualified":q,"native_evidence":row.get("native_evidence"),
                "blockers":row.get("blockers",[]),
                "absolute_qor":values,
                "delta_from_same_revision_baseline":costs,
                "resource_ratio_to_baseline":ratios,
            })
        results.append({
            "study":name,"expected_codegen_version":version,
            "functional_sha256":functional,
            "target_part":target,"clock_ns":clock,
            "numerical_tolerance":numerical_tolerance,
            "qualified_case_count":sum(e["routed_qualified"] for e in entries),
            "all_three_qualified":all(e["routed_qualified"] for e in entries),
            "cases":entries,
        })
    return {
        "schema":SCHEMA,
        "vendor_tools_executed":False,
        "native_source_and_evidence_reaudited_via_existing_case_reader":True,
        "medium_architectures_checked":6,
        "medium_architectures_qualified":sum(
            x["qualified_case_count"] for x in results),
        "ready_for_bounded_main_effect_campaign":all(
            x["all_three_qualified"] for x in results),
        "studies":results,
        "limitations":(
            "This check validates exactly six already executed native cases. "
            "Only in-revision tile-vs-baseline QoR differences are reported; "
            "no MatMul/Conv pooling, no board latency or power, no achieved "
            "per-loop II, no prediction about the other 78 medium requests. "
            "A positive slack delta is not a universal speedup. "
            "Source-fixed timing and numerical qualification gates apply."),
    }


def export_medium_tiling_gate(directory,output):
    root=Path(directory)
    dest=Path(output)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Require original prepared medium research suite")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Medium native report must be a fresh directory")
    aud=audit_suite(root,require_host=True,require_all_axes=False)
    if not aud.get("all_prepared_models_have_legal_executable_subsets"):
        raise ValueError("Prepared medium source study fails source audit")
    suite=root/"suite_results.json"
    if suite.is_symlink() or not suite.is_file():
        raise ValueError("Missing auditable medium suite_results.json")
    declared=json.loads(suite.read_text())
    named={x.get("id"):x for x in declared.get("studies",[])
           if isinstance(x,dict)}
    if (len(named)!=2 or set(named)!=set(STUDY_VERSIONS) or
            any(row.get("status")!="E1-PREPARED" for row in named.values())):
        raise ValueError("Expected isolated, source-prepared medium MatMul and Conv studies")
    rows={
        name:{ident:_case(root,name,ident) for ident in CASES}
        for name in STUDY_VERSIONS
    }
    result=analyze_medium_tiling_gate(rows)
    result["source_suite_sha256"]=sha256(suite.read_bytes()).hexdigest()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".medium-gate-") as work:
        output_path=Path(work)/"report"
        output_path.mkdir()
        (output_path/"gate.json").write_text(json.dumps(
            result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=["# Medium MatMul16 and Conv12: source-attested E4 tile gate","",
            "Existing native results only; no vendor tools executed by this report.",
            f"Qualified: {result['medium_architectures_qualified']} / 6.",
            "Every delta is measured relative to the SAME operator and SAME HLS "
            "generator revision, NEVER across models or source versions.","",
            "| Study | Case | E4? | HLS est. Δcycles | ΔLUT | ΔFF | ΔDSP | ΔWNS ns | Absolute LUT | LUT ratio |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in result["studies"]:
            for row in item["cases"]:
                d=row["delta_from_same_revision_baseline"]
                a=row["absolute_qor"]
                ratio=row["resource_ratio_to_baseline"]
                val=lambda k:("—" if d is None else
                              "0" if d[k]==0 else f"{d[k]:+g}")
                lines.append(
                    f"| {item['study']} | {row['case']} | {row['routed_qualified']} | "
                    + " | ".join(val(k) for k in (
                        "hls_estimated_cycles","postroute_lut","postroute_ff",
                        "postroute_dsp","postroute_wns_ns"))
                    + f" | {a['postroute_lut']:g} | "
                    + ("—" if ratio is None or ratio["postroute_lut"] is None
                       else f"{ratio['postroute_lut']:g}×") + " |")
        lines.extend(["",
            "Proceed with bounded native main-effect requests only if "
            "ready_for_bounded_main_effect_campaign=true in gate.json. "
            "Any unqualified tile retains missing numerical observations; "
            "this report never imputes values or predicts remaining configurations.",""])
        (output_path/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Destination appeared during read-only export")
        output_path.rename(dest)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    value=export_medium_tiling_gate(args.directory,args.output)
    print(json.dumps({
        "schema":SCHEMA,
        "qualified":value["medium_architectures_qualified"],
        "requested":value["medium_architectures_checked"],
        "ready_for_bounded_main_effect_campaign":
            value["ready_for_bounded_main_effect_campaign"],
        "vendor_tools_executed":False,
    },indent=2,sort_keys=True))
    return int(not value["ready_for_bounded_main_effect_campaign"])


if __name__=="__main__":
    raise SystemExit(main())
