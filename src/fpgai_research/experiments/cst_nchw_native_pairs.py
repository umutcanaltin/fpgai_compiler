"""Prepare source-pinned matched NCHW scratch pairs for ORIGINAL Vitis HLS.

Source-only, create-only, zero vendor invocations: two actually different
C++ materializations of the same graph/Phi/numerical testbench/part/clock.
The researcher may run original Vitis on an isolated workstation later;
no CPU test/report is promoted to CSynth evidence by preparation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.semantics import architecture_fingerprint
from fpgai_research.codegen.hls.nchw_shared_scratch import (
    emit_nchw_stage_scratch,greedy_nchw_stage_plan,
)
from fpgai_research.codegen.hls.nchw_vendor_directives import (
    POLICIES,PRAGMA,apply_native_scratch_binding,
)
from fpgai_research.experiments.cst_nchw_access_campaign import CASES,MAPPINGS
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.toolchains.vitis_hls.package import (
    make_vitis_tcl,write_vitis_package,
)

SCHEMA="fpgai-research/cst-NCHW-matched-source-native-Vitis-preflight/v1"
PART="xck26-sfvc784-2LV-c"
VARIANTS=("real_disjoint","real_shared")
CASE_RE=re.compile(r"case_[0-9]{3}\Z")


def source_revision():
    root=Path(__file__).resolve().parents[3]
    r=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
                     capture_output=True,text=True,check=True,timeout=5)
    return r.stdout.strip()


def _hash(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError("Missing or symlinked immutable Vitis package file")
    return sha256(p.read_bytes()).hexdigest()


def prepare(*,output:str|Path,part:str=PART,clock_ns:float=32.0,
            uncertainty_ns:float=8.64,tolerance:float=0.1,
            scratch_binding:str="auto"):
    """Only original source/Tcl/testbench files. NO Vitis execution."""
    dest=Path(output)
    if scratch_binding not in POLICIES:
        raise ValueError("Unsupported vendor HLS memory binding policy")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Original NCHW native package study is create-only")
    if (not all(type(v) in (float,int) and isfinite(v)
                for v in (clock_ns,uncertainty_ns,tolerance))
            or not 0<=uncertainty_ns<clock_ns or tolerance<=0):
        raise ValueError("Invalid clock/uncertainty/tolerance")
    # Fail before creating study root. This uses canonical Vitis syntax.
    make_vitis_tcl(part=part,clock_ns=clock_ns,
                   clock_uncertainty_ns=uncertainty_ns)
    plans=[]
    for graph in CASES:
        model=nchw_case(graph,height=4,width=4)
        x=np.asarray([(-1)**i*(i+1)/13 for i in range(16)],
                     dtype=np.float32).reshape(1,1,4,4)
        for mapping in MAPPINGS:
            base=ArchitectureRequest()
            overrides={}
            if mapping=="independent_conv_phi":
                for ordinal,op in enumerate(model.operators):
                    if op.kind=="Conv":
                        overrides[op.name]=replace(
                            base,pes=2,simd=2,
                            serialization=2 if ordinal%2==0 else 1)
            arch=resolve(model,base,overrides)
            sources=[]
            for variant in VARIANTS:
                offsets=greedy_nchw_stage_plan(
                    arch,reuse=variant=="real_shared")
                generated=emit_nchw_stage_scratch(
                    arch,offsets,trace_conv_access=False)
                emitted=apply_native_scratch_binding(
                    generated,policy=scratch_binding)
                sources.append((variant,emitted,offsets))
            if sources[0][1].manifest["cpp_sha256"]==sources[1][1].manifest["cpp_sha256"]:
                raise AssertionError("Native pair must use two distinct original HLS sources")
            plans.append((graph,mapping,model,x,sources))
    revision=source_revision()
    dest.mkdir(parents=True)
    cases=[]
    for group,(graph,mapping,model,x,variants) in enumerate(plans):
        for variant,emitted,offsets in variants:
            case=f"case_{len(cases):03d}"
            directory=dest/case
            package=write_vitis_package(
                str(directory),model,emitted,x,
                part=part,clock_ns=clock_ns,
                clock_uncertainty_ns=uncertainty_ns,tolerance=tolerance)
            cases.append({
                "case":case,"group":group,"graph":graph,
                "mapping":mapping,"source_storage":variant,
                "model_sha256":package["functional_sha256"],
                "architecture_sha256":architecture_fingerprint(arch),
                "source_cpp_sha256":_hash(directory/"model.cpp"),
                "testbench_sha256":_hash(directory/"tb.cpp"),
                "tcl_sha256":_hash(directory/"run_hls.tcl"),
                "case_manifest_sha256":_hash(directory/"manifest.json"),
                "input_sha256":sha256(x.tobytes()).hexdigest(),
                "real_scratch_elements":emitted.manifest["source_scratch_slots"],
                "requested_native_memory_binding":scratch_binding,
                "requested_native_pragma":emitted.manifest["requested_native_pragma"],
                "original_vitis_calls":0,
                "native_qualified":False,
            })
    for i in range(0,len(cases),2):
        a,b=cases[i:i+2]
        if not (
            a["group"]==b["group"] and a["mapping"]==b["mapping"]
            and a["model_sha256"]==b["model_sha256"]
            and a["testbench_sha256"]==b["testbench_sha256"]
            and a["tcl_sha256"]==b["tcl_sha256"]
            and a["source_cpp_sha256"]!=b["source_cpp_sha256"]
            and a["real_scratch_elements"]>b["real_scratch_elements"]
        ):
            raise AssertionError("Original Vitis study source pairs not matched")
    summary={
        "schema":SCHEMA,"source_revision":revision,
        "part":part,"requested_clock_ns":clock_ns,
        "explicit_clock_uncertainty_ns":uncertainty_ns,
        "numerical_tolerance":tolerance,
        "requested_native_memory_binding":scratch_binding,
        "source_pragma_does_not_qualify_vendor_binding":True,
        "cases":cases,"case_count":len(cases),
        "matched_source_pairs":len(cases)//2,
        "generated_vitis_package_count":len(cases),
        "original_vitis_calls":0,
        "csim_evidence_status":"NOT_RUN",
        "csynth_evidence_status":"NOT_RUN",
        "native_qualified_cases":0,
        "paper_qor_comparison_status":"NOT_RUN",
        "warnings":[
            "Prepare does not run original vendor tools, even when installed",
            "Original Vitis fixed-point CSim and CSynth may reject source aliases",
            "Source scratch savings are NOT BRAM/resource/timing measurements",
            "Run only in a clean source-pinned isolated Vitis research checkout",
        ],
    }
    (dest/"native_source_manifest.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def audit_prepared(study:str|Path,*,require_current_source:bool=True):
    """Read-only integrity check, NEVER launches the original Vitis tool."""
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing original study root")
    file=root/"native_source_manifest.json"
    if file.is_symlink() or not file.is_file():
        raise ValueError("Missing original package manifest")
    manifest=json.loads(file.read_text())
    if manifest.get("schema")!=SCHEMA or manifest.get("case_count")!=len(CASES)*len(MAPPINGS)*2:
        raise ValueError("Unexpected source-pinned study schema or denominator")
    if require_current_source and manifest.get("source_revision")!=source_revision():
        raise ValueError("Study source revision differs from this working compiler")
    rows=manifest["cases"]
    if len(rows)!=manifest["case_count"]:
        raise ValueError("Original case listing incomplete")
    for i,row in enumerate(rows):
        case=row.get("case")
        if not isinstance(case,str) or not CASE_RE.fullmatch(case) or case!=f"case_{i:03d}":
            raise ValueError("Unexpected/duplicate source case name")
        folder=root/case
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Unqualified native package directory")
        for name,field in (("model.cpp","source_cpp_sha256"),
                           ("tb.cpp","testbench_sha256"),
                           ("run_hls.tcl","tcl_sha256"),
                           ("manifest.json","case_manifest_sha256")):
            if _hash(folder/name)!=row[field]:
                raise ValueError("Tampered original Vitis source file "+case+"/"+name)
        case_manifest=json.loads((folder/"manifest.json").read_text())
        if (case_manifest.get("cpp_sha256")!=row["source_cpp_sha256"]
                or case_manifest.get("testbench_sha256")!=row["testbench_sha256"]
                or case_manifest.get("part")!=manifest["part"]
                or case_manifest.get("clock_ns")!=manifest["requested_clock_ns"]
                or case_manifest.get("explicit_clock_uncertainty_ns")!=manifest["explicit_clock_uncertainty_ns"]
                or case_manifest.get("numerical_tolerance")!=manifest["numerical_tolerance"]
                or case_manifest.get("requested_native_memory_binding")!=manifest["requested_native_memory_binding"]
                or row.get("requested_native_memory_binding")!=manifest["requested_native_memory_binding"]
                or row.get("requested_native_pragma")!=case_manifest.get("requested_native_pragma")):
            raise ValueError("Canonical Vitis case manifest differs from source-pinned study")
        source_text=(folder/"model.cpp").read_text()
        requested=manifest["requested_native_memory_binding"]
        if source_text.count(PRAGMA)!=(1 if requested=="bram_2p" else 0):
            raise ValueError("Vitis requested memory binding pragma missing or duplicated")
        if row.get("native_qualified") is not False or row.get("original_vitis_calls")!=0:
            raise ValueError("Source preparation cannot claim vendor results")
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        if not (a["group"]==b["group"]
                and (a["source_storage"],b["source_storage"])==VARIANTS
                and a["model_sha256"]==b["model_sha256"]
                and a["testbench_sha256"]==b["testbench_sha256"]
                and a["tcl_sha256"]==b["tcl_sha256"]
                and a["source_cpp_sha256"]!=b["source_cpp_sha256"]
                and a["real_scratch_elements"]>b["real_scratch_elements"]):
            raise ValueError("Unmatched NCHW source testbench/Tcl pair")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_PACKAGES_VERIFIED",
            "cases":len(rows),"matched_pairs":len(rows)//2,
            "source_revision":manifest["source_revision"],
            "original_vitis_calls":0,"native_qualified_cases":0,
            "native_evidence":"NOT_RUN"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output")
    group.add_argument("--audit")
    p.add_argument("--part",default=PART)
    p.add_argument("--clock-ns",type=float,default=32.0)
    p.add_argument("--uncertainty-ns",type=float,default=8.64)
    p.add_argument("--tolerance",type=float,default=0.1)
    p.add_argument("--scratch-binding",choices=POLICIES,default="auto")
    args=p.parse_args(argv)
    report=(prepare(output=args.output,part=args.part,clock_ns=args.clock_ns,
                    uncertainty_ns=args.uncertainty_ns,tolerance=args.tolerance,
                    scratch_binding=args.scratch_binding)
            if args.output else audit_prepared(args.audit))
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
