"""Prospective R2d 12 original GCC loop-driven Phi source cases.

Three fixed graph shapes × four independently mapped reader policies.
Unlike the R2b/R2c GCC studies, the emitted C++ computes H element
subscripts and consumer output positions *inside a runtime C++ loop*.
Actual stdout is verified by an independent algebraic Phi issue rule
and numerical NumPy FunctionalModel reference, not by generator-event
list replay. This is source correctness, not FPGA cycle scheduling.

Create-only, source-pinned, clean-checkout, no native Vitis/Vivado,
no historical artifact overwrites and no trained RL experiments.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.codegen.hls.r2d_loop_driven_branch import emit_loop_driven_cpp
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import SHAPES,POLICIES,_input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2d_loop_driven_cpp import verify_loop_driven_gcc

SCHEMA="fpgai-research/cst-R2d-original-loop-driven-GCC-source-cohort/v1"


def _sha(path:Path)->str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Original R2d artifact must exist and not be a symlink")
    return sha256(path.read_bytes()).hexdigest()


def run(*,output:str|Path)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original R2d source cohort must be create-only")
    clean=subprocess.run(
        ["git","status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=5).stdout
    if clean.strip():
        raise RuntimeError("Original R2d cohort requires a clean tracked source checkout")
    revision=subprocess.run(
        ["git","rev-parse","HEAD"],capture_output=True,text=True,
        check=True,timeout=5).stdout.strip()
    if len(revision)!=40:
        raise RuntimeError("Full original revision SHA is required")
    root.mkdir(parents=True)
    rows=[]
    for dims in SHAPES:
        m,k,n,q=dims
        model=sample_branch(*dims)
        x=_input(m,k)
        base=ArchitectureRequest()
        for policy,p0,p1 in POLICIES:
            case=f"case_{len(rows):03d}"
            folder=root/case
            folder.mkdir()
            overrides={
                model.operators[stage].name:replace(base,**vals)
                for stage,vals in ((1,p0),(2,p1)) if vals
            }
            arch=resolve(model,base,overrides)
            offsets=disjoint_branch_offsets(arch,reuse_after_last_read=True)
            emission=emit_loop_driven_cpp(arch,offsets=offsets)
            (folder/"source.cpp").write_text(emission.cpp)
            (folder/"case.json").write_text(
                json.dumps({
                    "source_revision":revision,
                    "case":case,"shape":dims,"policy":policy,
                    "input_float32":x.reshape(-1).astype(float).tolist(),
                    "input_sha256":sha256(x.tobytes()).hexdigest(),
                    "original_cpp_sha256":_sha(folder/"source.cpp"),
                    "source_manifest":emission.manifest,
                    "pre_execution_status":"SOURCE_PINNED",
                },sort_keys=True,indent=2)+"\n")
            try:
                receipt=verify_loop_driven_gcc(
                    arch,emission,offsets=offsets,runtime_input=x,
                    original_trace_output=folder/"original_gcc_trace.txt")
            except Exception as exc:
                (folder/"FAILED.json").write_text(
                    json.dumps({
                        "case":case,"status":"FAILED_ORIGINAL_R2D_E1",
                        "exception_type":type(exc).__name__,
                        "exception":str(exc)[:3000],
                        "original_cpp_sha256":_sha(folder/"source.cpp"),
                    },sort_keys=True,indent=2)+"\n")
                raise
            (folder/"verified.json").write_text(
                json.dumps(receipt,sort_keys=True,indent=2)+"\n")
            rows.append({
                "case":case,
                "shape":json.dumps(dims),
                "policy":policy,
                "source_revision":revision,
                "functional_sha256":receipt["functional_sha256"],
                "architecture_sha256":receipt["architecture_sha256"],
                "input_sha256":receipt["original_runtime_input_sha256"],
                "cpp_sha256":receipt["source_cpp_sha256"],
                "original_stdout_sha256":receipt["original_GCC_trace_sha256"],
                "original_H_reads":receipt["actual_H_read_count"],
                "actual_Phi_source_issue_sha256":receipt[
                    "actual_runtime_source_H_issue_signature_SHA256"],
                "numPy_absolute_error":receipt["original_numerical_max_absolute_error"],
                "runtime_loop_index_used":receipt["runtime_subscript_computed_by_CPP_loop"],
                "independent_issue_verified":receipt["independent_phi_source_rule"]==
                    "VERIFIED_E0_CLOSED_FORM",
                "independent_reduction_alias_verified":receipt[
                    "independent_branch_reduction_alias"]=="VERIFIED_E0",
                "original_GCC_compilations":receipt["original_GCC_compilations"],
                "original_GCC_executions":receipt["original_GCC_executions"],
                "original_vendor_calls":0,
            })
    summary={
        "schema":SCHEMA,
        "source_revision":revision,
        "prespecified_graphs":len(SHAPES),
        "prespecified_Phi_reader_policies":len(POLICIES),
        "prespecified_cases":len(rows),
        "original_GCC_compilations":len(rows),
        "original_GCC_executions":len(rows),
        "independent_runtime_Phi_issue_passes":sum(
            r["independent_issue_verified"] for r in rows),
        "source_reduction_alias_passes":sum(
            r["independent_reduction_alias_verified"] for r in rows),
        "original_numeric_passes":len(rows),
        "runtime_index_loop_cases":sum(r["runtime_loop_index_used"] for r in rows),
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "hardware_cycle_schedule_proven":False,
        "external_compiler_novelty_proven":False,
        "evidence":"E1-real-GCC-runtime-index-loop-vs-independent-closed-form-Phi-spec",
        "limitations":[
            "This source experiment is not an FPGA clock/port/PE-parallelism experiment",
            "All H producer values are fully materialized before readers start",
            "Only three fixed MatMul diamonds and fixed deterministic input per graph",
            "The rank decoder/source oracle implement the same stated sequential source policy, not arbitrary legal schedules",
            "No native Vitis/physical results, peer-compiler parity or trained RL outcomes",
        ],
    }
    with (root/"cases.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (root/"summary.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    print(json.dumps(run(output=args.output),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
