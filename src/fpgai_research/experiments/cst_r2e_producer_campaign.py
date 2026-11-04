"""Prospective R2e original H-producer contribution-completeness GCC study.

Prespecified 3 graph shapes x 4 independently mapped reader policies.
This is a NEW create-only 12-case original cohort. It preserves all
actual runtime C++ source, input, contribution+consumer stdout, source
hashes, independent E0 producer/reader safety and E1 NumPy results.

This must not rerun or overwrite historical R2b/R2c/R2d study files.
No native Vitis/Vivado calls and no claimed FPGA producer scheduling.
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
from fpgai_research.codegen.hls.r2e_producer_traced_branch import emit_producer_contribution_cpp
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import SHAPES,POLICIES,_input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2e_producer_cpp import verify_producer_traced_gcc

SCHEMA="fpgai-research/cst-R2e-original-complete-producer-GCC-cohort/v1"


def _sha(path:Path)->str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Original R2e source evidence must be an ordinary file")
    return sha256(path.read_bytes()).hexdigest()


def run(*,output:str|Path)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original R2e producer cohort must be create-only")
    clean=subprocess.run(
        ["git","status","--porcelain","--untracked-files=no"],
        check=True,capture_output=True,text=True,timeout=5).stdout
    if clean.strip():
        raise RuntimeError("Original R2e source cohort requires clean tracked files")
    revision=subprocess.run(
        ["git","rev-parse","HEAD"],check=True,capture_output=True,
        text=True,timeout=5).stdout.strip()
    if len(revision)!=40:
        raise RuntimeError("Original source revision must be full SHA")
    root.mkdir(parents=True)
    cases=[]
    for dims in SHAPES:
        m,k,n,q=dims
        model=sample_branch(*dims)
        x=_input(m,k)
        base=ArchitectureRequest()
        for policy,p0,p1 in POLICIES:
            case=f"case_{len(cases):03d}"
            folder=root/case
            folder.mkdir()
            edits={
                model.operators[stage].name:replace(base,**opts)
                for stage,opts in ((1,p0),(2,p1)) if opts
            }
            arch=resolve(model,base,edits)
            offsets=disjoint_branch_offsets(arch,reuse_after_last_read=True)
            emitted=emit_producer_contribution_cpp(arch,offsets=offsets)
            (folder/"source.cpp").write_text(emitted.cpp)
            (folder/"case.json").write_text(json.dumps({
                "case":case,
                "shape":dims,
                "policy":policy,
                "source_revision":revision,
                "input_float32_values":x.reshape(-1).astype(float).tolist(),
                "input_float32_sha256":sha256(x.tobytes()).hexdigest(),
                "original_source_sha256":_sha(folder/"source.cpp"),
                "source_manifest":emitted.manifest,
                "status_before_original_GCC":"SOURCE_PINNED",
            },sort_keys=True,indent=2)+"\n")
            try:
                proven=verify_producer_traced_gcc(
                    arch,emitted,offsets=offsets,runtime_input=x,
                    original_trace_output=folder/"original_runtime_trace.txt")
            except Exception as exc:
                (folder/"FAILED.json").write_text(
                    json.dumps({
                        "case":case,
                        "status":"FAILED_ORIGINAL_R2E_PRODUCER",
                        "exception_type":type(exc).__name__,
                        "exception":str(exc)[:3000],
                        "original_cpp_sha256":_sha(folder/"source.cpp"),
                    },sort_keys=True,indent=2)+"\n")
                raise
            (folder/"verified.json").write_text(
                json.dumps(proven,sort_keys=True,indent=2)+"\n")
            cases.append({
                "case":case,
                "shape":json.dumps(dims),
                "policy":policy,
                "source_revision":revision,
                "functional_sha256":proven["functional_sha256"],
                "architecture_sha256":proven["architecture_sha256"],
                "input_sha256":proven["original_input_sha256"],
                "source_cpp_sha256":proven["source_cpp_sha256"],
                "original_trace_sha256":proven["original_full_stdout_sha256"],
                "producer_event_sha256":proven["original_producer_trace_sha256"],
                "runtime_H_values_sha256":proven["runtime_H_values_sha256"],
                "original_H_numeric_max_abs_error":proven[
                    "original_intermediate_H_numeric_max_abs_error"],
                "actual_producer_contributions":proven[
                    "independent_complete_H_producer_contributions"],
                "actual_complete_H_producer_commits":proven[
                    "independent_complete_H_producer_commits"],
                "original_numerical_max_abs_error":proven[
                    "original_numeric_max_absolute_error"],
                "source_producer_complete":proven["E0_producer_completeness_witnessed"],
                "original_GCC_compilations":1,"original_GCC_executions":1,
                "original_vendor_calls":0,
            })
    summary={
        "schema":SCHEMA,
        "source_revision":revision,
        "prespecified_graphs":len(SHAPES),
        "prespecified_reader_Phi_policies":len(POLICIES),
        "prespecified_original_cases":len(cases),
        "original_GCC_compilations":len(cases),
        "original_GCC_executions":len(cases),
        "original_producer_H_commit_complete_cases":sum(
            c["source_producer_complete"] for c in cases),
        "original_complete_producer_contributions":sum(
            c["actual_producer_contributions"] for c in cases),
        "original_numeric_and_consumer_source_passes":len(cases),
        "original_intermediate_H_numeric_passes":len(cases),
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "hardware_producer_cycle_completeness_established":False,
        "arbitrary_graph_producer_completeness_established":False,
        "external_compiler_novelty_established":False,
        "evidence":"E1-original-C++-runtime-producer-terms-independent-MatMul+Phi+NumPy",
        "limitations":[
            "Runtime producer terms and intermediate H values are numerically checked, not a formal arbitrary-C++ or RTL proof",
            "Producer serially completes before readers start; no producer-consumer streaming",
            "Only three fixed small MatMul diamonds with four reader policies",
            "No native HLS, port timing, physical BRAM or trained RL performance",
        ],
    }
    with (root/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]))
        writer.writeheader();writer.writerows(cases)
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
