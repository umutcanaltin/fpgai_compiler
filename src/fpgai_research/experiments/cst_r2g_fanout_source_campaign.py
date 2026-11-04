"""Prospective original 36-case GCC runtime-source fanout study (R2g E1).

Same R2f 3 fanout × 3 shape × 4 mapped-reader cases, but with
independently audited real host C++ computation and runtime READ
subscripts. Creates immutable case source, trace, inputs and receipts.
Every case must pass E0 source safety and independent model numerical
verification; failed original cases remain pinned as failures.

No historical R2b/R2c/R2d/R2e files are modified or reexecuted.
No Vitis/Vivado or other native vendor calls.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.r2g_fanout_cpp import emit_fanout_cpp
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    FANOUTS,SHAPES,POLICY_NAMES,policy_widths,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2g_fanout_cpp import verify_fanout_gcc

SCHEMA="fpgai-research/cst-R2g-original-36-fanout-runtime-GCC-e1/v1"


def _sha(path:Path)->str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R2g original source artifact missing or symlink")
    return sha256(path.read_bytes()).hexdigest()


def run(*,output:str|Path)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original R2g study is create-only")
    tracked=subprocess.run(
        ["git","status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=5).stdout
    if tracked.strip():
        raise RuntimeError("Source-pinned original R2g requires clean tracked repository")
    revision=subprocess.run(
        ["git","rev-parse","HEAD"],capture_output=True,
        text=True,check=True,timeout=5).stdout.strip()
    if len(revision)!=40:
        raise ValueError("Original full source revision missing")
    root.mkdir(parents=True)
    rows=[]
    for readers in FANOUTS:
        for dims in SHAPES:
            m,k,n,q=dims
            model=sample_fanout(readers,m=m,k=k,n=n,q=q)
            x=_input(m,k)
            for policy in POLICY_NAMES:
                case=f"case_{len(rows):03d}"
                folder=root/case
                folder.mkdir()
                base=ArchitectureRequest()
                arch=resolve(model,base,{
                    f"reader{i}":replace(base,pes=width)
                    for i,width in enumerate(policy_widths(readers,policy))
                    if width!=1
                })
                emission=emit_fanout_cpp(arch)
                (folder/"source.cpp").write_text(emission.cpp)
                (folder/"case.json").write_text(
                    json.dumps({
                        "case":case,"fanout":readers,"shape":dims,
                        "policy":policy,
                        "source_revision":revision,
                        "input_float32_values":x.reshape(-1).astype(float).tolist(),
                        "input_sha256":sha256(x.tobytes()).hexdigest(),
                        "source_cpp_sha256":_sha(folder/"source.cpp"),
                        "source_manifest":emission.manifest,
                        "status_before_GCC":"SOURCE_AND_INPUT_PINNED",
                    },sort_keys=True,indent=2)+"\n")
                try:
                    checked=verify_fanout_gcc(
                        arch,emission,runtime_input=x,
                        original_trace_output=folder/"original_stdout.txt")
                except Exception as exc:
                    (folder/"FAILED.json").write_text(
                        json.dumps({
                            "case":case,"status":"FAILED_ORIGINAL_R2G",
                            "type":type(exc).__name__,
                            "message":str(exc)[:3000],
                            "source_cpp_sha256":_sha(folder/"source.cpp"),
                        },sort_keys=True,indent=2)+"\n")
                    raise
                (folder/"verified.json").write_text(
                    json.dumps(checked,sort_keys=True,indent=2)+"\n")
                rows.append({
                    "case":case,
                    "fanout":readers,
                    "shape":json.dumps(dims),
                    "policy":policy,
                    "source_revision":revision,
                    "functional_sha256":checked["functional_sha256"],
                    "architecture_sha256":checked["architecture_sha256"],
                    "source_cpp_sha256":checked["source_cpp_sha256"],
                    "original_stdout_sha256":checked["original_stdout_sha256"],
                    "original_input_sha256":checked["original_input_sha256"],
                    "runtime_event_sha256":checked["runtime_event_sha256"],
                    "producer_contributions":checked["original_producer_contributions"],
                    "reader_H_reads":checked["original_H_reader_reads"],
                    "intermediate_numeric_values":checked["source_intermediate_values"],
                    "max_intermediate_abs_error":checked["source_intermediate_max_absolute_error"],
                    "max_final_abs_error":checked["source_final_max_absolute_error"],
                    "all_source_oracles_passed":True,
                    "runtime_reader_subscripts_computed_in_CPP":True,
                    "source_disjoint_layout_only":True,
                    "actual_per_element_optimized_storage_executed":False,
                    "original_GCC_compilations":1,
                    "original_GCC_executions":1,
                    "original_Vitis_or_Vivado_calls":0,
                })
    if len(rows)!=len(FANOUTS)*len(SHAPES)*len(POLICY_NAMES):
        raise AssertionError("Original R2g complete denominator violated")
    summary={
        "schema":SCHEMA,"source_revision":revision,
        "prespecified_fanouts":list(FANOUTS),
        "prespecified_shapes":[list(x) for x in SHAPES],
        "prespecified_mapping_policies":list(POLICY_NAMES),
        "prespecified_original_cases":len(rows),
        "original_GCC_compilations":len(rows),
        "original_GCC_executions":len(rows),
        "independent_source_event_Phi_and_alias_passes":len(rows),
        "all_intermediate_and_final_NumPy_numeric_passes":len(rows),
        "runtime_dynamic_subscript_cases":len(rows),
        "producer_CONTRIBUTE_readiness_cases":len(rows),
        "source_disjoint_scratch_layout_cases":len(rows),
        "actual_Phi_conditioned_memory_capacity_savings_executed":False,
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "hardware_cycle_port_or_BRAM_qualified":False,
        "general_DAG_semantic_correctness_qualified":False,
        "external_novelty_qualified":False,
        "evidence":"E1-real-host-GCC-loop-driven-2-to-4-fanout-with-independent-Phi-and-all-intermediates",
        "limitations":[
            "New R2g source uses disjoint memory, not original E0-optimal scalar address packing",
            "Host GCC source order is not physical FPGA cycle/BRAM/port scheduling",
            "One MatMul producer with 2..4 MatMul readers and left Add chain, not arbitrary DAG",
            "One deterministic input per graph shape; no trained RL or peer compiler comparison",
        ],
    }
    with (root/"cases.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root/"summary.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
