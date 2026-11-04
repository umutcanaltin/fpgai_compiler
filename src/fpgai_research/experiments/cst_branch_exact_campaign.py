"""Prespecified L5f exact Phi-multi-reader scratch allocation + executed GCC.

Only the finite contiguous tensor-base source model may be called optimal.
Every selected source is recompared against NumPy and original GCC trace.
An explicit UNKNOWN_BUDGET control must remain unknown and never enter
the emitted-source arm.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import (
    disjoint_branch_offsets, emit_branch_matmul_hls,
)
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.model_ir.branch_exact_allocator import exact_mapped_branch_slots
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_branch_matmul_trace import trace_branch_cpp
from fpgai_research.validation.host import validate_host

CASES=(
    (1,1,2,1,False),
    (2,2,2,1,True),
    (2,2,3,1,True),
)


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("L5f original result directory is create-only")
    rows=[]
    for m,k,n,q,mapped in CASES:
        model=sample_branch(m,k,n,q)
        base=ArchitectureRequest()
        overrides={} if not mapped else {
            model.operators[0].name:replace(base,pes=2,simd=2),
            model.operators[1].name:replace(base,pes=2,simd=2),
            model.operators[2].name:replace(base,pes=2,serialization=2),
        }
        arch=resolve(model,base,overrides)
        optimum=exact_mapped_branch_slots(
            arch,max_slots=m*n+3*m*q,max_candidates=500_000,max_events=50_000)
        if optimum["status"]!="OPTIMAL_E0":
            raise AssertionError("Prespecified source E0 optimum not established: "+str(optimum["status"]))
        x=np.asarray([(-1)**i*(i+1)/7 for i in range(m*k)],
                     dtype=np.float32).reshape(m,k)
        source_options=(
            ("disjoint",disjoint_branch_offsets(arch)),
            ("exact_e0",optimum["offsets"]),
        )
        for layout,offsets in source_options:
            source=emit_branch_matmul_hls(arch,offsets=offsets)
            numeric=validate_host(model,source,x)
            trace=trace_branch_cpp(arch,source)
            if not numeric["accepted"] or trace["status"]!="EXECUTED_BRANCH_PHI_LIFETIMES_VERIFIED":
                raise AssertionError("Original matched source failed numerical/refinement controls")
            rows.append({
                "dims":json.dumps((m,k,n,q)),"mapped":mapped,"layout":layout,
                "functional_sha256":optimum["functional_sha256"],
                "architecture_sha256":optimum["architecture_sha256"],
                "source_event_sha256":trace["source_event_sha256"],
                "cpp_sha256":trace["cpp_sha256"],
                "source_slots":source.manifest["scratch_elements"],
                "optimal_slots_e0":optimum["physical_slots"],
                "disjoint_slots":optimum["total_disjoint_slots"],
                "explored_candidates":optimum["examined_candidates"],
                "mapped_h_reads":trace["h_mapped_reads_verified"],
                "source_numeric_pass":numeric["accepted"],
                "source_trace_pass":True,
                "numeric_max_abs_error":numeric["max_absolute_error"],
                "original_vitis_calls":0,
            })
    for i in range(0,len(rows),2):
        disjoint,chosen=rows[i:i+2]
        if (disjoint["source_slots"]<=chosen["source_slots"] or
                chosen["source_slots"]!=chosen["optimal_slots_e0"] or
                disjoint["source_event_sha256"]!=chosen["source_event_sha256"]):
            raise AssertionError("Non-saving or non-refining original E0/E1 layout pair")
    control_arch=resolve(sample_branch(2,2,3,2),ArchitectureRequest())
    unknown=exact_mapped_branch_slots(
        control_arch,max_slots=30,max_candidates=1)
    if unknown["status"]!="UNKNOWN_BUDGET" or unknown["offsets"] is not None:
        raise AssertionError("Budget exhaustion was silently treated as a legal allocation")
    summary={
        "schema":"fpgai-research/cst-multi-reader-exact-scratch-e0-e1/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
            check=True,capture_output=True,text=True,timeout=5).stdout.strip(),
        "cases":len(rows),"matched_pairs":len(CASES),
        "optimal_e0_proofs":len(CASES),
        "executed_host_numerical_passes":sum(r["source_numeric_pass"] for r in rows),
        "executed_source_trace_passes":sum(r["source_trace_pass"] for r in rows),
        "budget_unknown_control":unknown["status"],
        "native_vitis_calls":0,
        "evidence":"E0-finite-exact-Phi-multi-read-source-array-plus-E1-actual-GCC",
        "limitations":[
            "Only fixed sequential source-order and contiguous tensor base layout",
            "No bank-port/cycle model, no native Vitis CSim or measured BRAM/LUT/latency",
            "Not a general DAG synthesis algorithm or a firstness claim",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(output=parser.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
