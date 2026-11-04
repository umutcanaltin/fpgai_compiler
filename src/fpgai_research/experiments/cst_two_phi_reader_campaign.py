"""Prespecified causal Phi-on-both-sides fanout experiment (E0/E1, GCC).

Fixed Functional graph and weights; compare reader0 Phi changes alone,
reader1 Phi changes alone, and two independent policies. One source-pinned
record per configuration and a disjoint-vs-last-use layout pair. Evidence
is exactly observed GCC host-float source, NOT Vitis HLS resource/timing.
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
    emit_branch_matmul_hls,disjoint_branch_offsets,
)
from fpgai_research.model_ir.branch_matmul_contract import branch_contract
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_branch_matmul_trace import trace_branch_cpp
from fpgai_research.validation.host import validate_host
from fpgai_research.experiments.cst_branch_models import sample_branch

SHAPES=((2,2,3,2),(3,3,5,3),(2,4,4,3))
POLICIES=(
    ("scalar",{},{}),
    ("reader0_phi_only",{"pes":2,"simd":2},{}),
    ("reader1_phi_only",{},{"pes":2,"simd":2}),
    ("both_independent",{"pes":2,"simd":2},
       {"pes":2,"serialization":2}),
)


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Source-pinned two mapped readers study is create-only")
    rows=[]
    for dims in SHAPES:
        model=sample_branch(*dims)
        base=ArchitectureRequest()
        m,k,n,q=dims
        x=np.asarray([(-1)**i*(i+1)/7 for i in range(m*k)],
                     dtype=np.float32).reshape(m,k)
        for policy,c0,c1 in POLICIES:
            overrides={}
            for op,options in ((model.operators[1],c0),
                                (model.operators[2],c1)):
                if options:
                    overrides[op.name]=replace(base,**options)
            arch=resolve(model,base,overrides)
            for layout in ("disjoint","last_reader_reuse"):
                offsets=disjoint_branch_offsets(
                    arch,reuse_after_last_read=layout=="last_reader_reuse")
                e0=branch_contract(arch,offsets)
                emitted=emit_branch_matmul_hls(arch,offsets=offsets)
                e1=trace_branch_cpp(arch,emitted)
                numeric=validate_host(model,emitted,x)
                if (not numeric["accepted"] or
                        e1["status"]!="EXECUTED_BRANCH_PHI_LIFETIMES_VERIFIED" or
                        e0["source_order_event_sha256"]!=e1["source_event_sha256"]):
                    raise AssertionError("Prespecified GCC/mapping case failed")
                rows.append({
                    "shape":json.dumps(dims),"policy":policy,"layout":layout,
                    "functional_sha256":emitted.manifest["functional_sha256"],
                    "architecture_sha256":emitted.manifest["architecture_sha256"],
                    "cpp_sha256":emitted.manifest["cpp_sha256"],
                    "reader0_phi_sha256":e1["phi_iteration_sha256"]["reader0"],
                    "reader1_phi_sha256":e1["phi_iteration_sha256"]["reader1"],
                    "producer_phi_sha256":e1["phi_iteration_sha256"]["producer"],
                    "source_event_sha256":e1["source_event_sha256"],
                    "last_use_sha256":e1["last_use_sha256"],
                    "h_logical_values":e1["multi_reader_values"],
                    "h_required_reads":e1["h_mapped_reads_verified"],
                    "source_scratch_slots":e1["source_scratch_slots"],
                    "disjoint_slots":emitted.manifest["disjoint_source_slots"],
                    "numeric_pass":numeric["accepted"],
                    "max_float_abs_error":numeric["max_absolute_error"],
                    "executed_trace_pass":True,
                    "original_vitis_calls":0,
                })
    for i in range(0,len(rows),2):
        disjoint,reuse=rows[i:i+2]
        assert disjoint["layout"]=="disjoint" and reuse["layout"]=="last_reader_reuse"
        assert disjoint["source_scratch_slots"]>reuse["source_scratch_slots"]
        assert disjoint["functional_sha256"]==reuse["functional_sha256"]
        assert disjoint["architecture_sha256"]==reuse["architecture_sha256"]
        assert disjoint["source_event_sha256"]==reuse["source_event_sha256"]
    summary={
        "schema":"fpgai-research/cst-two-independent-phi-reader-source-e1/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
            capture_output=True,text=True,check=True,timeout=5).stdout.strip(),
        "graphs":len(SHAPES),"mapping_policies":len(POLICIES),
        "layout_pairs":len(rows)//2,"cases":len(rows),
        "numeric_passes":sum(r["numeric_pass"] for r in rows),
        "source_trace_passes":sum(r["executed_trace_pass"] for r in rows),
        "source_scratch_saving_pairs":len(rows)//2,
        "native_vitis_calls":0,
        "evidence":"E0-per-contribution-Phi-plus-E1-actual-GCC-two-reader-source",
        "limitations":[
            "Independent MatMul reader mappings with STATIC five-op graph only",
            "Compared logical scratch address extents, not actual HLS BRAM",
            "No original Vitis, fixed-point CSim, clock scheduling or bank-port proof",
            "No externally reproduced prior-art baseline or firstness claim",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),indent=2))


if __name__=="__main__":
    main()
