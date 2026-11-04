"""L6a: prespecified nested static DAG Phi/value-lifetime E0/E1 series.

Three non-diamond graphs with mixed independent MatMul Phi, scalar Relu/Add,
and paired disjoint vs post-last-reader scratch layouts. Original GCC
source is executed for every row; model arithmetic is checked with NumPy.
Zero Vitis or general-ONNX completeness claims.
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
from fpgai_research.codegen.hls.graph_phi import emit_graph_hls
from fpgai_research.experiments.cst_graph_phi_fixtures import sample_graph
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.graph_phi_contract import (
    graph_contract,graph_disjoint_offsets,
)
from fpgai_research.validation.cpp_graph_phi_trace import trace_graph_cpp
from fpgai_research.validation.host import validate_host

CASES=(("mixed",2,3),("triple_reader",2,4),("nested",3,4))
POLICIES=("serial","independent_phi")


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("General DAG source research is create-only")
    records=[]
    for kind,m,n in CASES:
        model=sample_graph(kind,m=m,n=n)
        for policy in POLICIES:
            base=ArchitectureRequest()
            overrides={}
            if policy=="independent_phi":
                for i,op in enumerate(model.operators):
                    if op.kind=="MatMul":
                        overrides[op.name]=replace(
                            base,pes=2 if i%2==0 else 3,
                            simd=2 if i%2==0 else 1,
                            serialization=2 if i%2 else 1)
            arch=resolve(model,base,overrides)
            default=graph_disjoint_offsets(arch)
            for layout,offsets in (("disjoint",default),
                                    ("release_reuse",{**default,"out":default["h"]})):
                cert=graph_contract(arch,offsets)
                source=emit_graph_hls(arch,offsets=offsets)
                x=np.asarray([(-1)**i*(i+1)/7 for i in range(m*n)],
                             dtype=np.float32).reshape(m,n)
                numeric=validate_host(model,source,x)
                trace=trace_graph_cpp(arch,source)
                if (not numeric["accepted"] or
                        trace["status"]!="EXECUTED_GRAPH_BOUNDARIES_VERIFIED" or
                        trace["source_event_sha256"]!=cert["source_event_sha256"]):
                    raise AssertionError("Graph E0/E1 numerical/source refinement failed")
                records.append({
                    "graph":kind,"rows":m,"cols":n,"policy":policy,"layout":layout,
                    "operator_count":cert["operator_count"],
                    "edge_count":cert["edge_count"],
                    "functional_sha256":cert["functional_sha256"],
                    "architecture_sha256":cert["architecture_sha256"],
                    "cpp_sha256":trace["cpp_sha256"],
                    "source_event_sha256":cert["source_event_sha256"],
                    "last_use_sha256":cert["last_use_sha256"],
                    "multi_reader_values":trace["multi_reader_values"],
                    "scratch_source_elements":trace["source_array_elements"],
                    "phi_sha256":json.dumps(cert["operator_phi_sha256"],sort_keys=True),
                    "actual_numeric_pass":numeric["accepted"],
                    "max_host_float_error":numeric["max_absolute_error"],
                    "actual_source_trace_pass":True,
                    "native_vitis_calls":0,
                })
    for i in range(0,len(records),2):
        disjoint,reuse=records[i:i+2]
        if not (disjoint["scratch_source_elements"]>reuse["scratch_source_elements"]
                and disjoint["functional_sha256"]==reuse["functional_sha256"]
                and disjoint["architecture_sha256"]==reuse["architecture_sha256"]
                and disjoint["source_event_sha256"]==reuse["source_event_sha256"]):
            raise AssertionError("Source-matched last-use graph layout pair invalid")
    summary={
        "schema":"fpgai-research/cst-bounded-DAG-Phi-source-e0-e1/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],check=True,
            capture_output=True,text=True,timeout=5).stdout.strip(),
        "graphs":len(CASES),"policies":len(POLICIES),
        "cases":len(records),"paired_source_layouts":len(records)//2,
        "original_numeric_passes":sum(x["actual_numeric_pass"] for x in records),
        "original_source_trace_passes":sum(x["actual_source_trace_pass"] for x in records),
        "actual_original_vitis_calls":0,
        "evidence":"E0-graph-Phi-readiness-last-use-plus-E1-actual-GCC-static-DAG",
        "limitations":[
            "Static rank-two MatMul(const RHS), scalar Relu, no-broadcast Add only",
            "No original Vitis fixed-point CSim/CSynth, HLS memory ports/timing or RTL proof",
            "Source-array savings are not BRAM/energy savings",
            "No external peer-system execution, learned RL or research priority claim",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as output_csv:
        writer=csv.DictWriter(output_csv,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (root/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(output=parser.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
