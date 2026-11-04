"""Source-attested rank-two non-identity graph access-map paper cases."""
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
from fpgai_research.experiments.cst_graph_access_fixtures import access_graph
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.graph_phi_contract import graph_contract,graph_disjoint_offsets
from fpgai_research.validation.cpp_graph_phi_trace import trace_graph_cpp
from fpgai_research.validation.host import validate_host

CASES=(("broadcast",2,3,1),("broadcast",3,4,1),
       ("concat",2,3,2),("concat",3,4,3))
POLICIES=("scalar","independent_phi")


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original rank-two access-map experiment is create-only")
    records=[]
    for kind,m,n,q in CASES:
        model=access_graph(kind,m=m,n=n,q=q)
        for policy in POLICIES:
            base=ArchitectureRequest()
            overrides={}
            if policy=="independent_phi":
                for i,op in enumerate(model.operators):
                    if op.kind=="MatMul":
                        overrides[op.name]=replace(
                            base,pes=2+(i%2),simd=2,serialization=1+i%2)
            arch=resolve(model,base,overrides)
            disjoint=graph_disjoint_offsets(arch)
            x=np.asarray([(-1)**i*(i+1)/9 for i in range(m*n)],
                         dtype=np.float32).reshape(m,n)
            for layout,offsets in (
                    ("disjoint",disjoint),
                    ("release_reuse",{**disjoint,"out":disjoint["h"]})):
                cert=graph_contract(arch,offsets)
                source=emit_graph_hls(arch,offsets=offsets)
                numeric=validate_host(model,source,x)
                trace=trace_graph_cpp(arch,source)
                if (not numeric["accepted"] or
                        trace["source_event_sha256"]!=cert["source_event_sha256"]
                        or trace["status"]!="EXECUTED_GRAPH_BOUNDARIES_VERIFIED"):
                    raise AssertionError("Nonidentity source access E0/E1 failed")
                records.append({
                    "graph":kind,"dims":json.dumps([m,n,q]),
                    "policy":policy,"layout":layout,
                    "functional_sha256":cert["functional_sha256"],
                    "architecture_sha256":cert["architecture_sha256"],
                    "source_cpp_sha256":source.manifest["cpp_sha256"],
                    "source_event_sha256":cert["source_event_sha256"],
                    "last_use_sha256":cert["last_use_sha256"],
                    "witness_sha256":trace["edge_witness_sha256"],
                    "independent_access_sha256":trace["independent_access_sha256"],
                    "source_scratch_slots":trace["source_array_elements"],
                    "logical_reads":trace["reads"],
                    "numerical_pass":numeric["accepted"],
                    "host_float_max_abs_error":numeric["max_absolute_error"],
                    "trace_pass":True,"native_vitis_calls":0,
                })
    for i in range(0,len(records),2):
        a,b=records[i:i+2]
        if not (a["source_scratch_slots"]>b["source_scratch_slots"]
                and a["functional_sha256"]==b["functional_sha256"]
                and a["architecture_sha256"]==b["architecture_sha256"]
                and a["source_event_sha256"]==b["source_event_sha256"]):
            raise AssertionError("Invalid safe source-array layout comparison")
    summary={
        "schema":"fpgai-research/cst-graph-nonidentity-access-E0-E1/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
            check=True,capture_output=True,text=True,timeout=5).stdout.strip(),
        "graph_shapes":len(CASES),"mapping_policies":len(POLICIES),
        "cases":len(records),"safe_layout_pairs":len(records)//2,
        "source_numerical_passes":sum(r["numerical_pass"] for r in records),
        "source_trace_passes":sum(r["trace_pass"] for r in records),
        "original_vitis_calls":0,
        "evidence":"E0-typed-rank2-broadcast-Concat-source-access-map-plus-original-GCC-E1",
        "limitations":[
            "Only rank-two axis-1 Concat and ONNX rank-two Add broadcast",
            "No Conv2D or general ONNX shape/streaming graph",
            "No original Vitis fixed-point CSim/CSynth, RTL/BRAM or timing evidence",
            "No external published baseline executed; no novelty or performance claim",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    print(json.dumps(run(output=p.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
