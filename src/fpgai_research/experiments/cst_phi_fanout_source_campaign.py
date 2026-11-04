"""Bounded E0 Phi -> E1 source-executed two-reader diamond causality study.

Changed only the producer's legal MatMul spatial/reduction mapping while
holding Functional graph, tensor storage offsets, precision and compiler
source revision fixed. Original MatMul lowering is used by the generated
diamond. All numerical and source-iteration checks use actual GCC C++.
The two branch consumers are scalar; no HLS QoR or learned RL is measured.
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
from fpgai_research.codegen.hls.fanout_diamond import (
    emit_fanout_diamond_hls, fanout_diamond_offsets,
)
from fpgai_research.experiments.cst_fanout_source_campaign import sample_diamond
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.fanout_phi_source import mapped_fanout_contract
from fpgai_research.validation.cpp_fanout_trace import trace_fanout_cpp
from fpgai_research.validation.host import validate_host

CASES=(
    (2,3,1,1,1,()),
    (3,5,2,2,1,()),
    (5,3,2,2,2,(2,2,2)),
)


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Mapped fanout source experiment is create-only")
    rows=[]
    for m,n,pes,simd,fold,tile in CASES:
        model=sample_diamond(m,n)
        base=ArchitectureRequest()
        producer=replace(base,pes=pes,simd=simd,serialization=fold,tile=tile)
        arch=resolve(model,base,{model.operators[0].name:producer})
        offsets=fanout_diamond_offsets(model,share_released=True)
        e0=mapped_fanout_contract(arch,offsets)
        source=emit_fanout_diamond_hls(
            model,offsets=offsets,architecture=arch)
        x=np.asarray([(-1)**i*(i+1)/4 for i in range(m*2)],
                     dtype=np.float32).reshape(m,2)
        numeric=validate_host(model,source,x)
        e1=trace_fanout_cpp(model,source,architecture=arch)
        if (not numeric["accepted"] or
                e1["status"]!="SOURCE_FANOUT_LIFETIME_VERIFIED" or
                e1["phi_iteration_sha256"]!=e0["producer_phi_iteration_sha256"]):
            raise AssertionError("Source E1 mapping/consumer lifetime failed")
        rows.append({
            "m":m,"n":n,"pes":pes,"simd":simd,"fold":fold,
            "tile":json.dumps(tile),
            "functional_sha256":source.manifest["functional_sha256"],
            "architecture_sha256":e0["architecture_sha256"],
            "cpp_sha256":source.manifest["cpp_sha256"],
            "phi_sha256":e1["phi_iteration_sha256"],
            "phi_iterations":e1["phi_reduction_iterations_verified"],
            "source_event_sha256":e1["observed_event_sha256"],
            "multi_reader_values":e1["multi_reader_values"],
            "numerical_pass":numeric["accepted"],
            "max_host_float_error":numeric["max_absolute_error"],
            "actual_source_trace_pass":e1["status"]=="SOURCE_FANOUT_LIFETIME_VERIFIED",
            "native_vitis_calls":0,
        })
    summary={
        "schema":"fpgai-research/cst-phi-fanout-source-study-e1/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
            check=True,capture_output=True,text=True,timeout=5).stdout.strip(),
        "cases":len(rows),"actual_host_numeric_pass":sum(x["numerical_pass"] for x in rows),
        "actual_mapped_phi_trace_pass":sum(x["actual_source_trace_pass"] for x in rows),
        "original_vitis_calls":0,
        "evidence":"E0-mapped-producer-Phi-plus-E1-original-executed-GCC-source",
        "limitations":[
            "Only producer MatMul has spatial Phi; both branches use scalar operator mapping",
            "Bounded static five-operator diamond; no arbitrary graph source lowering",
            "No source-qualified original HLS or measured resource/timing/port utilization",
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
