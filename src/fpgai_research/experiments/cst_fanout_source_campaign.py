"""E1 original generated-C++ two-reader fan-out controlled experiment.

Three pre-registered (M,N) sizes, two alias settings per task. Compiles
the actual restricted generated source with GCC, independently compares
its host-float result with NumPy Functional IR and executes source event
traces. Original Vitis and physical memory resources remain unmeasured.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.codegen.hls.fanout_diamond import (
    emit_fanout_diamond_hls, fanout_diamond_offsets,
)
from fpgai_research.model_ir.contracts import FunctionalModel, Operator, Tensor
from fpgai_research.model_ir.semantics import functional_fingerprint
from fpgai_research.validation.cpp_fanout_trace import trace_fanout_cpp
from fpgai_research.validation.host import validate_host

SHAPES = ((2,2),(3,3),(5,2))


def sample_diamond(m: int, n: int) -> FunctionalModel:
    if type(m) is not int or type(n) is not int or not 2 <= m <= 16 or not 2 <= n <= 16:
        raise ValueError("Prespecified bounded M,N >= 2 required")
    k=2
    return FunctionalModel(
        tensors=(
            Tensor("x",(m,k)),
            Tensor("w",(k,n),data=tuple(((-1)**i)*(i+1)/8 for i in range(k*n))),
            Tensor("h",(m,n)),Tensor("a",(m,n)),
            Tensor("b",(m,n)),Tensor("g",(m,n)),Tensor("out",(m,n)),
        ),
        operators=(
            Operator("producer","MatMul",("x","w"),("h",)),
            Operator("reader0","Relu",("h",),("a",)),
            Operator("reader1","Relu",("h",),("b",)),
            Operator("merge","Add",("a","b"),("g",)),
            Operator("final","Relu",("g",),("out",)),
        ),
        inputs=("x",), outputs=("out",),
    )


def run(*, output:str|Path) -> dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Fanout source study is create-only")
    rows=[]
    for m,n in SHAPES:
        model=sample_diamond(m,n)
        x=np.asarray([((-1)**i)*(i+1)/4 for i in range(m*2)],
                     dtype=np.float32).reshape(m,2)
        functional=functional_fingerprint(model)
        for alias in (False,True):
            source=emit_fanout_diamond_hls(
                model,offsets=fanout_diamond_offsets(model,share_released=alias))
            numerical=validate_host(model,source,x)
            trace=trace_fanout_cpp(model,source)
            if not numerical["accepted"] or trace["status"]!="SOURCE_FANOUT_LIFETIME_VERIFIED":
                raise AssertionError("Original generated C++ failed independent E1 controls")
            rows.append({
                "rows":m, "columns":n, "sharing":"shared_h_g" if alias else "disjoint",
                "functional_sha256":functional,
                "cpp_sha256":source.manifest["cpp_sha256"],
                "scratch_source_elements":source.manifest["scratch_elements"],
                "full_disjoint_source_elements":source.manifest["full_disjoint_scratch_elements"],
                "source_event_sha256":trace["observed_event_sha256"],
                "actual_host_numeric_pass":numerical["accepted"],
                "actual_host_max_abs_error":numerical["max_absolute_error"],
                "actual_host_trace_pass":trace["status"]=="SOURCE_FANOUT_LIFETIME_VERIFIED",
                "actual_trace_events":trace["observed_events"],
                "native_vitis_calls":0,
            })
    source_sha=subprocess.run(
        ["git","rev-parse","HEAD"],check=True,capture_output=True,
        text=True,timeout=5).stdout.strip()
    summary={
        "schema":"fpgai-research/cst-fanout-source-experiment-e1/v1",
        "source_revision":source_sha,
        "cases":len(rows),
        "distinct_graphs":len(SHAPES),
        "paired_source_layouts":len(SHAPES),
        "actual_gcc_numerical_passes":sum(x["actual_host_numeric_pass"] for x in rows),
        "actual_gcc_trace_passes":sum(x["actual_host_trace_pass"] for x in rows),
        "source_level_scratch_reduction_pairs":sum(
            rows[i]["scratch_source_elements"] > rows[i+1]["scratch_source_elements"]
            for i in range(0,len(rows),2)),
        "actual_vitis_cases":0,
        "evidence":"E1-compiled-generated-C++-two-reader-diamond-only",
        "limits":[
            "Static five-operator diamond only; no generic branch lowering",
            "Source scratch size is not measured HLS BRAM reduction",
            "No original Vitis, per-Phi fanout mapping or RTL bank-port proof",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),indent=2))


if __name__=="__main__":
    main()
