"""Prespecified bounded exact E0 source-order allocation with real E1 GCC checks.

The finite optimum is ONLY for contiguous tensor-base addresses under the
recorded per-element COMMIT/CONSUME order of the restricted C++ diamond.
It does not prove any resource improvement after Vitis/RTL synthesis.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.codegen.hls.fanout_diamond import (
    emit_fanout_diamond_hls, fanout_diamond_offsets, source_order_events,
)
from fpgai_research.experiments.cst_fanout_source_campaign import (
    SHAPES, sample_diamond,
)
from fpgai_research.model_ir.fanout_allocator import exact_fanout_slot_plan
from fpgai_research.validation.cpp_fanout_trace import trace_fanout_cpp
from fpgai_research.validation.host import validate_host


def run(*, output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Source-order exact allocation study is create-only")
    cases=[]
    for m,n in SHAPES:
        model=sample_diamond(m,n)
        events=source_order_events(model)
        optimum=exact_fanout_slot_plan(
            model,("h","a","b","g"),event_sequence=events,
            max_slots=4*m*n,max_candidates=500_000)
        if optimum["status"]!="OPTIMAL_E0":
            raise AssertionError("Prespecified source-order exact allocation not qualified")
        disjoint=emit_fanout_diamond_hls(
            model,offsets=fanout_diamond_offsets(model,share_released=False))
        optimized=emit_fanout_diamond_hls(model,offsets=optimum["offsets"])
        if optimized.manifest["scratch_elements"]!=optimum["physical_slots"]:
            raise AssertionError("Emitted scratch differs from E0 allocation")
        x=np.asarray([((-1)**i)*(i+1)/4 for i in range(m*2)],
                     dtype=np.float32).reshape(m,2)
        for layout,emission in (("disjoint",disjoint),("exact_e0",optimized)):
            numeric=validate_host(model,emission,x)
            trace=trace_fanout_cpp(model,emission)
            if not numeric["accepted"] or trace["status"]!="SOURCE_FANOUT_LIFETIME_VERIFIED":
                raise AssertionError("Source E1 numerical/trace proof failed")
            cases.append({
                "rows":m, "columns":n,"layout":layout,
                "functional_sha256":emission.manifest["functional_sha256"],
                "cpp_sha256":emission.manifest["cpp_sha256"],
                "source_scratch_elements":emission.manifest["scratch_elements"],
                "source_e0_schedule":optimum["event_schedule"],
                "source_e0_optimum_slots":optimum["physical_slots"],
                "source_e0_explored_candidates":optimum["examined_candidates"],
                "actual_gcc_numeric_pass":numeric["accepted"],
                "actual_gcc_trace_pass":trace["status"]=="SOURCE_FANOUT_LIFETIME_VERIFIED",
                "actual_trace_sha256":trace["observed_event_sha256"],
                "native_vitis_calls":0,
            })
    summary={
        "schema":"fpgai-research/cst-fanout-source-order-exact-e1/v1",
        "source_revision":subprocess.run(
            ["git","rev-parse","HEAD"],check=True,capture_output=True,
            text=True,timeout=5).stdout.strip(),
        "cases":len(cases),"paired_graphs":len(SHAPES),
        "successful_host_numeric":sum(x["actual_gcc_numeric_pass"] for x in cases),
        "successful_source_traces":sum(x["actual_gcc_trace_pass"] for x in cases),
        "source_optimality_certificates":len(SHAPES),
        "source_scratch_saving_pairs":sum(
            cases[i]["source_scratch_elements"]>cases[i+1]["source_scratch_elements"]
            for i in range(0,len(cases),2)),
        "native_vitis_calls":0,
        "evidence":"E0-exact-source-order-address-model-plus-E1-actual-GCC",
        "limitations":[
            "Bounded exact optimum applies ONLY to the stipulated sequential source event order",
            "No native HLS area, memory, bank-port, timing or RTL proof",
            "Restricted diamond only, not generic Phi-conditioned DAG compiler",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]))
        writer.writeheader();writer.writerows(cases)
    (root/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),indent=2))


if __name__=="__main__":
    main()
