"""37th source-pinned all-access GCC experiment, no native FPGA tools.

Three NCHW graph families and two independently selected Conv Phi
policies, each in *both* genuine disjoint and shared HLS-source scratch
allocations. Only executed GCC source READ/WRITE/NumPy E1 is qualified.
"""
from __future__ import annotations
import argparse,csv,json,subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.nchw_shared_scratch import (
    greedy_nchw_stage_plan,emit_nchw_stage_scratch,
)
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_nchw_full_access import verify_original_nchw_access
from fpgai_research.validation.nchw_access_instrument import instrument_nchw_source

CASES=("branch_concat","pool_fork","broadcast_fork")
MAPPINGS=("scalar","independent_conv_phi")
STORAGE=("actual_disjoint","actual_reused")


def run(*,output:str|Path):
    output=Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Original GCC real source event experiment is create-only")
    rows=[]
    for graph in CASES:
        model=nchw_case(graph,height=4,width=4)
        x=np.asarray([(-1)**i*(i+1)/13 for i in range(16)],
                     dtype=np.float32).reshape(1,1,4,4)
        for mapping in MAPPINGS:
            base=ArchitectureRequest()
            overrides={}
            if mapping=="independent_conv_phi":
                for ordinal,op in enumerate(model.operators):
                    if op.kind=="Conv":
                        overrides[op.name]=replace(
                            base,pes=2,simd=2,
                            serialization=2 if ordinal%2==0 else 1)
            arch=resolve(model,base,overrides)
            for plan in STORAGE:
                offsets=greedy_nchw_stage_plan(
                    arch,reuse=plan=="actual_reused")
                src=emit_nchw_stage_scratch(
                    arch,offsets,trace_conv_access=False)
                traced=instrument_nchw_source(arch,src)
                verified=verify_original_nchw_access(
                    arch,traced,x,scratch_offsets=offsets)
                if verified["status"]!="REAL_GCC_READ_WRITE_ORDER_VERIFIED_E1":
                    raise AssertionError("Actual source event evidence rejected")
                rows.append({
                    "graph":graph,"mapping":mapping,"real_source_layout":plan,
                    "functional_sha256":verified["functional_sha256"],
                    "architecture_sha256":verified["architecture_sha256"],
                    "original_cpp_sha256":traced.manifest["original_cpp_sha256"],
                    "instrumented_cpp_sha256":verified["source_cpp_sha256"],
                    "original_input_sha256":verified["input_sha256"],
                    "raw_source_trace_sha256":verified["raw_source_trace_sha256"],
                    "scratch_source_elements":src.manifest["source_scratch_slots"],
                    "real_read_events":verified["actual_reads"],
                    "real_write_events":verified["actual_writes"],
                    "real_all_source_events":verified["actual_events"],
                    "source_access_order_verified":True,
                    "source_final_numpy_match":verified["source_final_matches_numpy"],
                    "native_vitis_calls":0,
                    "hardware_qualified":False,
                })
    savings=[]
    for i in range(0,len(rows),2):
        disjoint,reused=rows[i:i+2]
        if not (reused["scratch_source_elements"]<disjoint["scratch_source_elements"]
                and reused["real_all_source_events"]==disjoint["real_all_source_events"]
                and reused["functional_sha256"]==disjoint["functional_sha256"]
                and reused["architecture_sha256"]==disjoint["architecture_sha256"]
                and reused["original_cpp_sha256"]!=disjoint["original_cpp_sha256"]):
            raise AssertionError("Malformed actual source scratch E1 comparison")
        savings.append(disjoint["scratch_source_elements"]-reused["scratch_source_elements"])
    summary={
        "schema":"fpgai-research/cst-NCHW-real-read-write-access-GCC-study/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
            check=True,capture_output=True,text=True,timeout=5).stdout.strip(),
        "graph_families":len(CASES),"mapping_policies":len(MAPPINGS),
        "real_storage_plans":len(STORAGE),"cases":len(rows),
        "genuine_original_GCC_executions":len(rows),
        "source_read_write_verified":len(rows),
        "real_shared_source_pairs":len(savings),
        "minimum_source_scratch_element_saving":min(savings),
        "observed_real_reads":sum(x["real_read_events"] for x in rows),
        "observed_real_writes":sum(x["real_write_events"] for x in rows),
        "original_Vitis_calls":0,"hardware_qualified_cases":0,
        "evidence":"E1-GCC-original-source-proxy-backed-access-order-and-NumPy-causality",
        "limitations":[
            "Source-level real READ/WRITE trace, not native Vitis clock/port scheduling",
            "ReLU conditional and extra MaxPool first tap counted by independent NumPy",
            "No FPGA BRAM savings or physical QoR evidenced by source scratch elements",
        ],
    }
    output.mkdir(parents=True)
    with (output/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (output/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(output=parser.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
