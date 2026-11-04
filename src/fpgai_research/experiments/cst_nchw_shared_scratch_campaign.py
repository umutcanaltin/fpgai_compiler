"""Original actual source-level NCHW scratch reuse comparison.

Whole-operator interval legality is checked independently of E0 value
readiness. Both source versions allocate one real C++ local scratch[]
and alias intermediate pointers into it, then ACTUALLY execute GCC,
run full NumPy intermediate check and check actual Conv input loads.
No original Vitis CSynth/BRAM/clock claim.
"""
from __future__ import annotations
import argparse,csv,json,subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.nchw_shared_scratch import (
    greedy_nchw_stage_plan,emit_nchw_stage_scratch,stage_intervals,
)
from fpgai_research.experiments.cst_nchw_graph_campaign import CASES,POLICIES
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_nchw_intermediate_trace import attest_nchw_intermediates
from fpgai_research.validation.cpp_nchw_conv_graph_trace import trace_nchw_conv_sources


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Actual source scratch experiment create-only")
    rows=[]
    for kind,height,width in CASES:
        model=nchw_case(kind,height=height,width=width)
        x=np.asarray([(-1)**i*(i+1)/13 for i in range(height*width)],
                     dtype=np.float32).reshape(1,1,height,width)
        for policy in POLICIES:
            base=ArchitectureRequest()
            overrides={}
            if policy=="independent_conv_phi":
                for i,op in enumerate(model.operators):
                    if op.kind=="Conv":
                        overrides[op.name]=replace(
                            base,pes=2,simd=2,serialization=2 if i%2==0 else 1)
            arch=resolve(model,base,overrides)
            for name,reuse in (("real_disjoint_source",False),
                               ("real_shared_source",True)):
                offsets=greedy_nchw_stage_plan(arch,reuse=reuse)
                emitted=emit_nchw_stage_scratch(arch,offsets)
                numerics=attest_nchw_intermediates(arch,emitted,x)
                loaded=trace_nchw_conv_sources(arch,emitted)
                if (numerics["status"]!="ACTUAL_NCHW_INTERMEDIATE_VALUES_VERIFIED_E1"
                        or loaded["status"]!="ACTUAL_CONV_SOURCE_READS_VERIFIED_E1"):
                    raise AssertionError("Actual C++ scratch materialization lost refinement")
                rows.append({
                    "graph":kind,"height":height,"width":width,"policy":policy,
                    "real_source_plan":name,
                    "functional_sha256":numerics["functional_sha256"],
                    "architecture_sha256":numerics["architecture_sha256"],
                    "source_cpp_sha256":numerics["source_cpp_sha256"],
                    "original_disjoint_cpp_sha256":emitted.manifest["original_disjoint_cpp_sha256"],
                    "logical_last_use_sha256":emitted.manifest["logical_event_sha256"],
                    "actual_local_scratch_slots":emitted.manifest["source_scratch_slots"],
                    "observed_tensor_elements":numerics["observed_tensor_elements"],
                    "real_original_gcc_intermediate_numerical_pass":True,
                    "real_original_gcc_conv_load_pass":True,
                    "hardware_qualified":False,
                    "original_vitis_calls":0,
                })
    saved=[]
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        if not (b["actual_local_scratch_slots"]<a["actual_local_scratch_slots"]
                and a["functional_sha256"]==b["functional_sha256"]
                and a["architecture_sha256"]==b["architecture_sha256"]
                and a["logical_last_use_sha256"]==b["logical_last_use_sha256"]
                and a["source_cpp_sha256"]!=b["source_cpp_sha256"]):
            raise AssertionError("Actual shared source address comparison invalid")
        saved.append(a["actual_local_scratch_slots"]-b["actual_local_scratch_slots"])
    summary={
        "schema":"fpgai-research/cst-NCHW-actual-C++-source-scratch-reuse-E1/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
            check=True,capture_output=True,text=True,timeout=5).stdout.strip(),
        "graph_settings":len(CASES),"architecture_policies":len(POLICIES),
        "actual_source_variants":2,"cases":len(rows),
        "real_original_gcc_executions":len(rows)*2,
        "actual_source_scratch_saving_pairs":len(saved),
        "minimum_actual_C++_scratch_element_saving":min(saved),
        "total_actual_C++_scratch_element_saving_across_pairs":sum(saved),
        "all_intermediate_numerical_checks":len(rows),
        "all_conv_actual_load_checks":len(rows),
        "native_vitis_calls":0,
        "hardware_qualified_cases":0,
        "evidence":"E1-two-real-GCC-C++-source-storage-variants-per-NCHW-graph",
        "limitations":[
            "Whole-stage first-fit reuse: NOT optimal and NOT a per-element schedule",
            "Actual C++ scratch element savings are NOT Vitis inferred BRAM savings",
            "No CSynth, actual memory-port/clock, RTL, physical proof or vendor QoR",
            "Actual tensor snapshot/Conv load checks, NOT full scalar READ/COMMIT trace",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    print(json.dumps(run(output=p.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
