"""Generalized NCHW CST mixed-operator graph source-pinned campaign.

Three graph families, a second channel/height setting, two mapping
policies and two mathematical scratch layout certificates. Each distinct
source graph emits genuine GCC-executed source and real Conv intermediate
load index traces; E0 scratch base reuse is NOT emitted physically.
"""
from __future__ import annotations

import argparse,csv,json,subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.nchw_graph_contract import (
    nchw_graph_events,disjoint_nchw_offsets,verify_nchw_graph,
    nchw_graph_contract,nchw_witness_receipt,
)
from fpgai_research.validation.nchw_access_oracle import independent_nchw_access
from fpgai_research.validation.cpp_nchw_conv_graph_trace import trace_nchw_conv_sources
from fpgai_research.validation.host import validate_host

CASES=(("branch_concat",4,4),("branch_concat",5,5),
       ("pool_fork",4,4),("broadcast_fork",4,4))
POLICIES=("scalar","independent_conv_phi")
LAYOUTS=("disjoint","post_last_use")


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("NCHW original experiment is create-only")
    records=[]
    for kind,height,width in CASES:
        model=nchw_case(kind,height=height,width=width)
        inputs=np.asarray([(-1)**i*(i+1)/13 for i in range(height*width)],
                          dtype=np.float32).reshape(1,1,height,width)
        for mapping in POLICIES:
            base=ArchitectureRequest()
            overrides={}
            if mapping=="independent_conv_phi":
                for index,op in enumerate(model.operators):
                    if op.kind=="Conv":
                        overrides[op.name]=replace(
                            base,pes=2,simd=2,
                            serialization=2 if index%2==0 else 1)
            arch=resolve(model,base,overrides)
            offsets=disjoint_nchw_offsets(arch)
            events=nchw_graph_events(arch)
            independent=independent_nchw_access(arch,events)
            if independent["status"]!="VERIFIED_INDEPENDENT_NCHW_ACCESS_E0":
                raise AssertionError("Independent semantic oracle rejects NCHW model")
            earlier=model.operators[1].outputs[0]
            if verify_nchw_graph(arch,{**offsets,earlier:offsets["h"]},
                                 events)["reason"]!="OVERWRITE_BEFORE_GLOBAL_LAST_READ":
                raise AssertionError("NCHW early alias not rejected")
            # Emit once per mathematical model and Phi: the source does not
            # accept or implement abstract scratch-base plans.
            source=emit_hls(lower(arch),trace_conv_access=True)
            numeric=validate_host(model,source,inputs)
            trace=trace_nchw_conv_sources(arch,source)
            if (not numeric["accepted"] or
                    trace["status"]!="ACTUAL_CONV_SOURCE_READS_VERIFIED_E1"):
                raise AssertionError("Real NCHW source evidence failed")
            for layout in LAYOUTS:
                chosen=(offsets if layout=="disjoint"
                        else {**offsets,"out":offsets["h"]})
                cert=nchw_graph_contract(arch,chosen)
                witness=nchw_witness_receipt(arch,chosen)
                records.append({
                    "graph":kind,"height":height,"width":width,"mapping":mapping,
                    "logical_layout":layout,"operator_count":cert["operators"],
                    "graph_edges":cert["edges"],"functional_sha256":cert["functional_sha256"],
                    "architecture_sha256":cert["architecture_sha256"],
                    "source_cpp_sha256":source.manifest["cpp_sha256"],
                    "logical_event_sha256":cert["event_sha256"],
                    "logical_last_use_sha256":cert["last_use_sha256"],
                    "logical_witness_sha256":witness["witness_sha256"],
                    "numpy_access_sha256":independent["access_multiset_sha256"],
                    "logical_scratch_slots":cert["source_scratch_elements"],
                    "actual_conv_source_trace_sha256":trace["observed_conv_trace_sha256"],
                    "original_gcc_numerical_pass":numeric["accepted"],
                    "original_gcc_conv_read_trace_pass":True,
                    "independent_e0_read_map_pass":True,
                    "logical_alias_e0_verified":True,
                    "physical_alias_emitted":False,
                    "native_vitis_calls":0,
                })
    for i in range(0,len(records),2):
        a,b=records[i:i+2]
        if not (a["logical_scratch_slots"]>b["logical_scratch_slots"]
                and a["functional_sha256"]==b["functional_sha256"]
                and a["architecture_sha256"]==b["architecture_sha256"]
                and a["logical_event_sha256"]==b["logical_event_sha256"]
                and a["source_cpp_sha256"]==b["source_cpp_sha256"]):
            raise AssertionError("Invalid source-matched LOGICAL scratch pair")
    summary={
        "schema":"fpgai-research/cst-general-NCHW-graph-E0-plus-partial-E1/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],check=True,
            capture_output=True,text=True,timeout=5).stdout.strip(),
        "graph_settings":len(CASES),"architectures":len(POLICIES),
        "logical_layouts":len(LAYOUTS),"cases":len(records),
        "unique_original_gcc_executions":len(CASES)*len(POLICIES),
        "source_matched_logical_layout_pairs":len(records)//2,
        "logical_e0_passes":len(records),
        "independent_numpy_e0_passes":len(records),
        "original_gcc_numerical_passes":sum(x["original_gcc_numerical_pass"] for x in records),
        "actual_original_gcc_conv_source_load_passes":sum(
            x["original_gcc_conv_read_trace_pass"] for x in records),
        "real_source_alias_emission_cases":0,
        "original_vitis_calls":0,
        "evidence":"E0-general-NCHW-mapped-value-read-and-last-use; partial-E1-original-Conv-LOAD+numerical",
        "limitations":[
            "General bounded 4D NCHW topological subset, not all ONNX/network operators",
            "Only actual Conv intermediate loads trace-attested; other accesses logical E0",
            "Logical scratch base comparison NOT materialized in canonical HLS source",
            "No actual Vitis CSim/CSynth, verified memory ports, clock, RTL or physical QoR",
            "No executed external compiler baselines or trained-network evidence",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(output=parser.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
