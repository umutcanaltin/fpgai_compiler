"""Original GCC stage-by-stage numerical evidence across mixed NCHW DAGs.

One real GCC source execution checks *all intermediate tensor snapshots*
for each graph / independently mapped architecture. This is separate
from the existing Conv read-index E1 tests; no full READ/COMMIT timeline,
real shared scratch alias or Vitis QoR is claimed.
"""
from __future__ import annotations

import argparse,csv,json,subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.experiments.cst_nchw_graph_campaign import CASES,POLICIES
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.cpp_nchw_intermediate_trace import (
    attest_nchw_intermediates,
)


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original GCC intermediate evidence create-only")
    rows=[]
    receipts=[]
    for graph,height,width in CASES:
        model=nchw_case(graph,height=height,width=width)
        inp=np.asarray([(-1)**i*(i+1)/13 for i in range(height*width)],
                       dtype=np.float32).reshape(1,1,height,width)
        for mapping in POLICIES:
            base=ArchitectureRequest()
            overrides={}
            if mapping=="independent_conv_phi":
                for index,op in enumerate(model.operators):
                    if op.kind=="Conv":
                        overrides[op.name]=replace(
                            base,pes=2,simd=2,serialization=2 if index%2==0 else 1)
            arch=resolve(model,base,overrides)
            source=emit_hls(lower(arch))
            receipt=attest_nchw_intermediates(arch,source,inp)
            if receipt["status"]!="ACTUAL_NCHW_INTERMEDIATE_VALUES_VERIFIED_E1":
                raise AssertionError("Actual emitted GCC source has unqualified intermediate")
            receipts.append(receipt)
            rows.append({
                "graph":graph,"height":height,"width":width,
                "mapping":mapping,
                "functional_sha256":receipt["functional_sha256"],
                "architecture_sha256":receipt["architecture_sha256"],
                "source_cpp_sha256":receipt["source_cpp_sha256"],
                "input_sha256":receipt["input_sha256"],
                "original_trace_sha256":receipt["observed_raw_trace_sha256"],
                "operator_count":receipt["operator_snapshots"],
                "source_tensor_elements":receipt["observed_tensor_elements"],
                "maximum_intermediate_abs_error":max(
                    x["host_float_max_abs_error"]
                    for x in receipt["individual_operator_receipts"]),
                "all_intermediates_match_numpy":True,
                "actual_global_READ_COMMIT_order_attested":False,
                "native_vitis_calls":0,
            })
    summary={
        "schema":"fpgai-research/cst-NCHW-original-GCC-intermediate-all-operators-E1/v1",
        "source_revision":subprocess.run(
            ["git","rev-parse","HEAD"],check=True,capture_output=True,text=True,
            timeout=5).stdout.strip(),
        "graph_shapes":len(CASES),
        "mapping_policies":len(POLICIES),
        "unique_original_gcc_source_executions":len(receipts),
        "cases":len(rows),
        "actual_intermediate_snapshot_passes":len(rows),
        "total_original_operator_snapshots":sum(x["operator_snapshots"] for x in receipts),
        "total_checked_intermediate_elements":sum(x["observed_tensor_elements"] for x in receipts),
        "original_Vitis_calls":0,
        "evidence":"E1-executed-GCC-post-operator-value-refinement-NOT-timestamped-store",
        "limitations":[
            "Stage-level C++ tensor observations, not actual WRITE/COMMIT events",
            "Other consumer READs remain E0 except the separately observed Conv kernel loads",
            "No actual source-level scratch reuse, timing, RAM bank/port, original Vitis/RTL",
            "No published competitor artifact or trained-model native comparison",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    (root/"operator_receipts.json").write_text(
        json.dumps(receipts,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(output=parser.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
