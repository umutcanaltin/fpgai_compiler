"""Source-pinned NCHW Conv fanout, E0 per-kernel edge checks and GCC numeric.

The original Vitis-compatible C++ is GCC-executed, numerically compared
against NumPy, and opt-in instrumented for ACTUAL Conv reader kernel LOAD
indices and Phi owners. Other scalar reads/commits and SOURCE-ORDER
lifetimes have NOT been fully attested to source; do not overclaim.
"""
from __future__ import annotations

import argparse,csv,json,subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.experiments.cst_conv_boundary_fixtures import conv_diamond
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.conv_boundary import (
    conv_contract,conv_logical_events,disjoint_conv_offsets,verify_conv_logical,
)
from fpgai_research.validation.conv_access_oracle import independent_conv_access
from fpgai_research.validation.cpp_conv_trace import trace_conv_cpp
from fpgai_research.validation.host import validate_host

CASES=((4,4,3,1,1),(5,6,3,0,1),(5,6,3,1,2),(3,4,1,0,1))
POLICIES=("scalar","independent_conv_phi")


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Conv source science result is create-only")
    records=[]
    for h,w,kh,pad,stride in CASES:
        model=conv_diamond(height=h,width=w,kernel=kh,
                           pad=pad,stride=stride)
        x=np.asarray([(-1)**i*(i+1)/17 for i in range(h*w)],
                     dtype=np.float32).reshape(1,1,h,w)
        for policy in POLICIES:
            default=ArchitectureRequest()
            overrides={} if policy=="scalar" else {
                "producer":replace(default,pes=2,simd=2,serialization=2),
                "reader1":replace(default,pes=2,simd=2),
            }
            arch=resolve(model,default,overrides)
            offsets=disjoint_conv_offsets(arch)
            e0=conv_contract(arch,offsets)
            events=conv_logical_events(arch)
            independent=independent_conv_access(arch,events)
            if independent["status"]!="VERIFIED_INDEPENDENT_CONV_READS_E0":
                raise AssertionError("Independent NumPy access geometry differs from E0 Conv map")
            legal_reuse=verify_conv_logical(
                arch,{**offsets,"out":offsets["h"]},events)
            bad_alias=verify_conv_logical(
                arch,{**offsets,"a":offsets["h"]},events)
            if (legal_reuse["status"]!="VERIFIED_E0" or
                    bad_alias["reason"]!="OVERWRITE_BEFORE_ALL_READS"):
                raise AssertionError("Conv fanout lifetime controls failed")
            source=emit_hls(lower(arch),trace_conv_access=True)
            numeric=validate_host(model,source,x)
            traced=trace_conv_cpp(arch,source)
            if traced["status"]!="ACTUAL_CONV_KERNEL_LOADS_VERIFIED_E1":
                raise AssertionError("Actual original GCC kernel source load trace did not qualify")
            if not numeric["accepted"]:
                raise AssertionError("Actual standard Conv HLS C++ failed host-float numerical reference")
            records.append({
                "h":h,"w":w,"kernel":kh,"pad":pad,"stride":stride,
                "mapping":policy,"functional_sha256":e0["functional_sha256"],
                "architecture_sha256":e0["architecture_sha256"],
                "e0_event_sha256":e0["source_event_sha256"],
                "numpy_access_sha256":independent["multiset_sha256"],
                "cpp_sha256":source.manifest["cpp_sha256"],
                "logical_commits":e0["logical_commits"],
                "logical_reads":e0["logical_reads"],
                "logical_source_reuse_e0_pass":True,
                "early_alias_reject_e0_pass":True,
                "independent_source_access_e0_pass":True,
                "actual_generated_cpp_numeric_pass":numeric["accepted"],
                "source_read_trace_verified":True,
                "actual_source_read_count":traced["source_read_count"],
                "source_read_trace_sha256":traced["raw_gcc_trace_sha256"],
                "host_float_max_absolute_error":numeric["max_absolute_error"],
                "native_vitis_calls":0,
            })
    summary={
        "schema":"fpgai-research/cst-conv-NCHW-phi-fanout-E0-plus-E1-numeric/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
            check=True,capture_output=True,text=True,timeout=5).stdout.strip(),
        "shapes":len(CASES),"architecture_policies":len(POLICIES),
        "cases":len(records),
        "independent_conv_e0_pass":sum(x["independent_source_access_e0_pass"] for x in records),
        "original_gcc_numeric_pass":sum(x["actual_generated_cpp_numeric_pass"] for x in records),
        "original_gcc_per_read_trace_pass":len(records),
        "logical_alias_and_early_reject_pass":len(records),
        "native_vitis_calls":0,
        "evidence":"E0-independent-NCHW-kernel-access-plus-E1-original-GCC-Conv-reader-LOAD-and-numeric",
        "limitations":[
            "Five-operator static Conv diamond, group1/dilation1; no general 4D DAG",
            "E1 actual Conv READER per-kernel loads only; other scalar reads/COMMITS not source attested",
            "No original Vitis CSim/CSynth, RTL bank-port, schedule, hardware QoR",
            "No executed Allo/HIDA/ST-Flow/StreamTensor/Wavelet peer baseline",
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
