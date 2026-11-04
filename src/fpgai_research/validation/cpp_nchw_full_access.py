"""Replay actual C++ NCHW READ/WRITE source events against NumPy and scratch lifetime.

Every event comes from an executed C++ array proxy conversion or store,
not from the abstract E0 event emitter. Per-stage access identities are
independently calculated from NumPy input-id grids plus dynamic ReLU
branch outcomes and MaxPool's extra seed read.

Source E1 is NOT FPGA cycle timing, RTL, inferred RAM ports, or Vitis.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from math import prod
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.model_ir.nchw_graph_contract import (
    admit_nchw_graph,disjoint_nchw_offsets,
)
from fpgai_research.model_ir.semantics import architecture_fingerprint,functional_fingerprint
from fpgai_research.validation.host import _float_literal
from fpgai_research.validation.nchw_dynamic_access_oracle import expected_actual_nchw_counts

SCHEMA="fpgai-research/observed-NCHW-C++-read-write-refinement-E1/v1"


def verify_original_nchw_access(arch,emission,runtime_input,*,
                                scratch_offsets=None,max_events=100_000,
                                timeout_seconds=120,cxx="g++"):
    model,produced=admit_nchw_graph(arch)
    m=emission.manifest
    if (not m.get("real_nchw_access_opt_in")
            or m.get("cpp_sha256")!=sha256(emission.cpp.encode()).hexdigest()
            or m.get("functional_sha256")!=functional_fingerprint(model)):
        raise ValueError("Original opt-in source fingerprint required")
    if type(max_events) is not int or max_events<1 or type(timeout_seconds) is not int or not 1<=timeout_seconds<=180:
        raise ValueError("Finite GCC source access budget/timeout required")
    x=np.asarray(runtime_input,dtype=np.float32)
    expected,reference=expected_actual_nchw_counts(arch,x)
    total=sum(expected.values())
    if total>max_events:
        return {"schema":SCHEMA,"status":"UNKNOWN_EVENT_BUDGET",
                "required_events":total,"max_events":max_events,
                "native_vitis_calls":0,"hardware_qualified":False}
    if scratch_offsets is None:
        scratch_offsets=m.get("source_scratch_offsets")
    if scratch_offsets is None:
        scratch_offsets=disjoint_nchw_offsets(arch)
    if m.get("source_scratch_offsets") is not None:
        if scratch_offsets!=m["source_scratch_offsets"]:
            raise ValueError("Claimed scratch offsets differ from actual generated source")
        from fpgai_research.codegen.hls.nchw_shared_scratch import validate_stage_scratch
        validate_stage_scratch(arch,scratch_offsets)
    if set(scratch_offsets)!=produced or any(type(v) is not int or v<0 for v in scratch_offsets.values()):
        raise ValueError("Complete nonnegative source scratch address mapping required")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Real host GCC required")
    names={i:t.name for i,t in enumerate(model.tensors) if t.name in produced}
    sizes={name:prod(model.tensor(name).shape) for name in produced}
    final_tensor=model.outputs[0]
    final_size=sizes[final_tensor]
    initializer=",".join(_float_literal(v) for v in x.flat)
    runner=(f"\nint main() {{float input[{x.size}]={{{initializer}}};"
            f"float output[{final_size}]={{}}; fpgai_model(input,output);"
            f'for(int i=0;i<{final_size};++i) std::printf("CST_FINAL %d %.9g%c",i,double(output[i]),10);'
            "return 0;}\n")
    with TemporaryDirectory(prefix="fpgai_full_source_rw_") as directory:
        src=Path(directory)/"source.cpp"
        exe=Path(directory)/"native_gcc"
        src.write_text(emission.cpp+runner)
        built=subprocess.run([compiler,"-std=c++17","-O0","-Wno-unknown-pragmas",
            "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_NCHW_EVENTS",str(src),"-o",str(exe)],
            text=True,capture_output=True,timeout=timeout_seconds)
        if built.returncode:
            raise RuntimeError("Source GCC access build failed: "+built.stderr[-5000:])
        run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=timeout_seconds)
        if run.returncode:
            raise RuntimeError("Source GCC access execution failed: "+run.stderr[-5000:])
    outstanding=Counter(expected)
    remaining=Counter()
    for (kind,stage,name,index),count in expected.items():
        if kind=="R":
            remaining[(name,index)]+=count
    seen=set()
    active={}
    observed=Counter()
    final=[]
    stage_floor=0
    observed_events=0
    for line_no,line in enumerate(run.stdout.splitlines()):
        fields=line.split()
        if len(fields)==5 and fields[0]=="CST_ACCESS":
            kind=fields[1]
            try:
                stage,tid,element=map(int,fields[2:])
            except ValueError as exc:
                raise AssertionError("Noninteger source event") from exc
            name=names.get(tid)
            if (kind not in ("R","W") or name is None or
                    not 0<=stage<len(model.operators) or
                    stage<stage_floor or not 0<=element<sizes[name]):
                raise AssertionError("INVALID_ACTUAL_SOURCE_ACCESS "+line)
            token=(kind,stage,name,element)
            if outstanding[token]<=0:
                raise AssertionError("WRONG_ACTUAL_SOURCE_ACCESS "+str(token))
            outstanding[token]-=1
            observed[token]+=1
            identity=(name,element)
            address=scratch_offsets[name]+element
            if kind=="R":
                if identity not in seen:
                    raise AssertionError("READ_BEFORE_ACTUAL_SOURCE_STORE "+str(token))
                if name!=final_tensor and active.get(address)!=identity:
                    raise AssertionError("ACTUAL_SOURCE_STALE_OR_OVERWRITTEN_READ "+str(token))
                remaining[identity]-=1
                if remaining[identity]<0:
                    raise AssertionError("OVERCOUNTED_REAL_SOURCE_READ")
                if remaining[identity]==0 and name!=final_tensor:
                    active.pop(address,None)
            else:
                if name!=model.operators[stage].outputs[0] or identity in seen:
                    raise AssertionError("INVALID_SOURCE_PRODUCER_OR_DUPLICATE_WRITE "+str(token))
                if name!=final_tensor:
                    if address in active:
                        raise AssertionError("REAL_SOURCE_PREMATURE_OVERWRITE "+str(token))
                    active[address]=identity
                seen.add(identity)
            observed_events+=1
            stage_floor=stage
        elif len(fields)==3 and fields[0]=="CST_FINAL":
            try:
                index=int(fields[1]); value=float(fields[2])
            except ValueError as exc:
                raise AssertionError("Malformed actual final output") from exc
            if (observed_events!=total or index!=len(final) or
                    not np.isfinite(value)):
                raise AssertionError("Final output before complete source trace")
            final.append(value)
        else:
            raise AssertionError("Unexpected original GCC trace row "+str(line_no))
    if (observed_events!=total or any(outstanding.values())
            or active or len(final)!=final_size):
        raise AssertionError("SOURCE_READ_WRITE_TRACE_INCOMPLETE")
    if len(seen)!=sum(sizes.values()):
        raise AssertionError("Actual source did not write every produced value")
    if not np.allclose(final,reference[final_tensor].ravel(),
                       atol=1e-5,rtol=1e-4):
        raise AssertionError("ACTUAL_SOURCE_FINAL_NUMERIC_MISMATCH")
    return {"schema":SCHEMA,"status":"REAL_GCC_READ_WRITE_ORDER_VERIFIED_E1",
            "source_cpp_sha256":m["cpp_sha256"],
            "functional_sha256":functional_fingerprint(model),
            "architecture_sha256":architecture_fingerprint(arch),
            "input_sha256":sha256(x.tobytes()).hexdigest(),
            "raw_source_trace_sha256":sha256(run.stdout.encode()).hexdigest(),
            "actual_events":observed_events,
            "actual_reads":sum(v for (kind,_,_,_),v in observed.items() if kind=="R"),
            "actual_writes":sum(v for (kind,_,_,_),v in observed.items() if kind=="W"),
            "max_events":max_events,
            "source_alias_identity_checked":True,
            "source_final_matches_numpy":True,
            "native_vitis_calls":0,
            "hardware_qualified":False,
            "limitations":[
                "Actual GCC source reads/writes, not RTL cycles or physical RAM ports",
                "Per-stage source-element multiplicity, not every reduction-output owner",
                "GCC HOST_FLOAT source wrapper does not establish Vitis equivalence",
            ]}
