"""L6a conservative materialized NCHW shared-scratch source experiment.

Use whole-operator producer-to-last-consumer intervals (not per-element
source-loop timing) as the actual alias legality condition. Reuse is
materialized in original generated HLS C++ by assigning each intermediate
a slice of one concrete local scratch array; the top-level output remains
a distinct external pointer. The original numerical and Phi codegen is
otherwise untouched.

GCC E1 and source audit may qualify this SOURCE storage plan, but only
original Vitis CSim/CSynth and RTL qualify synthesis/RAM banking/ports.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
from math import prod

from fpgai_research.codegen.hls.emitter import Emission,emit_hls
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.nchw_graph_contract import (
    admit_nchw_graph,disjoint_nchw_offsets,nchw_graph_contract,
)

SCHEMA="fpgai-research/cst-actual-source-NCHW-shared-scratch-e1/v1"


def stage_intervals(arch):
    model,_=admit_nchw_graph(arch)
    produced_at={op.outputs[0]:i for i,op in enumerate(model.operators)}
    result={}
    for name,first in produced_at.items():
        if name in model.outputs:
            continue
        last=max([i for i,op in enumerate(model.operators) if name in op.inputs])
        result[name]=(first,last)
    return result


def _overlap(a,b):
    return a[0]<=b[1] and b[0]<=a[1]


def _addresses_overlap(off_a,len_a,off_b,len_b):
    return off_a<off_b+len_b and off_b<off_a+len_a


def validate_stage_scratch(arch,offsets,*,max_slots:int=100_000):
    """Independent conservative whole-stage materialization gate."""
    model,_=admit_nchw_graph(arch)
    labels={op.outputs[0] for op in model.operators}
    if (type(offsets) is not dict or set(offsets)!=labels or
            any(type(v) is not int or v<0 for v in offsets.values()) or
            type(max_slots) is not int or max_slots<1):
        raise ValueError("Complete immutable nonnegative NCHW scratch plan required")
    intervals=stage_intervals(arch)
    sizes={name:prod(model.tensor(name).shape) for name in intervals}
    active=sorted(intervals)
    for i,name in enumerate(active):
        for other in active[i+1:]:
            if (_overlap(intervals[name],intervals[other]) and
                    _addresses_overlap(offsets[name],sizes[name],
                                       offsets[other],sizes[other])):
                raise ValueError(
                    "OVERLAPPING_SOURCE_STAGE_LIFETIMES "+
                    f"{name} {other} source={offsets[name]} "
                    f"other={offsets[other]}"
                )
    slots=max((offsets[name]+sizes[name] for name in sizes),default=0)
    if slots<1 or slots>max_slots:
        raise ValueError("Shared source scratch capacity out of bounded range")
    # All E0 semantic reader identities and last-use must ALSO be valid.
    cert=nchw_graph_contract(arch,offsets)
    return {"schema":SCHEMA,"status":"SOURCE_STAGE_LAYOUT_VERIFIED_E0",
            "intervals":dict(sorted(intervals.items())),
            "scratch_slots":slots,
            "logical_certificate_sha256":cert["event_sha256"],
            "native_vitis_calls":0,
            "evidence":"Conservative-topological-stage-order-and-E0-source-value-lifetimes"}


def greedy_nchw_stage_plan(arch,*,reuse:bool=True,max_slots:int=100_000):
    """Deterministic conservative first-fit, NOT an optimal allocator."""
    model,_=admit_nchw_graph(arch)
    sizes={op.outputs[0]:prod(model.tensor(op.outputs[0]).shape)
           for op in model.operators if op.outputs[0] not in model.outputs}
    intervals=stage_intervals(arch)
    offsets={}
    end=0
    for op in model.operators:
        name=op.outputs[0]
        if name in model.outputs:
            continue
        limit=end if reuse else end
        chosen=None
        for candidate in (range(limit+1) if reuse else (end,)):
            if all(not(
                _overlap(intervals[name],intervals[other]) and
                _addresses_overlap(candidate,sizes[name],
                                   offsets[other],sizes[other])
            ) for other in offsets):
                chosen=candidate
                break
        if chosen is None:
            raise AssertionError("First-fit failed to place an appendable tensor")
        offsets[name]=chosen
        end=max(end,chosen+sizes[name])
        if end>max_slots:
            raise ValueError("UNKNOWN_SOURCE_SCRATCH_CAPACITY_BOUND")
    # The external top-level output is NOT a local array. Give the E0
    # logical sink a disjoint address so it doesn't imply physical reuse.
    offsets[model.outputs[0]]=end
    validate_stage_scratch(arch,offsets,max_slots=max_slots)
    return offsets


def emit_nchw_stage_scratch(arch,offsets,*,trace_conv_access:bool=True,
                            max_slots:int=100_000):
    model,_=admit_nchw_graph(arch)
    receipt=validate_stage_scratch(arch,offsets,max_slots=max_slots)
    original=emit_hls(lower(arch),trace_conv_access=trace_conv_access)
    identifiers={tensor.name:f"tensor_{i}"
                 for i,tensor in enumerate(model.tensors)}
    cpp=original.cpp
    replacements=[]
    for op in model.operators:
        name=op.outputs[0]
        if name in model.outputs:
            continue
        symbol=identifiers[name]
        size=prod(model.tensor(name).shape)
        old=f"  data_t {symbol}[{size}];"
        if cpp.count(old)!=1:
            raise ValueError("Original HLS intermediate declaration was not uniquely identified")
        replacements.append((old,f"  data_t* {symbol}=__cst_scratch+{offsets[name]};"))
    if not replacements:
        raise ValueError("At least one intermediate tensor required")
    first=replacements[0][0]
    cpp=cpp.replace(first,
        f"  data_t __cst_scratch[{receipt['scratch_slots']}];\n"+
        replacements[0][1],1)
    for before,after in replacements[1:]:
        cpp=cpp.replace(before,after,1)
    manifest={
        **original.manifest,
        "schema":SCHEMA,
        "cpp_sha256":sha256(cpp.encode()).hexdigest(),
        "original_disjoint_cpp_sha256":original.manifest["cpp_sha256"],
        "source_scratch_slots":receipt["scratch_slots"],
        "source_scratch_offsets":dict(sorted(offsets.items())),
        "source_stage_intervals":receipt["intervals"],
        "logical_event_sha256":receipt["logical_certificate_sha256"],
        "shared_source_memory":"actual-local-C++-scratch-pointer-aliases",
        "hardware_qualified":False,
        "native_vitis_calls":0,
        "limitations":[
            "Source C++ aliases are executed under GCC; not synthesized Vitis RAM claims",
            "Whole-operator conservative interval plan, NOT per-element optimal",
            "Actual Conv input loads traceable; scalar READ and actual COMMIT ordering remain open",
        ],
    }
    return Emission(cpp,manifest)
