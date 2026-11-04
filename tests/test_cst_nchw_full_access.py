"""Falsifiable original GCC READ/WRITE source event refinement tests."""
from dataclasses import replace
from hashlib import sha256
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission,emit_hls
from fpgai_research.codegen.hls.nchw_shared_scratch import (
    emit_nchw_stage_scratch,greedy_nchw_stage_plan,
)
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.cpp_nchw_full_access import verify_original_nchw_access
from fpgai_research.validation.nchw_access_instrument import instrument_nchw_source


def case(kind="branch_concat",reuse=True,parallel=True):
    model=nchw_case(kind,height=4,width=4)
    base=ArchitectureRequest()
    override={}
    if parallel:
        for ordinal,op in enumerate(model.operators):
            if op.kind=="Conv":
                override[op.name]=replace(
                    base,pes=2,simd=2,serialization=2 if ordinal%2==0 else 1)
    arch=resolve(model,base,override)
    offsets=greedy_nchw_stage_plan(arch,reuse=reuse)
    orig=emit_nchw_stage_scratch(arch,offsets,trace_conv_access=False)
    traced=instrument_nchw_source(arch,orig)
    sample=np.asarray([(-1)**i*(i+1)/13 for i in range(16)],
                      dtype=np.float32).reshape(1,1,4,4)
    return arch,offsets,orig,traced,sample


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
@pytest.mark.parametrize("kind",["branch_concat","pool_fork","broadcast_fork"])
@pytest.mark.parametrize("reuse",[False,True])
def test_all_actual_source_reads_writes_after_gcc_execution(kind,reuse):
    arch,offsets,original,traced,x=case(kind,reuse)
    assert original.manifest["cpp_sha256"]==traced.manifest["original_cpp_sha256"]
    assert original.cpp!=traced.cpp
    result=verify_original_nchw_access(
        arch,traced,x,scratch_offsets=offsets)
    assert result["status"]=="REAL_GCC_READ_WRITE_ORDER_VERIFIED_E1"
    assert result["actual_writes"]==sum(
        int(np.prod(arch.functional.tensor(op.outputs[0]).shape))
        for op in arch.functional.operators)
    assert result["actual_reads"]>result["actual_writes"]//2
    assert result["native_vitis_calls"]==0
    assert result["hardware_qualified"] is False


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_corrupted_read_index_caught_with_forged_hash():
    arch,offsets,_,traced,x=case("branch_concat")
    # A valid-range shift changes real data, not the printed trace text.
    original="__cst_view_"+str(next(
        i for i,t in enumerate(arch.functional.tensors) if t.name=="h"))
    start=traced.cpp.index("  // OP ")
    snippet=traced.cpp[start:]
    assert original+"[" in snippet
    mutated=traced.cpp.replace(original+"[i]",original+"[(i+1)%32]",1)
    if mutated==traced.cpp:
        # Select first real source subscript for an arbitrary generated loop.
        import re
        match=re.search(r"(__cst_view_\d+)\[([^\]]+)\]",snippet)
        assert match is not None
        changed=match.group(1)+"[("+match.group(2)+"+1)%32]"
        mutated=traced.cpp[:start]+snippet.replace(match.group(0),changed,1)
    assert mutated!=traced.cpp
    forged=Emission(mutated,{**traced.manifest,
                             "cpp_sha256":sha256(mutated.encode()).hexdigest()})
    with pytest.raises((AssertionError,RuntimeError),match="SOURCE|ACCESS|read|operator|build"):
        verify_original_nchw_access(arch,forged,x,scratch_offsets=offsets)


def test_missing_source_and_forged_scratch_alias_fail_closed():
    arch,offsets,original,traced,x=case("pool_fork",parallel=False)
    budget=verify_original_nchw_access(arch,traced,x,max_events=1,
                                       scratch_offsets=offsets)
    assert budget["status"]=="UNKNOWN_EVENT_BUDGET"
    with pytest.raises(ValueError,match="source"):
        verify_original_nchw_access(arch,original,x,scratch_offsets=offsets)
    wrong={**offsets,"h":offsets["out"]}
    with pytest.raises(ValueError,match="offsets|LIFETIMES"):
        verify_original_nchw_access(arch,traced,x,scratch_offsets=wrong)
