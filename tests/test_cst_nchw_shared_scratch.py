"""Actual GCC compiled shared NCHW source memory, not just E0 logical offsets."""
from dataclasses import replace
from hashlib import sha256
import shutil
import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.nchw_shared_scratch import (
    greedy_nchw_stage_plan,stage_intervals,validate_stage_scratch,
    emit_nchw_stage_scratch,
)
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_nchw_intermediate_trace import attest_nchw_intermediates
from fpgai_research.validation.cpp_nchw_conv_graph_trace import trace_nchw_conv_sources
from fpgai_research.validation.host import validate_host


def arch_for(kind,independent=True):
    model=nchw_case(kind,height=4,width=4)
    base=ArchitectureRequest()
    overrides={}
    if independent:
        for i,op in enumerate(model.operators):
            if op.kind=="Conv":
                overrides[op.name]=replace(
                    base,pes=2,simd=2,serialization=2 if i%2==0 else 1)
    return resolve(model,base,overrides)


@pytest.mark.skipif(shutil.which("g++") is None,reason="Original GCC required")
@pytest.mark.parametrize("kind",["branch_concat","pool_fork","broadcast_fork"])
def test_actual_shared_hls_source_arrays_numerically_refine_all_intermediates(kind):
    arch=arch_for(kind)
    source_baseline=emit_nchw_stage_scratch(
        arch,greedy_nchw_stage_plan(arch,reuse=False))
    source_reuse=emit_nchw_stage_scratch(
        arch,greedy_nchw_stage_plan(arch,reuse=True))
    assert source_reuse.manifest["source_scratch_slots"]<(
        source_baseline.manifest["source_scratch_slots"])
    assert source_reuse.manifest["hardware_qualified"] is False
    x=np.asarray([(-1)**i*(i+1)/13 for i in range(16)],
                 dtype=np.float32).reshape(1,1,4,4)
    for emitted in (source_baseline,source_reuse):
        assert "__cst_scratch[" in emitted.cpp
        assert "data_t* tensor_" in emitted.cpp
        assert validate_host(arch.functional,emitted,x)["accepted"]
        tensor=attest_nchw_intermediates(arch,emitted,x)
        assert tensor["status"]=="ACTUAL_NCHW_INTERMEDIATE_VALUES_VERIFIED_E1"
        trace=trace_nchw_conv_sources(arch,emitted)
        assert trace["status"]=="ACTUAL_CONV_SOURCE_READS_VERIFIED_E1"
        assert trace["native_vitis_calls"]==0


def test_never_approve_stage_overlapping_alias_even_if_e0_scalar_happens_safe():
    arch=arch_for("branch_concat",independent=False)
    disjoint=greedy_nchw_stage_plan(arch,reuse=False)
    # The branch's first two outputs are simultaneously live.
    bad={**disjoint,"a":disjoint["h"]}
    with pytest.raises(ValueError,match="OVERLAPPING_SOURCE_STAGE_LIFETIMES"):
        validate_stage_scratch(arch,bad)
    with pytest.raises(ValueError,match="OVERLAPPING_SOURCE_STAGE_LIFETIMES"):
        emit_nchw_stage_scratch(arch,bad)
    with pytest.raises(ValueError,match="CAPACITY"):
        greedy_nchw_stage_plan(arch,max_slots=1)


def test_unknown_or_forged_scratch_layout_cannot_be_silently_ignored():
    arch=arch_for("pool_fork",independent=False)
    source=emit_nchw_stage_scratch(arch,greedy_nchw_stage_plan(arch))
    assert source.manifest["shared_source_memory"]=="actual-local-C++-scratch-pointer-aliases"
    assert source.manifest["original_disjoint_cpp_sha256"]!=source.manifest["cpp_sha256"]
    bad=greedy_nchw_stage_plan(arch)
    del bad["h"]
    with pytest.raises(ValueError,match="Complete"):
        validate_stage_scratch(arch,bad)
    wrong_schema=Emission(source.cpp.replace("data_t* tensor_","data_t* broken_",1),
                          dict(source.manifest))
    x=np.ones((1,1,4,4),dtype=np.float32)
    with pytest.raises(ValueError,match="digest"):
        attest_nchw_intermediates(arch,wrong_schema,x)
