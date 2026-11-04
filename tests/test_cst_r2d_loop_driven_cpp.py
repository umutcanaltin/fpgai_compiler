"""R2d F1: actual runtime C++ index generation, independent Phi rule and faults."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2d_loop_driven_branch import emit_loop_driven_cpp
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2d_loop_driven_cpp import verify_loop_driven_gcc


def _arch(m=2,k=2,n=3,q=4,r0=1,r1=1):
    model=sample_branch(m,k,n,q)
    base=ArchitectureRequest()
    overrides={}
    if r0!=1:
        overrides["reader0"]=replace(base,pes=r0)
    if r1!=1:
        overrides["reader1"]=replace(base,pes=r1)
    return resolve(model,base,overrides)


@pytest.mark.parametrize("dimensions,policies",[
    ((2,2,3,4),(1,1)),
    ((2,2,3,4),(2,1)),
    ((2,2,3,4),(1,2)),
    ((2,2,3,4),(2,2)),
    ((2,2,3,3),(2,2)), # incomplete final PE group
    ((2,3,5,4),(2,1)),
    ((3,3,4,4),(2,2)),
])
def test_gcc_actual_loop_runtime_subscripts_match_independent_issue_and_numpy(
        dimensions,policies,tmp_path):
    m,k,n,q=dimensions
    arch=_arch(*dimensions,r0=policies[0],r1=policies[1])
    offsets=disjoint_branch_offsets(arch,reuse_after_last_read=True)
    emitted=emit_loop_driven_cpp(arch,offsets=offsets)
    assert "for(int ordinal=0;ordinal<" in emitted.cpp
    assert "r2d_decode_issue(ordinal" in emitted.cpp
    assert "const int source_idx=row*" in emitted.cpp
    assert "scratch[source_addr]" in emitted.cpp
    assert "FPGAI_R2B READ h %d" in emitted.cpp
    assert emitted.manifest["source_additional_count_array_elements"]==2*m*q
    assert emitted.manifest["source_all_declared_mutable_array_elements"]==(
        emitted.manifest["scratch_unit_slots"]+6*m*q)
    trace=tmp_path/"actual_runtime_cpp_trace.txt"
    result=verify_loop_driven_gcc(
        arch,emitted,offsets=offsets,runtime_input=_input(m,k),
        original_trace_output=trace)
    assert result["status"]=="E1_GCC_RUNTIME_SUBSCRIPT_PHI_ORDER_AND_NUMPY_VERIFIED"
    assert result["original_GCC_executions"]==1
    assert result["actual_H_read_count"]==2*m*n*q
    assert result["runtime_subscript_computed_by_CPP_loop"] is True
    assert result["source_order_generated_event_replay_dependency"] is False
    assert result["independent_phi_source_rule"]=="VERIFIED_E0_CLOSED_FORM"
    assert result["independent_branch_reduction_alias"]=="VERIFIED_E0"
    assert result["original_numerical_max_absolute_error"]<1e-4
    assert sha256(trace.read_bytes()).hexdigest()==result["original_GCC_trace_sha256"]
    assert result["original_native_Vitis_calls"]==0
    assert result["hardware_execution_scheduling_proven"] is False


def test_forged_CPP_runtime_decoder_order_is_caught_independently_even_if_full_reductions_survive():
    arch=_arch(r0=2,r1=1)
    offsets=disjoint_branch_offsets(arch)
    generated=emit_loop_driven_cpp(arch,offsets=offsets)
    source=generated.cpp
    # Both reader outputs still receive all kk, but the per-group
    # source READ order differs. The C++ emitter/manifest is rehashed,
    # so only independent semantic checking can reject the mutation.
    assert "*reduction=within/active;" in source
    assert "*col=group+within%active;" in source
    bad=source.replace("*reduction=within/active;",
                       "*reduction=within%3;",1).replace(
                       "*col=group+within%active;",
                       "*col=group+within/3;",1)
    altered=Emission(bad,{**generated.manifest,
                         "cpp_sha256":sha256(bad.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="INDEPENDENT_RUNTIME_PHI_ISSUE_ORDER_MISMATCH"):
        verify_loop_driven_gcc(
            arch,altered,offsets=offsets,runtime_input=_input(2,2))


def test_wrong_numeric_output_is_caught_after_valid_runtime_Phi_issue():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    generated=emit_loop_driven_cpp(arch,offsets=offsets)
    target="output[idx]=v>data_t(0)?v:data_t(0);"
    assert target in generated.cpp
    altered_source=generated.cpp.replace(target,
                                        "output[idx]=data_t(999);",1)
    altered=Emission(altered_source,{
        **generated.manifest,
        "cpp_sha256":sha256(altered_source.encode()).hexdigest(),
    })
    with pytest.raises(AssertionError,match="R2D_INDEPENDENT_NUMPY_NUMERIC_MISMATCH"):
        verify_loop_driven_gcc(
            arch,altered,offsets=offsets,runtime_input=_input(2,2))


def test_early_reduction_commit_in_actual_CPP_runtime_loop_is_rejected():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    generated=emit_loop_driven_cpp(arch,offsets=offsets)
    target="if(count_a[output_idx]==3)"
    assert target in generated.cpp
    bad=generated.cpp.replace(target,"if(count_a[output_idx]==1)",1)
    altered=Emission(bad,{
        **generated.manifest,"cpp_sha256":sha256(bad.encode()).hexdigest(),
    })
    with pytest.raises(AssertionError,match="INDEPENDENT_RUNTIME_REDUCTION_OR_ALIAS_VIOLATION"):
        verify_loop_driven_gcc(
            arch,altered,offsets=offsets,runtime_input=_input(2,2))


def test_loop_source_refuses_nonmatching_sha_without_GCC():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    generated=emit_loop_driven_cpp(arch,offsets=offsets)
    corrupt=Emission(generated.cpp+"//tampered\n",generated.manifest)
    with pytest.raises(ValueError,match="lineage mismatch"):
        verify_loop_driven_gcc(
            arch,corrupt,offsets=offsets,runtime_input=_input(2,2))
