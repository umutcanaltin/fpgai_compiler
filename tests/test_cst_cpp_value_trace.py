"""Actual host-C++ Phi/value-readiness/reduction-lifetime refinement.

These are source-execution observations, NOT native HLS cycle or RTL proof.
"""
from __future__ import annotations

from dataclasses import replace
import shutil

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.experiments.cst_nonlinear_campaign import synthetic_nonlinear_model
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_value_trace import trace_cpp_values


@pytest.mark.skipif(shutil.which("g++") is None, reason="Actual GCC required")
@pytest.mark.parametrize("rows,block,widths",[
    (3,1,(3,5,2)),
    (3,2,(3,5,4,2)),
    (5,2,(3,5,4,2)),
    (5,5,(3,5,4,3,2)),
])
def test_real_matmul_source_proves_actual_mapped_value_readiness(rows,block,widths):
    model=make_chain(rows,widths)
    base=ArchitectureRequest()
    arch=resolve(model,base,{
        model.operators[0].name:replace(base,pes=2,simd=2,serialization=2),
    })
    compiled=compile_row_chain(arch,row_block=block)
    result=trace_cpp_values(compiled.ir,emission=compiled.emission)
    assert result["status"]=="SOURCE_VALUE_LIFETIMES_VERIFIED"
    assert result["cpp_sha256"]==compiled.emission.manifest["cpp_sha256"]
    assert result["row_block"]==block
    assert result["mapped_reduction_iterations_verified"]==sum(
        p.iteration_shape[0]*p.iteration_shape[1]*p.iteration_shape[2]
        for p in arch.points)
    assert result["element_commits_verified"]==sum(
        rows*model.tensor(op.outputs[0]).shape[1]
        for op in model.operators)
    assert result["last_reader_releases_verified"]==sum(
        rows*model.tensor(op.outputs[0]).shape[1]
        for op in model.operators[:-1])
    assert result["native_vitis_calls"]==0


@pytest.mark.skipif(shutil.which("g++") is None, reason="Actual GCC required")
@pytest.mark.parametrize("rows,block,transposed",[
    (2,1,1),(3,2,1),(5,2,0),(5,5,1)
])
def test_real_nonlinear_source_preserves_element_level_reuse(
        rows,block,transposed):
    model=synthetic_nonlinear_model(rows,trans_b=transposed)
    arch=resolve(model,ArchitectureRequest())
    c=compile_row_chain(arch,row_block=block)
    result=trace_cpp_values(c.ir,emission=c.emission)
    assert result["status"]=="SOURCE_VALUE_LIFETIMES_VERIFIED"
    assert result["element_commits_verified"]==rows*(6+4+4+2)
    assert result["last_reader_releases_verified"]==rows*(6+4+4)
    assert result["downstream_and_input_reads_verified"]>0
    assert result["row_block"]==block


@pytest.mark.skipif(shutil.which("g++") is None, reason="Actual GCC required")
def test_actual_source_trace_is_mapping_sensitive_but_not_a_hardware_timing_proof():
    model=make_chain(3,(3,5,2))
    base=ArchitectureRequest()
    first=compile_row_chain(resolve(model,base),row_block=1)
    second=compile_row_chain(resolve(model,base,{
        model.operators[0].name:replace(base,pes=2,simd=2),
    }),row_block=1)
    r1=trace_cpp_values(first.ir,emission=first.emission)
    r2=trace_cpp_values(second.ir,emission=second.emission)
    assert r1["status"]==r2["status"]
    assert r1["source_trace_sha256"]!=r2["source_trace_sha256"]
    assert r1["element_commits_verified"]==r2["element_commits_verified"]
    assert "HLS" in " ".join(r2["limits"])


def test_bound_and_source_provenance_fail_closed():
    model=make_chain(3,(3,5,2))
    c=compile_row_chain(resolve(model,ArchitectureRequest()))
    with pytest.raises(ValueError,match="budget"):
        trace_cpp_values(c.ir,emission=c.emission,max_events=1)
    with pytest.raises(ValueError,match="Positive finite"):
        trace_cpp_values(c.ir,emission=c.emission,max_events=0)
