"""R2c: real original GCC address witnesses, NumPy and safety mutations."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2c_allocated_branch import emit_r2c_allocated_cpp
from fpgai_research.codegen.hls.r2b_interleaved_branch import emit_r2b_interleaved_cpp
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.element_scratch_allocation import allocate_minimum_slots
from fpgai_research.model_ir.interleaved_branch_schedule import interleaved_branch_events
from fpgai_research.validation.r2c_allocated_cpp import (
    decode_allocated_stdout,verify_r2c_gcc,
)
from fpgai_research.validation.r2b_interleaved_cpp import verify_r2b_gcc


def _arch(*,m=2,k=2,n=3,q=4,r0=1,r1=1):
    model=sample_branch(m,k,n,q)
    base=ArchitectureRequest()
    overrides={}
    if r0!=1:
        overrides["reader0"]=replace(base,pes=r0)
    if r1!=1:
        overrides["reader1"]=replace(base,pes=r1)
    return resolve(model,base,overrides)


@pytest.mark.parametrize("r0,r1",[(1,1),(2,1),(1,2),(2,2)])
def test_actual_gcc_per_element_memory_and_matmul_numeric_with_baseline(r0,r1):
    arch=_arch(r0=r0,r1=r1)
    source=emit_r2c_allocated_cpp(arch)
    manifest=source.manifest
    assert manifest["source_allocation_is_optimal_unit_slots"] is True
    assert manifest["source_scratch_unit_slots"]==manifest[
        "source_peak_live_intermediates"]
    assert manifest["total_declared_mutable_source_array_elements"]==(
        manifest["source_scratch_unit_slots"]+4*2*4)
    assert "scratch[r2c_slot_" in source.cpp
    assert "FPGAI_SLOT READ" in source.cpp
    checked=verify_r2c_gcc(arch,source,runtime_input=_input(2,2))
    assert checked["status"]=="ORIGINAL_GCC_SOURCE_NUMERIC_AND_SCRATCH_SAFETY_VERIFIED_E1"
    assert checked["source_optimality_proven_for_fixed_event_order"] is True
    assert checked["original_float32_numeric_max_abs_error"]<1e-4
    assert checked["original_Vivado_calls"]==0
    baseline=emit_r2b_interleaved_cpp(arch,offsets=disjoint_branch_offsets(
        arch,reuse_after_last_read=True))
    old=verify_r2b_gcc(arch,baseline,
                       offsets=disjoint_branch_offsets(
                           arch,reuse_after_last_read=True),
                       runtime_input=_input(2,2))
    assert old["observed_event_sha256"]==checked["original_source_event_sha256"]
    assert source.manifest["source_scratch_unit_slots"]<=baseline.manifest[
        "scratch_slots"]


def test_source_slot_witness_parser_refuses_missing_reordered_or_wrong_index():
    good=("FPGAI_SLOT COMMIT h 0 1\n"
          "FPGAI_R2B COMMIT h 0\n"
          "FPGAI_R2B OUTPUT 0 0.5\n")
    events,slots,out=decode_allocated_stdout(good,n_outputs=1)
    assert slots==(("COMMIT","h",0,1),)
    assert len(events)==1 and out.tolist()==[0.5]
    with pytest.raises(AssertionError,match="Missing/malformed"):
        decode_allocated_stdout(
            "FPGAI_R2B COMMIT h 0\nFPGAI_R2B OUTPUT 0 0.5\n",
            n_outputs=1)
    with pytest.raises(AssertionError,match="disagree"):
        decode_allocated_stdout(good.replace("COMMIT h 0 1",
                                              "COMMIT h 1 1"),n_outputs=1)
    with pytest.raises(AssertionError,match="after numerical"):
        decode_allocated_stdout(
            good+"FPGAI_SLOT COMMIT h 1 0\nFPGAI_R2B COMMIT h 1\n",
            n_outputs=1)


def test_early_alias_rejected_before_any_gcc_call():
    arch=_arch(m=1,k=1,n=1,q=1)
    good=allocate_minimum_slots(interleaved_branch_events(arch).events)
    bad=good.slot_map()
    bad[("a",0)]=bad[("h",0)]
    with pytest.raises(ValueError,match="OVERWRITE_LIVE_VALUE"):
        emit_r2c_allocated_cpp(arch,value_slots=bad)


def test_manifest_cpp_tamper_rejected_before_execution():
    arch=_arch()
    emission=emit_r2c_allocated_cpp(arch)
    altered=emission.cpp.replace("FPGAI_SLOT READ h",
                                 "FPGAI_SLOT READ stale",1)
    with pytest.raises(ValueError,match="identity"):
        verify_r2c_gcc(arch,Emission(altered,emission.manifest),
                       runtime_input=_input(2,2))


def test_gcc_numeric_oracle_rejects_wrong_actual_scratch_load_even_if_trace_indices_match():
    arch=_arch()
    emission=emit_r2c_allocated_cpp(arch)
    source=emission.cpp
    # Mutate an actual model output write without changing any source
    # READ/COMMIT trace or witness, and re-hash the altered C++.
    target=next(line for line in source.splitlines()
                if "output[r2c_idx_" in line)
    new=source.replace(target,target.partition("=")[0]+"=data_t(123456);",1)
    assert new!=source
    manifest=dict(emission.manifest)
    manifest["cpp_sha256"]=sha256(new.encode()).hexdigest()
    with pytest.raises(AssertionError,match="R2C_INDEPENDENT_NUMPY_NUMERICAL_MISMATCH"):
        verify_r2c_gcc(arch,Emission(new,manifest),runtime_input=_input(2,2))
