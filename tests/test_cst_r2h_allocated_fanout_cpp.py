"""R2h: actual matched source disjoint/optimal memory and adversarial E1 controls."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    emit_allocated_fanout_cpp,source_layout,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2h_allocated_fanout_cpp import (
    decode_address_witnesses,verify_allocated_gcc,
)


def _arch(readers=3,*,m=2,k=2,n=3,q=4,policy="all"):
    model=sample_fanout(readers,m=m,k=k,n=n,q=q)
    base=ArchitectureRequest()
    widths=tuple(2 for _ in range(readers)) if policy=="all" else (1,)*readers
    return resolve(model,base,{
        f"reader{i}":replace(base,pes=width)
        for i,width in enumerate(widths) if width!=1
    })


@pytest.mark.parametrize("readers,dims,policy",[
    (2,(2,2,3,4),"scalar"),
    (2,(2,2,3,4),"all"),
    (3,(2,3,5,4),"all"),
    (4,(3,3,4,4),"all"),
    (4,(2,2,3,4),"scalar"),
])
def test_original_actual_CPP_matched_alias_source_address_and_intermediate_numeric(
        readers,dims,policy,tmp_path):
    m,k,n,q=dims
    arch=_arch(readers,m=m,k=k,n=n,q=q,policy=policy)
    results={}
    for layout in ("disjoint","per_element_optimal"):
        emitted=emit_allocated_fanout_cpp(arch,layout=layout)
        assert "r2h_slot_h[" in emitted.cpp
        assert "FPGAI_SLOT READ h" in emitted.cpp
        assert "FPGAI_SLOT COMMIT h" in emitted.cpp
        assert "scratch[h_addr]" in emitted.cpp
        assert "scratch[h_store_addr]" in emitted.cpp
        assert emitted.manifest["source_layout"]==layout
        path=tmp_path/(layout+"_original_stdout.txt")
        result=verify_allocated_gcc(
            arch,emitted,runtime_input=_input(m,k),
            original_trace_output=path)
        assert result["status"]=="R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED"
        assert result["source_scratch_unit_slots"]>0
        assert result["original_actual_address_witness_count"]>0
        assert result["original_all_intermediate_numeric_max_absolute_error"]<1e-4
        assert result["original_final_numeric_max_absolute_error"]<1e-4
        assert result["FPGA_hardware_BRAM_savings_proven"] is False
        assert sha256(path.read_bytes()).hexdigest()==result["original_stdout_sha256"]
        with pytest.raises(FileExistsError,match="create-only"):
            verify_allocated_gcc(
                arch,emitted,runtime_input=_input(m,k),
                original_trace_output=path)
        results[layout]=result
    assert results["disjoint"]["original_runtime_source_event_sha256"]==(
        results["per_element_optimal"]["original_runtime_source_event_sha256"])
    assert (results["disjoint"]["source_scratch_unit_slots"]>=
            results["per_element_optimal"]["source_scratch_unit_slots"])
    assert (results["disjoint"]["source_all_declared_mutable_array_elements"]-
            results["per_element_optimal"]["source_all_declared_mutable_array_elements"]
            ==results["disjoint"]["source_scratch_unit_slots"]-
              results["per_element_optimal"]["source_scratch_unit_slots"])


def test_independent_source_address_decoder_rejects_corrupt_runtime_address():
    arch=_arch(2)
    _,assigned,slots,_,_=source_layout(arch,layout="per_element_optimal")
    assert ("h",0) in assigned
    wrong=(assigned[("h",0)]+1)%slots
    if wrong==assigned[("h",0)]:
        pytest.skip("Degenerate one-slot source is not a 2-reader case")
    forged=(
        f"FPGAI_SLOT COMMIT h 0 {wrong}\n"
        "FPGAI_R2G COMMIT h 0\n"
        "FPGAI_R2G VALUE h 0 0\n"
    )
    with pytest.raises(ValueError,match="DIFFERS_FROM_CERTIFIED_VALUE"):
        decode_address_witnesses(
            forged,slot_map=assigned,capacity=slots)


def test_missing_runtime_address_witness_fails_even_if_typed_trace_is_valid():
    arch=_arch(2)
    _,slots_map,slots,_,_=source_layout(arch,layout="per_element_optimal")
    with pytest.raises(ValueError,match="WITHOUT_ACTUAL_ADDRESS_WITNESS"):
        decode_address_witnesses(
            "FPGAI_R2G COMMIT h 0\nFPGAI_R2G VALUE h 0 0\n",
            slot_map=slots_map,capacity=slots)


def test_runtime_source_arithmetic_mutation_rejected_at_intermediate_value():
    arch=_arch(2)
    emitted=emit_allocated_fanout_cpp(arch,layout="per_element_optimal")
    target="sum+=input[x_idx]*w0[w_idx];"
    assert target in emitted.cpp
    altered=emitted.cpp.replace(
        target,target.replace(";","+data_t(99);"),1)
    forged=Emission(altered,{
        **emitted.manifest,
        "source_cpp_sha256":sha256(altered.encode()).hexdigest(),
    })
    with pytest.raises(AssertionError,match="R2H_INDEPENDENT_INTERMEDIATE_NUMERIC_MISMATCH"):
        verify_allocated_gcc(arch,forged,runtime_input=_input(2,2))


def test_compiler_refuses_unknown_layout_or_wrong_original_source_sha():
    arch=_arch(2)
    with pytest.raises(ValueError,match="explicit disjoint"):
        emit_allocated_fanout_cpp(arch,layout="unsupported")
    emitted=emit_allocated_fanout_cpp(arch,layout="per_element_optimal")
    wrong=Emission(emitted.cpp+"// changed\n",emitted.manifest)
    with pytest.raises(ValueError,match="SHA lineage mismatch"):
        verify_allocated_gcc(arch,wrong,runtime_input=_input(2,2))
