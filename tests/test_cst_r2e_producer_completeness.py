"""R2e: independent producer reduction source-event and GCC/NumPy tests."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2e_producer_traced_branch import (
    emit_producer_contribution_cpp,
)
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2e_independent_producer_oracle import (
    inspect_producer_stdout,
)
from fpgai_research.validation.r2e_producer_cpp import verify_producer_traced_gcc


def _arch(m=2,k=2,n=3,q=4,r0=1,r1=1):
    model=sample_branch(m,k,n,q)
    base=ArchitectureRequest()
    changes={}
    if r0!=1:
        changes["reader0"]=replace(base,pes=r0)
    if r1!=1:
        changes["reader1"]=replace(base,pes=r1)
    return resolve(model,base,changes)


def _synthetic():
    return "\n".join([
        "FPGAI_PROD CONTRIBUTE 0 0 0 0",
        "FPGAI_PROD CONTRIBUTE 0 1 1 1",
        "FPGAI_PROD COMMIT 0",
        "FPGAI_R2B COMMIT h 0",
        "FPGAI_PROD VALUE 0 0.5",
        "FPGAI_R2B OUTPUT 0 0",
        "",
    ])


def test_independent_producer_oracle_accepts_exact_k_terms_and_typed_H_commit():
    result=inspect_producer_stdout(_synthetic(),m=1,k=2,n=1)
    assert result["status"]=="INDEPENDENT_COMPLETE_PRODUCER_TRACE_E0"
    assert result["required_contributions"]==2
    assert result["observed_contributions"]==2
    assert result["producer_commits"]==1
    assert result["H_typed_source_commits"]==1
    assert result["runtime_H_values"]==[0.5]
    assert "FPGAI_PROD" not in result["typed_branch_stdout"]
    assert "FPGAI_R2B COMMIT h 0" in result["typed_branch_stdout"]


@pytest.mark.parametrize("mutated,reason",[
    (
        "FPGAI_PROD CONTRIBUTE 0 0 0 0\n"
        "FPGAI_PROD COMMIT 0\nFPGAI_R2B COMMIT h 0\n"
        "FPGAI_R2B OUTPUT 0 0\n",
        "PREMATURE_H_PRODUCER_COMMIT_MISSING_TERMS",
    ),
    (
        "FPGAI_PROD CONTRIBUTE 0 0 0 0\n"
        "FPGAI_PROD CONTRIBUTE 0 0 0 0\n"
        "FPGAI_PROD CONTRIBUTE 0 1 1 1\n"
        "FPGAI_PROD COMMIT 0\nFPGAI_R2B COMMIT h 0\n",
        "PRODUCER_DUPLICATE_REDUCTION_TERM",
    ),
    (
        "FPGAI_PROD CONTRIBUTE 0 0 0 999\n",
        "PRODUCER_WRONG_INPUT_OR_WEIGHT_SUBSCRIPT",
    ),
    (
        "FPGAI_PROD CONTRIBUTE 0 0 0 0\n"
        "FPGAI_PROD CONTRIBUTE 0 1 1 1\n"
        "FPGAI_R2B COMMIT h 0\n",
        "H_SOURCE_COMMIT_WITHOUT_PRODUCER_COMPLETENESS",
    ),
    (
        "FPGAI_PROD CONTRIBUTE 0 0 0 0\n"
        "FPGAI_PROD CONTRIBUTE 0 1 1 1\n"
        "FPGAI_PROD COMMIT 0\n"
        "FPGAI_R2B OUTPUT 0 0\n",
        "PRODUCER_COMMIT_MISSING_ADJACENT_SOURCE_STORE_WITNESS",
    ),
    (
        "FPGAI_PROD CONTRIBUTE 0 0 0 0\n"
        "FPGAI_PROD CONTRIBUTE 0 1 1 1\n",
        "INCOMPLETE_PRODUCER_CONTRIBUTIONS_OR_COMMITS",
    ),
])
def test_independent_producer_rejects_missing_bad_or_uncommitted_contributions(mutated,reason):
    with pytest.raises(ValueError,match=reason):
        inspect_producer_stdout(mutated,m=1,k=2,n=1)


@pytest.mark.parametrize("shape,pe",[
    ((2,2,3,4),1),
    ((2,2,3,4),2),
    ((2,3,5,4),2),
    ((3,3,4,4),2),
    ((2,2,3,3),2),
])
def test_actual_GCC_producer_loop_contributions_and_independent_numpy(shape,pe,tmp_path):
    m,k,n,q=shape
    arch=_arch(*shape,r0=pe,r1=pe)
    offsets=disjoint_branch_offsets(arch,reuse_after_last_read=True)
    original=emit_producer_contribution_cpp(arch,offsets=offsets)
    assert "for(int kk=0;kk<" in original.cpp
    assert "sum+=input[x_address]*w0[weight_address];" in original.cpp
    assert "FPGAI_PROD CONTRIBUTE %d %d %d %d" in original.cpp
    assert "FPGAI_PROD COMMIT" in original.cpp
    trace=tmp_path/"new_r2e_original_runtime_trace.txt"
    result=verify_producer_traced_gcc(
        arch,original,offsets=offsets,runtime_input=_input(m,k),
        original_trace_output=trace)
    assert result["status"]=="ORIGINAL_R2E_GCC_PRODUCER_CONTRIBUTIONS_AND_NUMPY_VERIFIED_E1"
    assert result["independent_complete_H_producer_contributions"]==m*k*n
    assert result["independent_complete_H_producer_commits"]==m*n
    assert result["independent_H_store_witnesses"]==m*n
    assert result["E1_actual_source_producer_trace_and_numeric_verified"] is True
    assert result["original_numeric_max_absolute_error"]<1e-4
    assert result["original_intermediate_H_numeric_max_abs_error"]<1e-4
    assert result["original_Vitis_calls"]==0
    assert sha256(trace.read_bytes()).hexdigest()==result["original_full_stdout_sha256"]


def test_GCC_producer_rejects_missing_log_even_when_numerical_result_correct():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    original=emit_producer_contribution_cpp(arch,offsets=offsets)
    before=('        std::printf("FPGAI_PROD CONTRIBUTE %d %d %d %d\\n",\n'
            '                    idx,kk,x_address,weight_address);')
    assert before in original.cpp
    modified=original.cpp.replace(before,"        // deliberately omit producer term witness",1)
    forged=Emission(modified,{**original.manifest,
                             "cpp_sha256":sha256(modified.encode()).hexdigest()})
    with pytest.raises(ValueError,match="PREMATURE_H_PRODUCER_COMMIT_MISSING_TERMS"):
        verify_producer_traced_gcc(
            arch,forged,offsets=offsets,runtime_input=_input(2,2))


def test_GCC_producer_wrong_source_operand_refused_even_if_witnesses_complete():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    original=emit_producer_contribution_cpp(arch,offsets=offsets)
    old="sum+=input[x_address]*w0[weight_address];"
    assert old in original.cpp
    modified=original.cpp.replace(old,
        "sum+=input[x_address]*w0[weight_address]+data_t(99);",1)
    forged=Emission(modified,{**original.manifest,
                             "cpp_sha256":sha256(modified.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="R2E_INDEPENDENT_PRODUCER_H_NUMERIC_MISMATCH"):
        verify_producer_traced_gcc(
            arch,forged,offsets=offsets,runtime_input=_input(2,2))


def test_GCC_producer_wrong_runtime_operand_index_witness_refused():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    original=emit_producer_contribution_cpp(arch,offsets=offsets)
    old="idx,kk,x_address,weight_address);"
    assert old in original.cpp
    modified=original.cpp.replace(old,
                                  "idx,kk,x_address,weight_address+1);",1)
    forged=Emission(modified,{**original.manifest,
                             "cpp_sha256":sha256(modified.encode()).hexdigest()})
    with pytest.raises(ValueError,match="PRODUCER_WRONG_INPUT_OR_WEIGHT_SUBSCRIPT"):
        verify_producer_traced_gcc(
            arch,forged,offsets=offsets,runtime_input=_input(2,2))



def test_GCC_wrong_H_value_witness_rejected_even_when_final_numeric_is_correct():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    original=emit_producer_contribution_cpp(arch,offsets=offsets)
    before='std::printf("FPGAI_PROD VALUE %d %.9g\\n",idx,double(sum));'
    assert before in original.cpp
    changed=original.cpp.replace(
        before,'std::printf("FPGAI_PROD VALUE %d %.9g\\n",idx,double(sum)+9);',1)
    modified=Emission(changed,{
        **original.manifest,
        "cpp_sha256":sha256(changed.encode()).hexdigest(),
    })
    with pytest.raises(AssertionError,match="R2E_INDEPENDENT_PRODUCER_H_NUMERIC_MISMATCH"):
        verify_producer_traced_gcc(
            arch,modified,offsets=offsets,runtime_input=_input(2,2))
