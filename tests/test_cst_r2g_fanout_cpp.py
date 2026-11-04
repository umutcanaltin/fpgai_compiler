"""R2g original GCC dynamic fanout source and independent fault injection."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2g_fanout_cpp import emit_fanout_cpp
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2g_fanout_cpp import (
    decode_stdout,verify_fanout_gcc,
)


def _arch(readers=3,*,m=2,k=2,n=3,q=4,policies=None):
    model=sample_fanout(readers,m=m,k=k,n=n,q=q)
    base=ArchitectureRequest()
    return resolve(model,base,{
        f"reader{i}":replace(base,pes=width)
        for i,width in enumerate(policies or (1,)*readers) if width!=1
    })


def _alter(source,old,new):
    assert old in source.cpp
    cpp=source.cpp.replace(old,new,1)
    return Emission(cpp,{
        **source.manifest,
        "source_cpp_sha256":sha256(cpp.encode()).hexdigest(),
    })


@pytest.mark.parametrize("readers,policies,q",[
    (2,(1,1),4),(2,(2,1),4),(2,(1,2),3),
    (3,(1,2,1),4),(3,(2,2,2),3),
    (4,(1,1,1,2),4),(4,(2,2,2,2),3),
])
def test_actual_GCC_2_to_4_runtime_fanout_reads_intermediates_and_numerics(
        readers,policies,q,tmp_path):
    arch=_arch(readers,q=q,policies=policies)
    emission=emit_fanout_cpp(arch)
    assert "r2g_decode_issue(ordinal" in emission.cpp
    assert "scratch[h_addr]" in emission.cpp
    assert "FPGAI_R2G VALUE" in emission.cpp
    assert emission.manifest["runtime_C++_indices_computed"] is True
    original=tmp_path/"original_stdout.txt"
    report=verify_fanout_gcc(
        arch,emission,runtime_input=_input(2,2),
        original_trace_output=original)
    assert report["status"]=="ORIGINAL_GCC_RUNTIME_2_TO_4_FANOUT_E1_VERIFIED"
    assert report["fanout_readers"]==readers
    assert report["original_producer_contributions"]==2*2*3
    assert report["original_H_reader_reads"]==readers*2*3*q
    assert report["source_intermediate_max_absolute_error"]<1e-4
    assert report["source_final_max_absolute_error"]<1e-4
    assert report["independent_all_intermediate_numeric_verification"] is True
    assert report["source_runtime_index_generator_event_replay"] is False
    assert report["hardware_cycle_or_BRAM_savings_proven"] is False
    assert sha256(original.read_bytes()).hexdigest()==report["original_stdout_sha256"]
    with pytest.raises(FileExistsError,match="create-only"):
        verify_fanout_gcc(
            arch,emission,runtime_input=_input(2,2),
            original_trace_output=original)


def test_actual_runtime_wrong_but_complete_Phi_reader_order_is_rejected():
    arch=_arch(3,policies=(2,1,2))
    source=emit_fanout_cpp(arch)
    bad=_alter(
        source,
        "*kk=within/active;",
        "*kk=within%3;",
    )
    bad=_alter(
        bad,
        "*col=start+within%active;",
        "*col=start+within/3;",
    )
    # This performs every reduction term for PE2 groups but the C++
    # ordinal-to-coordinate algorithm now uses a different order.
    with pytest.raises(AssertionError,match="R2G_INDEPENDENT_RUNTIME_PHI_ISSUE_MISMATCH"):
        verify_fanout_gcc(arch,bad,runtime_input=_input(2,2))


def test_missing_actual_GCC_producer_term_is_rejected_before_H_commit():
    arch=_arch()
    source=emit_fanout_cpp(arch)
    bad=_alter(
        source,'std::printf("FPGAI_R2G CONTRIBUTE ',
        'if(kk!=1) std::printf("FPGAI_R2G CONTRIBUTE ',
    )
    with pytest.raises(AssertionError,match="R2G_RUNTIME_REDUCTION_OR_ALIAS_MISMATCH"):
        verify_fanout_gcc(arch,bad,runtime_input=_input(2,2))


def test_arithmetic_mutation_keeps_event_trace_but_fails_intermediate_oracle():
    arch=_arch()
    source=emit_fanout_cpp(arch)
    bad=_alter(source,"sum+=input[x_idx]*w0[w_idx];",
               "sum+=input[x_idx]*w0[w_idx]+data_t(99);")
    with pytest.raises(AssertionError,match="R2G_INDEPENDENT_INTERMEDIATE_NUMERIC_MISMATCH"):
        verify_fanout_gcc(arch,bad,runtime_input=_input(2,2))


def test_wrong_H_runtime_value_witness_fails_even_when_final_is_correct():
    arch=_arch()
    source=emit_fanout_cpp(arch)
    bad=_alter(
        source,'h_idx,double(sum));',
        'h_idx,double(sum)+9);',
    )
    with pytest.raises(AssertionError,match="R2G_INDEPENDENT_INTERMEDIATE_NUMERIC_MISMATCH"):
        verify_fanout_gcc(arch,bad,runtime_input=_input(2,2))


def test_add_chain_arithmetic_mutation_is_detected_by_all_intermediate_values():
    arch=_arch(4)
    source=emit_fanout_cpp(arch)
    bad=_alter(source,"const data_t sum=left+right;",
               "const data_t sum=left-right;")
    with pytest.raises(AssertionError,match="R2G_INDEPENDENT_INTERMEDIATE_NUMERIC_MISMATCH"):
        verify_fanout_gcc(arch,bad,runtime_input=_input(2,2))


def test_early_runtime_reader_commit_fails_branch_readiness():
    arch=_arch()
    source=emit_fanout_cpp(arch)
    bad=_alter(
        source,"if(count[0][output_idx]==3)",
        "if(count[0][output_idx]==1)",
    )
    with pytest.raises(AssertionError,match="R2G_RUNTIME_REDUCTION_OR_ALIAS_MISMATCH"):
        verify_fanout_gcc(arch,bad,runtime_input=_input(2,2))


def test_source_SHA_or_noncomplete_runtime_value_fails_closed():
    arch=_arch()
    source=emit_fanout_cpp(arch)
    bad=Emission(source.cpp+"\n// corrupted\n",source.manifest)
    with pytest.raises(ValueError,match="SHA lineage mismatch"):
        verify_fanout_gcc(arch,bad,runtime_input=_input(2,2))
    with pytest.raises(ValueError,match="MISSING_ADJACENT"):
        decode_stdout("FPGAI_R2G COMMIT h 0\n"
                      "FPGAI_R2G OUTPUT 0 0\n")
