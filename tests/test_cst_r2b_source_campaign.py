"""Prospective original GCC E1 R2b: real source, not E0 schedule hashes."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import csv
import json

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2b_interleaved_branch import emit_r2b_interleaved_cpp
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import run, _input
from fpgai_research.experiments.cst_r2b_source_audit import audit
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2b_interleaved_cpp import (
    verify_r2b_gcc, _decode_trace,
)


def _arch(r0=None, r1=None):
    model=sample_branch(2,2,3,4)
    base=ArchitectureRequest()
    override={}
    if r0:
        override["reader0"]=replace(base,**r0)
    if r1:
        override["reader1"]=replace(base,**r1)
    return resolve(model,base,override)


def test_actual_original_gcc_interleaving_changed_source_last_read(tmp_path):
    scalar=_arch()
    changed=_arch(r1={"pes":2})
    positions=[]
    for i,arch in enumerate((scalar,changed)):
        offsets=disjoint_branch_offsets(arch)
        emitted=emit_r2b_interleaved_cpp(arch,offsets=offsets)
        # The READ must use the same C++ index symbol as its dynamic witness.
        assert "+r2b_index_" in emitted.cpp
        assert 'std::printf("FPGAI_R2B READ h %d' in emitted.cpp
        proof=verify_r2b_gcc(
            arch,emitted,offsets=offsets,
            runtime_input=_input(2,2),
            original_trace_output=tmp_path/f"source_{i}.txt")
        assert proof["status"]=="E1_ORIGINAL_GCC_SOURCE_EVENT_AND_NUMERIC_VERIFIED"
        assert proof["original_gcc_compilations"]==1
        assert proof["original_vitis_calls"]==0
        assert proof["original_source_h_reads"]==2*2*3*4
        assert proof["max_absolute_numeric_error"]<1e-4
        assert sha256((tmp_path/f"source_{i}.txt").read_bytes()).hexdigest()==(
            proof["raw_original_gcc_stdout_sha256"])
        positions.append(proof["observed_h_final_read_rank_and_owner"])
    assert positions[0]!=positions[1]


def test_reader0_only_phi_change_does_not_automatically_change_actual_last_read():
    scalar=_arch()
    only_reader0=_arch(r0={"pes":2})
    evidence=[]
    for arch in (scalar,only_reader0):
        offsets=disjoint_branch_offsets(arch)
        emitted=emit_r2b_interleaved_cpp(arch,offsets=offsets)
        evidence.append(verify_r2b_gcc(
            arch,emitted,offsets=offsets,runtime_input=_input(2,2)))
    # The event SHA may change due to intermediate COMMIT insertion,
    # but the independent H-only final READ rank/consumer can tie.
    assert evidence[0]["observed_h_final_read_rank_and_owner"]==(
        evidence[1]["observed_h_final_read_rank_and_owner"])


def test_actual_reused_source_memory_matches_independent_numpy(tmp_path):
    arch=_arch(r0={"pes":2},r1={"pes":2})
    disjoint=disjoint_branch_offsets(arch)
    shared=disjoint_branch_offsets(arch,reuse_after_last_read=True)
    a=emit_r2b_interleaved_cpp(arch,offsets=disjoint)
    b=emit_r2b_interleaved_cpp(arch,offsets=shared)
    assert a.manifest["scratch_slots"]>b.manifest["scratch_slots"]
    pa=verify_r2b_gcc(arch,a,offsets=disjoint,runtime_input=_input(2,2))
    pb=verify_r2b_gcc(arch,b,offsets=shared,runtime_input=_input(2,2))
    assert pa["observed_event_sha256"]==pb["observed_event_sha256"]
    assert pa["observed_h_last_use_sha256"]==pb["observed_h_last_use_sha256"]
    assert pa["max_absolute_numeric_error"]<1e-4
    assert pb["max_absolute_numeric_error"]<1e-4


def test_original_trace_parser_rejects_forged_or_missing_event_and_output():
    with pytest.raises(AssertionError):
        _decode_trace("FPGAI_R2B READD h 0 reader0 0 0 0 0 0\n",
                      n_outputs=1,max_events=100)
    with pytest.raises(AssertionError,match="cardinality"):
        _decode_trace("FPGAI_R2B COMMIT h 0\n",
                      n_outputs=1,max_events=100)
    with pytest.raises(AssertionError,match="Duplicate"):
        _decode_trace(
            "FPGAI_R2B OUTPUT 0 1.0\nFPGAI_R2B OUTPUT 0 2.0\n",
            n_outputs=1,max_events=100)


def test_mutated_runtime_trace_and_source_identity_are_rejected():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    emitted=emit_r2b_interleaved_cpp(arch,offsets=offsets)
    corrupted=emitted.cpp.replace("FPGAI_R2B READ h %d",
                                  "FPGAI_R2B READ stale %d",1)
    assert corrupted!=emitted.cpp
    with pytest.raises(ValueError,match="source/architecture"):
        verify_r2b_gcc(
            arch,Emission(corrupted,dict(emitted.manifest)),
            offsets=offsets,runtime_input=_input(2,2))
    updated={**emitted.manifest,
             "cpp_sha256":sha256(corrupted.encode()).hexdigest()}
    # The new source-independent Phi oracle rejects the forged READ
    # before the older byte-for-byte generated-event equality check.
    # Preserve this earlier and more meaningful certificate failure.
    with pytest.raises(
            AssertionError,match="Independent Phi source issue proof rejects GCC trace"):
        verify_r2b_gcc(arch,Emission(corrupted,updated),
                       offsets=offsets,runtime_input=_input(2,2))


def test_original_gcc_numPy_oracle_catches_wrong_source_load_with_intact_trace():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    emitted=emit_r2b_interleaved_cpp(arch,offsets=offsets)
    # Keep the real observed READ index and every source trace token
    # unchanged while deliberately corrupting an actual C++ operand.
    line=next(
        item for item in emitted.cpp.splitlines()
        if "data_t r2b_value_" in item and "scratch[0+r2b_index_" in item)
    broken=line.partition("=")[0]+"=data_t(999);"
    bad=emitted.cpp.replace(line,broken,1)
    assert bad!=emitted.cpp
    manifest={**emitted.manifest,"cpp_sha256":sha256(bad.encode()).hexdigest()}
    with pytest.raises(AssertionError,match="INDEPENDENT_NUMPY_SOURCE_NUMERIC_MISMATCH"):
        verify_r2b_gcc(
            arch,Emission(bad,manifest),offsets=offsets,
            runtime_input=_input(2,2))


def test_unsafe_source_memory_alias_is_rejected_before_gcc():
    arch=_arch()
    offsets=disjoint_branch_offsets(arch)
    offsets["a"]=0
    with pytest.raises(ValueError,match="unsafe"):
        emit_r2b_interleaved_cpp(arch,offsets=offsets)


def test_prospective_complete_original_E1_series_create_only_and_auditable(tmp_path):
    output=tmp_path/"r2b_prospective"
    summary=run(output=output)
    assert summary["graphs"]==3
    assert summary["prespecified_cases"]==24
    assert summary["original_gcc_compilations"]==24
    assert summary["original_gcc_executions"]==24
    assert summary["original_numeric_and_event_passes"]==24
    assert summary["original_controlled_phi_contrasts"]==18
    assert (summary["changed_original_executed_h_last_read_contrasts"]+
            summary["unchanged_original_executed_h_last_read_contrasts"])==18
    assert (summary["source_read_only_rank_or_owner_changed_contrasts"]+
            summary["source_read_only_rank_or_owner_unchanged_contrasts"])==18
    assert summary["source_read_only_rank_or_owner_unchanged_contrasts"]>=1
    assert summary["source_read_only_rank_or_owner_changed_contrasts"]>=1
    assert summary["global_event_index_only_effects_do_not_qualify"] is True
    assert summary["source_executed_lifetime_change_established"] is True
    assert summary["mapping_conditioned_source_capacity_difference_established"] is False
    assert summary["original_vitis_calls"]==0
    assert summary["original_vivado_calls"]==0
    with (output/"cases.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==24
    for case in rows:
        folder=output/case["case"]
        assert (folder/"source.cpp").is_file()
        assert sha256((folder/"source.cpp").read_bytes()).hexdigest()==(
            case["source_cpp_sha256"])
        assert sha256((folder/"original_gcc_trace.txt").read_bytes()).hexdigest()==(
            case["original_gcc_stdout_sha256"])
        assert json.loads((folder/"verified_e1.json").read_text())["original_gcc_executions"]==1
    assert json.loads((output/"summary.json").read_text())==summary
    checked=audit(output)
    assert checked["original_cases_verified"]==24
    assert checked["source_executed_last_use_recomputed"] is True
    assert checked["independent_H_only_read_rank_and_last_reader_recomputed"] is True
    assert checked["original_vendor_tool_calls_during_audit"]==0
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=output)
    # Preserve original source files in the failed audit case and refuse
    # to treat even one edited stdout as scientific evidence.
    original=output/"case_000"/"original_gcc_trace.txt"
    original.write_text(original.read_text().replace("FPGAI_R2B", "SPOOFED",1))
    with pytest.raises(ValueError,match="lineage"):
        audit(output)
