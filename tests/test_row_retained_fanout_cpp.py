"""Prospective row-retained H lowering: source proof and actual GCC numerical tests.

All cases are NEW tests, not the historical R2g/R2h source-pinned study.
They do not run Vitis/Vivado or infer any physical FPGA BRAM savings.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2g_fanout_cpp import emit_fanout_cpp
from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    emit_row_retained_fanout_cpp, row_retained_plan,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    project_source_value_events,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.element_scratch_allocation import verify_allocation
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout, verify_fanout_issue,
)
from fpgai_research.validation.row_retained_fanout_cpp import (
    verify_row_retained_gcc,
)


def _arch(readers, shape, mapped=True):
    m, k, n, q = shape
    model = sample_fanout(readers, m=m, k=k, n=n, q=q)
    default = ArchitectureRequest()
    requested = {}
    if mapped:
        requested["reader0"] = replace(default, pes=2)
        requested["reader1"] = replace(default, simd=2)
        if readers >= 3:
            requested["reader2"] = replace(default, pes=2, simd=2)
        if readers == 4:
            requested["reader3"] = replace(default, pes=2)
    return resolve(model, default, requested)


@pytest.mark.parametrize("readers", (2, 3, 4))
@pytest.mark.parametrize("shape", ((2, 3, 3, 4), (3, 2, 4, 4)))
def test_row_plan_reuses_only_last_read_values_and_preserves_phi(readers, shape):
    arch = _arch(readers, shape)
    m, k, n, q = shape
    plan = row_retained_plan(arch)
    baseline = emit_fanout_cpp(arch)
    emitted = emit_row_retained_fanout_cpp(arch)
    assert plan["source_saved_scratch_elements"] == (m-1)*n
    assert plan["source_H_row_buffer_elements"] == n
    assert emitted.manifest["source_scratch_array_elements"] == (
        baseline.manifest["source_scratch_array_elements"] - (m-1)*n)
    assert emitted.manifest["source_all_declared_mutable_array_elements"] == (
        baseline.manifest["source_all_declared_mutable_array_elements"] - (m-1)*n)
    assert emitted.manifest["HLS_E3_qualified"] is None
    assert emitted.manifest["physical_E4_qualified"] is None
    assert emitted.manifest["source_cpp_sha256"] == sha256(
        emitted.cpp.encode()).hexdigest()
    assert plan["source_event_sha256"] == (
        emitted.manifest["source_event_order_sha256"])
    assert verify_fanout_issue(arch, plan["source_events"])["status"] == "VERIFIED_E0"
    spec = admit_fanout(arch)
    assert all(plan["source_slot_map"][(spec.h, row*n+col)] == col
               for row in range(m) for col in range(n))
    assert verify_allocation(
        project_source_value_events(plan["source_events"]),
        plan["source_slot_map"], unit_slots=plan["source_scratch_array_elements"],
    )["status"] == "EXACT_SOURCE_SLOT_REPLAY_VERIFIED"
    assert emitted.cpp != baseline.cpp
    # The original unchanged full-buffer emitter is not rewritten by
    # creating this alternative source arm.
    assert emit_fanout_cpp(arch).cpp == baseline.cpp


def test_accidental_within_row_alias_rejected_by_unit_slot_verifier():
    arch = _arch(3, (2, 2, 3, 4))
    plan = row_retained_plan(arch)
    bad = dict(plan["source_slot_map"])
    spec = admit_fanout(arch)
    bad[(spec.h, 1)] = bad[(spec.h, 0)]
    with pytest.raises(ValueError, match="OVERWRITE_LIVE_VALUE"):
        verify_allocation(
            project_source_value_events(plan["source_events"]),
            bad, unit_slots=plan["source_scratch_array_elements"],
        )


def test_early_row_release_breaks_architecture_order_and_source_contract():
    arch = _arch(2, (2, 2, 3, 4))
    plan = row_retained_plan(arch)
    events = list(plan["source_events"])
    spec = admit_fanout(arch)
    first_reads = [i for i, e in enumerate(events)
                   if e.kind == "READ" and e.tensor == spec.h]
    assert len(first_reads) >= 2
    a, b = first_reads[:2]
    events[a], events[b] = events[b], events[a]
    assert verify_fanout_issue(arch, tuple(events))["status"] == "REJECTED"


@pytest.mark.skipif(shutil.which("g++") is None, reason="Host g++ absent")
@pytest.mark.parametrize("readers,shape,mapped", (
    (2, (2, 3, 3, 4), False),
    (2, (3, 2, 4, 4), True),
    (3, (2, 2, 3, 4), True),
    (4, (3, 2, 3, 4), True),
))
def test_real_original_gcc_all_intermediates_and_actual_source_witnesses(
    readers, shape, mapped,
):
    arch = _arch(readers, shape, mapped)
    emission = emit_row_retained_fanout_cpp(arch)
    m, k, n, q = shape
    # Nontrivial signed input; selected to avoid all-zero outputs.
    x = np.asarray([[((row+2)*2 - col*3)/7.0
                     for col in range(k)] for row in range(m)],
                   dtype=np.float32)
    report = verify_row_retained_gcc(arch, emission, runtime_input=x)
    assert report["original_GCC_compilations"] == 1
    assert report["original_GCC_executions"] == 1
    assert report["original_Vitis_calls"] == 0
    assert report["original_Vivado_calls"] == 0
    assert report["HLS_BRAM_result"] is None
    assert report["source_saved_scratch_elements"] == (m-1)*n
    assert report["original_address_witnesses"] > 0
    assert report["final_max_absolute_error"] < 1e-3


@pytest.mark.skipif(shutil.which("g++") is None, reason="Host g++ absent")
def test_corrupted_actual_H_load_is_not_validated_merely_by_address_trace():
    arch = _arch(2, (2, 2, 3, 4))
    emitted = emit_row_retained_fanout_cpp(arch)
    original = "      data_t h_val=scratch[h_addr];"
    assert emitted.cpp.count(original) == 2
    changed = emitted.cpp.replace(
        original, "      data_t h_val=scratch[(h_addr+1)%3];", 1)
    forged = Emission(
        changed, {**emitted.manifest,
                  "source_cpp_sha256": sha256(changed.encode()).hexdigest()})
    x = np.asarray([[0.3, -0.5], [-0.7, 1.1]], dtype=np.float32)
    with pytest.raises(AssertionError, match="numerical mismatch"):
        verify_row_retained_gcc(arch, forged, runtime_input=x)
