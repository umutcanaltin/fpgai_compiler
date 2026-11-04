"""Independent cross-operator source mutation checks for CST IR research.

The generated source, not a copied Model IR hash, is deliberately modified.
This measures whether numerical and temporal host-based checks can expose
specific defects. Tests never invoke Vitis and are not full mutation coverage.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.compiler.composition import compile_composed_architecture
from fpgai_research.compiler.row_interleaved import compile_row_interleaved
from fpgai_research.experiments.cst_row_campaign import model_for
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_row_trace import trace_row_epochs
from fpgai_research.validation.host import validate_host

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ required to execute source")


def model():
    # Nonuniform weights ensure a wrong physical read cannot be hidden by
    # constant-valued channels.
    return model_for(3, 3, 5, 2)


def _mutate_emission(emission, before, after):
    assert before in emission.cpp
    changed = emission.cpp.replace(before, after, 1)
    assert changed != emission.cpp
    manifest = dict(emission.manifest)
    manifest["cpp_sha256"] = sha256(changed.encode()).hexdigest()
    return Emission(changed, manifest)


def test_row_epoch_trace_attests_actual_cxx_execution_and_detects_order_mutation():
    graph = model()
    arch = resolve(graph, ArchitectureRequest())
    compiled = compile_row_interleaved(arch)
    actual = trace_row_epochs(compiled.ir, compiled.emission)
    assert actual["row_count"] == 3
    assert actual["actual_events"] == 6
    assert actual["row_ready_before_consumed"]
    assert actual["native_vitis_executed"] is False
    # Tamper a source-level C++ trace event. A mere typed IR contract remains
    # unchanged, but actual executed event ordering now violates that contract.
    mutant = _mutate_emission(
        compiled.emission,
        'std::printf("FPGAI_ROW_READY %d\\n", mi);',
        'std::printf("FPGAI_ROW_CONSUMED %d\\n", mi);')
    with pytest.raises(AssertionError, match="ROW_LIFETIME_ORDER_VIOLATION"):
        trace_row_epochs(compiled.ir, mutant)


def test_wrong_real_producer_consumer_address_is_numerically_detected():
    graph = model()
    compiled = compile_composed_architecture(
        resolve(graph, ArchitectureRequest()),
        intermediate_layouts={"h": "column_major"})
    x = np.array([[0.5, -0.25, 0.75],
                  [-0.2, 0.9, 0.1],
                  [0.15, 0.35, -0.6]], dtype=np.float32)
    valid = validate_host(graph, compiled.emission, x, atol=1e-6)
    assert valid["accepted"]
    # Change one physical consumer-side address rule while leaving the
    # producer's correct column-major writes unchanged.
    source = compiled.emission.cpp
    source_expr = "tensor_2[kk*3+mi]"
    assert source_expr in source
    mutant = _mutate_emission(
        compiled.emission, source_expr, "tensor_2[mi*5+kk]")
    checked = validate_host(graph, mutant, x, atol=1e-6, rtol=1e-6)
    assert not checked["accepted"]
    assert checked["max_absolute_error"] > 1e-6


def test_wrong_row_scratch_read_is_numerically_detected():
    graph = model()
    compiled = compile_row_interleaved(resolve(graph, ArchitectureRequest()))
    x = np.array([[0.5, -0.25, 0.75],
                  [-0.2, 0.9, 0.1],
                  [0.15, 0.35, -0.6]], dtype=np.float32)
    valid = validate_host(graph, compiled.emission, x, atol=1e-6)
    assert valid["accepted"]
    mutant = _mutate_emission(compiled.emission,
                              "row_buffer[kk]*w1[",
                              "row_buffer[(kk+1)%5]*w1[")
    incorrect = validate_host(graph, mutant, x, atol=1e-6, rtol=1e-6)
    assert not incorrect["accepted"]
