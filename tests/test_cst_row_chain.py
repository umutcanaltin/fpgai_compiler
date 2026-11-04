"""Three- and four-operator CST row-chain E0/E1 regression tests."""
from __future__ import annotations

from dataclasses import replace
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.validation.cpp_phi_trace import trace_cpp_phi
from fpgai_research.validation.cpp_chain_trace import trace_chain_epochs
from fpgai_research.validation.host import validate_host
from fpgai_research.model_ir.row_chain_liveness import verify_row_chain_liveness


def chain(rows=3, widths=(3, 5, 4, 2)):
    tensors = [Tensor("x", (rows, widths[0]))]
    ops = []
    for i, (k, n) in enumerate(zip(widths[:-1], widths[1:])):
        name = f"w{i}"
        tensors.append(Tensor(
            name, (k, n),
            data=tuple(0.015 * ((index % 9) - 4) for index in range(k*n))))
        out_name = "y" if i == len(widths)-2 else f"hidden{i}"
        tensors.append(Tensor(out_name, (rows, n)))
        ops.append(Operator(f"op{i}", "MatMul",
                            (tensors[0].name if i == 0 else f"hidden{i-1}", name),
                            (out_name,)))
    return FunctionalModel(tuple(tensors), tuple(ops), ("x",), ("y",))


def heterogeneous(f):
    base = ArchitectureRequest()
    return resolve(f, base, {
        op.name: replace(
            base, pes=(i % 3)+1, parallelism=(i % 2)+1,
            simd=(i % 3)+1, serialization=(i % 2)+1)
        for i, op in enumerate(f.operators)
    })


@pytest.mark.parametrize("widths", [(3, 5, 4, 2), (3, 5, 4, 3, 2)])
def test_chain_ir_subsets_genuinely_materialize_distinct_graph_lifetimes(widths):
    graph = chain(widths=widths)
    compiled = compile_row_chain(heterogeneous(graph))
    ir = compiled.ir
    assert len(ir.chain) == len(widths)-1
    assert len(ir.scratch_by_tensor) == len(widths)-2
    assert ir.row_count == 3
    assert ir.physical_scratch_elements == sum(widths[1:-1])
    assert ir.full_intermediate_elements == 3*sum(widths[1:-1])
    assert "for (int mi=0; mi<3; ++mi)" in compiled.emission.cpp
    assert f"data_t scratch_0[{widths[1]}]" in compiled.emission.cpp
    assert f"data_t scratch_{len(widths)-3}[{widths[-2]}]" in compiled.emission.cpp
    assert "FPGAI_CHAIN_ROW_READY 0" in compiled.emission.cpp
    assert "FPGAI_CHAIN_ROW_CONSUMED 0" in compiled.emission.cpp
    assert compiled.evidence["vitis_calls"] == 0
    witness = compiled.evidence["independent_liveness_witness"]
    assert witness["status"] == "EXHAUSTIVE_BOUNDED_LIVENESS_VERIFIED"
    assert witness["physical_scratch_elements"] == sum(widths[1:-1])
    assert witness["peak_simultaneously_live_scratch_values"] == max(
        (widths[i] + widths[i+1] for i in range(1, len(widths)-2)),
        default=widths[1])

    assert witness["all_live_values_released_each_row"]
    assert all(e.storage_layout == "row_buffer"
               and e.exchange_policy == "producer_row_complete_then_consumer_row"
               for e in ir.composed.boundaries)


@pytest.mark.parametrize("widths", [(3, 5, 4, 2), (3, 5, 4, 3, 2)])
def test_actual_generated_chain_cpp_matches_independent_numeric_reference_and_phi(widths):
    if shutil.which("g++") is None:
        pytest.skip("GCC is unavailable")
    graph = chain(rows=3, widths=widths)
    compiled = compile_row_chain(heterogeneous(graph))
    x = np.array([[.2, .1, -.25], [.5, -.15, .35], [.4, .05, -.45]],
                 dtype=np.float32)
    receipt = validate_host(graph, compiled.emission, x,
                            atol=1e-5, rtol=1e-4)
    assert receipt["accepted"]
    assert receipt["max_absolute_error"] < 1e-5
    traced = trace_cpp_phi(compiled.ir.composed.lowered,
                           emission=compiled.emission)
    event_trace = trace_chain_epochs(compiled.ir, compiled.emission)
    assert event_trace["checked_edges"] == len(compiled.ir.chain)-1
    assert event_trace["actual_event_count"] == 3*2*(len(compiled.ir.chain)-1)
    assert traced["source_phi_agreement"]
    assert traced["operators"] == {
        name: 3*widths[i]*widths[i+1] for i, name in enumerate(compiled.ir.chain)
    }


def test_non_chain_fanout_and_non_row_local_tile_fail_closed():
    graph = chain()
    base = ArchitectureRequest()
    illegal = resolve(graph, base, {
        "op0": replace(base, tile=(2, 5, 3)),
    })
    with pytest.raises(ValueError, match="row tile extent"):
        compile_row_chain(illegal)
    # A value consumed by two MatMuls is not a legal single-lifetime
    # chain; do not silently reorder or overwrite it.
    branched = FunctionalModel(
        tensors=graph.tensors + (
            Tensor("other_w", (5, 2), data=(.01,)*10),
            Tensor("other_y", (3, 2)),
        ),
        operators=graph.operators + (
            Operator("extra", "MatMul", ("hidden0", "other_w"), ("other_y",)),
        ),
        inputs=("x",), outputs=("y",),
    )
    with pytest.raises(ValueError, match="exactly one internal|sequential MatMul|single|must start|requires"):
        compile_row_chain(resolve(branched, base))


def test_bounded_liveness_witness_does_not_treat_unverified_as_pass():
    graph = chain(rows=3, widths=(3, 5, 4, 2))
    ir = compile_row_chain(heterogeneous(graph)).ir
    bounded = verify_row_chain_liveness(ir, max_events=1)
    assert bounded["status"] == "BOUND_EXCEEDED"
    assert bounded["vitis_executed"] is False
    with pytest.raises(ValueError, match="max_events"):
        verify_row_chain_liveness(ir, max_events=0)


def test_corrupt_lifetime_contract_is_rejected_independently():
    from dataclasses import replace
    from fpgai_research.model_ir.composition import ComposedModelIR
    graph = chain(rows=3, widths=(3, 5, 4, 2))
    original = compile_row_chain(heterogeneous(graph)).ir
    edges = list(original.composed.boundaries)
    edges[0] = replace(edges[0], physical_elements=edges[0].physical_elements-1)
    malformed = replace(
        original,
        composed=ComposedModelIR(
            lowered=original.composed.lowered,
            boundaries=tuple(edges)),
    )
    with pytest.raises(ValueError, match="Invalid physical scratch"):
        verify_row_chain_liveness(malformed)
