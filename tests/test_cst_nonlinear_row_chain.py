"""Scientific E0/E1 tests for full sequential Flatten/Gemm/Relu/Gemm.

Do not replace nonlinear operations with linear MatMul for a false model
equivalence. Actual generated host C++ and a separate NumPy Functional-IR
reference must agree for B=1,2,M under distinct spatial choices.
"""
from __future__ import annotations

from dataclasses import replace
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.experiments.cst_trained_model_preflight import inspect_model
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.validation.cpp_chain_trace import trace_chain_epochs
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.reference import evaluate


def network(rows=3, *, transposed=True):
    # [N,1,2,3] -> Flatten [N,6] -> biased Gemm [N,4]
    # -> ReLU -> biased Gemm [N,2]. Nonuniform +/- weights and biases
    # ensure removing ReLU or changing transB does not pass accidentally.
    a = np.array([
        [.20, -.12, .04, -.19, .25, .31],
        [-.07, .13, -.27, .35, .02, -.11],
        [.21, -.16, .06, -.04, .17, -.33],
        [-.18, .28, .09, -.23, -.14, .05],
    ], dtype=np.float32)
    b = np.array([
        [.13, -.23, .32, .10],
        [-.15, .29, -.09, .26],
    ], dtype=np.float32)
    first = a if transposed else a.T.copy()
    second = b if transposed else b.T.copy()
    attrs = (("transB", 1),) if transposed else ()
    return FunctionalModel(
        tensors=(
            Tensor("x", (rows, 1, 2, 3)),
            Tensor("flat", (rows, 6)),
            Tensor("w0", first.shape, data=tuple(first.flat)),
            Tensor("b0", (4,), data=(.07, -.15, .13, -.04)),
            Tensor("linear0", (rows, 4)),
            Tensor("activated", (rows, 4)),
            Tensor("w1", second.shape, data=tuple(second.flat)),
            Tensor("b1", (2,), data=(.11, -.08)),
            Tensor("y", (rows, 2)),
        ),
        operators=(
            Operator("flatten", "Flatten", ("x",), ("flat",),
                     attributes=(("axis", 1),)),
            Operator("fc0", "Gemm", ("flat", "w0", "b0"),
                     ("linear0",), attributes=attrs),
            Operator("relu", "Relu", ("linear0",), ("activated",)),
            Operator("fc1", "Gemm", ("activated", "w1", "b1"),
                     ("y",), attributes=attrs),
        ),
        inputs=("x",), outputs=("y",),
    )


@pytest.mark.parametrize("rows,block,transposed", [
    (1, 1, True), (3, 1, True), (3, 2, True), (3, 3, True),
    (5, 2, False), (5, 5, False),
])
def test_complete_nonlinear_network_source_is_numerically_equivalent(
        rows, block, transposed):
    if shutil.which("g++") is None:
        pytest.skip("g++ host C++ required")
    graph = network(rows, transposed=transposed)
    request = ArchitectureRequest()
    arch = resolve(graph, request, {
        "fc0": replace(request, pes=2, simd=2),
        "fc1": replace(request, pes=2, simd=2),
    })
    compiled = compile_row_chain(arch, row_block=block)
    assert compiled.evidence["schedule_witness"]["status"] == "VERIFIED"
    assert compiled.ir.chain == ("flatten", "fc0", "relu", "fc1")
    assert compiled.emission.manifest["row_block"] == block
    assert [x["kind"] for x in compiled.emission.manifest["operators"]] == [
        "Flatten", "Gemm", "Relu", "Gemm",
    ]
    assert "if (value < data_t(0))" in compiled.emission.cpp
    assert "bias1[ni]" in compiled.emission.cpp
    x = np.linspace(-0.9, .8, rows*6, dtype=np.float32).reshape(rows,1,2,3)
    expected = evaluate(graph, {"x": x})["y"]
    # Verify this graph actually uses a nontrivial nonlinear clipping stage.
    before = evaluate(graph, {"x": x}, return_all=True)["linear0"]
    assert np.any(before < 0.0)
    assert np.any(before > 0.0)
    receipt = validate_host(graph, compiled.emission, x,
                            atol=1e-5, rtol=1e-4)
    assert receipt["accepted"], receipt
    assert receipt["max_absolute_error"] <= 1e-5
    timeline = trace_chain_epochs(compiled.ir, compiled.emission)
    assert timeline["producer_ready_before_consumer_finished"]
    assert timeline["emitted_row_block"] == block
    assert np.isfinite(expected).all()
    admission = inspect_model(graph)
    assert admission["row_lifetime_status"] == "SUPPORTED_SEQUENTIAL_CST_ROW_CHAIN"
    assert admission["non_matmul_operators"] == ["Flatten", "Gemm", "Relu"]


def test_unimplemented_branching_and_convolution_are_not_misrepresented():
    graph = network()
    base = ArchitectureRequest()
    # Identity is supported by the original Functional IR but not by this
    # restricted CST source transformation. It must not be silently removed.
    modified = FunctionalModel(
        tensors=graph.tensors + (Tensor("identity", (3, 4)),),
        operators=graph.operators[:2] + (
            Operator("identity", "Identity", ("linear0",), ("identity",)),
            Operator("relu", "Relu", ("identity",), ("activated",)),
        ) + graph.operators[3:],
        inputs=graph.inputs, outputs=graph.outputs,
    )
    with pytest.raises(ValueError, match="MatMul/Gemm/Relu"):
        compile_row_chain(resolve(modified, base))
    with pytest.raises(ValueError, match="Non-compute operator supports scalar mapping only|spatial mapping"):
        compile_row_chain(resolve(graph, base, {
            "relu": replace(base, pes=2)
        }))


def test_exact_nonlinear_ir_is_detectably_different_from_clipped_output():
    graph = network(rows=3)
    x = np.linspace(-.9, .8, 18, dtype=np.float32).reshape(3,1,2,3)
    original = evaluate(graph, {"x":x})["y"]
    # Deliberately drop the ReLU from the graph and rewire consumers.
    without = FunctionalModel(
        tensors=tuple(t for t in graph.tensors if t.name != "activated"),
        operators=graph.operators[:2] + (
            Operator("fc1", "Gemm", ("linear0", "w1", "b1"), ("y",),
                     attributes=(("transB",1),)),
        ),
        inputs=graph.inputs, outputs=graph.outputs,
    )
    wrong = evaluate(without, {"x":x})["y"]
    assert not np.allclose(original, wrong, atol=1e-6, rtol=1e-6)
