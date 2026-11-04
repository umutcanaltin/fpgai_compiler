"""Regression for Vitis HLS 2023.2 ap_fixed conditional arithmetic.

FPGAI_HOST_FLOAT verifies only the generated host-float implementation.
The native tool must separately validate actual ap_fixed CSim/E2 and E3.
"""
import re
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.host import validate_host


def _model(kind):
    if kind == "MatMul":
        model = FunctionalModel(
            (Tensor("x", (1, 3)),
             Tensor("w", (3, 2), data=(.25, -.5, 1., .75, -.1, .125)),
             Tensor("y", (1, 2))),
            (Operator("compute", "MatMul", ("x", "w"), ("y",)),),
            ("x",), ("y",))
        samples = np.array([[1., -.5, .25]], dtype=np.float32)
    elif kind == "Gemm":
        model = FunctionalModel(
            (Tensor("x", (1, 3)),
             Tensor("w", (2, 3), data=(.25, 1., -.1, -.5, .75, .125)),
             Tensor("bias", (2,), data=(.1, -.2)),
             Tensor("y", (1, 2))),
            (Operator("compute", "Gemm", ("x", "w", "bias"),
                      ("y",), (("transB", 1),)),),
            ("x",), ("y",))
        samples = np.array([[1., -.5, .25]], dtype=np.float32)
    elif kind == "Conv":
        model = FunctionalModel(
            (Tensor("x", (1, 1, 3, 3)),
             Tensor("w", (2, 1, 2, 2),
                    data=(.5, -.25, .125, 0., -.1, .2, .3, .4)),
             Tensor("y", (1, 2, 2, 2))),
            (Operator("compute", "Conv", ("x", "w"), ("y",)),),
            ("x",), ("y",))
        samples = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 8
    else:
        raise AssertionError(kind)
    model.validate()
    return model, samples


@pytest.mark.parametrize("kind", ("MatMul", "Gemm", "Conv"))
@pytest.mark.parametrize("precision_bits", (8, 12, 16))
def test_generated_products_do_not_mix_wide_ap_fixed_expression_with_zero(kind, precision_bits):
    model, _ = _model(kind)
    numerical = ArchitectureRequest(precision_bits=precision_bits, integer_bits=4)
    vector = ArchitectureRequest(precision_bits=precision_bits, integer_bits=4, simd=2)
    source = emit_hls(lower(resolve(model, numerical, {"compute": vector}))).cpp
    assert "partial[lane] = data_t(0);" in source
    guarded = [line for line in source.splitlines()
               if "partial[lane] =" in line and "*" in line]
    assert len(guarded) == 1
    assert "if (" in guarded[0] and "?" not in guarded[0]
    assert " ? " not in guarded[0]
    assert re.search(r"partial\[lane\] = tensor_\d+\[", guarded[0])
    # A SIMD tail is guarded by if, avoiding invalid memory accesses.
    assert "partial[lane] = (kk<" not in source
    assert "partial[lane]=(ic<" not in source


@pytest.mark.skipif(shutil.which("g++") is None, reason="host GCC unavailable")
@pytest.mark.parametrize("kind", ("MatMul", "Gemm", "Conv"))
def test_guarded_products_preserve_gcc_host_float_numerics(kind):
    model, samples = _model(kind)
    arch = resolve(model, ArchitectureRequest(),
                   {"compute": ArchitectureRequest(simd=2)})
    emitted = emit_hls(lower(arch))
    receipt = validate_host(model, emitted, samples)
    assert receipt["accepted"] is True
    assert receipt["strongest_evidence"] == "E1-plus-host-float-check"


def test_leaky_relu_negative_product_uses_explicit_branch():
    model = FunctionalModel(
        (Tensor("x", (1, 4)), Tensor("y", (1, 4))),
        (Operator("act", "LeakyRelu", ("x",), ("y",), (("alpha", .125),)),),
        ("x",), ("y",))
    source = emit_hls(lower(resolve(model, ArchitectureRequest()))).cpp
    lines = [line for line in source.splitlines() if "else " in line and "*" in line]
    assert len(lines) == 1
    assert "?" not in lines[0]
    assert "data_t(0.125)" in lines[0]


@pytest.mark.skipif(shutil.which("g++") is None, reason="host GCC unavailable")
def test_leaky_relu_explicit_branch_preserves_host_numerics():
    model = FunctionalModel(
        (Tensor("x", (1, 4)), Tensor("y", (1, 4))),
        (Operator("act", "LeakyRelu", ("x",), ("y",), (("alpha", .125),)),),
        ("x",), ("y",))
    emitted = emit_hls(lower(resolve(model, ArchitectureRequest())))
    receipt = validate_host(model, emitted,
                            np.array([[-1., -.25, 0., 2.]], dtype=np.float32))
    assert receipt["accepted"] is True
