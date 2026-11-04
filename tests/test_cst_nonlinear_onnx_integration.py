"""Independent ONNX Runtime → FPGAI CST HLS host E1 integration.

These are deterministic **synthetic** ONNX models, not original trained
MNIST assets. They exercise the actual strict ONNX frontend, retained
Gemm orientation and bias, nonlinear operator preservation and compiled
C++ code across row-block schedules. Never substitute this for fixed-point
CSim, native HLS, or actual trained-network performance.
"""
import shutil
import numpy as np
import pytest


@pytest.mark.parametrize("rows,transposed", [(1, True), (3, True), (3, False)])
def test_real_onnx_frontend_to_nonlinear_cst_host(rows, transposed, tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if shutil.which("g++") is None:
        pytest.skip("Actual host C++ compiler missing")
    from onnx import helper, numpy_helper, TensorProto

    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.compiler.row_chain import compile_row_chain
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.model_ir.contracts import ArchitectureRequest
    from fpgai_research.validation.host import validate_host
    from fpgai_research.validation.reference import evaluate

    first = np.array([
        [.20, -.12, .04, -.19, .25, .31],
        [-.07, .13, -.27, .35, .02, -.11],
        [.21, -.16, .06, -.04, .17, -.33],
        [-.18, .28, .09, -.23, -.14, .05]], dtype=np.float32)
    second = np.array([
        [.13, -.23, .32, .10],
        [-.15, .29, -.09, .26]], dtype=np.float32)
    if not transposed:
        first = first.T.copy()
        second = second.T.copy()
    nodes = [
        helper.make_node("Flatten", ["image"], ["flat"],
                         name="flatten", axis=1),
        helper.make_node("Gemm", ["flat", "w0", "b0"], ["linear0"],
                         name="gemm0", transB=int(transposed)),
        helper.make_node("Relu", ["linear0"], ["act"],
                         name="relu"),
        helper.make_node("Gemm", ["act", "w1", "b1"], ["output"],
                         name="gemm1", transB=int(transposed)),
    ]
    graph = helper.make_graph(
        nodes, "synthetic-cst-mlp",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT,
                                       [rows, 1, 2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [rows, 2])],
        initializer=[
            numpy_helper.from_array(first, "w0"),
            numpy_helper.from_array(
                np.array([.07, -.15, .13, -.04], dtype=np.float32), "b0"),
            numpy_helper.from_array(second, "w1"),
            numpy_helper.from_array(
                np.array([.11, -.08], dtype=np.float32), "b1"),
        ],
    )
    proto = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)], ir_version=8,
    )
    filename = tmp_path / "synthetic_mlp.onnx"
    onnx.save(proto, str(filename))
    onnx.checker.check_model(str(filename))
    imported = load_onnx(str(filename))
    assert [op.kind for op in imported.operators] == [
        "Flatten", "Gemm", "Relu", "Gemm",
    ]
    inp = np.linspace(-.9, .8, rows*6, dtype=np.float32).reshape(
        rows, 1, 2, 3)
    expected = evaluate(imported, {"image": inp})["output"]
    session = ort.InferenceSession(
        str(filename), providers=["CPUExecutionProvider"])
    observed = session.run(None, {"image":inp})[0]
    np.testing.assert_allclose(expected, observed, atol=1e-5, rtol=1e-4)
    arch = resolve(imported, ArchitectureRequest())
    for block in sorted({1, min(2,rows), rows}):
        compiled = compile_row_chain(arch, row_block=block)
        numerical = validate_host(imported, compiled.emission, inp,
                                  atol=1e-5, rtol=1e-4)
        assert numerical["accepted"], numerical
        assert compiled.evidence["schedule_witness"]["status"] == "VERIFIED"
        assert compiled.emission.manifest["row_block"] == block
