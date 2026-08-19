from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper

from fpgai.frontend.onnx import import_onnx


def test_onnx_attention_primitives_import_and_preserve_general_matmul(tmp_path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])
    wq = numpy_helper.from_array(np.eye(8, dtype=np.float32), name="wq")
    wk = numpy_helper.from_array(np.eye(8, dtype=np.float32), name="wk")
    wv = numpy_helper.from_array(np.eye(8, dtype=np.float32), name="wv")
    gamma = numpy_helper.from_array(np.ones((8,), dtype=np.float32), name="gamma")
    beta = numpy_helper.from_array(np.zeros((8,), dtype=np.float32), name="beta")
    scale = numpy_helper.from_array(np.asarray(0.35355339, dtype=np.float32), name="scale")

    nodes = [
        helper.make_node("MatMul", ["x", "wq"], ["q"], name="q_proj"),
        helper.make_node("MatMul", ["x", "wk"], ["k"], name="k_proj"),
        helper.make_node("MatMul", ["x", "wv"], ["v"], name="v_proj"),
        helper.make_node("Transpose", ["k"], ["kt"], name="kt", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "kt"], ["scores"], name="scores"),
        helper.make_node("Mul", ["scores", "scale"], ["scaled"], name="scale_scores"),
        helper.make_node("Softmax", ["scaled"], ["probs"], name="softmax", axis=-1),
        helper.make_node("MatMul", ["probs", "v"], ["ctx"], name="context"),
        helper.make_node("Add", ["ctx", "x"], ["res"], name="residual"),
        helper.make_node("LayerNormalization", ["res", "gamma", "beta"], ["y"], name="norm", axis=-1),
    ]
    graph = helper.make_graph(nodes, "attention", [x], [y], [wq, wk, wv, gamma, beta, scale])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = tmp_path / "attention.onnx"
    onnx.save(model, path)

    ir = import_onnx(str(path), canonicalize=True, infer_shapes=True)
    types = [op.op_type for op in ir.ops]
    assert types.count("MatMul") == 5
    assert "Transpose" in types
    assert "Mul" in types
    assert "Softmax" in types
    assert "LayerNormalization" in types
    assert ir.get_tensor("scores").shape == (1, 4, 4)
    assert ir.get_tensor("ctx").shape == (1, 4, 8)
    assert ir.get_tensor("y").shape == (1, 4, 8)
