from __future__ import annotations

import pytest
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from fpgai.frontend.onnx import import_onnx


def test_static_onnx_split_lowers_to_layerwise_slices(tmp_path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 2])
    node = helper.make_node("Split", ["x"], ["a", "b"], name="split", axis=1, split=[2, 2])
    graph = helper.make_graph([node], "split_graph", [x], [a, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "split.onnx"; onnx.save(model, path)

    ir = import_onnx(str(path), canonicalize=True, infer_shapes=True)
    assert [op.op_type for op in ir.ops] == ["Slice", "Slice"]
    assert ir.ops[0].attrs["canonicalized_from"] == "Split"
    assert ir.ops[0].attrs["starts"] == [0]
    assert ir.ops[1].attrs["starts"] == [2]
    assert ir.get_tensor("a").shape == (1, 2, 2)
    assert ir.get_tensor("b").shape == (1, 2, 2)


def test_onnx_constant_node_materializes_as_compile_time_tensor(tmp_path) -> None:
    import numpy as np

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    payload = helper.make_tensor("constant_payload", TensorProto.FLOAT, [1], [2.0])
    constant = helper.make_node("Constant", [], ["two"], name="constant_two", value=payload)
    add = helper.make_node("Add", ["x", "two"], ["y"], name="add_two")
    graph = helper.make_graph([constant, add], "constant_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "constant.onnx"
    onnx.save(model, path)

    ir = import_onnx(str(path), canonicalize=True, infer_shapes=True)

    assert [op.op_type for op in ir.ops] == ["Add"]
    assert "two" in ir.constants
    np.testing.assert_allclose(np.asarray(ir.constants["two"]).reshape(-1), [2.0])
