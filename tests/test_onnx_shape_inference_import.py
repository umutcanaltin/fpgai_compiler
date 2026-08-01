from pathlib import Path

import pytest


def test_importer_populates_standard_intermediate_shape_before_custom_operator(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from fpgai.frontend.onnx import import_onnx

    model_path = tmp_path / "standard_then_custom.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["relu_out"], name="relu_0"),
            helper.make_node(
                "UnknownCustom",
                ["relu_out"],
                ["output"],
                name="custom_0",
                domain="community.test",
            ),
        ],
        "standard_then_custom",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("community.test", 1),
        ],
    )
    onnx.save(model, model_path)

    imported = import_onnx(str(model_path), infer_shapes=True)

    relu_spec = imported.get_tensor("relu_out")
    assert relu_spec is not None
    assert tuple(relu_spec.shape) == (1, 4)
