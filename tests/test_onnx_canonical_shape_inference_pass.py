from pathlib import Path

import pytest


def test_canonical_import_runs_final_ir_shape_pass(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    import fpgai.frontend.onnx.importer as importer

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Relu", ["input"], ["relu_out"]),
                helper.make_node("Identity", ["relu_out"], ["output"]),
            ],
            "canonical_shape_pass",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)

    called = {"value": False}
    real = importer.infer_ir_shapes

    def wrapped(graph):
        called["value"] = True
        return real(graph)

    monkeypatch.setattr(importer, "infer_ir_shapes", wrapped)
    graph = importer.import_onnx(str(model_path), canonicalize=True, infer_shapes=True)

    assert called["value"] is True
    assert graph.get_tensor("relu_out").shape == (1, 4)
