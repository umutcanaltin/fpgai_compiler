from pathlib import Path

import pytest


def test_residual_example_yaml_uses_automatic_liveness_buffer_mode():
    text = Path("configs/examples/mixed_external_residual_add.yml").read_text(encoding="utf-8")
    assert "community.scale_bias_operator" in text
    assert "community.scale_bias_hls" in text
    assert "\nhls:" not in text
    assert "Add" in text


def test_residual_model_generator_writes_branch_add_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    from scripts.make_mixed_external_residual_hls_example import write_model

    path = write_model(tmp_path / "residual.onnx")
    model = onnx.load(path)
    assert [node.op_type for node in model.graph.node] == ["Relu", "ScaleBias", "Add", "Sigmoid"]
    assert list(model.graph.node[0].input) == ["input"]
    assert list(model.graph.node[1].input) == ["input"]
    assert list(model.graph.node[2].input) == ["relu_out", "scaled"]
