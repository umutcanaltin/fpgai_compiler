from pathlib import Path
import pytest


def test_maintained_yaml_declares_real_validation_flow():
    text=Path("configs/examples/mixed_external_relu_scale_bias_sigmoid.yml").read_text()
    assert "community.scale_bias_operator" in text
    assert "community.scale_bias_hls" in text
    assert "run_vitis_csim" in text
    assert "input_values" in text


def test_model_generator_writes_mixed_onnx(tmp_path):
    onnx=pytest.importorskip("onnx")
    from scripts.make_mixed_external_hls_example import write_model
    path=write_model(tmp_path/"mixed.onnx")
    model=onnx.load(path)
    assert [n.op_type for n in model.graph.node]==["Relu","ScaleBias","Sigmoid"]
    assert model.graph.node[1].domain=="community.fpgai"
