from pathlib import Path
import pytest


def test_residual_cnn_config_is_maintained():
    text=Path('configs/examples/mixed_external_residual_cnn.yml').read_text()
    assert 'mixed_external_residual_cnn.onnx' in text
    assert 'community.scale_bias_hls' in text
    assert 'Conv' in text and 'Add' in text


def test_residual_cnn_model_contains_two_convs_skip_and_add(tmp_path):
    pytest.importorskip('onnx')
    import onnx
    from scripts.make_mixed_external_residual_cnn_example import write_model
    path=write_model(tmp_path/'cnn.onnx'); model=onnx.load(path)
    ops=[n.op_type for n in model.graph.node]
    assert ops.count('Conv')==2
    assert 'ScaleBias' in ops and 'Add' in ops
