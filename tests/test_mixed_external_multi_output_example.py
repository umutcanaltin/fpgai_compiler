from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")


def _load_script():
    path = Path("scripts/make_mixed_external_multi_output_example.py")
    spec = importlib.util.spec_from_file_location("make_multi_output", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_maintained_multi_output_example_has_two_external_outputs(tmp_path):
    module = _load_script()
    path = module.build_model(tmp_path / "model.onnx")
    model = onnx.load(path)
    nodes = list(model.graph.node)
    assert nodes[0].op_type == "SplitScale"
    assert list(nodes[0].output) == ["identity", "scaled"]
    assert nodes[1].op_type == "Add"
    assert nodes[2].op_type == "Relu"
