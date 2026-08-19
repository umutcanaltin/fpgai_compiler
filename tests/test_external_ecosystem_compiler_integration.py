from __future__ import annotations

import json
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from fpgai.config.loader import load_config
from fpgai.engine.compiler import Compiler


def _write_model(path: Path) -> None:
    node = helper.make_node(
        "ScaleBias",
        ["input"],
        ["output"],
        domain="community.fpgai",
        scale=2.0,
        bias=1.0,
    )
    graph = helper.make_graph(
        [node],
        "external_scale_bias",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("community.fpgai", 1),
        ],
    )
    onnx.save(model, path)


def test_one_compile_connects_external_operator_to_hls_package(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "scale_bias.onnx"
    out_dir = tmp_path / "out"
    _write_model(model_path)
    config_path = tmp_path / "compile.yml"
    config_path.write_text(
        f"""
version: 1
project:
  out_dir: {out_dir}
  clean: true
model:
  path: {model_path}
pipeline:
  mode: inference
  outputs: {{top_kernel_name: deeplearn}}
operators: {{supported: [ScaleBias]}}
numerics:
  kind: float
targets:
  platform:
    board: kv260
    part: xck26-sfvc784-2LV-c
    clocks: [{{name: pl_clk0, target_mhz: 200}}]
build:
  stages:
    cpp: true
    hls_project: true
    hls_synthesis: false
    reports: true
    runtime_package: false
ecosystem:
  enabled: true
  project_root: {repo}
  package_directories: [{repo / 'examples/packages'}]
  strict_discovery: true
  operator_packages:
    enable: [community.scale_bias_operator]
  trust:
    community.scale_bias_operator: approved_for_reference
implementations:
  enable: [community.scale_bias_hls]
  operators:
    community.operator.scale_bias:
      preferred: [community.scale_bias_hls]
      allow_fallback: false
""",
        encoding="utf-8",
    )
    result = Compiler.from_yaml(str(config_path)).compile()
    assert result.hls_project_dir == out_dir / "hls"
    assert (out_dir / "hls/src/deeplearn.cpp").is_file()
    assert (out_dir / "package-lock.yml").is_file()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["external_ecosystem"]["operator"]["operator_id"] == "community.operator.scale_bias"
    assert manifest["external_ecosystem"]["selected_implementation"]["package_id"] == "community.scale_bias_hls"


def test_external_ecosystem_uses_compiler_numeric_validation_contract(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "scale_bias.onnx"
    out_dir = tmp_path / "out_numeric"
    _write_model(model_path)
    config_path = tmp_path / "compile_numeric.yml"
    config_path.write_text(
        f"""
version: 1
project:
  out_dir: {out_dir}
  clean: true
model:
  path: {model_path}
pipeline:
  mode: inference
  outputs: {{top_kernel_name: deeplearn}}
operators: {{supported: [ScaleBias]}}
numerics:
  kind: float
targets:
  platform:
    board: kv260
    part: xck26-sfvc784-2LV-c
    clocks: [{{name: pl_clk0, target_mhz: 200}}]
build:
  stages:
    cpp: true
    hls_project: true
    hls_synthesis: false
    reports: true
    runtime_package: false
validation:
  numeric:
    enabled: true
    levels: [model, layer, intermediate]
ecosystem:
  enabled: true
  project_root: {repo}
  package_directories: [{repo / 'examples/packages'}]
  strict_discovery: true
  operator_packages:
    enable: [community.scale_bias_operator]
  trust:
    community.scale_bias_operator: approved_for_reference
implementations:
  enable: [community.scale_bias_hls]
  operators:
    community.operator.scale_bias:
      preferred: [community.scale_bias_hls]
      allow_fallback: false
""",
        encoding="utf-8",
    )
    Compiler.from_yaml(str(config_path)).compile()
    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert numeric["status"] == "passed"
    validation = manifest["external_ecosystem"]["validation"]
    assert validation["compiler_numeric_validation"]["status"] == "passed"
    assert validation["host_cpp"]["status"] == "passed"
    assert manifest["status"] == "passed"


def test_external_ecosystem_training_numeric_enforce_validates_forward_and_backward(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "scale_bias_train.onnx"
    out_dir = tmp_path / "out_training_numeric"
    _write_model(model_path)
    config_path = tmp_path / "compile_training_numeric.yml"
    config_path.write_text(
        f"""
version: 1
project:
  out_dir: {out_dir}
  clean: true
model:
  path: {model_path}
pipeline:
  mode: training_on_device
  outputs: {{top_kernel_name: deeplearn}}
operators: {{supported: [ScaleBias]}}
numerics:
  kind: float
training:
  optimizer: {{type: sgd, learning_rate: 0.01}}
  loss: {{type: mse}}
  batch: {{size: 1, epochs: 1}}
memory:
  weight_storage: bram
  activation_storage: bram
  gradient_storage: bram
  optimizer_state_storage: none
weights: {{mode: embedded}}
targets:
  platform:
    board: kv260
    part: xck26-sfvc784-2LV-c
    clocks: [{{name: pl_clk0, target_mhz: 200}}]
build:
  stages:
    cpp: true
    hls_project: true
    hls_synthesis: false
    reports: true
    runtime_package: false
validation:
  numeric:
    enabled: true
    policy: enforce
    levels: [model, layer, intermediate, state]
ecosystem:
  enabled: true
  project_root: {repo}
  package_directories: [{repo / 'examples/packages'}]
  strict_discovery: true
  operator_packages:
    enable: [community.scale_bias_operator]
  trust:
    community.scale_bias_operator: approved_for_reference
implementations:
  enable: [community.scale_bias_hls]
  operators:
    community.operator.scale_bias:
      preferred: [community.scale_bias_hls]
      allow_fallback: false
""",
        encoding="utf-8",
    )
    result = Compiler.from_yaml(str(config_path)).compile()
    assert result.hls_project_dir == out_dir / "hls"
    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert numeric["status"] == "passed"
    assert numeric["ecosystem"]["forward"]["status"] == "passed"
    assert numeric["ecosystem"]["backward_input"]["status"] == "passed"
    assert manifest["status"] == "passed"
