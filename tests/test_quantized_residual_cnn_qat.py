from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from fpgai.benchmark.training_qat_reference import execute_frozen_qat_reference, run_qat_training_dataset_reference
from fpgai.ir.graph import Graph
from fpgai.quantization import apply_model_qat_to_hls_graph, execute_quantized_hls_reference
from fpgai.quantization.hardware import quantization_parameters_from_tensor
from fpgai.quantization.ptq import quantize
from scripts.run_quantized_residual_cnn_qat import _numeric_comparison


def _residual_graph() -> Graph:
    graph = Graph("qat_residual_cnn")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "conv0", "relu0", "conv1", "sum", "output"):
        graph.add_tensor(name, (1, 1, 4, 4), "float32")
    graph.add_tensor("w0", (1, 1, 3, 3), "float32")
    graph.add_tensor("b0", (1,), "float32")
    graph.add_tensor("w1", (1, 1, 3, 3), "float32")
    graph.add_tensor("b1", (1,), "float32")
    w0 = np.zeros((1, 1, 3, 3), dtype=np.float32); w0[0, 0, 1, 1] = 1.0
    w1 = np.zeros((1, 1, 3, 3), dtype=np.float32); w1[0, 0, 1, 1] = 0.5
    graph.constants.update({"w0": w0, "b0": np.zeros(1, np.float32), "w1": w1, "b1": np.zeros(1, np.float32)})
    graph.add_op("Conv", ["input", "w0", "b0"], ["conv0"], name="conv0", attrs={"pads": (1, 1, 1, 1), "strides": (1, 1)})
    graph.add_op("Relu", ["conv0"], ["relu0"], name="relu0")
    graph.add_op("Conv", ["relu0", "w1", "b1"], ["conv1"], name="conv1", attrs={"pads": (1, 1, 1, 1), "strides": (1, 1)})
    graph.add_op("Add", ["conv1", "input"], ["sum"], name="add0")
    graph.add_op("Relu", ["sum"], ["output"], name="relu1")
    return graph


def _cfg() -> dict:
    return yaml.safe_load(Path("configs/examples/quantized_residual_cnn_qat.yml").read_text(encoding="utf-8"))


def test_quantized_residual_cnn_qat_config_is_explicit_and_public():
    raw = _cfg()
    assert raw["numerics"]["quantization"]["mode"] == "qat"
    assert raw["numerics"]["quantization"]["qat"]["fake_quant"] is True
    assert raw["numerics"]["quantization"]["qat"]["straight_through_estimator"] is True
    assert raw["targets"]["mixed_backend"] == {"residual_add_backend": "vhdl", "final_relu_backend": "vhdl"}


def test_quantized_residual_cnn_qat_runner_supports_direct_help():
    completed = subprocess.run([sys.executable, "scripts/run_quantized_residual_cnn_qat.py", "--help"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "--skip-hls" in completed.stdout


def test_qat_residual_export_reuses_integer_lowering_and_matches_frozen_output_domain(tmp_path):
    graph = _residual_graph()
    raw = _cfg()
    inputs = np.asarray([
        np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(1, 1, 4, 4),
        np.linspace(1.0, -1.0, 16, dtype=np.float32).reshape(1, 1, 4, 4),
    ])
    targets = np.maximum(inputs, 0.0).astype(np.float32)
    training = run_qat_training_dataset_reference(graph=graph, raw_cfg=raw, out_dir=tmp_path, inputs=inputs, targets=targets)
    hardware_graph = training.trained_graph
    import copy
    hardware_graph = copy.deepcopy(hardware_graph)
    lowering = apply_model_qat_to_hls_graph(hardware_graph, training.qat_result)
    assert lowering.quantized_add_nodes == ("add0",)
    assert lowering.quantized_relu_nodes == ("relu0", "relu1")

    validation_input = np.linspace(-1.5, 1.5, 16, dtype=np.float32).reshape(1, 1, 4, 4)
    fake_output = execute_frozen_qat_reference(
        graph=training.trained_graph, qat_result=training.qat_result, raw_cfg=raw,
        out_dir=tmp_path / "frozen", x_input=validation_input,
    )
    input_qparams = quantization_parameters_from_tensor(hardware_graph.get_tensor("input"))
    output_qparams = quantization_parameters_from_tensor(hardware_graph.get_tensor("output"))
    integer_output = execute_quantized_hls_reference(hardware_graph, quantize(validation_input, input_qparams))
    comparison = _numeric_comparison(fake_output=fake_output, integer_output=integer_output, output_qparams=output_qparams)
    assert comparison["within_one_lsb"] is True
    assert comparison["max_integer_abs_error_lsb"] <= 1
    assert comparison["max_abs_error"] >= 0.0


def test_qat_hardware_result_emits_common_hls_contract_source():
    source = Path("scripts/run_quantized_residual_cnn_qat.py").read_text(encoding="utf-8")
    assert '"quantized_hls_tool_result.json"' in source
    assert '"schema": "fpgai.quantized-residual-cnn-hls-result/v1"' in source
    assert '"quantization_source": "qat"' in source
    assert '"qat_hardware_result"' in source
