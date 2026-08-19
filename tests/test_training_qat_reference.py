from __future__ import annotations

import json

import numpy as np

from fpgai.benchmark.training_qat_reference import execute_frozen_qat_reference, run_qat_training_dataset_reference
from fpgai.ir.graph import Graph


def _graph() -> Graph:
    graph = Graph("qat_train_conv")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 1, 2, 2), "float32")
    graph.add_tensor("w0", (1, 1, 1, 1), "float32")
    graph.add_tensor("b0", (1,), "float32")
    graph.add_tensor("conv", (1, 1, 2, 2), "float32")
    graph.add_tensor("output", (1, 1, 2, 2), "float32")
    graph.constants["w0"] = np.asarray([[[[0.5]]]], dtype=np.float32)
    graph.constants["b0"] = np.asarray([0.0], dtype=np.float32)
    graph.add_op("Conv", ["input", "w0", "b0"], ["conv"], name="conv0", attrs={"strides": (1, 1), "pads": (0, 0, 0, 0)})
    graph.add_op("Relu", ["conv"], ["output"], name="relu0")
    return graph


def _cfg(optimizer: str = "sgd") -> dict:
    return {
        "numerics": {
            "quantization": {
                "mode": "qat",
                "activations": {"bits": 8, "scheme": "symmetric", "granularity": "per_tensor", "signed": True},
                "weights": {"bits": 8, "scheme": "symmetric", "granularity": "per_channel", "signed": True, "axis": 0},
                "calibration": {"method": "min_max"},
                "qat": {"fake_quant": True, "straight_through_estimator": True, "freeze_after_updates": 2},
            }
        },
        "training": {
            "batch": {"size": 1, "epochs": 2, "mode": "accumulate", "shuffle": False},
            "optimizer": {"type": optimizer, "learning_rate": 0.05, "momentum": 0.9, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            "loss": {"type": "mse"},
        },
    }


def test_qat_dataset_training_updates_master_weights_freezes_and_reuses_hls_lowering(tmp_path):
    graph = _graph()
    inputs = np.asarray([
        [[[0.5, 1.0], [0.25, 0.75]]],
        [[[1.0, 0.5], [0.75, 0.25]]],
    ], dtype=np.float32)
    targets = inputs.copy()
    result = run_qat_training_dataset_reference(
        graph=graph, raw_cfg=_cfg(), out_dir=tmp_path, inputs=inputs, targets=targets
    )

    payload = json.loads(result.summary_json.read_text())
    assert payload["schema"] == "fpgai.qat-training-reference/v1"
    assert payload["optimizer_updates"] == 4
    assert payload["observers_frozen"] is True
    assert payload["master_weight_policy"] == "float_master_weights_fake_quant_forward_ste_backward"
    assert payload["common_hls_lowering"]["status"] == "passed"
    assert payload["common_hls_lowering"]["quantized_conv_nodes"] == ["conv0"]
    assert payload["common_hls_lowering"]["quantized_relu_nodes"] == ["relu0"]

    before = np.fromfile(result.master_weights_before_path, dtype=np.float32)
    after = np.fromfile(result.master_weights_after_path, dtype=np.float32)
    assert before.dtype == np.float32 and after.dtype == np.float32
    assert not np.array_equal(before, after)
    qat_export = json.loads(result.qat_report_path.read_text())
    assert qat_export["schema"] == "fpgai.model-qat/v1"
    assert qat_export["optimizer_updates"] == 4


def test_qat_dataset_training_supports_adam_stateful_updates(tmp_path):
    graph = _graph()
    inputs = np.asarray([[[[0.25, 0.5], [0.75, 1.0]]]], dtype=np.float32)
    targets = np.asarray([[[[0.5, 0.75], [1.0, 1.25]]]], dtype=np.float32)
    result = run_qat_training_dataset_reference(
        graph=graph, raw_cfg=_cfg("adam"), out_dir=tmp_path, inputs=inputs, targets=targets
    )
    payload = json.loads(result.summary_json.read_text())
    assert payload["optimizer"]["type"] == "adam"
    assert result.optimizer_updates == 2
    assert np.isfinite(result.final_dataset_loss)


def test_qat_dataset_training_exposes_trained_graph_and_frozen_fake_quant_reference(tmp_path):
    graph = _graph()
    inputs = np.asarray([[[[0.25, 0.5], [0.75, 1.0]]]], dtype=np.float32)
    targets = inputs.copy()
    cfg = _cfg()
    result = run_qat_training_dataset_reference(
        graph=graph, raw_cfg=cfg, out_dir=tmp_path, inputs=inputs, targets=targets
    )
    assert result.trained_graph is not graph
    output = execute_frozen_qat_reference(
        graph=result.trained_graph,
        qat_result=result.qat_result,
        raw_cfg=cfg,
        out_dir=tmp_path / "frozen_eval",
        x_input=inputs[0],
    )
    assert output.shape == (1, 1, 2, 2)
    assert np.all(np.isfinite(output))
