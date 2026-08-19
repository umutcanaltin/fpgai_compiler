from __future__ import annotations

import numpy as np

from fpgai.ir.graph import Graph
from fpgai.quantization import (
    ModelQATSession,
    QATSchedule,
    QuantizationSpec,
    apply_model_qat_to_hls_graph,
)


def _graph() -> Graph:
    graph = Graph("qat_conv_relu")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 1, 2, 2), "float32")
    graph.add_tensor("w0", (1, 1, 1, 1), "float32")
    graph.add_tensor("b0", (1,), "float32")
    graph.add_tensor("conv", (1, 1, 2, 2), "float32")
    graph.add_tensor("output", (1, 1, 2, 2), "float32")
    graph.constants["w0"] = np.asarray([[[[0.73]]]], dtype=np.float32)
    graph.constants["b0"] = np.asarray([0.11], dtype=np.float32)
    graph.add_op("Conv", ["input", "w0", "b0"], ["conv"], name="conv0", attrs={"strides": (1, 1), "pads": (0, 0, 0, 0)})
    graph.add_op("Relu", ["conv"], ["output"], name="relu0")
    return graph


def _session(*, freeze_after_updates: int | None = 2) -> ModelQATSession:
    return ModelQATSession(
        activation_spec=QuantizationSpec(bits=8, scheme="symmetric", granularity="per_tensor", signed=True),
        weight_spec=QuantizationSpec(bits=8, scheme="symmetric", granularity="per_channel", signed=True, axis=0),
        schedule=QATSchedule(fake_quant=True, straight_through_estimator=True, freeze_after_updates=freeze_after_updates),
    )


def test_qat_keeps_master_weights_float_and_freezes_on_optimizer_update():
    graph = _graph()
    session = _session()
    master_before = graph.constants["w0"].copy()

    fake_constants = session.fake_quant_graph_constants(graph)
    session.observe_activation_trace({
        "input": np.asarray([[[[-1.0, -0.2], [0.4, 1.0]]]], dtype=np.float32),
        "conv": np.asarray([[[[-0.5, -0.1], [0.3, 0.8]]]], dtype=np.float32),
        "output": np.asarray([[[[0.0, 0.0], [0.3, 0.8]]]], dtype=np.float32),
    })
    session.complete_optimizer_update()
    assert not session.observers_frozen

    # A second update reaches the configured freeze boundary.
    session.fake_quant_graph_constants(graph)
    session.complete_optimizer_update()
    assert session.observers_frozen
    assert np.array_equal(graph.constants["w0"], master_before)
    assert fake_constants["w0"].dtype == np.float32


def test_qat_ste_routes_gradient_without_modifying_shape():
    session = _session(freeze_after_updates=None)
    gradient = np.asarray([-2.0, 0.5, 4.0], dtype=np.float32)
    routed = session.backward_gradient(gradient)
    assert np.array_equal(routed, gradient)
    assert routed is not gradient


def test_qat_export_attaches_quantized_ir_and_reuses_hls_lowering():
    graph = _graph()
    session = _session(freeze_after_updates=1)
    session.fake_quant_graph_constants(graph)
    session.observe_activation_trace({
        "input": np.asarray([[[[-1.0, -0.5], [0.5, 1.0]]]], dtype=np.float32),
        "conv": np.asarray([[[[-0.6, -0.2], [0.4, 0.9]]]], dtype=np.float32),
        "output": np.asarray([[[[0.0, 0.0], [0.4, 0.9]]]], dtype=np.float32),
    })
    session.complete_optimizer_update()

    result = session.export_to_graph(graph)
    assert result.to_dict()["schema"] == "fpgai.model-qat/v1"
    assert graph.get_tensor("input").quantization["spec"]["bits"] == 8
    assert graph.get_tensor("w0").quantization["spec"]["granularity"] == "per_channel"
    assert np.asarray(graph.constants["w0"]).dtype == np.float32

    lowering = apply_model_qat_to_hls_graph(graph, result)
    assert lowering.quantized_conv_nodes == ("conv0",)
    assert lowering.quantized_relu_nodes == ("relu0",)
    assert np.asarray(graph.constants["w0"]).dtype.kind in {"i", "u"}
    assert "quantized_conv" in graph.ops[0].attrs
    assert "quantized_relu" in graph.ops[1].attrs


def test_qat_frozen_observer_parameters_do_not_drift():
    session = _session(freeze_after_updates=1)
    first = session.fake_quant_activation("x", np.asarray([-1.0, 1.0], dtype=np.float32))
    session.complete_optimizer_update()
    params_before = session._states["x"].parameters.to_dict()
    second = session.fake_quant_activation("x", np.asarray([-100.0, 100.0], dtype=np.float32))
    params_after = session._states["x"].parameters.to_dict()
    assert params_after == params_before
    assert np.max(np.abs(second)) <= np.max(np.abs(first)) + 1e-6


def test_qat_schedule_mapping_rejects_string_booleans():
    import pytest
    with pytest.raises(ValueError, match="fake_quant must be a boolean"):
        QATSchedule.from_mapping({"fake_quant": "false"})


def test_qat_report_is_machine_readable(tmp_path):
    import json
    from fpgai.quantization import write_model_qat_report

    graph = _graph()
    session = _session(freeze_after_updates=1)
    session.fake_quant_graph_constants(graph)
    session.observe_activation_trace({
        "input": np.asarray([[[[-1.0, -0.5], [0.5, 1.0]]]], dtype=np.float32),
        "conv": np.asarray([[[[-0.6, -0.2], [0.4, 0.9]]]], dtype=np.float32),
        "output": np.asarray([[[[0.0, 0.0], [0.4, 0.9]]]], dtype=np.float32),
    })
    session.complete_optimizer_update()
    result = session.export_to_graph(graph)
    path = write_model_qat_report(result, tmp_path / "qat.json")
    payload = json.loads(path.read_text())
    assert payload["schema"] == "fpgai.model-qat/v1"
    assert payload["schedule"]["freeze_after_updates"] == 1
    assert payload["optimizer_updates"] == 1


def test_qat_session_builds_from_existing_quantization_yaml_contract():
    from fpgai.quantization import model_qat_session_from_config

    session = model_qat_session_from_config({
        "numerics": {"quantization": {
            "mode": "qat",
            "activations": {"bits": 8, "scheme": "symmetric", "granularity": "per_tensor", "signed": True},
            "weights": {"bits": 8, "scheme": "symmetric", "granularity": "per_channel", "signed": True, "axis": 0},
            "calibration": {"method": "min_max"},
            "qat": {"fake_quant": True, "straight_through_estimator": True, "freeze_after_updates": 3},
        }}
    })
    assert session.activation_spec.bits == 8
    assert session.weight_spec.granularity == "per_channel"
    assert session.schedule.freeze_after_updates == 3
