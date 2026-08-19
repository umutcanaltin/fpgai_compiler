import numpy as np

from fpgai.ir.graph import Graph
from fpgai.quantization import QuantizationSpec, calibrate_model_ptq, dequantized_constant


def _graph() -> Graph:
    graph = Graph("ptq_probe")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (2,), "float32")
    graph.add_tensor("weight", (2, 2), "float32")
    graph.add_tensor("hidden", (2,), "float32")
    graph.add_tensor("output", (2,), "float32")
    graph.constants["weight"] = np.asarray([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    return graph


def _trace(graph: Graph, sample: np.ndarray):
    weight = graph.constants["weight"]
    hidden = np.asarray(sample, dtype=np.float32) @ weight.T
    output = np.maximum(hidden, 0.0)
    return {"input": sample, "hidden": hidden, "output": output}


def test_model_ptq_calibrates_activations_and_per_channel_weights() -> None:
    graph = _graph()
    result = calibrate_model_ptq(
        graph,
        [np.asarray([1.0, -1.0], dtype=np.float32), np.asarray([2.0, 0.5], dtype=np.float32)],
        trace_fn=_trace,
        activation_spec=QuantizationSpec(bits=8, granularity="per_tensor"),
        weight_spec=QuantizationSpec(bits=8, granularity="per_channel", axis=0),
        method="min_max",
    )
    assert result.sample_count == 2
    assert {item.tensor for item in result.activations} == {"input", "hidden", "output"}
    assert {item.tensor for item in result.weights} == {"weight"}
    assert graph.tensors["hidden"].quantization is not None
    assert graph.tensors["weight"].quantization["spec"]["granularity"] == "per_channel"
    assert result.quantized_constants["weight"].shape == (2, 2)
    restored = dequantized_constant(result, "weight")
    assert restored.shape == (2, 2)
    assert np.max(np.abs(restored - graph.constants["weight"])) < 0.05


def test_model_ptq_honors_sample_limit() -> None:
    graph = _graph()
    result = calibrate_model_ptq(
        graph,
        [np.asarray([1.0, 1.0], dtype=np.float32), np.asarray([100.0, 100.0], dtype=np.float32)],
        trace_fn=_trace,
        activation_spec=QuantizationSpec(bits=8),
        weight_spec=QuantizationSpec(bits=8, granularity="per_channel", axis=0),
        sample_limit=1,
    )
    input_params = next(item.parameters for item in result.activations if item.tensor == "input")
    assert input_params.observed_max == 1.0


def test_model_ptq_report_is_json_serializable(tmp_path) -> None:
    import json
    from fpgai.quantization import write_model_ptq_report

    graph = _graph()
    result = calibrate_model_ptq(
        graph,
        [np.asarray([1.0, -1.0], dtype=np.float32)],
        trace_fn=_trace,
        activation_spec=QuantizationSpec(bits=8),
        weight_spec=QuantizationSpec(bits=8, granularity="per_channel", axis=0),
    )
    path = write_model_ptq_report(result, tmp_path / "model_ptq.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fpgai.model-ptq/v1"
    assert payload["sample_count"] == 1
    assert payload["quantized_constants"]["weight"]["shape"] == [2, 2]


def test_model_ptq_reports_conv_biases_separately_from_weights() -> None:
    graph = Graph("ptq_conv_roles")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 1, 1, 1), "float32")
    graph.add_tensor("weight", (1, 1, 1, 1), "float32")
    graph.add_tensor("bias", (1,), "float32")
    graph.add_tensor("output", (1, 1, 1, 1), "float32")
    graph.constants["weight"] = np.asarray([[[[0.5]]]], dtype=np.float32)
    graph.constants["bias"] = np.asarray([0.25], dtype=np.float32)
    graph.add_op("Conv", ["input", "weight", "bias"], ["output"], name="conv0")

    result = calibrate_model_ptq(
        graph,
        [np.asarray([[[[1.0]]]], dtype=np.float32)],
        trace_fn=lambda _g, sample: {"input": sample, "output": sample * 0.5 + 0.25},
        activation_spec=QuantizationSpec(bits=8),
        weight_spec=QuantizationSpec(bits=8, granularity="per_channel", axis=0),
    )

    assert {item.tensor for item in result.weights} == {"weight"}
    assert {item.tensor for item in result.biases} == {"bias"}
    assert result.biases[0].role == "bias"
    payload = result.to_dict()
    assert [item["tensor"] for item in payload["biases"]] == ["bias"]
    assert payload["accumulators"]["policy"] == "operator_lowering"
