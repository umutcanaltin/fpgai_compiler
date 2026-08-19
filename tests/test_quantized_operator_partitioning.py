from fpgai.ir.graph import Graph
from fpgai.quantization import partition_terminal_relu, partition_residual_add_and_terminal_relu


def _q(scale: float) -> dict:
    return {
        "spec": {
            "bits": 8,
            "scheme": "symmetric",
            "granularity": "per_tensor",
            "signed": True,
            "axis": None,
            "rounding": "nearest",
            "saturation": "saturate",
        },
        "scale": scale,
        "zero_point": 0,
        "observed_min": -1.0,
        "observed_max": 1.0,
    }


def test_partition_terminal_relu_preserves_quantized_tensor_contracts() -> None:
    graph = Graph("g")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_tensor("x", (1, 4), "int8", quantization=_q(0.25))
    graph.add_tensor("sum", (1, 4), "int8", quantization=_q(0.5))
    graph.add_tensor("y", (1, 4), "int8", quantization=_q(0.5))
    graph.add_op("Add", ["x", "x"], ["sum"], name="add0")
    graph.add_op("Relu", ["sum"], ["y"], name="relu1")

    partition = partition_terminal_relu(graph, backend="vhdl")

    assert graph.outputs == ["sum"]
    assert [op.name for op in graph.ops] == ["add0"]
    assert partition.node_name == "relu1"
    assert partition.input_tensor == "sum"
    assert partition.output_tensor == "y"
    assert partition.input_quantization["scale"] == 0.5
    assert partition.output_quantization["scale"] == 0.5
    assert partition.backend == "vhdl"


def test_partition_residual_add_and_relu_exposes_skip_and_hls_body() -> None:
    graph = Graph("residual")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    for name, scale in (("x", 0.25), ("main", 0.125), ("sum", 0.5), ("y", 0.5)):
        graph.add_tensor(name, (1, 4), "int8", quantization=_q(scale))
    graph.add_op("Identity", ["x"], ["main"], name="body")
    add = graph.add_op("Add", ["main", "x"], ["sum"], name="add0")
    add.attrs["quantized_add"] = {
        "left_zero": 0, "left_multiplier": 1, "left_shift": 2,
        "right_zero": 0, "right_multiplier": 1, "right_shift": 1,
        "output_zero": 0, "qmin": -128, "qmax": 127,
        "rounding_mode": 0, "saturation_mode": 0,
    }
    graph.add_op("Relu", ["sum"], ["y"], name="relu1")
    partition = partition_residual_add_and_terminal_relu(graph)
    assert graph.outputs == ["main"]
    assert [op.name for op in graph.ops] == ["body"]
    payload = partition.to_dict()
    assert payload["partition_type"] == "residual_add_relu"
    assert payload["add"]["left_tensor"] == "main"
    assert payload["add"]["right_tensor"] == "x"
    assert payload["add"]["backend"] == "vhdl"
    assert payload["relu"]["backend"] == "vhdl"
