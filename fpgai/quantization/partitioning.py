from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fpgai.quantization.hardware import derive_requantization_contract, quantization_parameters_from_tensor


@dataclass(frozen=True)
class TerminalQuantizedOperatorPartition:
    node_name: str
    op_type: str
    input_tensor: str
    output_tensor: str
    input_quantization: dict
    output_quantization: dict
    backend: str

    def to_dict(self) -> dict:
        return {
            "schema": "fpgai.quantized-operator-partition/v1",
            "node": self.node_name,
            "op_type": self.op_type,
            "input_tensor": self.input_tensor,
            "output_tensor": self.output_tensor,
            "input_quantization": self.input_quantization,
            "output_quantization": self.output_quantization,
            "backend": self.backend,
        }


def partition_terminal_relu(graph: Any, *, backend: str = "vhdl") -> TerminalQuantizedOperatorPartition:
    """Remove a terminal quantized ReLU from the HLS graph for external implementation.

    The graph must have exactly one output produced by a one-input/one-output ReLU.
    Quantization metadata is preserved on both tensors so the external backend can
    implement the same integer semantics.
    """
    outputs = list(getattr(graph, "outputs", ()) or ())
    if len(outputs) != 1:
        raise ValueError("QPART001: terminal ReLU partition requires exactly one graph output")
    output_tensor = str(outputs[0])
    matches = [
        op for op in getattr(graph, "ops", ()) or []
        if str(getattr(op, "op_type", "")) == "Relu"
        and list(getattr(op, "outputs", ()) or []) == [output_tensor]
    ]
    if len(matches) != 1:
        raise ValueError("QPART002: graph output must be produced by exactly one terminal Relu")
    op = matches[0]
    if len(getattr(op, "inputs", ()) or []) != 1:
        raise ValueError("QPART003: terminal Relu must have exactly one input")
    input_tensor = str(op.inputs[0])
    input_spec = graph.get_tensor(input_tensor)
    output_spec = graph.get_tensor(output_tensor)
    if input_spec is None or output_spec is None:
        raise ValueError("QPART004: terminal Relu tensors must exist in graph.tensors")
    input_q = quantization_parameters_from_tensor(input_spec)
    output_q = quantization_parameters_from_tensor(output_spec)
    graph.ops = [candidate for candidate in graph.ops if candidate is not op]
    graph.outputs = [input_tensor]
    return TerminalQuantizedOperatorPartition(
        node_name=str(op.name),
        op_type="Relu",
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        input_quantization=input_q.to_dict(),
        output_quantization=output_q.to_dict(),
        backend=str(backend),
    )


@dataclass(frozen=True)
class ResidualQuantizedOperatorPartition:
    add_node_name: str
    relu_node_name: str
    hls_output_tensor: str
    skip_tensor: str
    sum_tensor: str
    output_tensor: str
    main_quantization: dict
    skip_quantization: dict
    sum_quantization: dict
    output_quantization: dict
    add_backend: str
    relu_backend: str
    add_lowering: dict
    relu_lowering: dict

    def to_dict(self) -> dict:
        return {
            "schema": "fpgai.quantized-residual-operator-partition/v1",
            "partition_type": "residual_add_relu",
            "add": {
                "node": self.add_node_name,
                "op_type": "Add",
                "backend": self.add_backend,
                "left_tensor": self.hls_output_tensor,
                "right_tensor": self.skip_tensor,
                "output_tensor": self.sum_tensor,
                "left_quantization": self.main_quantization,
                "right_quantization": self.skip_quantization,
                "output_quantization": self.sum_quantization,
                "lowering": self.add_lowering,
            },
            "relu": {
                "node": self.relu_node_name,
                "op_type": "Relu",
                "backend": self.relu_backend,
                "input_tensor": self.sum_tensor,
                "output_tensor": self.output_tensor,
                "input_quantization": self.sum_quantization,
                "output_quantization": self.output_quantization,
                "lowering": self.relu_lowering,
            },
            "hls_output_tensor": self.hls_output_tensor,
            "skip_tensor": self.skip_tensor,
        }


def partition_residual_add_and_terminal_relu(
    graph: Any,
    *,
    add_backend: str = "vhdl",
    relu_backend: str = "vhdl",
) -> ResidualQuantizedOperatorPartition:
    """Partition a residual Add + terminal ReLU from the HLS graph.

    The maintained profile requires one Add input to be the original graph
    input (skip branch) and one to be produced by the HLS body.
    """
    outputs = list(getattr(graph, "outputs", ()) or ())
    if len(outputs) != 1:
        raise ValueError("QPART010: residual Add partition requires exactly one graph output")
    final_output = str(outputs[0])
    relus = [
        op for op in getattr(graph, "ops", ()) or []
        if str(getattr(op, "op_type", "")) == "Relu"
        and list(getattr(op, "outputs", ()) or []) == [final_output]
    ]
    if len(relus) != 1:
        raise ValueError("QPART011: graph output must be produced by exactly one terminal Relu")
    relu = relus[0]
    if len(getattr(relu, "inputs", ()) or []) != 1:
        raise ValueError("QPART012: terminal Relu must have exactly one input")
    sum_tensor = str(relu.inputs[0])
    adds = [
        op for op in getattr(graph, "ops", ()) or []
        if str(getattr(op, "op_type", "")) == "Add"
        and list(getattr(op, "outputs", ()) or []) == [sum_tensor]
    ]
    if len(adds) != 1:
        raise ValueError("QPART013: terminal Relu input must be produced by exactly one residual Add")
    add = adds[0]
    runtime_inputs = [str(name) for name in (getattr(add, "inputs", ()) or [])]
    if len(runtime_inputs) != 2:
        raise ValueError("QPART014: residual Add must have exactly two runtime tensor inputs")
    graph_inputs = {str(name) for name in (getattr(graph, "inputs", ()) or [])}
    skip_candidates = [name for name in runtime_inputs if name in graph_inputs]
    main_candidates = [name for name in runtime_inputs if name not in graph_inputs]
    if len(skip_candidates) != 1 or len(main_candidates) != 1:
        raise ValueError("QPART015: maintained residual partition requires one graph-input skip and one HLS-produced Add input")
    skip_tensor = skip_candidates[0]
    hls_output_tensor = main_candidates[0]
    add_lowering = (getattr(add, "attrs", {}) or {}).get("quantized_add")
    if not isinstance(add_lowering, dict):
        raise ValueError("QPART016: residual Add is missing quantized lowering metadata")
    relu_lowering = (getattr(relu, "attrs", {}) or {}).get("quantized_relu")
    if not isinstance(relu_lowering, dict):
        relu_input_q = quantization_parameters_from_tensor(graph.get_tensor(sum_tensor))
        relu_output_q = quantization_parameters_from_tensor(graph.get_tensor(final_output))
        contract = derive_requantization_contract(relu_input_q, relu_output_q)
        rounding_codes = {"nearest": 0, "floor": 1, "ceil": 2}
        saturation_codes = {"saturate": 0, "wrap": 1}
        relu_lowering = {
            "input_zero": int(relu_input_q.zero_point),
            "multiplier": int(contract.multiplier),
            "shift": int(contract.shift),
            "output_zero": int(relu_output_q.zero_point),
            "qmin": int(relu_output_q.spec.qmin),
            "qmax": int(relu_output_q.spec.qmax),
            "rounding_mode": int(rounding_codes[relu_output_q.spec.rounding]),
            "saturation_mode": int(saturation_codes[relu_output_q.spec.saturation]),
        }

    def q(name: str) -> dict:
        spec = graph.get_tensor(name)
        if spec is None:
            raise ValueError(f"QPART017: partition tensor {name!r} is missing from graph.tensors")
        return quantization_parameters_from_tensor(spec).to_dict()

    result = ResidualQuantizedOperatorPartition(
        add_node_name=str(add.name),
        relu_node_name=str(relu.name),
        hls_output_tensor=hls_output_tensor,
        skip_tensor=skip_tensor,
        sum_tensor=sum_tensor,
        output_tensor=final_output,
        main_quantization=q(hls_output_tensor),
        skip_quantization=q(skip_tensor),
        sum_quantization=q(sum_tensor),
        output_quantization=q(final_output),
        add_backend=str(add_backend),
        relu_backend=str(relu_backend),
        add_lowering=dict(add_lowering),
        relu_lowering=dict(relu_lowering),
    )
    graph.ops = [candidate for candidate in graph.ops if candidate is not add and candidate is not relu]
    graph.outputs = [hls_output_tensor]
    return result
