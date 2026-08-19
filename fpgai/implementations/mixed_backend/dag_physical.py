from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping

from fpgai.implementations.vhdl_integration import (
    VHDLTensorPortsReadyValidABI,
    parse_vhdl_tensor_ports_ready_valid_abi,
    validate_vhdl_integration_contract,
)

from .graph_physical import GraphPhysicalIssue, HLSPhysicalBinding, VHDLPhysicalBinding, RequantizationPhysicalBinding
from fpgai.quantization.hardware import (
    derive_requantization_contract,
    quantization_parameters_from_tensor,
)

from .physical import (
    _find_hls_top,
    _ready_valid_hls_ports,
    _multi_ready_valid_hls_ports,
    _safe_package_source,
    _verilog_ports,
)


_PROFILE = "dag_grouped_ready_valid_v1"


@dataclass(frozen=True)
class DAGMixedBackendPhysicalRequest:
    out_dir: str | Path
    graph: Any
    bindings: Mapping[str, HLSPhysicalBinding | VHDLPhysicalBinding | RequantizationPhysicalBinding]
    part: str = "xck26-sfvc784-2LV-c"
    top_name: str = "fpgai_dag_mixed_backend_top"
    clock_period_ns: float = 5.0
    input_value: int = 7
    expected_output: int = 43
    input_values: tuple[int, ...] | None = None
    expected_outputs: tuple[int, ...] | None = None
    run_implementation: bool = False
    physical_profile: str = _PROFILE
    fanout_buffer_depths: Mapping[str, int] | None = None


@dataclass(frozen=True)
class DAGMixedBackendPhysicalResult:
    ok: bool
    project_dir: Path | None
    wrapper: Path | None
    testbench: Path | None
    run_tcl: Path | None
    report_path: Path | None
    issues: tuple[GraphPhysicalIssue, ...] = ()


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"n_{cleaned}"


def _tensor_width(dtype: str) -> int | None:
    return {
        "int8": 8,
        "uint8": 8,
        "int16": 16,
        "uint16": 16,
        "float16": 16,
        "fp16": 16,
        "int32": 32,
        "uint32": 32,
        "float32": 32,
        "fp32": 32,
        "float": 32,
    }.get(str(dtype).lower().strip())


def _runtime_inputs(graph: Any, op: Any) -> list[str]:
    constants = getattr(graph, "constants", {}) or {}
    return [name for name in getattr(op, "inputs", ()) if name not in constants]


def _tensor_physical_width(tensor: Any) -> int | None:
    quantization = getattr(tensor, "quantization", None)
    if isinstance(quantization, Mapping):
        spec = quantization.get("spec")
        if isinstance(spec, Mapping) and spec.get("bits") is not None:
            bits = int(spec["bits"])
            if bits <= 0:
                raise ValueError("MIXDAG008: quantized tensor bit width must be positive")
            return bits
    return _tensor_width(getattr(tensor, "dtype", ""))


def _graph_topology(
    graph: Any,
) -> tuple[list[Any], dict[str, str | None], dict[str, list[str]], dict[str, int]]:
    ops = list(getattr(graph, "ops", ()))
    graph_inputs = list(getattr(graph, "inputs", ()))
    graph_outputs = list(getattr(graph, "outputs", ()))
    if len(graph_inputs) != 1 or len(graph_outputs) != 1:
        raise ValueError("MIXDAG001: maintained DAG physical profile requires one graph input and one graph output")
    if not ops:
        raise ValueError("MIXDAG002: mixed-backend DAG must contain at least one operation")

    producer: dict[str, str | None] = {graph_inputs[0]: None}
    consumers: dict[str, list[str]] = {graph_inputs[0]: []}
    tensors = getattr(graph, "tensors", {}) or {}

    for op in ops:
        runtime_inputs = _runtime_inputs(graph, op)
        outputs = list(getattr(op, "outputs", ()))
        if not runtime_inputs or not outputs:
            raise ValueError(f"MIXDAG003: node {op.name!r} requires at least one runtime input and output")
        for tensor_name in runtime_inputs:
            if tensor_name not in producer:
                raise ValueError(
                    f"MIXDAG004: node {op.name!r} consumes {tensor_name!r} before a graph input or producer defines it"
                )
            consumers.setdefault(tensor_name, []).append(op.name)
        for tensor_name in outputs:
            if tensor_name in producer:
                raise ValueError(f"MIXDAG005: tensor {tensor_name!r} has multiple producers")
            producer[tensor_name] = op.name
            consumers.setdefault(tensor_name, [])

    if graph_outputs[0] not in producer:
        raise ValueError(f"MIXDAG006: graph output {graph_outputs[0]!r} has no producer")

    implicit_fanout = sorted(name for name, users in consumers.items() if len(users) > 1)
    if implicit_fanout:
        raise ValueError(
            "MIXDAG007: implicit tensor fanout is not supported by the grouped ready/valid profile; "
            "insert an explicit multi-output implementation for: " + ", ".join(implicit_fanout)
        )

    widths: dict[str, int] = {}
    for tensor_name in producer:
        tensor = tensors.get(tensor_name)
        if tensor is None:
            raise ValueError(f"MIXDAG008: tensor {tensor_name!r} is missing from graph.tensors")
        tensor_width = _tensor_physical_width(tensor)
        if tensor_width is None:
            raise ValueError(
                f"MIXDAG008: tensor {tensor_name!r} has unsupported physical dtype {getattr(tensor, 'dtype', None)!r}"
            )
        widths[tensor_name] = int(tensor_width)
    return ops, producer, consumers, widths


def _stage_hls_binding(
    *,
    op: Any,
    binding: HLSPhysicalBinding,
    rtl_dir: Path,
    stage_index: int,
    tensor_widths: Mapping[str, int],
    read_lines: list[str],
    staged_files: list[str],
    source_cache: dict[Path, str],
) -> dict[str, Any]:
    runtime_inputs = list(getattr(op, "inputs", ()))
    outputs = list(getattr(op, "outputs", ()))

    source_dir = Path(binding.rtl_dir).expanduser().resolve()
    top_path = _find_hls_top(source_dir, binding.top)
    if top_path is None:
        raise ValueError(f"MIXDAG011: HLS RTL top {binding.top!r} not found for node {op.name!r}")
    port_info = _verilog_ports(top_path, binding.top)
    multi_ports = _multi_ready_valid_hls_ports(
        port_info,
        input_prefixes=binding.input_streams,
        output_prefixes=binding.output_streams,
        input_count=len(runtime_inputs),
        output_count=len(outputs),
    )
    input_ports = tuple(multi_ports["inputs"])
    output_ports = tuple(multi_ports["outputs"])
    input_packet_words = tuple(int(v) for v in getattr(binding, "input_packet_words", ()) or ())
    output_packet_words = tuple(int(v) for v in getattr(binding, "output_packet_words", ()) or ())
    if input_packet_words and len(input_packet_words) != len(runtime_inputs):
        raise ValueError(
            f"MIXDAG024: HLS node {op.name!r} declares {len(input_packet_words)} input packet counts "
            f"for {len(runtime_inputs)} runtime inputs"
        )
    if output_packet_words and len(output_packet_words) != len(outputs):
        raise ValueError(
            f"MIXDAG024: HLS node {op.name!r} declares {len(output_packet_words)} output packet counts "
            f"for {len(outputs)} outputs"
        )
    if any(v <= 0 for v in (*input_packet_words, *output_packet_words)):
        raise ValueError(f"MIXDAG024: HLS node {op.name!r} packet word counts must be positive")
    for logical, names, groups in (
        ("input", runtime_inputs, input_ports),
        ("output", outputs, output_ports),
    ):
        for port_index, (tensor_name, group) in enumerate(zip(names, groups)):
            actual_width = port_info[group["data"]][1]
            expected_width = int(tensor_widths[tensor_name])
            if actual_width != expected_width:
                raise ValueError(
                    f"MIXDAG012: HLS node {op.name!r} {logical} port {port_index} width {actual_width} "
                    f"does not match tensor {tensor_name!r} width {expected_width}"
                )

    copied: list[str] = []
    for sequence, source in enumerate(sorted(source_dir.rglob("*"))):
        if not source.is_file() or source.suffix.lower() not in {".v", ".sv", ".vhd", ".vhdl"}:
            continue
        source_key = source.resolve()
        if source_key in source_cache:
            copied.append(source_cache[source_key])
            continue
        destination = rtl_dir / f"hls_{stage_index:03d}_{sequence:03d}_{source.name}"
        shutil.copy2(source, destination)
        source_cache[source_key] = destination.name
        copied.append(destination.name)
        staged_files.append(destination.name)
        command = "read_vhdl" if destination.suffix.lower() in {".vhd", ".vhdl"} else "read_verilog"
        read_lines.append(f'{command} "./rtl/{destination.name}"')

    return {
        "node_name": op.name,
        "op_type": op.op_type,
        "backend": "vitis_hls",
        "top": binding.top,
        "inputs": runtime_inputs,
        "outputs": outputs,
        "ports": {
            "clock": multi_ports.get("clock"),
            "reset": multi_ports.get("reset"),
            "inputs": input_ports,
            "outputs": output_ports,
        },
        "port_info": port_info,
        "input_packet_words": input_packet_words,
        "output_packet_words": output_packet_words,
        "source_files": copied,
        "package_id": None,
        "package_version": None,
        "handshake_policy": "axis_independent_ports",
    }


def _stage_vhdl_binding(
    *,
    op: Any,
    binding: VHDLPhysicalBinding,
    rtl_dir: Path,
    stage_index: int,
    tensor_widths: Mapping[str, int],
    read_lines: list[str],
    staged_files: list[str],
    source_cache: dict[Path, str],
) -> dict[str, Any]:
    issues = validate_vhdl_integration_contract(binding.contract)
    if issues:
        issue = issues[0]
        raise ValueError(f"MIXDAG013: VHDL node {op.name!r} contract invalid: {issue.code} {issue.message}")
    abi = parse_vhdl_tensor_ports_ready_valid_abi(binding.contract)
    runtime_inputs = _runtime_inputs_placeholder(op)
    outputs = list(getattr(op, "outputs", ()))
    if len(runtime_inputs) != len(abi.inputs) or len(outputs) != len(abi.outputs):
        raise ValueError(
            f"MIXDAG014: VHDL node {op.name!r} graph arity {len(runtime_inputs)}->{len(outputs)} does not match ABI {len(abi.inputs)}->{len(abi.outputs)}"
        )
    for tensor_name, port in zip(runtime_inputs, abi.inputs):
        expected_width = int(tensor_widths[tensor_name])
        if port.data_width != expected_width:
            raise ValueError(
                f"MIXDAG015: VHDL node {op.name!r} input port {port.name!r} width {port.data_width} "
                f"does not match tensor {tensor_name!r} width {expected_width}"
            )
    for tensor_name, port in zip(outputs, abi.outputs):
        expected_width = int(tensor_widths[tensor_name])
        if port.data_width != expected_width:
            raise ValueError(
                f"MIXDAG015: VHDL node {op.name!r} output port {port.name!r} width {port.data_width} "
                f"does not match tensor {tensor_name!r} width {expected_width}"
            )

    package_root = Path(binding.contract.package_root)
    copied: list[str] = []
    for sequence, relative in enumerate(binding.contract.source_order or binding.contract.sources):
        source = _safe_package_source(package_root, relative)
        source_key = source.resolve()
        if source_key in source_cache:
            copied.append(source_cache[source_key])
            continue
        destination = rtl_dir / f"vhdl_{stage_index:03d}_{sequence:03d}_{source.name}"
        shutil.copy2(source, destination)
        source_cache[source_key] = destination.name
        copied.append(destination.name)
        staged_files.append(destination.name)
        read_lines.append(f'read_vhdl "./rtl/{destination.name}"')

    return {
        "node_name": op.name,
        "op_type": op.op_type,
        "backend": "vhdl",
        "top": binding.contract.top,
        "inputs": runtime_inputs,
        "outputs": outputs,
        "abi": abi,
        "source_files": copied,
        "package_id": binding.contract.package_id,
        "package_version": binding.contract.version,
        "handshake_policy": abi.handshake_policy,
    }


def _stage_requantization_binding(
    *,
    graph: Any,
    op: Any,
    binding: RequantizationPhysicalBinding,
    tensor_widths: Mapping[str, int],
) -> dict[str, Any]:
    runtime_inputs = list(getattr(op, "inputs", ()))
    outputs = list(getattr(op, "outputs", ()))
    if len(runtime_inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"MIXDAG020: requantization node {op.name!r} requires exactly one runtime input and one output"
        )
    source_tensor = getattr(graph, "tensors", {})[runtime_inputs[0]]
    destination_tensor = getattr(graph, "tensors", {})[outputs[0]]
    source = quantization_parameters_from_tensor(source_tensor)
    destination = quantization_parameters_from_tensor(destination_tensor)
    contract = derive_requantization_contract(source, destination)
    if int(tensor_widths[runtime_inputs[0]]) != contract.source_bits:
        raise ValueError(
            f"MIXDAG021: requantization source tensor width does not match quantization contract for {op.name!r}"
        )
    if int(tensor_widths[outputs[0]]) != contract.destination_bits:
        raise ValueError(
            f"MIXDAG021: requantization destination tensor width does not match quantization contract for {op.name!r}"
        )
    return {
        "node_name": op.name,
        "op_type": op.op_type,
        "backend": binding.backend,
        "top": None,
        "inputs": runtime_inputs,
        "outputs": outputs,
        "source_files": [],
        "package_id": None,
        "package_version": None,
        "handshake_policy": "direct_ready_valid",
        "requantization": contract,
    }


def _runtime_inputs_placeholder(op: Any) -> list[str]:
    # DAG physical examples are parameter-free.  Constants are filtered during
    # topology analysis; this helper keeps VHDL arity code independent from the
    # concrete Graph class while preserving input order.
    return list(getattr(op, "inputs", ()))


def _tensor_signal(name: str, field: str) -> str:
    return f"tensor_{_sanitize(name)}_{field}"


def _and(expressions: list[str]) -> str:
    if not expressions:
        return "1'b1"
    return " & ".join(f"({expr})" for expr in expressions)


def _signed_verilog_literal(width: int, value: int) -> str:
    """Return a syntactically valid signed decimal Verilog literal."""
    width = int(width)
    value = int(value)
    if width <= 0:
        raise ValueError("Verilog literal width must be positive")
    magnitude = abs(value)
    literal = f"{width}'sd{magnitude}"
    return f"-{literal}" if value < 0 else literal


def _requantization_stage_lines(
    *,
    index: int,
    stage: Mapping[str, Any],
    tensor_widths: Mapping[str, int],
) -> list[str]:
    inputs = list(stage["inputs"])
    outputs = list(stage["outputs"])
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("MIXDAG020: requantization stages require exactly one input and one output")
    source_tensor = inputs[0]
    destination_tensor = outputs[0]
    contract = stage["requantization"]
    source_width = int(tensor_widths[source_tensor])
    destination_width = int(tensor_widths[destination_tensor])
    if source_width != contract.source_bits or destination_width != contract.destination_bits:
        raise ValueError(
            f"MIXDAG021: requantization node {stage['node_name']!r} tensor widths "
            f"{source_width}->{destination_width} do not match contract "
            f"{contract.source_bits}->{contract.destination_bits}"
        )

    src_data = _tensor_signal(source_tensor, "data")
    src_valid = _tensor_signal(source_tensor, "valid")
    src_ready = _tensor_signal(source_tensor, "ready")
    dst_data = _tensor_signal(destination_tensor, "data")
    dst_valid = _tensor_signal(destination_tensor, "valid")
    dst_ready = _tensor_signal(destination_tensor, "ready")

    source_zero = int(contract.source.zero_point)
    destination_zero = int(contract.destination.zero_point)
    qmin = int(contract.destination.spec.qmin)
    qmax = int(contract.destination.spec.qmax)
    prefix = f"node_{index}_requant"
    lines = [
        f"  // compiler-owned requantization bridge: {source_width} -> {destination_width} bits",
        f"  localparam signed [63:0] {prefix}_src_zero = 64'sd{source_zero};",
        f"  localparam signed [63:0] {prefix}_dst_zero = 64'sd{destination_zero};",
        f"  localparam signed [63:0] {prefix}_multiplier = 64'sd{int(contract.multiplier)};",
        f"  localparam integer {prefix}_shift = {int(contract.shift)};",
        f"  wire signed [63:0] {prefix}_centered = $signed({src_data}) - {prefix}_src_zero;",
        f"  wire signed [127:0] {prefix}_product = {prefix}_centered * {prefix}_multiplier;",
    ]
    rounding = contract.destination.spec.rounding
    if contract.shift == 0:
        lines.append(f"  wire signed [127:0] {prefix}_scaled = {prefix}_product;")
    elif rounding == "nearest":
        half = 1 << (contract.shift - 1)
        lines.extend(
            [
                f"  wire signed [127:0] {prefix}_round_bias = "
                f"({prefix}_product >= 0) ? 128'sd{half} : -128'sd{half};",
                f"  wire signed [127:0] {prefix}_scaled = "
                f"({prefix}_product + {prefix}_round_bias) >>> {prefix}_shift;",
            ]
        )
    elif rounding == "floor":
        lines.append(
            f"  wire signed [127:0] {prefix}_scaled = {prefix}_product >>> {prefix}_shift;"
        )
    elif rounding == "ceil":
        mask = (1 << contract.shift) - 1
        lines.extend(
            [
                f"  wire signed [127:0] {prefix}_floor = {prefix}_product >>> {prefix}_shift;",
                f"  wire {prefix}_has_fraction = (({prefix}_product & 128'sd{mask}) != 0);",
                f"  wire signed [127:0] {prefix}_scaled = "
                f"{prefix}_floor + (({prefix}_product > 0 && {prefix}_has_fraction) ? 1 : 0);",
            ]
        )
    else:
        raise ValueError(f"MIXDAG022: unsupported requantization rounding {rounding!r}")

    lines.append(
        f"  wire signed [127:0] {prefix}_biased = {prefix}_scaled + {prefix}_dst_zero;"
    )
    if contract.destination.spec.saturation == "saturate":
        lines.extend(
            [
                f"  wire signed [127:0] {prefix}_clamped = "
                f"({prefix}_biased < {_signed_verilog_literal(128, qmin)}) ? {_signed_verilog_literal(128, qmin)} : "
                f"(({prefix}_biased > {_signed_verilog_literal(128, qmax)}) ? {_signed_verilog_literal(128, qmax)} : {prefix}_biased);",
                f"  assign {dst_data} = {prefix}_clamped[{destination_width - 1}:0];",
            ]
        )
    elif contract.destination.spec.saturation == "wrap":
        lines.append(f"  assign {dst_data} = {prefix}_biased[{destination_width - 1}:0];")
    else:
        raise ValueError(
            f"MIXDAG023: unsupported requantization saturation {contract.destination.spec.saturation!r}"
        )
    lines.extend(
        [
            f"  assign {dst_valid} = {src_valid};",
            f"  assign {src_ready} = {dst_ready};",
            "",
        ]
    )
    return lines


def _dag_wrapper_source(
    *,
    top_name: str,
    graph: Any,
    stages: list[dict[str, Any]],
    tensor_widths: Mapping[str, int],
    fanout_buffer_depths: Mapping[str, int] | None = None,
) -> str:
    graph_input = list(getattr(graph, "inputs", ()))[0]
    graph_output = list(getattr(graph, "outputs", ()))[0]
    tensors: list[str] = []
    for name in [graph_input, *[output for stage in stages for output in stage["outputs"]]]:
        if name not in tensors:
            tensors.append(name)

    input_width = int(tensor_widths[graph_input])
    output_width = int(tensor_widths[graph_output])
    lines = [
        f"module {top_name}(",
        "    input wire clk,",
        "    input wire rst_n,",
        "    input wire input_valid,",
        "    output wire input_ready,",
        f"    input wire signed [{input_width - 1}:0] input_data,",
        "    output wire output_valid,",
        "    input wire output_ready,",
        f"    output wire signed [{output_width - 1}:0] output_data",
        ");",
        "",
    ]
    for tensor in tensors:
        width = int(tensor_widths[tensor])
        lines.extend(
            [
                f"  wire signed [{width - 1}:0] {_tensor_signal(tensor, 'data')};",
                f"  wire {_tensor_signal(tensor, 'valid')};",
                f"  wire {_tensor_signal(tensor, 'ready')};",
            ]
        )
    lines.extend(
        [
            "",
            f"  assign {_tensor_signal(graph_input, 'data')} = input_data;",
            f"  assign {_tensor_signal(graph_input, 'valid')} = input_valid;",
            f"  assign input_ready = {_tensor_signal(graph_input, 'ready')};",
            f"  assign output_data = {_tensor_signal(graph_output, 'data')};",
            f"  assign output_valid = {_tensor_signal(graph_output, 'valid')};",
            f"  assign {_tensor_signal(graph_output, 'ready')} = output_ready;",
            "",
        ]
    )

    protocol_error_signals: list[str] = []
    fanout_buffer_depths = {str(name): int(depth) for name, depth in (fanout_buffer_depths or {}).items()}
    if any(depth <= 0 for depth in fanout_buffer_depths.values()):
        raise ValueError("MIXDAG025: fanout buffer depths must be positive")

    for index, stage in enumerate(stages):
        instance = f"u_{index}_{_sanitize(stage['node_name'])}"
        inputs = stage["inputs"]
        outputs = stage["outputs"]
        if stage["backend"] == "requantization":
            lines.extend(
                _requantization_stage_lines(
                    index=index,
                    stage=stage,
                    tensor_widths=tensor_widths,
                )
            )
            continue

        if stage["backend"] == "vitis_hls":
            ports = stage["ports"]
            port_info = stage.get("port_info", {})
            connections: list[str] = []
            clock_port = ports.get("clock")
            reset_port = ports.get("reset")
            if clock_port:
                connections.append(f".{clock_port}(clk)")
            if reset_port:
                reset_expression = "rst_n" if str(reset_port).endswith("_n") else "~rst_n"
                connections.append(f".{reset_port}({reset_expression})")

            input_counts = tuple(stage.get("input_packet_words") or ())
            output_counts = tuple(stage.get("output_packet_words") or ())
            for input_index, (tensor, group) in enumerate(zip(inputs, ports["inputs"])):
                connections.extend(
                    [
                        f".{group['data']}({_tensor_signal(tensor, 'data')})",
                        f".{group['valid']}({_tensor_signal(tensor, 'valid')})",
                        f".{group['ready']}({_tensor_signal(tensor, 'ready')})",
                    ]
                )
                for field in ("keep", "strb"):
                    port = group.get(field)
                    if port:
                        width = int(port_info[port][1])
                        signal = f"node_{index}_input_{field}_{input_index}"
                        lines.append(f"  wire [{width - 1}:0] {signal} = {{{width}{{1'b1}}}};")
                        connections.append(f".{port}({signal})")
                last_port = group.get("last")
                if last_port:
                    count = int(input_counts[input_index]) if input_counts else 1
                    signal = f"node_{index}_input_last_{input_index}"
                    if count == 1:
                        lines.append(f"  wire {signal} = 1'b1;")
                    else:
                        counter = f"node_{index}_input_word_index_{input_index}"
                        width = max(1, (count - 1).bit_length())
                        lines.append(f"  reg [{width - 1}:0] {counter};")
                        lines.append(f"  wire {signal} = ({counter} == {width}'d{count - 1});")
                        lines.extend([
                            "  always @(posedge clk) begin",
                            f"    if (!rst_n) {counter} <= {width}'d0;",
                            f"    else if ({_tensor_signal(tensor, 'valid')} && {_tensor_signal(tensor, 'ready')}) begin",
                            f"      if ({signal}) {counter} <= {width}'d0;",
                            f"      else {counter} <= {counter} + 1'b1;",
                            "    end",
                            "  end",
                        ])
                    connections.append(f".{last_port}({signal})")

            for output_index, (tensor, group) in enumerate(zip(outputs, ports["outputs"])):
                connections.extend(
                    [
                        f".{group['data']}({_tensor_signal(tensor, 'data')})",
                        f".{group['valid']}({_tensor_signal(tensor, 'valid')})",
                        f".{group['ready']}({_tensor_signal(tensor, 'ready')})",
                    ]
                )
                count = int(output_counts[output_index]) if output_counts else 1
                sideband_signals: dict[str, str] = {}
                for field in ("keep", "strb", "last"):
                    port = group.get(field)
                    if port:
                        width = int(port_info[port][1])
                        signal = f"node_{index}_output_{field}_{output_index}"
                        lines.append(f"  wire [{width - 1}:0] {signal};")
                        connections.append(f".{port}({signal})")
                        sideband_signals[field] = signal
                if sideband_signals:
                    counter = f"node_{index}_output_word_index_{output_index}"
                    cwidth = max(1, (count - 1).bit_length())
                    lines.append(f"  reg [{cwidth - 1}:0] {counter};")
                    protocol_error_signal = f"node_{index}_output_protocol_error_{output_index}"
                    protocol_error_signals.append(protocol_error_signal)
                    lines.append(f"  reg {protocol_error_signal};")
                    checks = []
                    for field in ("keep", "strb"):
                        signal = sideband_signals.get(field)
                        if signal:
                            width = int(port_info[group[field]][1])
                            checks.append(f"({signal} != {{{width}{{1'b1}}}})")
                    last_signal = sideband_signals.get("last")
                    if last_signal:
                        checks.append(f"({last_signal}[0] != ({counter} == {cwidth}'d{count - 1}))")
                    error_expr = " || ".join(checks) if checks else "1'b0"
                    lines.extend([
                        "  always @(posedge clk) begin",
                        f"    if (!rst_n) begin {counter} <= {cwidth}'d0; {protocol_error_signal} <= 1'b0; end",
                        f"    else if ({_tensor_signal(tensor, 'valid')} && {_tensor_signal(tensor, 'ready')}) begin",
                        f"      if ({error_expr}) {protocol_error_signal} <= 1'b1;",
                        f"      if ({counter} == {cwidth}'d{count - 1}) {counter} <= {cwidth}'d0;",
                        f"      else {counter} <= {counter} + 1'b1;",
                        "    end",
                        "  end",
                    ])
            lines.append(f"  {stage['top']} {instance} (")
            lines.append("    " + ",\n    ".join(connections))
            lines.append("  );")
            lines.append("")
            continue

        abi: VHDLTensorPortsReadyValidABI = stage["abi"]
        input_valid_signal = f"node_{index}_input_valid"
        input_ready_signal = f"node_{index}_input_ready"
        output_valid_signal = f"node_{index}_output_valid"
        output_ready_signal = f"node_{index}_output_ready"
        lines.extend(
            [
                f"  wire {input_valid_signal};",
                f"  wire {input_ready_signal};",
                f"  wire {output_valid_signal};",
                f"  wire {output_ready_signal};",
                f"  assign {input_valid_signal} = {_and([_tensor_signal(name, 'valid') for name in inputs])};",
            ]
        )
        for input_index, tensor in enumerate(inputs):
            peer_valid = [
                _tensor_signal(other, "valid")
                for peer_index, other in enumerate(inputs)
                if peer_index != input_index
            ]
            lines.append(
                f"  assign {_tensor_signal(tensor, 'ready')} = {input_ready_signal} & {_and(peer_valid)};"
            )

        vhdl_output_data_signals: list[str] = []
        if len(outputs) > 1:
            # Grouped multi-output VHDL has one output VALID/READY transaction for all
            # outputs.  Each logical branch needs its own elastic queue so a delayed
            # residual/merge branch cannot backpressure a faster branch before a full
            # packet has entered the graph.  Depth defaults to one, preserving the
            # previous single-beat elastic fanout semantics.
            can_accept_signals: list[str] = []
            push_signal = f"node_{index}_fanout_push"
            for output_index, tensor in enumerate(outputs):
                width = int(tensor_widths[tensor])
                depth = int(fanout_buffer_depths.get(tensor, 1))
                count_width = max(1, depth.bit_length())
                ptr_width = max(1, (depth - 1).bit_length())
                data_signal = f"node_{index}_output_data_{output_index}"
                mem = f"node_{index}_fanout_fifo_data_{output_index}"
                count = f"node_{index}_fanout_fifo_count_{output_index}"
                rd_ptr = f"node_{index}_fanout_fifo_rd_{output_index}"
                wr_ptr = f"node_{index}_fanout_fifo_wr_{output_index}"
                pop = f"node_{index}_fanout_fifo_pop_{output_index}"
                can_accept = f"node_{index}_fanout_fifo_can_accept_{output_index}"
                vhdl_output_data_signals.append(data_signal)
                can_accept_signals.append(can_accept)
                lines.extend([
                    f"  wire signed [{width - 1}:0] {data_signal};",
                    f"  reg signed [{width - 1}:0] {mem} [0:{depth - 1}];",
                    f"  reg [{count_width - 1}:0] {count};",
                    f"  reg [{ptr_width - 1}:0] {rd_ptr};",
                    f"  reg [{ptr_width - 1}:0] {wr_ptr};",
                    f"  wire {pop} = ({count} != 0) & {_tensor_signal(tensor, 'ready')};",
                    f"  wire {can_accept} = ({count} < {count_width}'d{depth}) | {pop};",
                    f"  assign {_tensor_signal(tensor, 'data')} = {mem}[{rd_ptr}];",
                    f"  assign {_tensor_signal(tensor, 'valid')} = ({count} != 0);",
                ])
            lines.append(f"  assign {output_ready_signal} = {_and(can_accept_signals)};")
            lines.append(f"  wire {push_signal} = {output_valid_signal} & {output_ready_signal};")
            for output_index, tensor in enumerate(outputs):
                depth = int(fanout_buffer_depths.get(tensor, 1))
                count_width = max(1, depth.bit_length())
                ptr_width = max(1, (depth - 1).bit_length())
                mem = f"node_{index}_fanout_fifo_data_{output_index}"
                count = f"node_{index}_fanout_fifo_count_{output_index}"
                rd_ptr = f"node_{index}_fanout_fifo_rd_{output_index}"
                wr_ptr = f"node_{index}_fanout_fifo_wr_{output_index}"
                pop = f"node_{index}_fanout_fifo_pop_{output_index}"
                data_signal = f"node_{index}_output_data_{output_index}"
                lines.extend([
                    "  always @(posedge clk) begin",
                    "    if (!rst_n) begin",
                    f"      {count} <= {count_width}'d0;",
                    f"      {rd_ptr} <= {ptr_width}'d0;",
                    f"      {wr_ptr} <= {ptr_width}'d0;",
                    "    end else begin",
                    f"      if ({push_signal}) begin",
                    f"        {mem}[{wr_ptr}] <= {data_signal};",
                    (f"        {wr_ptr} <= {ptr_width}'d0;" if depth == 1 else f"        if ({wr_ptr} == {ptr_width}'d{depth - 1}) {wr_ptr} <= {ptr_width}'d0; else {wr_ptr} <= {wr_ptr} + 1'b1;"),
                    "      end",
                    f"      if ({pop}) begin",
                    (f"        {rd_ptr} <= {ptr_width}'d0;" if depth == 1 else f"        if ({rd_ptr} == {ptr_width}'d{depth - 1}) {rd_ptr} <= {ptr_width}'d0; else {rd_ptr} <= {rd_ptr} + 1'b1;"),
                    "      end",
                    f"      case ({{{push_signal}, {pop}}})",
                    f"        2'b10: {count} <= {count} + 1'b1;",
                    f"        2'b01: {count} <= {count} - 1'b1;",
                    "        default: ;",
                    "      endcase",
                    "    end",
                    "  end",
                ])
        else:
            lines.append(
                f"  assign {output_ready_signal} = {_and([_tensor_signal(name, 'ready') for name in outputs])};"
            )
            for tensor in outputs:
                lines.append(
                    f"  assign {_tensor_signal(tensor, 'valid')} = {output_valid_signal};"
                )
                vhdl_output_data_signals.append(_tensor_signal(tensor, "data"))

        connections = [
            f".{abi.clock}(clk)",
            f".{abi.reset_n}(rst_n)",
            f".{abi.input_valid}({input_valid_signal})",
            f".{abi.input_ready}({input_ready_signal})",
            f".{abi.output_valid}({output_valid_signal})",
            f".{abi.output_ready}({output_ready_signal})",
        ]
        connections.extend(
            f".{port.data}({_tensor_signal(tensor, 'data')})"
            for port, tensor in zip(abi.inputs, inputs)
        )
        connections.extend(
            f".{port.data}({data_signal})"
            for port, data_signal in zip(abi.outputs, vhdl_output_data_signals)
        )
        lines.append(f"  {stage['top']} {instance} (")
        lines.append("    " + ",\n    ".join(connections))
        lines.append("  );")
        lines.append("")

    if protocol_error_signals:
        lines.append(
            "  wire fpgai_axis_protocol_error = " + " | ".join(protocol_error_signals) + ";"
        )
    else:
        lines.append("  wire fpgai_axis_protocol_error = 1'b0;")
    lines.append("")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def _sv_word_literal(value: int, width: int) -> str:
    mask = (1 << int(width)) - 1
    digits = max(1, (int(width) + 3) // 4)
    return f"{int(width)}'h{(int(value) & mask):0{digits}x}"


def _testbench_source(
    top_name: str,
    input_width: int,
    output_width: int,
    input_value: int,
    expected_output: int,
    *,
    input_values: tuple[int, ...] | None = None,
    expected_outputs: tuple[int, ...] | None = None,
) -> str:
    vector_mode = input_values is not None or expected_outputs is not None
    if not vector_mode:
        input_values = (int(input_value),)
        expected_outputs = (int(expected_output),)
    if input_values is None or expected_outputs is None or not input_values or not expected_outputs:
        raise ValueError("MIXDAG021: vector numeric validation requires non-empty input_values and expected_outputs")
    in_literals = ", ".join(_sv_word_literal(value, input_width) for value in input_values)
    out_literals = ", ".join(_sv_word_literal(value, output_width) for value in expected_outputs)
    return f'''`timescale 1ns/1ps
module {top_name}_tb;
  localparam integer INPUT_COUNT = {len(input_values)};
  localparam integer OUTPUT_COUNT = {len(expected_outputs)};
  reg clk = 0;
  reg rst_n = 0;
  reg input_valid = 0;
  wire input_ready;
  reg [{input_width - 1}:0] input_data = 0;
  wire output_valid;
  reg output_ready = 0;
  wire [{output_width - 1}:0] output_data;
  reg [{input_width - 1}:0] input_words [0:INPUT_COUNT-1];
  reg [{output_width - 1}:0] expected_words [0:OUTPUT_COUNT-1];
  integer i;
  integer cycles;
  integer cycle_count = 0;
  integer first_input_accept_cycle = -1;
  integer last_input_accept_cycle = -1;
  integer first_output_accept_cycle = -1;
  integer last_output_accept_cycle = -1;

  always #2.5 clk = ~clk;
  always @(posedge clk) cycle_count = cycle_count + 1;
  {top_name} dut(
    .clk(clk), .rst_n(rst_n),
    .input_valid(input_valid), .input_ready(input_ready), .input_data(input_data),
    .output_valid(output_valid), .output_ready(output_ready), .output_data(output_data)
  );

  initial begin
    input_words = '{{ {in_literals} }};
    expected_words = '{{ {out_literals} }};
    repeat (3) @(posedge clk);
    rst_n <= 1;
    @(posedge clk);

    for (i = 0; i < INPUT_COUNT; i = i + 1) begin
      // Drive each input word before the sampling edge and keep VALID asserted
      // until one rising edge observes READY.  The previous checker could wait
      // for READY to become high on a rising edge and then hold VALID for one
      // additional rising edge, duplicating that AXIS beat.
      @(negedge clk);
      input_data = input_words[i];
      input_valid = 1;
      cycles = 0;
      while (!input_ready && cycles < 400) begin
        @(posedge clk);
        cycles = cycles + 1;
        if (!input_ready) @(negedge clk);
      end
      if (!input_ready) $fatal(1, "FPGAI DAG mixed-backend input handshake timeout at word %0d", i);
      // READY is high before the next rising edge, so that edge consumes exactly
      // one transaction.  Deassert only afterwards to avoid a testbench/DUT race.
      @(posedge clk);
      #1;
      if (first_input_accept_cycle < 0) first_input_accept_cycle = cycle_count;
      last_input_accept_cycle = cycle_count;
      @(negedge clk);
      input_valid = 0;
    end

    // Drive READY synchronously before validating the first transaction.
    // Use blocking assignment here because the checker observes the combinational
    // ready/valid state in the same simulation time slot.
    output_ready = 1;
    #1;
    for (i = 0; i < OUTPUT_COUNT; i = i + 1) begin
      cycles = 0;
      while (!(output_valid && output_ready) && cycles < 4000) begin
        @(posedge clk);
        #1;
        cycles = cycles + 1;
      end
      if (!(output_valid && output_ready))
        $fatal(1, "FPGAI DAG mixed-backend output handshake timeout at word %0d", i);
      if (output_data !== expected_words[i])
        $fatal(1, "FPGAI DAG mixed-backend numeric mismatch at word %0d: expected %h got %h", i, expected_words[i], output_data);
      // Consume exactly one transaction, then wait for NBA/combinational updates
      // before inspecting the next output word.  Without this settling point, a
      // multi-word producer may be checked twice against the same held beat.
      @(posedge clk);
      #1;
      if (first_output_accept_cycle < 0) first_output_accept_cycle = cycle_count;
      last_output_accept_cycle = cycle_count;
    end

    if (dut.fpgai_axis_protocol_error)
      $fatal(1, "FPGAI DAG mixed-backend AXIS sideband protocol mismatch");
    $display("FPGAI_DAG_MIXED_BACKEND_SIM_METRICS first_input_cycle=%0d last_input_cycle=%0d first_output_cycle=%0d last_output_cycle=%0d input_count=%0d output_count=%0d",
      first_input_accept_cycle, last_input_accept_cycle, first_output_accept_cycle, last_output_accept_cycle, INPUT_COUNT, OUTPUT_COUNT);
    $display("FPGAI_DAG_MIXED_BACKEND_SIM_PASS");
    #5;
    $finish;
  end
endmodule
'''


def _edge_report(
    *,
    graph: Any,
    stages_by_name: Mapping[str, dict[str, Any]],
    producer: Mapping[str, str | None],
    consumers: Mapping[str, list[str]],
    tensor_widths: Mapping[str, int],
    fanout_buffer_depths: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    fanout_buffer_depths = fanout_buffer_depths or {}
    graph_inputs = set(getattr(graph, "inputs", ()))
    graph_outputs = set(getattr(graph, "outputs", ()))
    for tensor, source_node in producer.items():
        targets = consumers.get(tensor, [])
        if tensor in graph_outputs and not targets:
            targets = [None]
        for target_node in targets:
            source_backend = "graph_input" if tensor in graph_inputs else stages_by_name[source_node]["backend"]
            target_backend = "graph_output" if target_node is None else stages_by_name[target_node]["backend"]
            source_stage = stages_by_name.get(source_node) if source_node is not None else None
            physical_bridge = "direct_ready_valid_channel"
            if (
                source_stage is not None
                and source_stage.get("backend") == "vhdl"
                and len(source_stage.get("outputs", ())) > 1
            ):
                physical_bridge = "elastic_grouped_fanout"
            edges.append(
                {
                    "tensor": tensor,
                    "from_node": source_node,
                    "to_node": target_node,
                    "from_backend": source_backend,
                    "to_backend": target_backend,
                    "data_width": int(tensor_widths[tensor]),
                    "elastic_buffer_depth_words": int(fanout_buffer_depths.get(tensor, 1)),
                    "interface": "grouped_ready_valid_v1",
                    "physical_bridge": physical_bridge,
                }
            )
    return edges


def emit_dag_mixed_backend_physical_project(
    request: DAGMixedBackendPhysicalRequest,
) -> DAGMixedBackendPhysicalResult:
    try:
        if request.physical_profile != _PROFILE:
            raise ValueError(f"MIXDAG016: unsupported physical profile {request.physical_profile!r}")
        ops, producer, consumers, tensor_widths = _graph_topology(request.graph)
        op_names = {op.name for op in ops}
        binding_names = set(request.bindings)
        missing = sorted(op_names - binding_names)
        extra = sorted(binding_names - op_names)
        if missing:
            raise ValueError(f"MIXDAG017: physical bindings missing for nodes: {', '.join(missing)}")
        if extra:
            raise ValueError(f"MIXDAG018: physical bindings reference unknown nodes: {', '.join(extra)}")

        root = Path(request.out_dir).expanduser().resolve()
        project = root / "dag_mixed_backend"
        rtl_dir = project / "rtl"
        sim_dir = project / "sim"
        reports = root / "reports"
        rtl_dir.mkdir(parents=True, exist_ok=True)
        sim_dir.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)

        stages: list[dict[str, Any]] = []
        read_lines: list[str] = []
        staged_files: list[str] = []
        source_cache: dict[Path, str] = {}
        constants = getattr(request.graph, "constants", {}) or {}
        for index, op in enumerate(ops):
            binding = request.bindings[op.name]
            # Keep runtime-input ordering consistent with the graph contract.
            op_for_stage = type("PhysicalOpView", (), {})()
            op_for_stage.name = op.name
            op_for_stage.op_type = op.op_type
            op_for_stage.inputs = [name for name in op.inputs if name not in constants]
            op_for_stage.outputs = list(op.outputs)
            if isinstance(binding, HLSPhysicalBinding):
                stage = _stage_hls_binding(
                    op=op_for_stage,
                    binding=binding,
                    rtl_dir=rtl_dir,
                    stage_index=index,
                    tensor_widths=tensor_widths,
                    read_lines=read_lines,
                    staged_files=staged_files,
                    source_cache=source_cache,
                )
            elif isinstance(binding, VHDLPhysicalBinding):
                stage = _stage_vhdl_binding(
                    op=op_for_stage,
                    binding=binding,
                    rtl_dir=rtl_dir,
                    stage_index=index,
                    tensor_widths=tensor_widths,
                    read_lines=read_lines,
                    staged_files=staged_files,
                    source_cache=source_cache,
                )
            elif isinstance(binding, RequantizationPhysicalBinding):
                stage = _stage_requantization_binding(
                    graph=request.graph,
                    op=op_for_stage,
                    binding=binding,
                    tensor_widths=tensor_widths,
                )
            else:
                raise ValueError(f"MIXDAG019: unsupported physical binding type for node {op.name!r}")
            stages.append(stage)

        wrapper = rtl_dir / f"{request.top_name}.sv"
        wrapper.write_text(
            _dag_wrapper_source(
                top_name=request.top_name,
                graph=request.graph,
                stages=stages,
                tensor_widths=tensor_widths,
                fanout_buffer_depths=request.fanout_buffer_depths,
            ),
            encoding="utf-8",
        )
        testbench = sim_dir / f"{request.top_name}_tb.sv"
        testbench.write_text(
            _testbench_source(
                request.top_name,
                int(tensor_widths[list(getattr(request.graph, "inputs", ()))[0]]),
                int(tensor_widths[list(getattr(request.graph, "outputs", ()))[0]]),
                request.input_value,
                request.expected_output,
                input_values=request.input_values,
                expected_outputs=request.expected_outputs,
            ),
            encoding="utf-8",
        )
        read_lines.append(f'read_verilog -sv "./rtl/{wrapper.name}"')
        read_lines.append(f'add_files -fileset sim_1 -norecurse "./sim/{testbench.name}"')

        run_tcl = project / "run_vivado.tcl"
        run_tcl.write_text(
            f"create_project -force fpgai_dag_mixed_backend ./vivado_proj -part {request.part}\n"
            + "\n".join(read_lines)
            + f'''\nset_property top {request.top_name} [current_fileset]
set_property top {request.top_name}_tb [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
set_property xsim.simulate.runtime {{100 us}} [get_filesets sim_1]
launch_simulation
close_sim
synth_design -mode out_of_context -top {request.top_name} -part {request.part}
create_clock -name clk -period {float(request.clock_period_ns):.6f} [get_ports clk]
report_utilization -file ../reports/dag_mixed_backend_utilization_synth.rpt
report_timing_summary -file ../reports/dag_mixed_backend_timing_synth.rpt
'''
            + (
                '''opt_design
place_design
route_design
report_utilization -file ../reports/dag_mixed_backend_utilization_impl.rpt
report_timing_summary -file ../reports/dag_mixed_backend_timing_impl.rpt
report_power -file ../reports/dag_mixed_backend_power_impl.rpt
'''
                if request.run_implementation
                else ""
            )
            + '''exit
''',
            encoding="utf-8",
        )

        stages_by_name = {stage["node_name"]: stage for stage in stages}
        edges = _edge_report(
            graph=request.graph,
            stages_by_name=stages_by_name,
            producer=producer,
            consumers=consumers,
            tensor_widths=tensor_widths,
            fanout_buffer_depths=request.fanout_buffer_depths,
        )
        report_path = reports / "dag_mixed_backend_physical.json"
        report_payload = {
            "schema": "fpgai.dag-mixed-backend-physical/v1",
            "status": "generated",
            "graph_name": str(getattr(request.graph, "name", "main")),
            "physical_profile": request.physical_profile,
            "handshake_policy": "grouped_transaction",
            "implementation_context": "out_of_context_accelerator_core",
            "data_width": (
                next(iter(set(tensor_widths.values())))
                if len(set(tensor_widths.values())) == 1
                else None
            ),
            "tensor_widths": {name: int(width) for name, width in sorted(tensor_widths.items())},
            "graph_inputs": list(getattr(request.graph, "inputs", ())),
            "graph_outputs": list(getattr(request.graph, "outputs", ())),
            "nodes": [
                {
                    "node": stage["node_name"],
                    "op_type": stage["op_type"],
                    "backend": stage["backend"],
                    "top": stage["top"],
                    "inputs": list(stage["inputs"]),
                    "outputs": list(stage["outputs"]),
                    "input_count": len(stage["inputs"]),
                    "output_count": len(stage["outputs"]),
                    "handshake_policy": stage["handshake_policy"],
                    "package_id": stage.get("package_id"),
                    "package_version": stage.get("package_version"),
                    "generated_or_staged_rtl": list(stage["source_files"]),
                    "requantization": (
                        stage["requantization"].to_dict()
                        if stage.get("requantization") is not None
                        else None
                    ),
                    "axis_sidebands": (
                        {
                            "inputs": [
                                {
                                    "keep": "keep" in group,
                                    "strb": "strb" in group,
                                    "last": "last" in group,
                                    "packet_words": (
                                        int(stage.get("input_packet_words", ())[idx])
                                        if stage.get("input_packet_words")
                                        else 1
                                    ),
                                }
                                for idx, group in enumerate(stage.get("ports", {}).get("inputs", ()))
                            ],
                            "outputs": [
                                {
                                    "keep": "keep" in group,
                                    "strb": "strb" in group,
                                    "last": "last" in group,
                                    "packet_words": (
                                        int(stage.get("output_packet_words", ())[idx])
                                        if stage.get("output_packet_words")
                                        else 1
                                    ),
                                }
                                for idx, group in enumerate(stage.get("ports", {}).get("outputs", ()))
                            ],
                            "policy": "explicit_keep_strb_last_when_present",
                        }
                        if stage["backend"] == "vitis_hls"
                        else None
                    ),
                }
                for stage in stages
            ],
            "edges": edges,
            "numeric_validation": {
                "input": int(request.input_value) if request.input_values is None else None,
                "expected_output": int(request.expected_output) if request.expected_outputs is None else None,
                "input_values": list(request.input_values) if request.input_values is not None else None,
                "expected_outputs": list(request.expected_outputs) if request.expected_outputs is not None else None,
                "comparison": (
                    "xsim_exact_word_sequence"
                    if request.input_values is not None or request.expected_outputs is not None
                    else "xsim_exact_signed_integer"
                ),
            },
            "support": {
                "dag": True,
                "backpressure": True,
                "multi_input_vhdl": any(stage["backend"] == "vhdl" and len(stage["inputs"]) > 1 for stage in stages),
                "multi_output_vhdl": any(stage["backend"] == "vhdl" and len(stage["outputs"]) > 1 for stage in stages),
                "implicit_fanout": False,
                "explicit_split": True,
                "explicit_merge": True,
                "multi_port_hls": any(
                    stage["backend"] == "vitis_hls" and (len(stage["inputs"]) > 1 or len(stage["outputs"]) > 1)
                    for stage in stages
                ),
                "hls_multi_port_handshake": "axis_independent_ports",
                "heterogeneous_tensor_widths": len(set(tensor_widths.values())) > 1,
                "requantization_bridges": any(stage["backend"] == "requantization" for stage in stages),
                "hls_axis_sidebands": any(
                    stage["backend"] == "vitis_hls"
                    and any(
                        any(field in group for field in ("keep", "strb", "last"))
                        for group in (
                            *stage.get("ports", {}).get("inputs", ()),
                            *stage.get("ports", {}).get("outputs", ()),
                        )
                    )
                    for stage in stages
                ),
            },
            "artifacts": {
                "wrapper": str(wrapper),
                "testbench": str(testbench),
                "run_tcl": str(run_tcl),
                "staged_rtl": staged_files,
            },
            "validation_level": "mixed_dag_rtl_project_generated",
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return DAGMixedBackendPhysicalResult(True, project, wrapper, testbench, run_tcl, report_path)
    except (ValueError, OSError) as exc:
        code, separator, message = str(exc).partition(": ")
        issue = GraphPhysicalIssue(code if separator else "MIXDAG000", "dag_physical", message if separator else str(exc))
        return DAGMixedBackendPhysicalResult(False, None, None, None, None, None, (issue,))


_SIM_METRICS_RE = re.compile(
    r"FPGAI_DAG_MIXED_BACKEND_SIM_METRICS\s+"
    r"first_input_cycle=(?P<first_input>-?\d+)\s+"
    r"last_input_cycle=(?P<last_input>-?\d+)\s+"
    r"first_output_cycle=(?P<first_output>-?\d+)\s+"
    r"last_output_cycle=(?P<last_output>-?\d+)\s+"
    r"input_count=(?P<input_count>\d+)\s+"
    r"output_count=(?P<output_count>\d+)"
)


def _parse_simulation_metrics(text: str) -> dict[str, Any] | None:
    match = _SIM_METRICS_RE.search(text)
    if not match:
        return None
    values = {key: int(value) for key, value in match.groupdict().items()}
    first_input = values["first_input"]
    last_input = values["last_input"]
    first_output = values["first_output"]
    last_output = values["last_output"]
    output_count = values["output_count"]
    output_span = last_output - first_output
    return {
        "schema": "fpgai.dag-mixed-backend-simulation-metrics/v1",
        "measurement": "cycle_accurate_behavioral_xsim",
        "first_input_accept_cycle": first_input,
        "last_input_accept_cycle": last_input,
        "first_output_accept_cycle": first_output,
        "last_output_accept_cycle": last_output,
        "input_count_words": values["input_count"],
        "output_count_words": output_count,
        "first_output_latency_cycles": first_output - first_input,
        "packet_completion_latency_cycles": last_output - first_input,
        "post_input_drain_cycles": last_output - last_input,
        "input_accept_span_cycles": last_input - first_input,
        "output_accept_span_cycles": output_span,
        "mean_output_interbeat_cycles": (float(output_span) / float(output_count - 1)) if output_count > 1 else None,
        "initiation_interval": None,
        "initiation_interval_status": "not_measured_single_packet_testbench",
    }


def run_dag_mixed_backend_physical_project(
    result: DAGMixedBackendPhysicalResult,
    *,
    vivado_executable: str = "vivado",
    timeout: int = 900,
) -> dict[str, Any]:
    if not result.ok or result.run_tcl is None:
        return {"schema": "fpgai.dag-mixed-backend-tool-result/v1", "status": "not_run", "returncode": None}

    cwd = result.run_tcl.parent
    reports = cwd.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [vivado_executable, "-mode", "batch", "-source", result.run_tcl.name],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    stdout = reports / "dag_mixed_backend_vivado_stdout.log"
    stderr = reports / "dag_mixed_backend_vivado_stderr.log"
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")

    marker = "FPGAI_DAG_MIXED_BACKEND_SIM_PASS"
    simulation_passed = marker in proc.stdout
    simulation_log: Path | None = stdout if simulation_passed else None
    simulation_metrics = _parse_simulation_metrics(proc.stdout)
    if not simulation_passed or simulation_metrics is None:
        for candidate in sorted(cwd.glob("vivado_proj/**/*.log")):
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not simulation_passed and marker in text:
                simulation_passed = True
                simulation_log = candidate
            if simulation_metrics is None:
                simulation_metrics = _parse_simulation_metrics(text)
            if simulation_passed and simulation_metrics is not None:
                break

    simulation_metrics_report: Path | None = None
    if simulation_metrics is not None:
        simulation_metrics_report = reports / "dag_mixed_backend_simulation_metrics.json"
        simulation_metrics_report.write_text(
            json.dumps(simulation_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    utilization = reports / "dag_mixed_backend_utilization_synth.rpt"
    timing = reports / "dag_mixed_backend_timing_synth.rpt"
    utilization_impl = reports / "dag_mixed_backend_utilization_impl.rpt"
    timing_impl = reports / "dag_mixed_backend_timing_impl.rpt"
    power_impl = reports / "dag_mixed_backend_power_impl.rpt"
    synthesis_reports_present = utilization.is_file() and timing.is_file()
    implementation_reports_present = utilization_impl.is_file() and timing_impl.is_file()
    passed = proc.returncode == 0 and simulation_passed and synthesis_reports_present
    payload = {
        "schema": "fpgai.dag-mixed-backend-tool-result/v1",
        "status": "passed" if passed else "failed",
        "returncode": proc.returncode,
        "mixed_language_simulation_passed": simulation_passed,
        "synthesis_reports_present": synthesis_reports_present,
        "implementation_reports_present": implementation_reports_present,
        "validation_level": (
            "vivado_implemented"
            if passed and implementation_reports_present
            else (
                "vivado_synthesized"
                if passed
                else ("rtl_simulated" if simulation_passed else "mixed_dag_rtl_project_generated")
            )
        ),
        "stdout_log": str(stdout),
        "stderr_log": str(stderr),
        "simulation_log": str(simulation_log) if simulation_log else None,
        "simulation_metrics": simulation_metrics,
        "simulation_metrics_report": str(simulation_metrics_report) if simulation_metrics_report else None,
        "utilization_report": str(utilization) if utilization.is_file() else None,
        "timing_report": str(timing) if timing.is_file() else None,
        "utilization_impl_report": str(utilization_impl) if utilization_impl.is_file() else None,
        "timing_impl_report": str(timing_impl) if timing_impl.is_file() else None,
        "power_impl_report": str(power_impl) if power_impl.is_file() else None,
    }
    (reports / "dag_mixed_backend_tool_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "DAGMixedBackendPhysicalRequest",
    "DAGMixedBackendPhysicalResult",
    "emit_dag_mixed_backend_physical_project",
    "run_dag_mixed_backend_physical_project",
]
