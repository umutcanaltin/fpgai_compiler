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

from .graph_physical import GraphPhysicalIssue, HLSPhysicalBinding, VHDLPhysicalBinding
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
    bindings: Mapping[str, HLSPhysicalBinding | VHDLPhysicalBinding]
    part: str = "xck26-sfvc784-2LV-c"
    top_name: str = "fpgai_dag_mixed_backend_top"
    clock_period_ns: float = 5.0
    input_value: int = 7
    expected_output: int = 43
    physical_profile: str = _PROFILE


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


def _graph_topology(graph: Any) -> tuple[list[Any], dict[str, str | None], dict[str, list[str]], int]:
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
    width: int | None = None

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

    # Physical fanout must be represented by an explicit multi-output node so
    # transaction replication and backpressure semantics remain deterministic.
    implicit_fanout = sorted(name for name, users in consumers.items() if len(users) > 1)
    if implicit_fanout:
        raise ValueError(
            "MIXDAG007: implicit tensor fanout is not supported by the grouped ready/valid profile; "
            "insert an explicit multi-output implementation for: " + ", ".join(implicit_fanout)
        )

    for tensor_name, tensor in tensors.items():
        if tensor_name not in producer:
            continue
        tensor_width = _tensor_width(getattr(tensor, "dtype", ""))
        if tensor_width is None:
            raise ValueError(
                f"MIXDAG008: tensor {tensor_name!r} has unsupported physical dtype {getattr(tensor, 'dtype', None)!r}"
            )
        if width is None:
            width = tensor_width
        elif tensor_width != width:
            raise ValueError(
                f"MIXDAG009: maintained DAG profile requires one stream width; tensor {tensor_name!r} uses {tensor_width}, expected {width}"
            )
    return ops, producer, consumers, int(width or 16)


def _stage_hls_binding(
    *,
    op: Any,
    binding: HLSPhysicalBinding,
    rtl_dir: Path,
    stage_index: int,
    data_width: int,
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
    for logical, groups in (("input", input_ports), ("output", output_ports)):
        for port_index, group in enumerate(groups):
            actual_width = port_info[group["data"]][1]
            if actual_width != data_width:
                raise ValueError(
                    f"MIXDAG012: HLS node {op.name!r} {logical} port {port_index} width {actual_width} does not match graph width {data_width}"
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
    data_width: int,
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
    for port in (*abi.inputs, *abi.outputs):
        if port.data_width != data_width:
            raise ValueError(
                f"MIXDAG015: VHDL node {op.name!r} port {port.name!r} width {port.data_width} does not match graph width {data_width}"
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


def _dag_wrapper_source(
    *,
    top_name: str,
    graph: Any,
    stages: list[dict[str, Any]],
    data_width: int,
) -> str:
    graph_input = list(getattr(graph, "inputs", ()))[0]
    graph_output = list(getattr(graph, "outputs", ()))[0]
    tensors: list[str] = []
    for name in [graph_input, *[output for stage in stages for output in stage["outputs"]]]:
        if name not in tensors:
            tensors.append(name)

    lines = [
        f"module {top_name}(",
        "    input wire clk,",
        "    input wire rst_n,",
        "    input wire input_valid,",
        "    output wire input_ready,",
        f"    input wire signed [{data_width - 1}:0] input_data,",
        "    output wire output_valid,",
        "    input wire output_ready,",
        f"    output wire signed [{data_width - 1}:0] output_data",
        ");",
        "",
    ]
    for tensor in tensors:
        lines.extend(
            [
                f"  wire signed [{data_width - 1}:0] {_tensor_signal(tensor, 'data')};",
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

    for index, stage in enumerate(stages):
        instance = f"u_{index}_{_sanitize(stage['node_name'])}"
        inputs = stage["inputs"]
        outputs = stage["outputs"]
        if stage["backend"] == "vitis_hls":
            ports = stage["ports"]
            connections: list[str] = []
            clock_port = ports.get("clock")
            reset_port = ports.get("reset")
            if clock_port:
                connections.append(f".{clock_port}(clk)")
            if reset_port:
                reset_expression = "rst_n" if str(reset_port).endswith("_n") else "~rst_n"
                connections.append(f".{reset_port}({reset_expression})")
            for tensor, group in zip(inputs, ports["inputs"]):
                connections.extend(
                    [
                        f".{group['data']}({_tensor_signal(tensor, 'data')})",
                        f".{group['valid']}({_tensor_signal(tensor, 'valid')})",
                        f".{group['ready']}({_tensor_signal(tensor, 'ready')})",
                    ]
                )
            for tensor, group in zip(outputs, ports["outputs"]):
                connections.extend(
                    [
                        f".{group['data']}({_tensor_signal(tensor, 'data')})",
                        f".{group['valid']}({_tensor_signal(tensor, 'valid')})",
                        f".{group['ready']}({_tensor_signal(tensor, 'ready')})",
                    ]
                )
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
            # A grouped VHDL output represents one transaction containing every
            # output tensor.  Converting that transaction to independent
            # ready/valid channels with combinational peer-ready gating creates
            # a zero-valid combinational loop when a grouped split feeds a
            # grouped join directly.  Use a compiler-owned elastic fanout
            # buffer instead: accept the grouped transaction once, then track
            # delivery of each output independently until every consumer has
            # acknowledged it.
            fanout_active = f"node_{index}_fanout_active"
            fanout_pending = f"node_{index}_fanout_pending"
            fanout_next_pending = f"node_{index}_fanout_next_pending"
            lines.extend(
                [
                    f"  reg {fanout_active};",
                    f"  reg [{len(outputs) - 1}:0] {fanout_pending};",
                    f"  wire [{len(outputs) - 1}:0] {fanout_next_pending};",
                    f"  assign {output_ready_signal} = ~{fanout_active};",
                ]
            )
            consume_terms: list[str] = []
            for output_index, tensor in enumerate(outputs):
                data_signal = f"node_{index}_output_data_{output_index}"
                data_reg = f"node_{index}_fanout_data_{output_index}"
                vhdl_output_data_signals.append(data_signal)
                lines.extend(
                    [
                        f"  wire signed [{data_width - 1}:0] {data_signal};",
                        f"  reg signed [{data_width - 1}:0] {data_reg};",
                        f"  assign {_tensor_signal(tensor, 'data')} = {data_reg};",
                        f"  assign {_tensor_signal(tensor, 'valid')} = {fanout_active} & {fanout_pending}[{output_index}];",
                    ]
                )
                consume_terms.append(
                    f"({fanout_pending}[{output_index}] & {_tensor_signal(tensor, 'ready')})"
                )
            next_bits = [
                f"{fanout_pending}[{idx}] & ~({_tensor_signal(tensor, 'ready')})"
                for idx, tensor in enumerate(outputs)
            ]
            lines.append(
                f"  assign {fanout_next_pending} = {{{', '.join(reversed(next_bits))}}};"
            )
            lines.extend(
                [
                    "  always @(posedge clk) begin",
                    "    if (!rst_n) begin",
                    f"      {fanout_active} <= 1'b0;",
                    f"      {fanout_pending} <= {len(outputs)}'b0;",
                ]
            )
            for output_index in range(len(outputs)):
                lines.append(f"      node_{index}_fanout_data_{output_index} <= '0;")
            lines.extend(
                [
                    "    end else if (!" + fanout_active + ") begin",
                    f"      if ({output_valid_signal}) begin",
                    f"        {fanout_active} <= 1'b1;",
                    f"        {fanout_pending} <= {{{len(outputs)}{{1'b1}}}};",
                ]
            )
            for output_index in range(len(outputs)):
                lines.append(
                    f"        node_{index}_fanout_data_{output_index} <= node_{index}_output_data_{output_index};"
                )
            lines.extend(
                [
                    "      end",
                    f"    end else if ({fanout_next_pending} == {len(outputs)}'b0) begin",
                    f"      {fanout_active} <= 1'b0;",
                    f"      {fanout_pending} <= {len(outputs)}'b0;",
                    "    end else begin",
                    f"      {fanout_pending} <= {fanout_next_pending};",
                    "    end",
                    "  end",
                ]
            )
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

    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def _testbench_source(top_name: str, data_width: int, input_value: int, expected_output: int) -> str:
    return f'''`timescale 1ns/1ps
module {top_name}_tb;
  reg clk = 0;
  reg rst_n = 0;
  reg input_valid = 0;
  wire input_ready;
  reg signed [{data_width - 1}:0] input_data = 0;
  wire output_valid;
  reg output_ready = 0;
  wire signed [{data_width - 1}:0] output_data;
  reg signed [{data_width - 1}:0] held_output = 0;
  integer cycles = 0;

  always #2.5 clk = ~clk;
  {top_name} dut(
    .clk(clk), .rst_n(rst_n),
    .input_valid(input_valid), .input_ready(input_ready), .input_data(input_data),
    .output_valid(output_valid), .output_ready(output_ready), .output_data(output_data)
  );

  initial begin
    repeat (3) @(posedge clk);
    rst_n <= 1;
    @(posedge clk);
    input_data <= {int(input_value)};
    input_valid <= 1;
    while (!input_ready && cycles < 80) begin
      @(posedge clk);
      cycles = cycles + 1;
    end
    if (!input_ready) $fatal(1, "FPGAI DAG mixed-backend input_ready timeout");
    @(posedge clk);
    input_valid <= 0;

    cycles = 0;
    while (!output_valid && cycles < 160) begin
      @(posedge clk);
      cycles = cycles + 1;
    end
    if (!output_valid) $fatal(1, "FPGAI DAG mixed-backend output_valid timeout");
    if ($signed(output_data) !== {int(expected_output)})
      $fatal(1, "FPGAI DAG mixed-backend numeric mismatch: expected {int(expected_output)} got %0d", $signed(output_data));

    held_output = output_data;
    repeat (3) begin
      @(posedge clk);
      if (!output_valid) $fatal(1, "FPGAI DAG output_valid dropped under backpressure");
      if (output_data !== held_output) $fatal(1, "FPGAI DAG output_data changed under backpressure");
    end

    output_ready <= 1;
    @(posedge clk);
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
    data_width: int,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
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
                    "data_width": data_width,
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
        ops, producer, consumers, data_width = _graph_topology(request.graph)
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
                    data_width=data_width,
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
                    data_width=data_width,
                    read_lines=read_lines,
                    staged_files=staged_files,
                    source_cache=source_cache,
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
                data_width=data_width,
            ),
            encoding="utf-8",
        )
        testbench = sim_dir / f"{request.top_name}_tb.sv"
        testbench.write_text(
            _testbench_source(
                request.top_name,
                data_width,
                request.input_value,
                request.expected_output,
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
launch_simulation
run 500 ns
close_sim
synth_design -top {request.top_name} -part {request.part}
create_clock -name clk -period {float(request.clock_period_ns):.6f} [get_ports clk]
report_utilization -file ../reports/dag_mixed_backend_utilization_synth.rpt
report_timing_summary -file ../reports/dag_mixed_backend_timing_synth.rpt
exit
''',
            encoding="utf-8",
        )

        stages_by_name = {stage["node_name"]: stage for stage in stages}
        edges = _edge_report(
            graph=request.graph,
            stages_by_name=stages_by_name,
            producer=producer,
            consumers=consumers,
            data_width=data_width,
        )
        report_path = reports / "dag_mixed_backend_physical.json"
        report_payload = {
            "schema": "fpgai.dag-mixed-backend-physical/v1",
            "status": "generated",
            "graph_name": str(getattr(request.graph, "name", "main")),
            "physical_profile": request.physical_profile,
            "handshake_policy": "grouped_transaction",
            "data_width": data_width,
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
                }
                for stage in stages
            ],
            "edges": edges,
            "numeric_validation": {
                "input": int(request.input_value),
                "expected_output": int(request.expected_output),
                "comparison": "xsim_exact_signed_integer",
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
    simulation_log: Path | None = None
    if not simulation_passed:
        for candidate in sorted(cwd.glob("vivado_proj/**/*.log")):
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if marker in text:
                simulation_passed = True
                simulation_log = candidate
                break

    utilization = reports / "dag_mixed_backend_utilization_synth.rpt"
    timing = reports / "dag_mixed_backend_timing_synth.rpt"
    synthesis_reports_present = utilization.is_file() and timing.is_file()
    passed = proc.returncode == 0 and simulation_passed and synthesis_reports_present
    payload = {
        "schema": "fpgai.dag-mixed-backend-tool-result/v1",
        "status": "passed" if passed else "failed",
        "returncode": proc.returncode,
        "mixed_language_simulation_passed": simulation_passed,
        "synthesis_reports_present": synthesis_reports_present,
        "validation_level": (
            "vivado_synthesized"
            if passed
            else ("rtl_simulated" if simulation_passed else "mixed_dag_rtl_project_generated")
        ),
        "stdout_log": str(stdout),
        "stderr_log": str(stderr),
        "simulation_log": str(simulation_log) if simulation_log else None,
        "utilization_report": str(utilization) if utilization.is_file() else None,
        "timing_report": str(timing) if timing.is_file() else None,
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
