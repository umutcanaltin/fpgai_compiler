from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping

from fpgai.implementations.implementation_contract import ImplementationContract
from fpgai.implementations.vhdl_integration.integration import (
    parse_vhdl_scalar_stream_abi,
    validate_vhdl_integration_contract,
)
from .physical import (
    _find_hls_top,
    _ready_valid_hls_ports,
    _required_hls_ports,
    _safe_package_source,
    _verilog_ports,
)


@dataclass(frozen=True)
class GraphPhysicalIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class HLSPhysicalBinding:
    node_name: str
    rtl_dir: str | Path
    top: str
    backend: str = "vitis_hls"
    input_streams: tuple[str, ...] = ()
    output_streams: tuple[str, ...] = ()
    input_packet_words: tuple[int, ...] = ()
    output_packet_words: tuple[int, ...] = ()


@dataclass(frozen=True)
class VHDLPhysicalBinding:
    node_name: str
    contract: ImplementationContract
    backend: str = "vhdl"


@dataclass(frozen=True)
class RequantizationPhysicalBinding:
    node_name: str
    backend: str = "requantization"


@dataclass(frozen=True)
class GraphMixedBackendPhysicalRequest:
    out_dir: str | Path
    graph: Any
    bindings: Mapping[str, HLSPhysicalBinding | VHDLPhysicalBinding]
    part: str = "xck26-sfvc784-2LV-c"
    top_name: str = "fpgai_graph_mixed_backend_top"
    clock_period_ns: float = 5.0
    input_value: int = 7
    expected_output: int = 15
    physical_profile: str = "linear_scalar_valid_data_v1"


@dataclass(frozen=True)
class GraphMixedBackendPhysicalResult:
    ok: bool
    project_dir: Path | None
    wrapper: Path | None
    testbench: Path | None
    run_tcl: Path | None
    report_path: Path | None
    issues: tuple[GraphPhysicalIssue, ...] = ()


def _tensor_width(dtype: str) -> int | None:
    value = str(dtype).lower().strip()
    table = {
        "int8": 8, "uint8": 8,
        "int16": 16, "uint16": 16,
        "int32": 32, "uint32": 32,
        "float16": 16, "fp16": 16,
        "float32": 32, "fp32": 32, "float": 32,
    }
    return table.get(value)


def _runtime_inputs(graph: Any, op: Any) -> list[str]:
    constants = getattr(graph, "constants", {}) or {}
    return [name for name in getattr(op, "inputs", ()) if name not in constants]


def _linear_chain(graph: Any) -> tuple[list[Any], list[dict[str, Any]], int]:
    ops = list(getattr(graph, "ops", ()))
    graph_inputs = list(getattr(graph, "inputs", ()))
    graph_outputs = list(getattr(graph, "outputs", ()))
    if len(graph_inputs) != 1 or len(graph_outputs) != 1:
        raise ValueError("MIXGRAPH001: scalar_stream_v1 graph profile requires exactly one graph input and one graph output")
    if not ops:
        raise ValueError("MIXGRAPH002: mixed-backend physical graph must contain at least one operation")

    current = graph_inputs[0]
    boundaries: list[dict[str, Any]] = []
    width: int | None = None
    tensors = getattr(graph, "tensors", {}) or {}
    for index, op in enumerate(ops):
        runtime_inputs = _runtime_inputs(graph, op)
        outputs = list(getattr(op, "outputs", ()))
        if runtime_inputs != [current] or len(outputs) != 1:
            raise ValueError(
                f"MIXGRAPH003: node {op.name!r} is not compatible with the maintained linear scalar-stream physical profile"
            )
        tensor = tensors.get(current)
        if tensor is not None:
            tensor_width = _tensor_width(getattr(tensor, "dtype", ""))
            if tensor_width is None:
                raise ValueError(f"MIXGRAPH004: tensor {current!r} has unsupported physical dtype {tensor.dtype!r}")
            if width is None:
                width = tensor_width
            elif tensor_width != width:
                raise ValueError(f"MIXGRAPH005: tensor {current!r} changes stream width from {width} to {tensor_width}")
        if index:
            boundaries.append({"tensor": current, "from_node": ops[index - 1].name, "to_node": op.name})
        current = outputs[0]
    if current != graph_outputs[0]:
        raise ValueError(
            f"MIXGRAPH006: linear physical chain terminates at {current!r}, expected graph output {graph_outputs[0]!r}"
        )
    out_tensor = tensors.get(current)
    if out_tensor is not None:
        out_width = _tensor_width(getattr(out_tensor, "dtype", ""))
        if out_width is None:
            raise ValueError(f"MIXGRAPH004: tensor {current!r} has unsupported physical dtype {out_tensor.dtype!r}")
        if width is None:
            width = out_width
        elif out_width != width:
            raise ValueError(f"MIXGRAPH005: output tensor {current!r} changes stream width from {width} to {out_width}")
    return ops, boundaries, int(width or 16)


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _stage_wrapper(
    *,
    top_name: str,
    stages: list[dict[str, Any]],
    data_width: int,
) -> str:
    lines = [
        f"module {top_name}(",
        "    input wire clk,",
        "    input wire rst_n,",
        "    input wire input_valid,",
        f"    input wire signed [{data_width - 1}:0] input_data,",
        "    output wire output_valid,",
        f"    output wire signed [{data_width - 1}:0] output_data",
        ");",
        "",
    ]
    for idx in range(len(stages) - 1):
        lines += [f"  wire stage_{idx}_valid;", f"  wire signed [{data_width - 1}:0] stage_{idx}_data;"]
    if len(stages) > 1:
        lines.append("")

    for idx, stage in enumerate(stages):
        in_valid = "input_valid" if idx == 0 else f"stage_{idx - 1}_valid"
        in_data = "input_data" if idx == 0 else f"stage_{idx - 1}_data"
        out_valid = "output_valid" if idx == len(stages) - 1 else f"stage_{idx}_valid"
        out_data = "output_data" if idx == len(stages) - 1 else f"stage_{idx}_data"
        instance = f"u_{idx}_{_sanitize(stage['node_name'])}"
        if stage["backend"] == "vitis_hls":
            ports = stage["ports"]
            conns: list[str] = []
            if "clock" in ports:
                conns.append(f".{ports['clock']}(clk)")
            if "reset" in ports:
                reset_name = ports["reset"]
                reset_expr = "rst_n" if reset_name.endswith("_n") else "~rst_n"
                conns.append(f".{reset_name}({reset_expr})")
            conns += [
                f".{ports['input_data']}({in_data})",
                f".{ports['input_valid']}({in_valid})",
                f".{ports['output_data']}({out_data})",
                f".{ports['output_valid']}({out_valid})",
            ]
            lines.append(f"  {stage['top']} {instance} (")
            lines.append("    " + ",\n    ".join(conns))
            lines.append("  );")
        elif stage["backend"] == "vhdl":
            abi = stage["abi"]
            lines += [
                f"  {stage['top']} {instance} (",
                f"    .{abi.clock}(clk),",
                f"    .{abi.reset_n}(rst_n),",
                f"    .{abi.input_valid}({in_valid}),",
                f"    .{abi.input_data}({in_data}),",
                f"    .{abi.output_valid}({out_valid}),",
                f"    .{abi.output_data}({out_data})",
                "  );",
            ]
        else:
            raise ValueError(f"MIXGRAPH010: unsupported physical backend {stage['backend']!r}")
        lines.append("")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def _ready_valid_stage_wrapper(*, top_name: str, stages: list[dict[str, Any]], data_width: int) -> str:
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
    for idx in range(len(stages) - 1):
        lines += [
            f"  wire stage_{idx}_valid;",
            f"  wire stage_{idx}_ready;",
            f"  wire signed [{data_width - 1}:0] stage_{idx}_data;",
        ]
    if len(stages) > 1:
        lines.append("")

    for idx, stage in enumerate(stages):
        in_valid = "input_valid" if idx == 0 else f"stage_{idx - 1}_valid"
        in_ready = "input_ready" if idx == 0 else f"stage_{idx - 1}_ready"
        in_data = "input_data" if idx == 0 else f"stage_{idx - 1}_data"
        out_valid = "output_valid" if idx == len(stages) - 1 else f"stage_{idx}_valid"
        out_ready = "output_ready" if idx == len(stages) - 1 else f"stage_{idx}_ready"
        out_data = "output_data" if idx == len(stages) - 1 else f"stage_{idx}_data"
        instance = f"u_{idx}_{_sanitize(stage['node_name'])}"
        if stage["backend"] == "vitis_hls":
            ports = stage["ports"]
            conns: list[str] = []
            if "clock" in ports:
                conns.append(f".{ports['clock']}(clk)")
            if "reset" in ports:
                reset_name = ports["reset"]
                reset_expr = "rst_n" if reset_name.endswith("_n") else "~rst_n"
                conns.append(f".{reset_name}({reset_expr})")
            conns += [
                f".{ports['input_data']}({in_data})",
                f".{ports['input_valid']}({in_valid})",
                f".{ports['input_ready']}({in_ready})",
                f".{ports['output_data']}({out_data})",
                f".{ports['output_valid']}({out_valid})",
                f".{ports['output_ready']}({out_ready})",
            ]
            lines.append(f"  {stage['top']} {instance} (")
            lines.append("    " + ",\n    ".join(conns))
            lines.append("  );")
        elif stage["backend"] == "vhdl":
            abi = stage["abi"]
            if abi.abi != "scalar_ready_valid_v1":
                raise ValueError(
                    f"MIXGRAPH015: VHDL node {stage['node_name']!r} must use scalar_ready_valid_v1 for backpressure profile"
                )
            lines += [
                f"  {stage['top']} {instance} (",
                f"    .{abi.clock}(clk),",
                f"    .{abi.reset_n}(rst_n),",
                f"    .{abi.input_valid}({in_valid}),",
                f"    .{abi.input_ready}({in_ready}),",
                f"    .{abi.input_data}({in_data}),",
                f"    .{abi.output_valid}({out_valid}),",
                f"    .{abi.output_ready}({out_ready}),",
                f"    .{abi.output_data}({out_data})",
                "  );",
            ]
        else:
            raise ValueError(f"MIXGRAPH010: unsupported physical backend {stage['backend']!r}")
        lines.append("")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def _ready_valid_tb_source(top_name: str, data_width: int, input_value: int, expected_output: int) -> str:
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
  integer cycles = 0;
  reg signed [{data_width - 1}:0] held_data = 0;

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
    while (!input_ready && cycles < 40) begin
      @(posedge clk);
      cycles = cycles + 1;
    end
    if (!input_ready) $fatal(1, "FPGAI ready/valid input_ready timeout");
    @(posedge clk);
    input_valid <= 0;

    repeat (3) @(posedge clk);
    if (output_valid) begin
      held_data = output_data;
      repeat (2) begin
        @(posedge clk);
        if (!output_valid) $fatal(1, "FPGAI ready/valid output_valid dropped under backpressure");
        if (output_data !== held_data) $fatal(1, "FPGAI ready/valid output_data changed under backpressure");
      end
    end

    output_ready <= 1;
    cycles = 0;
    while (!output_valid && cycles < 40) begin
      @(posedge clk);
      cycles = cycles + 1;
    end
    if (!output_valid) $fatal(1, "FPGAI ready/valid output_valid timeout");
    if ($signed(output_data) !== {int(expected_output)}) $fatal(1, "FPGAI ready/valid numeric mismatch: expected {int(expected_output)} got %0d", $signed(output_data));
    $display("FPGAI_GRAPH_MIXED_BACKEND_SIM_PASS");
    #5;
    $finish;
  end
endmodule
'''


def _tb_source(top_name: str, data_width: int, input_value: int, expected_output: int) -> str:
    return f'''`timescale 1ns/1ps
module {top_name}_tb;
  reg clk = 0;
  reg rst_n = 0;
  reg input_valid = 0;
  reg signed [{data_width - 1}:0] input_data = 0;
  wire output_valid;
  wire signed [{data_width - 1}:0] output_data;
  integer cycles = 0;

  always #2.5 clk = ~clk;
  {top_name} dut(.clk(clk), .rst_n(rst_n), .input_valid(input_valid), .input_data(input_data), .output_valid(output_valid), .output_data(output_data));

  initial begin
    repeat (3) @(posedge clk);
    rst_n <= 1;
    @(posedge clk);
    input_data <= {int(input_value)};
    input_valid <= 1;
    @(posedge clk);
    input_valid <= 0;
    while (!output_valid && cycles < 40) begin
      @(posedge clk);
      cycles = cycles + 1;
    end
    if (!output_valid) $fatal(1, "FPGAI graph mixed-backend output_valid timeout");
    if ($signed(output_data) !== {int(expected_output)}) $fatal(1, "FPGAI graph mixed-backend numeric mismatch: expected {int(expected_output)} got %0d", $signed(output_data));
    $display("FPGAI_GRAPH_MIXED_BACKEND_SIM_PASS");
    #5;
    $finish;
  end
endmodule
'''


def emit_graph_mixed_backend_physical_project(
    request: GraphMixedBackendPhysicalRequest,
) -> GraphMixedBackendPhysicalResult:
    try:
        ops, boundaries, data_width = _linear_chain(request.graph)
        if request.physical_profile not in {"linear_scalar_valid_data_v1", "linear_scalar_ready_valid_v1"}:
            raise ValueError(f"MIXGRAPH016: unsupported physical profile {request.physical_profile!r}")
        ready_valid = request.physical_profile == "linear_scalar_ready_valid_v1"
        binding_names = set(request.bindings)
        op_names = {op.name for op in ops}
        missing = sorted(op_names - binding_names)
        extra = sorted(binding_names - op_names)
        if missing:
            raise ValueError(f"MIXGRAPH007: physical bindings missing for nodes: {', '.join(missing)}")
        if extra:
            raise ValueError(f"MIXGRAPH008: physical bindings reference unknown nodes: {', '.join(extra)}")

        root = Path(request.out_dir).expanduser().resolve()
        project = root / "graph_mixed_backend"
        rtl = project / "rtl"
        sim = project / "sim"
        reports = root / "reports"
        rtl.mkdir(parents=True, exist_ok=True)
        sim.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)

        stages: list[dict[str, Any]] = []
        read_lines: list[str] = []
        staged_files: list[str] = []
        for index, op in enumerate(ops):
            binding = request.bindings[op.name]
            if isinstance(binding, HLSPhysicalBinding):
                rtl_dir = Path(binding.rtl_dir).expanduser().resolve()
                top_path = _find_hls_top(rtl_dir, binding.top)
                if top_path is None:
                    raise ValueError(f"MIXGRAPH011: HLS RTL top {binding.top!r} not found for node {op.name!r}")
                port_info = _verilog_ports(top_path, binding.top)
                ports = _ready_valid_hls_ports(port_info) if ready_valid else _required_hls_ports(port_info)
                for pname in ("input_data", "output_data"):
                    actual_width = port_info[ports[pname]][1]
                    if actual_width != data_width:
                        raise ValueError(
                            f"MIXGRAPH012: HLS node {op.name!r} {pname} width {actual_width} does not match IR tensor width {data_width}"
                        )
                copied: list[str] = []
                for seq, src in enumerate(sorted(rtl_dir.rglob("*"))):
                    if not src.is_file() or src.suffix.lower() not in {".v", ".sv", ".vhd", ".vhdl"}:
                        continue
                    dst = rtl / f"hls_{index:03d}_{seq:03d}_{src.name}"
                    shutil.copy2(src, dst)
                    copied.append(dst.name)
                    staged_files.append(dst.name)
                    cmd = "read_vhdl" if dst.suffix.lower() in {".vhd", ".vhdl"} else "read_verilog"
                    read_lines.append(f'{cmd} "./rtl/{dst.name}"')
                stages.append({
                    "node_name": op.name,
                    "op_type": op.op_type,
                    "backend": "vitis_hls",
                    "top": binding.top,
                    "ports": ports,
                    "source_files": copied,
                })
            elif isinstance(binding, VHDLPhysicalBinding):
                issues = validate_vhdl_integration_contract(binding.contract)
                if issues:
                    first = issues[0]
                    raise ValueError(f"MIXGRAPH013: VHDL node {op.name!r} contract invalid: {first.code} {first.message}")
                abi = parse_vhdl_scalar_stream_abi(binding.contract)
                if ready_valid and abi.abi != "scalar_ready_valid_v1":
                    raise ValueError(
                        f"MIXGRAPH015: VHDL node {op.name!r} must use scalar_ready_valid_v1 for backpressure profile"
                    )
                if not ready_valid and abi.abi != "scalar_stream_v1":
                    raise ValueError(
                        f"MIXGRAPH017: VHDL node {op.name!r} must use scalar_stream_v1 for valid/data profile"
                    )
                if abi.data_width != data_width:
                    raise ValueError(
                        f"MIXGRAPH014: VHDL node {op.name!r} width {abi.data_width} does not match IR tensor width {data_width}"
                    )
                package_root = Path(binding.contract.package_root)
                copied = []
                for seq, rel in enumerate(binding.contract.source_order or binding.contract.sources):
                    src = _safe_package_source(package_root, rel)
                    dst = rtl / f"vhdl_{index:03d}_{seq:03d}_{src.name}"
                    shutil.copy2(src, dst)
                    copied.append(dst.name)
                    staged_files.append(dst.name)
                    read_lines.append(f'read_vhdl "./rtl/{dst.name}"')
                stages.append({
                    "node_name": op.name,
                    "op_type": op.op_type,
                    "backend": "vhdl",
                    "top": binding.contract.top,
                    "abi": abi,
                    "package_id": binding.contract.package_id,
                    "package_version": binding.contract.version,
                    "source_files": copied,
                })
            else:
                raise ValueError(f"MIXGRAPH009: unsupported physical binding type for node {op.name!r}")

        for boundary in boundaries:
            from_stage = next(stage for stage in stages if stage["node_name"] == boundary["from_node"])
            to_stage = next(stage for stage in stages if stage["node_name"] == boundary["to_node"])
            boundary["from_backend"] = from_stage["backend"]
            boundary["to_backend"] = to_stage["backend"]
            boundary["data_width"] = data_width
            boundary["interface"] = "scalar_ready_valid_v1" if ready_valid else "scalar_valid_data_v1"
            boundary["physical_bridge"] = "direct_ready_valid_stream" if ready_valid else "direct_scalar_stream"

        wrapper = rtl / f"{request.top_name}.sv"
        wrapper_source = (
            _ready_valid_stage_wrapper(top_name=request.top_name, stages=stages, data_width=data_width)
            if ready_valid
            else _stage_wrapper(top_name=request.top_name, stages=stages, data_width=data_width)
        )
        wrapper.write_text(wrapper_source, encoding="utf-8")
        tb = sim / f"{request.top_name}_tb.sv"
        tb_source = (
            _ready_valid_tb_source(request.top_name, data_width, request.input_value, request.expected_output)
            if ready_valid
            else _tb_source(request.top_name, data_width, request.input_value, request.expected_output)
        )
        tb.write_text(tb_source, encoding="utf-8")
        read_lines.append(f'read_verilog -sv "./rtl/{wrapper.name}"')
        read_lines.append(f'add_files -fileset sim_1 -norecurse "./sim/{tb.name}"')

        tcl = project / "run_vivado.tcl"
        tcl.write_text(
            f"create_project -force fpgai_graph_mixed_backend ./vivado_proj -part {request.part}\n"
            + "\n".join(read_lines)
            + f'''\nset_property top {request.top_name} [current_fileset]\nset_property top {request.top_name}_tb [get_filesets sim_1]\nupdate_compile_order -fileset sources_1\nupdate_compile_order -fileset sim_1\nlaunch_simulation\nrun 300 ns\nclose_sim\nsynth_design -top {request.top_name} -part {request.part}\ncreate_clock -name clk -period {float(request.clock_period_ns):.6f} [get_ports clk]\nreport_utilization -file ../reports/graph_mixed_backend_utilization_synth.rpt\nreport_timing_summary -file ../reports/graph_mixed_backend_timing_synth.rpt\nexit\n''',
            encoding="utf-8",
        )

        segment_payload: list[dict[str, Any]] = []
        current_backend: str | None = None
        current_nodes: list[str] = []
        for stage in stages:
            if current_backend is None or stage["backend"] == current_backend:
                current_backend = stage["backend"]
                current_nodes.append(stage["node_name"])
            else:
                segment_payload.append({"backend": current_backend, "nodes": list(current_nodes)})
                current_backend = stage["backend"]
                current_nodes = [stage["node_name"]]
        if current_nodes:
            segment_payload.append({"backend": current_backend, "nodes": list(current_nodes)})

        report = reports / "graph_mixed_backend_physical.json"
        report_payload = {
            "schema": "fpgai.graph-mixed-backend-physical/v1",
            "status": "generated",
            "graph_name": str(getattr(request.graph, "name", "main")),
            "graph_input": list(getattr(request.graph, "inputs", ())),
            "graph_output": list(getattr(request.graph, "outputs", ())),
            "physical_profile": request.physical_profile,
            "data_width": data_width,
            "segments": segment_payload,
            "nodes": [
                {
                    "node": stage["node_name"],
                    "op_type": stage["op_type"],
                    "backend": stage["backend"],
                    "top": stage["top"],
                    "package_id": stage.get("package_id"),
                    "package_version": stage.get("package_version"),
                    "generated_or_staged_rtl": list(stage["source_files"]),
                }
                for stage in stages
            ],
            "boundaries": boundaries,
            "numeric_validation": {
                "input": int(request.input_value),
                "expected_output": int(request.expected_output),
                "comparison": "xsim_exact_signed_integer",
            },
            "artifacts": {
                "wrapper": str(wrapper),
                "testbench": str(tb),
                "run_tcl": str(tcl),
                "staged_rtl": staged_files,
            },
            "validation_level": "mixed_graph_rtl_project_generated",
            "support": {
                "hls_to_vhdl": any(b["from_backend"] == "vitis_hls" and b["to_backend"] == "vhdl" for b in boundaries),
                "vhdl_to_hls": any(b["from_backend"] == "vhdl" and b["to_backend"] == "vitis_hls" for b in boundaries),
                "multi_input": False,
                "multi_output": False,
                "backpressure": ready_valid,
            },
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }
        report.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return GraphMixedBackendPhysicalResult(True, project, wrapper, tb, tcl, report)
    except (ValueError, OSError) as exc:
        code, sep, message = str(exc).partition(": ")
        issue = GraphPhysicalIssue(code if sep else "MIXGRAPH000", "graph_physical", message if sep else str(exc))
        return GraphMixedBackendPhysicalResult(False, None, None, None, None, None, (issue,))


def run_graph_mixed_backend_physical_project(
    result: GraphMixedBackendPhysicalResult,
    *,
    vivado_executable: str = "vivado",
    timeout: int = 900,
) -> dict[str, Any]:
    if not result.ok or result.run_tcl is None:
        return {"schema": "fpgai.graph-mixed-backend-tool-result/v1", "status": "not_run", "returncode": None}
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
    stdout = reports / "graph_mixed_backend_vivado_stdout.log"
    stderr = reports / "graph_mixed_backend_vivado_stderr.log"
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    marker = "FPGAI_GRAPH_MIXED_BACKEND_SIM_PASS"
    sim_pass = marker in proc.stdout
    simulation_log: Path | None = None
    if not sim_pass:
        for candidate in sorted(cwd.glob("vivado_proj/**/*.log")):
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if marker in text:
                sim_pass = True
                simulation_log = candidate
                break
    util = reports / "graph_mixed_backend_utilization_synth.rpt"
    timing = reports / "graph_mixed_backend_timing_synth.rpt"
    synth_present = util.is_file() and timing.is_file()
    passed = proc.returncode == 0 and sim_pass and synth_present
    payload = {
        "schema": "fpgai.graph-mixed-backend-tool-result/v1",
        "status": "passed" if passed else "failed",
        "returncode": proc.returncode,
        "mixed_language_simulation_passed": sim_pass,
        "synthesis_reports_present": synth_present,
        "validation_level": "vivado_synthesized" if passed else ("rtl_simulated" if sim_pass else "mixed_graph_rtl_project_generated"),
        "stdout_log": str(stdout),
        "stderr_log": str(stderr),
        "simulation_log": str(simulation_log) if simulation_log else None,
        "utilization_report": str(util) if util.is_file() else None,
        "timing_report": str(timing) if timing.is_file() else None,
    }
    (reports / "graph_mixed_backend_tool_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
