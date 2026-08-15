from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from fpgai.implementations.implementation_contract import ImplementationContract
from fpgai.implementations.vhdl_integration.integration import (
    parse_vhdl_scalar_stream_abi,
    validate_vhdl_integration_contract,
)


@dataclass(frozen=True)
class MixedBackendPhysicalIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class MixedBackendPhysicalRequest:
    out_dir: str | Path
    hls_rtl_dir: str | Path
    hls_top: str
    vhdl_contract: ImplementationContract
    part: str = "xck26-sfvc784-2LV-c"
    top_name: str = "fpgai_mixed_backend_top"
    clock_period_ns: float = 5.0


@dataclass(frozen=True)
class MixedBackendPhysicalResult:
    ok: bool
    project_dir: Path | None
    wrapper: Path | None
    testbench: Path | None
    run_tcl: Path | None
    report_path: Path | None
    issues: tuple[MixedBackendPhysicalIssue, ...] = ()


def _find_hls_top(rtl_dir: Path, top: str) -> Path | None:
    for suffix in (".v", ".sv"):
        candidate = rtl_dir / f"{top}{suffix}"
        if candidate.is_file():
            return candidate
    for candidate in sorted(rtl_dir.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in {".v", ".sv"} and candidate.stem == top:
            return candidate
    return None


def _verilog_ports(path: Path, module: str) -> dict[str, tuple[str, int]]:
    """Parse simple HLS-generated Verilog port declarations.

    Vitis HLS emits non-ANSI declarations for its RTL top in supported versions,
    e.g. `input ap_clk;` and `output [15:0] output_data;`.  We only need the
    maintained scalar validation kernel's well-defined ports here.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if not re.search(rf"\bmodule\s+{re.escape(module)}\b", text):
        raise ValueError(f"MIXRTL003: HLS top module {module!r} not found in {path.name}")
    ports: dict[str, tuple[str, int]] = {}
    decl = re.compile(
        r"\b(input|output|inout)\s+(?:wire\s+|reg\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*;"
    )
    for direction, msb, lsb, name in decl.findall(text):
        width = abs(int(msb) - int(lsb)) + 1 if msb and lsb else 1
        ports[name] = (direction, width)
    return ports


def _pick_port(ports: dict[str, tuple[str, int]], candidates: tuple[str, ...], *, direction: str | None = None) -> str | None:
    for name in candidates:
        info = ports.get(name)
        if info is not None and (direction is None or info[0] == direction):
            return name
    return None


def _axis_groups(ports: dict[str, tuple[str, int]]) -> dict[str, dict[str, str]]:
    """Return AXI-stream-like TDATA/TVALID/TREADY groups keyed by RTL prefix.

    Vitis HLS may rewrite C/C++ argument names in generated RTL (for example
    ``input`` -> ``input_r``).  Physical composition therefore discovers an
    interface from the AXIS signal triplet and port directions instead of
    depending on an exact argument prefix.
    """
    groups: dict[str, dict[str, str]] = {}
    for name in ports:
        upper = name.upper()
        for suffix, field in (("_TDATA", "data"), ("_TVALID", "valid"), ("_TREADY", "ready")):
            if upper.endswith(suffix):
                prefix = name[: -len(suffix)]
                groups.setdefault(prefix, {})[field] = name
                break
    return groups


def _axis_group_by_direction(
    ports: dict[str, tuple[str, int]], *, sink_from_wrapper: bool
) -> dict[str, str] | None:
    """Find one complete AXIS group by signal directions at the HLS boundary.

    ``sink_from_wrapper=True`` selects an HLS input stream: TDATA/TVALID are
    RTL inputs and TREADY is an RTL output.  False selects an HLS output
    stream with the opposite handshake directions.
    """
    expected = ("input", "input", "output") if sink_from_wrapper else ("output", "output", "input")
    matches: list[tuple[str, dict[str, str]]] = []
    for prefix, group in _axis_groups(ports).items():
        if set(group) != {"data", "valid", "ready"}:
            continue
        directions = (
            ports[group["data"]][0],
            ports[group["valid"]][0],
            ports[group["ready"]][0],
        )
        if directions == expected:
            matches.append((prefix, group))
    if len(matches) == 1:
        return matches[0][1]
    if not matches:
        return None
    # Maintained scalar profile has one input and one output stream.  Ambiguous
    # multi-stream RTL must be handled by the future multi-port ABI, not guessed.
    raise ValueError(
        "MIXRTL009: HLS RTL contains multiple AXI-stream groups compatible with the scalar ready/valid profile: "
        + ", ".join(prefix for prefix, _ in matches)
    )


def _axis_groups_by_direction(
    ports: dict[str, tuple[str, int]], *, sink_from_wrapper: bool
) -> dict[str, dict[str, str]]:
    """Return every complete AXI-stream group matching one interface direction."""
    expected = ("input", "input", "output") if sink_from_wrapper else ("output", "output", "input")
    matches: dict[str, dict[str, str]] = {}
    for prefix, group in _axis_groups(ports).items():
        if set(group) != {"data", "valid", "ready"}:
            continue
        directions = (
            ports[group["data"]][0],
            ports[group["valid"]][0],
            ports[group["ready"]][0],
        )
        if directions == expected:
            matches[prefix] = group
    return matches


def _select_axis_groups(
    ports: dict[str, tuple[str, int]],
    *,
    sink_from_wrapper: bool,
    requested_prefixes: tuple[str, ...],
    expected_count: int,
    path: str,
) -> tuple[dict[str, str], ...]:
    groups = _axis_groups_by_direction(ports, sink_from_wrapper=sink_from_wrapper)
    if requested_prefixes:
        if len(requested_prefixes) != expected_count:
            raise ValueError(
                f"MIXRTL010: {path} declares {len(requested_prefixes)} AXI-stream prefixes for {expected_count} graph ports"
            )
        selected: list[dict[str, str]] = []
        resolved_prefixes: list[str] = []
        for requested in requested_prefixes:
            candidates = [
                prefix
                for prefix in groups
                if prefix == requested
                or prefix.rstrip("_r") == requested
                or prefix.startswith(requested + "_")
            ]
            if len(candidates) != 1:
                available = ", ".join(sorted(groups)) or "<none>"
                raise ValueError(
                    f"MIXRTL011: {path} AXI-stream prefix {requested!r} resolves to {len(candidates)} groups; "
                    f"available prefixes: {available}"
                )
            resolved = candidates[0]
            resolved_prefixes.append(resolved)
            selected.append(groups[resolved])
        if len(set(resolved_prefixes)) != len(resolved_prefixes):
            raise ValueError(f"MIXRTL012: {path} resolves multiple logical ports to the same AXI-stream prefix")
        return tuple(selected)
    if expected_count == 1 and len(groups) == 1:
        return (next(iter(groups.values())),)
    if len(groups) == expected_count == 1:
        return (next(iter(groups.values())),)
    raise ValueError(
        f"MIXRTL013: {path} requires explicit AXI-stream prefix mapping for {expected_count} graph ports; "
        f"discovered prefixes: {', '.join(sorted(groups)) or '<none>'}"
    )


def _multi_ready_valid_hls_ports(
    ports: dict[str, tuple[str, int]],
    *,
    input_prefixes: tuple[str, ...],
    output_prefixes: tuple[str, ...],
    input_count: int,
    output_count: int,
) -> dict[str, object]:
    inputs = _select_axis_groups(
        ports,
        sink_from_wrapper=True,
        requested_prefixes=input_prefixes,
        expected_count=input_count,
        path="input_streams",
    )
    outputs = _select_axis_groups(
        ports,
        sink_from_wrapper=False,
        requested_prefixes=output_prefixes,
        expected_count=output_count,
        path="output_streams",
    )
    return {
        "clock": _pick_port(ports, ("ap_clk", "clk"), direction="input"),
        "reset": _pick_port(ports, ("ap_rst", "ap_rst_n", "rst", "rst_n"), direction="input"),
        "inputs": inputs,
        "outputs": outputs,
    }


def _ready_valid_hls_ports(ports: dict[str, tuple[str, int]]) -> dict[str, str]:
    input_axis = _axis_group_by_direction(ports, sink_from_wrapper=True)
    output_axis = _axis_group_by_direction(ports, sink_from_wrapper=False)
    mapping = {
        "clock": _pick_port(ports, ("ap_clk", "clk"), direction="input"),
        "reset": _pick_port(ports, ("ap_rst", "ap_rst_n", "rst", "rst_n"), direction="input"),
        "input_data": input_axis["data"] if input_axis else None,
        "input_valid": input_axis["valid"] if input_axis else None,
        "input_ready": input_axis["ready"] if input_axis else None,
        "output_data": output_axis["data"] if output_axis else None,
        "output_valid": output_axis["valid"] if output_axis else None,
        "output_ready": output_axis["ready"] if output_axis else None,
    }
    missing = [key for key, value in mapping.items() if value is None and key not in {"clock", "reset"}]
    if missing:
        available = ", ".join(sorted(ports))
        raise ValueError(
            f"MIXRTL008: HLS RTL is missing required ready/valid bridge ports: {', '.join(missing)}; "
            f"available RTL ports: {available}"
        )
    return {key: value for key, value in mapping.items() if value is not None}


def _required_hls_ports(ports: dict[str, tuple[str, int]]) -> dict[str, str]:
    mapping = {
        "clock": _pick_port(ports, ("ap_clk", "clk"), direction="input"),
        "reset": _pick_port(ports, ("ap_rst", "ap_rst_n", "rst", "rst_n"), direction="input"),
        "input_data": _pick_port(ports, ("input_data",), direction="input"),
        "input_valid": _pick_port(ports, ("input_valid",), direction="input"),
        "output_data": _pick_port(ports, ("output_data",), direction="output"),
        "output_valid": _pick_port(ports, ("output_valid",), direction="output"),
    }
    missing = [
        key for key, value in mapping.items()
        if value is None and key not in {"clock", "reset"}
    ]
    if missing:
        raise ValueError(f"MIXRTL004: HLS RTL is missing required scalar bridge ports: {', '.join(missing)}")
    return {key: value for key, value in mapping.items() if value is not None}


def _safe_package_source(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    resolved = root.resolve()
    try:
        candidate.relative_to(resolved)
    except ValueError as exc:
        raise ValueError(f"MIXRTL006: VHDL source escapes package root: {rel}") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"MIXRTL007: invalid VHDL package source: {rel}")
    return candidate


def _wrapper_source(
    *, top_name: str, hls_top: str, hls_ports: dict[str, str], hls_port_info: dict[str, tuple[str, int]], vhdl_top: str, data_width: int
) -> str:
    hls_data_width = hls_port_info[hls_ports["output_data"]][1]
    if hls_data_width != data_width:
        raise ValueError(
            f"MIXRTL005: HLS output width {hls_data_width} does not match VHDL scalar_stream_v1 width {data_width}"
        )
    clock_connection = ""
    if "clock" in hls_ports:
        clock_connection = f"    .{hls_ports['clock']}(clk),\n"
    reset_connection = ""
    if "reset" in hls_ports:
        reset_name = hls_ports["reset"]
        reset_expr = "rst_n" if reset_name.endswith("_n") else "~rst_n"
        reset_connection = f"    .{reset_name}({reset_expr}),\n"
    return f'''module {top_name}(
    input  wire clk,
    input  wire rst_n,
    input  wire input_valid,
    input  wire signed [{data_width-1}:0] input_data,
    output wire output_valid,
    output wire signed [{data_width-1}:0] output_data
);
  wire signed [{data_width-1}:0] hls_data;
  wire hls_valid;

  {hls_top} u_hls (
{clock_connection}{reset_connection}    .{hls_ports['input_data']}(input_data),
    .{hls_ports['input_valid']}(input_valid),
    .{hls_ports['output_data']}(hls_data),
    .{hls_ports['output_valid']}(hls_valid)
  );

  {vhdl_top} u_vhdl (
    .clk(clk),
    .rst_n(rst_n),
    .input_valid(hls_valid),
    .input_data(hls_data),
    .output_valid(output_valid),
    .output_data(output_data)
  );
endmodule
'''


def _testbench_source(top_name: str, data_width: int) -> str:
    return f'''`timescale 1ns/1ps
module {top_name}_tb;
  reg clk = 0;
  reg rst_n = 0;
  reg input_valid = 0;
  reg signed [{data_width-1}:0] input_data = 0;
  wire output_valid;
  wire signed [{data_width-1}:0] output_data;
  integer cycles = 0;

  always #2.5 clk = ~clk;
  {top_name} dut(.clk(clk), .rst_n(rst_n), .input_valid(input_valid), .input_data(input_data), .output_valid(output_valid), .output_data(output_data));

  initial begin
    repeat (3) @(posedge clk);
    rst_n <= 1;
    @(posedge clk);
    input_data <= 7;
    input_valid <= 1;
    @(posedge clk);
    input_valid <= 0;
    while (!output_valid && cycles < 20) begin
      @(posedge clk);
      cycles = cycles + 1;
    end
    if (!output_valid) $fatal(1, "FPGAI mixed-backend output_valid timeout");
    if ($signed(output_data) !== 14) $fatal(1, "FPGAI mixed-backend numeric mismatch: expected 14 got %0d", $signed(output_data));
    $display("FPGAI_MIXED_BACKEND_SIM_PASS");
    #5;
    $finish;
  end
endmodule
'''


def emit_mixed_backend_physical_project(request: MixedBackendPhysicalRequest) -> MixedBackendPhysicalResult:
    issues = validate_vhdl_integration_contract(request.vhdl_contract)
    if issues:
        converted = tuple(MixedBackendPhysicalIssue(i.code, i.path, i.message) for i in issues)
        return MixedBackendPhysicalResult(False, None, None, None, None, None, converted)
    try:
        rtl_source_dir = Path(request.hls_rtl_dir).expanduser().resolve()
        hls_top_path = _find_hls_top(rtl_source_dir, request.hls_top)
        if hls_top_path is None:
            raise ValueError(f"MIXRTL002: HLS RTL top {request.hls_top!r} not found under {rtl_source_dir}")
        ports = _verilog_ports(hls_top_path, request.hls_top)
        hls_ports = _required_hls_ports(ports)
        abi = parse_vhdl_scalar_stream_abi(request.vhdl_contract)

        root = Path(request.out_dir).expanduser().resolve()
        project = root / "mixed_backend"
        rtl = project / "rtl"
        sim = project / "sim"
        reports = root / "reports"
        rtl.mkdir(parents=True, exist_ok=True)
        sim.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)

        staged_hls: list[Path] = []
        for src in sorted(rtl_source_dir.rglob("*")):
            if src.is_file() and src.suffix.lower() in {".v", ".sv", ".vhd", ".vhdl"}:
                dst = rtl / f"hls_{src.name}"
                shutil.copy2(src, dst)
                staged_hls.append(dst)

        staged_vhdl: list[Path] = []
        package_root = Path(request.vhdl_contract.package_root)
        for index, rel in enumerate(request.vhdl_contract.source_order or request.vhdl_contract.sources):
            src = _safe_package_source(package_root, rel)
            dst = rtl / f"vhdl_{index:03d}_{src.name}"
            shutil.copy2(src, dst)
            staged_vhdl.append(dst)

        wrapper = rtl / f"{request.top_name}.sv"
        wrapper.write_text(
            _wrapper_source(
                top_name=request.top_name,
                hls_top=request.hls_top,
                hls_ports=hls_ports,
                hls_port_info=ports,
                vhdl_top=request.vhdl_contract.top,
                data_width=abi.data_width,
            ),
            encoding="utf-8",
        )
        tb = sim / f"{request.top_name}_tb.sv"
        tb.write_text(_testbench_source(request.top_name, abi.data_width), encoding="utf-8")

        read_lines: list[str] = []
        for path in staged_hls:
            command = "read_vhdl" if path.suffix.lower() in {".vhd", ".vhdl"} else "read_verilog"
            read_lines.append(f'{command} "./rtl/{path.name}"')
        for path in staged_vhdl:
            read_lines.append(f'read_vhdl "./rtl/{path.name}"')
        read_lines.append(f'read_verilog -sv "./rtl/{wrapper.name}"')
        read_lines.append(f'add_files -fileset sim_1 -norecurse "./sim/{tb.name}"')
        tcl = project / "run_vivado.tcl"
        tcl.write_text(
            f'''create_project -force fpgai_mixed_backend ./vivado_proj -part {request.part}\n'''
            + "\n".join(read_lines)
            + f'''\nset_property top {request.top_name} [current_fileset]\nset_property top {request.top_name}_tb [get_filesets sim_1]\nupdate_compile_order -fileset sources_1\nupdate_compile_order -fileset sim_1\nlaunch_simulation\nrun 200 ns\nclose_sim\nsynth_design -top {request.top_name} -part {request.part}\ncreate_clock -name clk -period {float(request.clock_period_ns):.6f} [get_ports clk]\nreport_utilization -file ../reports/mixed_backend_utilization_synth.rpt\nreport_timing_summary -file ../reports/mixed_backend_timing_synth.rpt\nexit\n''',
            encoding="utf-8",
        )

        report = reports / "mixed_backend_physical_integration.json"
        payload = {
            "schema": "fpgai.mixed-backend-physical/v1",
            "status": "generated",
            "hls": {"top": request.hls_top, "rtl_top": str(hls_top_path), "ports": hls_ports},
            "vhdl": {"top": request.vhdl_contract.top, "package_id": request.vhdl_contract.package_id, "abi": abi.abi},
            "bridge": {"data_width": abi.data_width, "handshake": "valid_data", "order": ["hls", "vhdl"]},
            "artifacts": {"wrapper": str(wrapper), "testbench": str(tb), "run_tcl": str(tcl)},
            "validation_level": "mixed_rtl_project_generated",
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }
        report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return MixedBackendPhysicalResult(True, project, wrapper, tb, tcl, report)
    except ValueError as exc:
        code, _, message = str(exc).partition(": ")
        issue = MixedBackendPhysicalIssue(code or "MIXRTL001", "physical", message or str(exc))
        return MixedBackendPhysicalResult(False, None, None, None, None, None, (issue,))


def run_mixed_backend_physical_project(
    result: MixedBackendPhysicalResult, *, vivado_executable: str = "vivado", timeout: int = 900
) -> dict[str, Any]:
    if not result.ok or result.run_tcl is None:
        return {"schema": "fpgai.mixed-backend-tool-result/v1", "status": "not_run", "returncode": None}
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
    stdout = reports / "mixed_backend_vivado_stdout.log"
    stderr = reports / "mixed_backend_vivado_stderr.log"
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    marker = "FPGAI_MIXED_BACKEND_SIM_PASS"
    simulation_log: Path | None = None
    sim_pass = marker in proc.stdout
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
    util = reports / "mixed_backend_utilization_synth.rpt"
    timing = reports / "mixed_backend_timing_synth.rpt"
    synth_present = util.is_file() and timing.is_file()
    passed = proc.returncode == 0 and sim_pass and synth_present
    payload = {
        "schema": "fpgai.mixed-backend-tool-result/v1",
        "status": "passed" if passed else "failed",
        "returncode": proc.returncode,
        "mixed_language_simulation_passed": sim_pass,
        "synthesis_reports_present": synth_present,
        "validation_level": "vivado_synthesized" if passed else ("rtl_simulated" if sim_pass else "mixed_rtl_project_generated"),
        "stdout_log": str(stdout),
        "stderr_log": str(stderr),
        "simulation_log": str(simulation_log) if simulation_log else None,
        "utilization_report": str(util) if util.is_file() else None,
        "timing_report": str(timing) if timing.is_file() else None,
    }
    (reports / "mixed_backend_tool_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
