from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

from fpgai.implementations.implementation_contract import ImplementationContract, resolve_architecture_parameters

from .abi import (
    VHDLScalarStreamABI,
    VHDLTensorPortsReadyValidABI,
    parse_vhdl_abi,
    parse_vhdl_scalar_stream_abi,
    parse_vhdl_tensor_ports_ready_valid_abi,
)


@dataclass(frozen=True)
class VHDLIntegrationIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class ExternalVHDLProjectRequest:
    out_dir: str | Path
    contract: ImplementationContract
    wrapper_top: str = "fpgai_vhdl_wrapper"
    part: str = "xck26-sfvc784-2LV-c"
    clock_period_ns: float = 5.0
    architecture: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExternalVHDLProjectResult:
    ok: bool
    rtl_dir: Path | None
    wrapper: Path | None
    run_tcl: Path | None
    report_path: Path | None
    copied_sources: tuple[Path, ...] = ()
    issues: tuple[VHDLIntegrationIssue, ...] = ()
    testbench: Path | None = None


def validate_vhdl_integration_contract(contract: ImplementationContract) -> tuple[VHDLIntegrationIssue, ...]:
    issues: list[VHDLIntegrationIssue] = []
    if contract.language != "vhdl":
        issues.append(VHDLIntegrationIssue("VHDLINT001", "language", "VHDL integration requires language=vhdl"))
    if contract.backend != "vhdl":
        issues.append(VHDLIntegrationIssue("VHDLINT002", "backend", "VHDL integration requires backend=vhdl"))
    if not all(str(source).lower().endswith((".vhd", ".vhdl")) for source in contract.sources):
        issues.append(
            VHDLIntegrationIssue("VHDLINT006", "sources", "VHDL sources must use .vhd/.vhdl extensions")
        )
    try:
        parse_vhdl_abi(contract)
    except ValueError as exc:
        code, _, message = str(exc).partition(": ")
        issues.append(VHDLIntegrationIssue(code or "VHDLINT003", "integration.vhdl", message or str(exc)))
    return tuple(issues)


def _safe(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("VHDLINT007: source escapes package root") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"VHDLINT008: invalid package source {relative}")
    return path


def _tb_source(top: str, abi: VHDLScalarStreamABI) -> str:
    hi = abi.data_width - 1
    data_type = "signed" if abi.signed else "std_logic_vector"
    if abi.abi == "scalar_ready_valid_v1":
        return f'''library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity {top}_tb is end entity;
architecture sim of {top}_tb is
  signal clk: std_logic := '0'; signal rst_n: std_logic := '0';
  signal input_valid: std_logic := '0'; signal input_ready: std_logic;
  signal input_data: {data_type}({hi} downto 0) := (others=>'0');
  signal output_valid: std_logic; signal output_ready: std_logic := '0';
  signal output_data: {data_type}({hi} downto 0);
begin
  clk <= not clk after 2.5 ns;
  dut: entity work.{top} port map(clk=>clk,rst_n=>rst_n,input_valid=>input_valid,input_ready=>input_ready,input_data=>input_data,output_valid=>output_valid,output_ready=>output_ready,output_data=>output_data);
  process
  begin
    wait for 10 ns; rst_n <= '1'; wait until rising_edge(clk);
    input_data <= to_signed(7,{abi.data_width}); input_valid <= '1';
    wait until rising_edge(clk) and input_ready='1'; input_valid <= '0';
    wait for 15 ns;
    assert output_valid='1' report "FPGAI VHDL ready/valid output_valid mismatch" severity failure;
    assert output_data=to_signed(7,{abi.data_width}) report "FPGAI VHDL ready/valid numeric mismatch" severity failure;
    output_ready <= '1'; wait until rising_edge(clk);
    report "FPGAI_VHDL_SIM_PASS" severity note;
    wait; end process;
end architecture;
'''
    return f'''library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity {top}_tb is end entity;
architecture sim of {top}_tb is
  signal clk: std_logic := '0'; signal rst_n: std_logic := '0';
  signal input_valid: std_logic := '0'; signal input_data: {data_type}({hi} downto 0) := (others=>'0');
  signal output_valid: std_logic; signal output_data: {data_type}({hi} downto 0);
begin
  clk <= not clk after 2.5 ns;
  dut: entity work.{top} port map(clk=>clk,rst_n=>rst_n,input_valid=>input_valid,input_data=>input_data,output_valid=>output_valid,output_data=>output_data);
  process
  begin
    wait for 10 ns; rst_n <= '1'; wait until rising_edge(clk);
    input_data <= to_signed(7,{abi.data_width}); input_valid <= '1'; wait until rising_edge(clk); input_valid <= '0';
    wait until rising_edge(clk);
    assert output_valid='1' report "FPGAI VHDL output_valid mismatch" severity failure;
    assert output_data=to_signed(7,{abi.data_width}) report "FPGAI VHDL numeric mismatch" severity failure;
    report "FPGAI_VHDL_SIM_PASS" severity note;
    wait; end process;
end architecture;
'''


def _vhdl_generic_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    raise ValueError(f"VHDLINT018: unsupported architecture generic value {value!r}; use integer/boolean FPGAI architecture values")


def _generic_map(contract: ImplementationContract, architecture: Mapping[str, Any] | None) -> str:
    values = resolve_architecture_parameters(contract, architecture).get("vhdl_generic", {})
    if not values:
        return ""
    assignments = ", ".join(f"{name}=>{_vhdl_generic_literal(value)}" for name, value in sorted(values.items()))
    return f" generic map ({assignments})"


def _scalar_wrapper_source(top: str, contract_top: str, abi: VHDLScalarStreamABI, *, generic_map: str = "") -> str:
    hi = abi.data_width - 1
    data_type = "signed" if abi.signed else "std_logic_vector"
    if abi.abi == "scalar_ready_valid_v1":
        return f'''library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity {top} is port (
  clk: in std_logic; rst_n: in std_logic;
  input_valid: in std_logic; input_ready: out std_logic;
  input_data: in {data_type}({hi} downto 0);
  output_valid: out std_logic; output_ready: in std_logic;
  output_data: out {data_type}({hi} downto 0)
); end entity;
architecture rtl of {top} is begin
  u_impl: entity work.{contract_top}{generic_map} port map (
    {abi.clock}=>clk, {abi.reset_n}=>rst_n,
    {abi.input_valid}=>input_valid, {abi.input_ready}=>input_ready, {abi.input_data}=>input_data,
    {abi.output_valid}=>output_valid, {abi.output_ready}=>output_ready, {abi.output_data}=>output_data
  );
end architecture;
'''
    return f'''library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity {top} is port (
  clk: in std_logic; rst_n: in std_logic;
  input_valid: in std_logic; input_data: in {data_type}({hi} downto 0);
  output_valid: out std_logic; output_data: out {data_type}({hi} downto 0)
); end entity;
architecture rtl of {top} is begin
  u_impl: entity work.{contract_top}{generic_map} port map (
    {abi.clock}=>clk, {abi.reset_n}=>rst_n,
    {abi.input_valid}=>input_valid, {abi.input_data}=>input_data,
    {abi.output_valid}=>output_valid, {abi.output_data}=>output_data
  );
end architecture;
'''


def emit_external_vhdl_operator_project(request: ExternalVHDLProjectRequest) -> ExternalVHDLProjectResult:
    issues = validate_vhdl_integration_contract(request.contract)
    if issues:
        return ExternalVHDLProjectResult(False, None, None, None, None, issues=issues)

    try:
        contract = request.contract
        parsed_abi = parse_vhdl_abi(contract)
        if isinstance(parsed_abi, VHDLTensorPortsReadyValidABI):
            raise ValueError(
                "VHDLINT017: standalone VHDL project emission currently supports scalar ABIs; "
                "tensor_ports_ready_valid_v1 is validated through graph physical composition"
            )
        abi = parsed_abi
        root = Path(request.out_dir)
        vhdl_dir = root / "vhdl"
        rtl_dir = vhdl_dir / "rtl"
        sim_dir = vhdl_dir / "sim"
        reports = root / "reports"
        rtl_dir.mkdir(parents=True, exist_ok=True)
        sim_dir.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        package_root = Path(contract.package_root)
        for index, relative in enumerate(contract.source_order or contract.sources):
            source = _safe(package_root, relative)
            destination = rtl_dir / f"{index:03d}_{source.name}"
            shutil.copy2(source, destination)
            copied.append(destination)

        wrapper = rtl_dir / f"{request.wrapper_top}.vhd"
        generic_map = _generic_map(contract, request.architecture)
        wrapper.write_text(_scalar_wrapper_source(request.wrapper_top, contract.top, abi, generic_map=generic_map), encoding="utf-8")
        testbench = sim_dir / f"{request.wrapper_top}_tb.vhd"
        testbench.write_text(_tb_source(request.wrapper_top, abi), encoding="utf-8")

        run_tcl = vhdl_dir / "run_vivado.tcl"
        read_lines = "\n".join(f'read_vhdl "./rtl/{path.name}"' for path in copied)
        run_tcl.write_text(
            f'''create_project -force fpgai_vhdl ./vivado_proj -part {request.part}
{read_lines}
read_vhdl "./rtl/{wrapper.name}"
add_files -fileset sim_1 -norecurse "./sim/{testbench.name}"
set_property top {request.wrapper_top} [current_fileset]
set_property top {request.wrapper_top}_tb [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation
run 50 ns
close_sim
synth_design -top {request.wrapper_top} -part {request.part}
report_utilization -file ../reports/utilization_synth.rpt
report_timing_summary -file ../reports/timing_synth.rpt
exit
''',
            encoding="utf-8",
        )

        report_path = reports / "external_vhdl_integration.json"
        payload = {
            "schema": "fpgai.external-vhdl-integration/v2",
            "status": "generated",
            "abi": abi.__dict__,
            "implementation": contract.to_dict(),
            "artifacts": {
                "wrapper": str(wrapper),
                "testbench": str(testbench),
                "run_tcl": str(run_tcl),
                "sources": [str(path) for path in copied],
            },
            "architecture_parameters": resolve_architecture_parameters(contract, request.architecture),
            "validation_level": "simulation_and_synthesis_project_generated",
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return ExternalVHDLProjectResult(
            True,
            rtl_dir,
            wrapper,
            run_tcl,
            report_path,
            tuple(copied),
            testbench=testbench,
        )
    except ValueError as exc:
        code, _, message = str(exc).partition(": ")
        return ExternalVHDLProjectResult(
            False,
            None,
            None,
            None,
            None,
            issues=(VHDLIntegrationIssue(code or "VHDLINT009", "integration", message or str(exc)),),
        )


def run_external_vhdl_project(
    result: ExternalVHDLProjectResult,
    *,
    vivado_executable: str = "vivado",
    timeout: int = 600,
) -> dict[str, Any]:
    if not result.ok or result.run_tcl is None:
        return {"schema": "fpgai.external-vhdl-tool-result/v1", "status": "not_run", "returncode": None}

    cwd = result.run_tcl.parent
    proc = subprocess.run(
        [vivado_executable, "-mode", "batch", "-source", result.run_tcl.name],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    reports = cwd.parent / "reports"
    stdout = reports / "vivado_stdout.log"
    stderr = reports / "vivado_stderr.log"
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")

    marker = "FPGAI_VHDL_SIM_PASS"
    simulation_log: Path | None = None
    simulation_passed = marker in proc.stdout
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

    utilization = reports / "utilization_synth.rpt"
    timing = reports / "timing_synth.rpt"
    synthesis_reports_present = utilization.is_file() and timing.is_file()
    passed = proc.returncode == 0 and simulation_passed and synthesis_reports_present
    payload = {
        "schema": "fpgai.external-vhdl-tool-result/v2",
        "status": "passed" if passed else "failed",
        "returncode": proc.returncode,
        "rtl_simulation_passed": simulation_passed,
        "synthesis_reports_present": synthesis_reports_present,
        "validation_level": (
            "vivado_synthesized"
            if passed
            else ("rtl_simulated" if simulation_passed else "project_generated")
        ),
        "stdout_log": str(stdout),
        "stderr_log": str(stderr),
        "simulation_log": str(simulation_log) if simulation_log else None,
    }
    (reports / "external_vhdl_tool_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "VHDLIntegrationIssue",
    "VHDLScalarStreamABI",
    "VHDLTensorPortsReadyValidABI",
    "ExternalVHDLProjectRequest",
    "ExternalVHDLProjectResult",
    "parse_vhdl_abi",
    "parse_vhdl_scalar_stream_abi",
    "parse_vhdl_tensor_ports_ready_valid_abi",
    "validate_vhdl_integration_contract",
    "emit_external_vhdl_operator_project",
    "run_external_vhdl_project",
]
