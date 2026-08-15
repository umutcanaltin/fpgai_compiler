from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Sequence

from fpgai.backends.vivado.boards import get_board


@dataclass(frozen=True)
class ClockProbePoint:
    requested_mhz: float
    actual_mhz: float | None
    actual_period_ns: float | None
    status: str
    project_dir: str
    returncode: int | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mhz": self.requested_mhz,
            "actual_mhz": self.actual_mhz,
            "actual_period_ns": self.actual_period_ns,
            "status": self.status,
            "project_dir": self.project_dir,
            "returncode": self.returncode,
            "failure_reason": self.failure_reason,
        }


def _probe_tcl(*, board_name: str, requested_mhz: float) -> str:
    board = get_board(board_name)
    board_part = board.board_part or ""
    lines = [
        f"create_project -force fpgai_clock_probe ./vivado_proj -part {board.part}",
    ]
    if board_part:
        lines.append(f"catch {{set_property board_part {board_part} [current_project]}} board_part_msg")
        lines.append('puts "FPGAI-CLOCK-PROBE board-part: $board_part_msg"')
    lines.append("create_bd_design clock_probe_bd")
    if board.family == "zynqmp":
        lines += [
            "create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0",
            'catch {apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells zynq_ultra_ps_e_0]} ps_auto_msg',
            'puts "FPGAI-CLOCK-PROBE ps-auto: $ps_auto_msg"',
            f'if {{[catch {{set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {{0}} CONFIG.PSU__USE__S_AXI_GP0 {{0}} CONFIG.PSU__FPGA_PL0_ENABLE {{1}} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {{{requested_mhz:g}}}] [get_bd_cells zynq_ultra_ps_e_0]}} cfg_msg]}} {{ puts stderr "FPGAI-CLOCK-PROBE ERROR: PS clock configuration failed: $cfg_msg"; exit 21 }}',
            # Board presets can leave PS AXI clock pins enabled even in a clock-only probe.
            # Vivado validation requires every enabled ACLK pin to have a valid source, so
            # mirror the production bridge and fan the requested PL clock into any
            # remaining unconnected PS clock inputs before validating the BD.
            'set probe_clk [get_bd_pins -quiet zynq_ultra_ps_e_0/pl_clk0]',
            'if {$probe_clk eq ""} { puts stderr "FPGAI-CLOCK-PROBE ERROR: ZynqMP pl_clk0 pin not found"; exit 25 }',
            'foreach pin [concat [get_bd_pins -quiet zynq_ultra_ps_e_0/*aclk] [get_bd_pins -quiet zynq_ultra_ps_e_0/*ACLK]] { if {$pin ne "" && [llength [get_bd_nets -quiet -of_objects $pin]] == 0} { catch {connect_bd_net $probe_clk $pin} fanout_msg } }',
            'catch {make_bd_intf_pins_external [get_bd_intf_pins zynq_ultra_ps_e_0/DDR]} ddr_ext_msg',
            'catch {make_bd_intf_pins_external [get_bd_intf_pins zynq_ultra_ps_e_0/FIXED_IO]} fixed_ext_msg',
            'puts "FPGAI-CLOCK-PROBE DDR external: $ddr_ext_msg"',
            'puts "FPGAI-CLOCK-PROBE FIXED_IO external: $fixed_ext_msg"',
        ]
    elif board.family == "zynq7000":
        lines += [
            "create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* processing_system7_0",
            'catch {apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} [get_bd_cells processing_system7_0]} ps_auto_msg',
            'puts "FPGAI-CLOCK-PROBE ps-auto: $ps_auto_msg"',
            f'if {{[catch {{set_property -dict [list CONFIG.PCW_EN_CLK0_PORT {{1}} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {{{requested_mhz:g}}}] [get_bd_cells processing_system7_0]}} cfg_msg]}} {{ puts stderr "FPGAI-CLOCK-PROBE ERROR: PS clock configuration failed: $cfg_msg"; exit 21 }}',
        ]
    else:
        raise ValueError(f"Clock probe does not support board family {board.family!r}")
    lines += [
        # Mirror the known-working PS portion of the real FPGAI bridge.  Kria/ZynqMP
        # target generation expects a structurally valid PS design with the board-facing
        # interfaces materialized even though no accelerator is present in this probe.
        'if {[catch {validate_bd_design} validate_msg]} { puts stderr "FPGAI-CLOCK-PROBE ERROR: validate_bd_design failed: $validate_msg"; exit 24 }',
        "save_bd_design",
        "set bd_file [get_files -quiet */clock_probe_bd.bd]",
        "if {[llength $bd_file] == 0} { set bd_file [get_files -quiet clock_probe_bd.bd] }",
        "if {[llength $bd_file] == 0} { puts stderr {FPGAI-CLOCK-PROBE ERROR: generated BD file not found}; exit 22 }",
        'if {[catch {generate_target all $bd_file} gen_msg]} { puts stderr \"FPGAI-CLOCK-PROBE ERROR: generate_target failed: $gen_msg\"; exit 23 }',
        "puts {FPGAI-CLOCK-PROBE TARGET-GENERATED}",
        "close_project",
        "exit 0",
    ]
    return "\n".join(lines) + "\n"


def _parse_generated_period(project_dir: Path) -> float | None:
    patterns = (
        re.compile(r'create_clock\s+-name\s+clk_pl_0\s+-period\s+"?([0-9.]+)"?'),
        re.compile(r'create_clock[^\n]*-period\s+"?([0-9.]+)"?[^\n]*(?:FCLK_CLK0|PLCLK\[0\])'),
    )
    for path in sorted(project_dir.glob("**/*.xdc")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pat in patterns:
            match = pat.search(text)
            if match:
                try:
                    period = float(match.group(1))
                except ValueError:
                    continue
                if period > 0:
                    return period
    return None


def _failure_reason(stdout: str, stderr: str) -> str | None:
    combined = "\n".join(part for part in (stderr, stdout) if part)
    preferred = []
    fallback = []
    for raw in combined.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "FPGAI-CLOCK-PROBE ERROR:" in line:
            preferred.append(line.split("FPGAI-CLOCK-PROBE ERROR:", 1)[1].strip())
        elif line.startswith("ERROR:"):
            fallback.append(line)
    if preferred:
        # Keep the probe-owned error plus nearby native Vivado errors when available.
        context = fallback[-4:] + [preferred[-1]]
        return " | ".join(context)
    if fallback:
        return " | ".join(fallback[-5:])
    return None


def run_clock_probe(
    *,
    board_name: str,
    requested_mhz: float,
    out_dir: str | Path,
    vivado_executable: str = "vivado",
    timeout: int = 300,
) -> ClockProbePoint:
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tcl = root / "probe_clock.tcl"
    tcl.write_text(_probe_tcl(board_name=board_name, requested_mhz=requested_mhz), encoding="utf-8")
    exe = shutil.which(vivado_executable) or vivado_executable
    try:
        proc = subprocess.run(
            [exe, "-mode", "batch", "-source", tcl.name],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ClockProbePoint(float(requested_mhz), None, None, "tool_failed", str(root), None, "Vivado process could not be started or timed out")
    (root / "vivado_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (root / "vivado_stderr.log").write_text(proc.stderr, encoding="utf-8")
    period = _parse_generated_period(root)
    actual = 1000.0 / period if period else None
    status = "probed" if proc.returncode == 0 and actual is not None else "tool_failed"
    reason = None if status == "probed" else _failure_reason(proc.stdout, proc.stderr)
    if status == "tool_failed" and proc.returncode == 0 and actual is None and reason is None:
        reason = "Vivado completed but no generated PS clock constraint was found"
    return ClockProbePoint(float(requested_mhz), actual, period, status, str(root), proc.returncode, reason)


def choose_realizable_clock(
    requested_mhz: float,
    realizable_mhz: Sequence[float],
    *,
    policy: str = "exact_only",
    tolerance_percent: float = 1.0,
) -> float | None:
    values = sorted({float(v) for v in realizable_mhz if float(v) > 0})
    if not values:
        return None
    requested = float(requested_mhz)
    tol = max(0.05, abs(requested) * float(tolerance_percent) / 100.0)
    exact = [v for v in values if abs(v - requested) <= tol]
    if policy == "exact_only":
        return min(exact, key=lambda v: abs(v - requested)) if exact else None
    if policy == "nearest":
        return min(values, key=lambda v: (abs(v - requested), v))
    if policy == "nearest_below":
        below = [v for v in values if v <= requested + tol]
        return max(below) if below else None
    raise ValueError("clock policy must be exact_only, nearest, or nearest_below")


def summarize_clock_probes(points: Iterable[ClockProbePoint]) -> dict[str, Any]:
    rows = list(points)
    actuals = sorted({round(float(p.actual_mhz), 6) for p in rows if p.actual_mhz is not None})
    aliases: dict[str, list[float]] = {}
    for actual in actuals:
        aliases[f"{actual:g}"] = [
            p.requested_mhz for p in rows
            if p.actual_mhz is not None and abs(p.actual_mhz - actual) <= max(0.05, actual * 1e-6)
        ]
    return {
        "schema": "fpgai.board-clock-probe/v1",
        "status": "passed" if actuals else "failed",
        "points": [p.to_dict() for p in rows],
        "realizable_mhz": actuals,
        "request_aliases": aliases,
        "measurement_semantics": (
            "Realizable frequencies are extracted from Vivado-generated PS clock constraints before HLS/Vivado implementation. "
            "They are board/toolchain observations, not generic device guarantees."
        ),
    }


def write_clock_probe_report(points: Iterable[ClockProbePoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summarize_clock_probes(points), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def realizable_targets_from_probes(points: Iterable[ClockProbePoint], *, precision_digits: int = 6) -> tuple[float, ...]:
    values = {
        round(float(p.actual_mhz), precision_digits)
        for p in points
        if p.status == "probed" and p.actual_mhz is not None and p.actual_mhz > 0
    }
    return tuple(sorted(values))
