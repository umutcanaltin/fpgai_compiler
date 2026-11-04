"""Pure Vitis-HLS Tcl construction: no vendor invocation, no file writes.

Separated from testbench generation and XML report ingestion. Exact script
bytes remain stable for identical arguments. Original-HLS provenance is
still checked by rl_environment.timing against executed script hashes.
"""
from __future__ import annotations

def make_vitis_tcl(part: str = "xck26-sfvc784-2LV-c", clock_ns: float = 5.0,
                   run_csim: bool = True, run_csynth: bool = True,
                   clock_uncertainty_ns: float | None = None) -> str:
    if not part or any(x in part for x in (" ", "\n", "\r", ";", "{", "}")):
        raise ValueError("Invalid part")
    from math import isfinite
    if not isfinite(clock_ns) or clock_ns <= 0:
        raise ValueError("Clock period must be finite and positive")
    if clock_uncertainty_ns is not None and (
            type(clock_uncertainty_ns) not in (int, float)
            or not isfinite(clock_uncertainty_ns)
            or not 0. <= clock_uncertainty_ns < clock_ns):
        raise ValueError("Explicit HLS clock uncertainty must be within [0, clock)")
    commands = ["open_project -reset fpgai_hls_project", "set_top fpgai_model",
                "add_files model.cpp", "add_files -tb tb.cpp", "open_solution -reset solution1",
                f"set_part {{{part}}}", f"create_clock -period {clock_ns:g} -name default"]
    if clock_uncertainty_ns is not None:
        # The 2023.2 Tcl command is executed under the opened solution, so
        # timing evidence never relies on silently presumed default margins.
        commands.append(f"set_clock_uncertainty {clock_uncertainty_ns:.3f}")
    if run_csim:
        # Numerical rejection must not suppress independent synthesis measurements.
        commands.extend((
            'set fpgai_csim_failed [catch {csim_design} fpgai_csim_reason]',
            'if {$fpgai_csim_failed} { puts "FPGAI_CSIM_STATUS=failed $fpgai_csim_reason" } else { puts "FPGAI_CSIM_STATUS=completed" }',
        ))
    if run_csynth:
        commands.append("csynth_design")
    commands.append("exit")
    return "\n".join(commands) + "\n"

