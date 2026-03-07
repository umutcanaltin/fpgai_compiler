from __future__ import annotations

from pathlib import Path
from typing import List


def emit_run_hls_tcl(
    *,
    top_name: str,
    proj_name: str,
    part: str,
    clk_mhz: int,
    include_dirs: List[Path],
    design_files: List[Path],
    tb_files: List[Path],
) -> str:
    inc = " ".join([f"-I{p.as_posix()}" for p in include_dirs])

    def add_design(p: Path) -> str:
        # IMPORTANT: no -quiet (unsupported in Vitis HLS add_files)
        return f'add_files "{p.as_posix()}" -cflags "{inc}"'

    def add_tb(p: Path) -> str:
        # -tb is supported; no -quiet
        return f'add_files -tb "{p.as_posix()}" -cflags "{inc}"'

    lines = []
    lines.append(f"open_project -reset {proj_name}")
    lines.append(f"set_top {top_name}")

    for f in design_files:
        lines.append(add_design(f))

    for f in tb_files:
        lines.append(add_tb(f))

    lines.append("open_solution -reset sol1")
    lines.append(f"set_part {part}")
    lines.append(f"create_clock -period {1000.0/float(clk_mhz):.6f} -name default")
    lines.append("")
    lines.append("# Run C-sim")
    lines.append("csim_design")
    lines.append("")
    lines.append("# You can enable these later:")
    lines.append("# csynth_design")
    lines.append("# cosim_design")
    lines.append("# export_design -format ip_catalog")
    lines.append("")
    lines.append("exit")
    return "\n".join(lines) + "\n"
