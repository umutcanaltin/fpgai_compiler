from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fpgai.ir.graph import Graph


@dataclass(frozen=True)
class CompileResult:
    out_dir: Path
    graph: Graph
    hls_project_dir: Optional[Path] = None
    host_project_dir: Optional[Path] = None

    hls_ran: bool = False
    hls_ok: Optional[bool] = None
    hls_returncode: Optional[int] = None
    hls_stdout_log: Optional[str] = None
    hls_stderr_log: Optional[str] = None
    hls_csynth_report: Optional[str] = None

    def summary(self) -> str:
        lines = []
        lines.append("\n=============== FPGAI Compile Result ===============")
        lines.append(f"Out dir           : {self.out_dir}")
        lines.append(f"Ops               : {len(self.graph.ops)}")
        lines.append(f"Params            : {len(self.graph.params)}")
        lines.append(f"Inputs            : {self.graph.inputs}")
        lines.append(f"Outputs           : {self.graph.outputs}")
        lines.append("---------------------------------------------------")
        lines.append(f"HLS project dir   : {self.hls_project_dir}")
        lines.append(f"Host C++ dir      : {self.host_project_dir}")
        lines.append("---------------------------------------------------")
        lines.append(f"HLS ran           : {self.hls_ran}")
        lines.append(f"HLS ok            : {self.hls_ok}")
        lines.append(f"HLS returncode    : {self.hls_returncode}")
        lines.append(f"HLS stdout log    : {self.hls_stdout_log}")
        lines.append(f"HLS stderr log    : {self.hls_stderr_log}")
        lines.append(f"HLS csynth report : {self.hls_csynth_report}")
        lines.append("===================================================")
        return "\n".join(lines)