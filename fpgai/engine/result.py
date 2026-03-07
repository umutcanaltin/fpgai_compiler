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
        lines.append("===================================================")
        return "\n".join(lines)
