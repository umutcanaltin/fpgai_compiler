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

    quant_report_dir: Optional[Path] = None
    quant_metrics_json: Optional[Path] = None
    quant_summary_txt: Optional[Path] = None
    quant_layerwise_csv: Optional[Path] = None

    precision_sweep_dir: Optional[Path] = None
    precision_sweep_results_json: Optional[Path] = None
    precision_sweep_summary_txt: Optional[Path] = None
    precision_sweep_results_csv: Optional[Path] = None

    design_space_dir: Optional[Path] = None
    design_space_results_json: Optional[Path] = None
    design_space_summary_txt: Optional[Path] = None
    design_space_results_csv: Optional[Path] = None
    design_space_terminal_summary: Optional[str] = None

    training_plan_json: Optional[Path] = None
    training_summary_txt: Optional[Path] = None

    def summary(self) -> str:
        lines = []
        lines.append("\n=============== FPGAI Compile Result ===============")
        lines.append(f"Out dir              : {self.out_dir}")
        lines.append(f"Ops                  : {len(self.graph.ops)}")
        lines.append(f"Params               : {len(self.graph.params)}")
        lines.append(f"Inputs               : {self.graph.inputs}")
        lines.append(f"Outputs              : {self.graph.outputs}")
        lines.append("---------------------------------------------------")
        lines.append(f"HLS project dir      : {self.hls_project_dir}")
        lines.append(f"Host C++ dir         : {self.host_project_dir}")
        lines.append("---------------------------------------------------")
        lines.append(f"HLS ran              : {self.hls_ran}")
        lines.append(f"HLS ok               : {self.hls_ok}")
        lines.append(f"HLS returncode       : {self.hls_returncode}")
        lines.append(f"HLS stdout log       : {self.hls_stdout_log}")
        lines.append(f"HLS stderr log       : {self.hls_stderr_log}")
        lines.append(f"HLS csynth report    : {self.hls_csynth_report}")
        lines.append("---------------------------------------------------")
        lines.append(f"Quant report dir     : {self.quant_report_dir}")
        lines.append(f"Quant metrics JSON   : {self.quant_metrics_json}")
        lines.append(f"Quant summary TXT    : {self.quant_summary_txt}")
        lines.append(f"Quant layerwise CSV  : {self.quant_layerwise_csv}")
        lines.append("---------------------------------------------------")
        lines.append(f"Precision sweep dir  : {self.precision_sweep_dir}")
        lines.append(f"Sweep results JSON   : {self.precision_sweep_results_json}")
        lines.append(f"Sweep summary TXT    : {self.precision_sweep_summary_txt}")
        lines.append(f"Sweep results CSV    : {self.precision_sweep_results_csv}")
        lines.append("---------------------------------------------------")
        lines.append(f"Design space dir     : {self.design_space_dir}")
        lines.append(f"Design space JSON    : {self.design_space_results_json}")
        lines.append(f"Design space TXT     : {self.design_space_summary_txt}")
        lines.append(f"Design space CSV     : {self.design_space_results_csv}")
        lines.append("---------------------------------------------------")
        lines.append(f"Training plan JSON   : {self.training_plan_json}")
        lines.append(f"Training summary TXT : {self.training_summary_txt}")
        lines.append("===================================================")
        return "\n".join(lines)