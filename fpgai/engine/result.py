from __future__ import annotations

from dataclasses import dataclass
import json
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
    hls_stdout_log: Optional[Path] = None
    hls_stderr_log: Optional[Path] = None
    hls_csynth_report: Optional[Path] = None

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
    design_space_layer_breakdown_csv: Optional[Path] = None
    design_space_terminal_summary: Optional[str] = None

    estimate_vs_hls_dir: Optional[Path] = None
    estimate_vs_hls_results_json: Optional[Path] = None
    estimate_vs_hls_summary_txt: Optional[Path] = None

    hls_module_breakdown_dir: Optional[Path] = None
    hls_module_breakdown_json: Optional[Path] = None
    hls_module_breakdown_csv: Optional[Path] = None
    hls_module_breakdown_summary_txt: Optional[Path] = None

    training_plan_json: Optional[Path] = None
    training_summary_txt: Optional[Path] = None

    def _prediction_artifact_summary_lines(self) -> list[str]:
        manifest_path = self.out_dir / "manifest.json"
        if not manifest_path.exists():
            return []

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        artifacts = manifest.get("prediction_artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return []

        labels = [
            ("model_profile_json", "Model profile JSON"),
            ("resource_prediction_json", "Resource prediction"),
            ("timing_prediction_json", "Timing prediction"),
            ("prediction_summary_md", "Prediction summary"),
        ]

        lines = ["Prediction artifacts:"]
        for key, label in labels:
            value = artifacts.get(key)
            if value:
                lines.append(f"  - {label}: {value}")

        return lines if len(lines) > 1 else []

    def _pipeline_stage_summary_lines(self) -> list[str]:
        manifest_path = self.out_dir / "manifest.json"
        if not manifest_path.exists():
            return []

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        stages = manifest.get("pipeline_stages")
        if not isinstance(stages, list) or not stages:
            return []

        stage_lines = ["Pipeline stages     :"]
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            name = stage.get("name")
            status = stage.get("status")
            if not name or not status:
                continue
            stage_lines.append(f"  - {name}: {status}")

        return stage_lines if len(stage_lines) > 1 else []

    def summary(self) -> str:
        lines = [
            "",
            "=============== FPGAI Compile Result ===============",
            f"Out dir              : {self.out_dir}",
            f"Ops                  : {len(self.graph.ops)}",
            f"Params               : {len(self.graph.params)}",
            f"Inputs               : {self.graph.inputs}",
            f"Outputs              : {self.graph.outputs}",
            "---------------------------------------------------",
            f"HLS project dir      : {self.hls_project_dir}",
            f"Host C++ dir         : {self.host_project_dir}",
            "---------------------------------------------------",
            f"HLS ran              : {self.hls_ran}",
            f"HLS ok               : {self.hls_ok}",
            f"HLS returncode       : {self.hls_returncode}",
            f"HLS stdout log       : {self.hls_stdout_log}",
            f"HLS stderr log       : {self.hls_stderr_log}",
            f"HLS csynth report    : {self.hls_csynth_report}",
            "---------------------------------------------------",
            f"Quant report dir     : {self.quant_report_dir}",
            f"Quant metrics JSON   : {self.quant_metrics_json}",
            f"Quant summary TXT    : {self.quant_summary_txt}",
            f"Quant layerwise CSV  : {self.quant_layerwise_csv}",
            "---------------------------------------------------",
            f"Precision sweep dir  : {self.precision_sweep_dir}",
            f"Sweep results JSON   : {self.precision_sweep_results_json}",
            f"Sweep summary TXT    : {self.precision_sweep_summary_txt}",
            f"Sweep results CSV    : {self.precision_sweep_results_csv}",
            "---------------------------------------------------",
            f"Design space dir     : {self.design_space_dir}",
            f"Design space JSON    : {self.design_space_results_json}",
            f"Design space TXT     : {self.design_space_summary_txt}",
            f"Design space CSV     : {self.design_space_results_csv}",
            (
                "Layer breakdown CSV  : "
                f"{self.design_space_layer_breakdown_csv}"
            ),
            "---------------------------------------------------",
            f"Estimate/HLS dir     : {self.estimate_vs_hls_dir}",
            f"Estimate/HLS JSON    : {self.estimate_vs_hls_results_json}",
            f"Estimate/HLS TXT     : {self.estimate_vs_hls_summary_txt}",
            "---------------------------------------------------",
            f"HLS modules dir      : {self.hls_module_breakdown_dir}",
            f"HLS modules JSON     : {self.hls_module_breakdown_json}",
            f"HLS modules CSV      : {self.hls_module_breakdown_csv}",
            (
                "HLS modules TXT      : "
                f"{self.hls_module_breakdown_summary_txt}"
            ),
            "---------------------------------------------------",
            f"Training plan JSON   : {self.training_plan_json}",
            f"Training summary TXT : {self.training_summary_txt}",
            "===================================================",
        ]

        prediction_lines = self._prediction_artifact_summary_lines()
        if prediction_lines:
            lines.insert(-1, "---------------------------------------------------")
            lines[-1:-1] = prediction_lines

        pipeline_lines = self._pipeline_stage_summary_lines()
        if pipeline_lines:
            lines.insert(-1, "---------------------------------------------------")
            lines[-1:-1] = pipeline_lines

        return "\n".join(lines)
