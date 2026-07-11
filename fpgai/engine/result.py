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

    def _read_manifest(self) -> dict:
        manifest_path = self.out_dir / "manifest.json"
        if not manifest_path.exists():
            return {}

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        return data if isinstance(data, dict) else {}

    def _manifest_summary_lines(self) -> list[str]:
        manifest_path = self.out_dir / "manifest.json"
        if not manifest_path.exists():
            return []

        manifest = self._read_manifest()
        lines = [f"Manifest             : {manifest_path}"]

        if manifest.get("pipeline_mode"):
            lines.append(f"Pipeline mode        : {manifest.get('pipeline_mode')}")
        if manifest.get("top_kernel_name"):
            lines.append(f"Top kernel           : {manifest.get('top_kernel_name')}")
        if manifest.get("seconds") is not None:
            lines.append(f"Compile seconds      : {manifest.get('seconds')}")

        return lines

    def _prediction_artifact_summary_lines(self) -> list[str]:
        manifest = self._read_manifest()
        if not manifest:
            return []

        artifacts = manifest.get("prediction_artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return []

        labels = [
            ("model_profile_json", "Model profile JSON"),
            ("resource_prediction_json", "Resource prediction"),
            ("timing_prediction_json", "Timing prediction"),
            ("prediction_summary_md", "Prediction summary"),
            ("board_fit_json", "Board fit JSON"),
            ("board_fit_markdown", "Board fit summary"),
        ]

        lines = ["Prediction artifacts:"]
        for key, label in labels:
            value = artifacts.get(key)
            if value:
                lines.append(f"  - {label}: {value}")

        board_fit = artifacts.get("board_fit")
        if isinstance(board_fit, dict):
            status = board_fit.get("status")
            limiting = board_fit.get("limiting_dimension")
            vivado_allowed = board_fit.get("vivado_allowed")
            if status is not None:
                lines.append(f"  - Board fit status: {status}")
            if limiting:
                lines.append(f"  - Board fit limiting: {limiting}")
            if vivado_allowed is not None:
                lines.append(f"  - Vivado allowed by fit: {vivado_allowed}")

        return lines if len(lines) > 1 else []

    def _hardware_knob_contract_summary_lines(self) -> list[str]:
        manifest = self._read_manifest()
        if not manifest:
            return []

        contract = manifest.get("hardware_knob_contract")
        if not isinstance(contract, dict) or not contract:
            return []

        lines = ["Hardware knob contract: available"]

        for key, label in [
            ("json", "JSON"),
            ("markdown", "Summary"),
            ("knob_count", "Knob count"),
            ("manual_yaml_count", "Manual YAML knobs"),
            ("changed_or_clamped_count", "Changed/clamped"),
            ("report_only_count", "Report-only"),
        ]:
            value = contract.get(key)
            if value is not None:
                lines.append(f"  - {label}: {value}")

        return lines


    def _design_space_summary_lines(self) -> list[str]:
        manifest = self._read_manifest()
        if not manifest:
            return []

        design_space = manifest.get("design_space")
        if not isinstance(design_space, dict) or not design_space:
            return []

        lines = ["Design space        : available"]
        for key in [
            "recommendation_policy",
            "recommended_balanced",
            "recommended_smallest_valid",
            "recommended_best_accuracy",
            "layer_breakdown_csv",
        ]:
            value = design_space.get(key)
            if value:
                lines.append(f"  - {key}: {value}")
        return lines

    def _hls_artifacts_summary_lines(self) -> list[str]:
        manifest = self._read_manifest()
        if not manifest:
            return []

        hls = manifest.get("hls_artifacts")
        if not isinstance(hls, dict) or not hls:
            return []

        lines = [
            "HLS artifacts       : available",
            f"  - hls_ran: {hls.get('hls_ran')}",
            f"  - hls_ok: {hls.get('hls_ok')}",
            f"  - hls_returncode: {hls.get('hls_returncode')}",
        ]

        for key in ["stdout_log", "stderr_log", "csynth_report"]:
            value = hls.get(key)
            if value:
                lines.append(f"  - {key}: {value}")

        artifact_metadata = hls.get("artifact_metadata")
        if isinstance(artifact_metadata, dict):
            path = artifact_metadata.get("path")
            file_count = artifact_metadata.get("file_count")
            if path:
                lines.append(f"  - artifact_metadata: {path}")
            if file_count is not None:
                lines.append(f"  - hls_file_count: {file_count}")

        schedule_summary = hls.get("schedule_summary")
        if isinstance(schedule_summary, dict) and schedule_summary.get("path"):
            lines.append(f"  - schedule_summary: {schedule_summary.get('path')}")

        ii_comparison = hls.get("ii_comparison")
        if isinstance(ii_comparison, dict) and ii_comparison.get("path"):
            lines.append(f"  - ii_comparison: {ii_comparison.get('path')}")

        return lines

    def _fit_policy_gate_summary_lines(self) -> list[str]:
        manifest = self._read_manifest()
        if not manifest:
            return []

        gate = manifest.get("fit_policy_gate")
        if not isinstance(gate, dict) or not gate:
            return []

        lines = ["Fit policy gate    : available"]
        for key, label in [
            ("policy", "policy"),
            ("policy_source", "policy_source"),
            ("requested_policy", "requested_policy"),
            ("board_fit_status", "board_fit_status"),
            ("board_fit_limiting_dimension", "limiting"),
            ("vivado_allowed_by_board_fit", "vivado_allowed_by_board_fit"),
            ("over_limit", "over_limit"),
            ("blocked", "blocked"),
            ("warning", "warning"),
            ("severity", "severity"),
            ("reason", "reason"),
        ]:
            if key in gate:
                lines.append(f"  - {label}: {gate.get(key)}")

        blocked_stages = gate.get("blocked_stages")
        if blocked_stages:
            lines.append(f"  - blocked_stages: {blocked_stages}")

        return lines

    def _vivado_bridge_summary_lines(self) -> list[str]:
        manifest = self._read_manifest()
        if not manifest:
            return []

        bridge = manifest.get("vivado_bridge")
        if not isinstance(bridge, dict) or not bridge:
            return ["Vivado bridge       : not_requested"]

        lines = ["Vivado bridge       : available"]
        for key in [
            "board",
            "part",
            "ps_type",
            "vivado_bridge_generated",
            "vivado_synth_requested",
            "vivado_impl_requested",
            "bitstream_requested",
            "bitstream_exists",
            "xsa_exists",
        ]:
            if key in bridge:
                lines.append(f"  - {key}: {bridge.get(key)}")
        return lines

    def _runtime_package_summary_lines(self) -> list[str]:
        manifest = self._read_manifest()
        if not manifest:
            return []

        package = manifest.get("runtime_package")
        if not isinstance(package, dict) or not package:
            return []

        lines = [
            f"Runtime package     : {package.get('status', 'unknown')}",
            f"  - package: {package.get('path')}",
            f"  - deployable_overlay_present: {package.get('deployable_overlay_present')}",
            f"  - bitstream_present: {package.get('bitstream_present')}",
            f"  - hwh_present: {package.get('hwh_present')}",
            f"  - xsa_present: {package.get('xsa_present')}",
            f"  - file_count: {package.get('file_count')}",
        ]
        return [line for line in lines if not line.endswith(": None")]


    def _pipeline_stage_summary_lines(self) -> list[str]:
        manifest = self._read_manifest()
        if not manifest:
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

        manifest_lines = self._manifest_summary_lines()
        if manifest_lines:
            lines.insert(-1, "---------------------------------------------------")
            lines[-1:-1] = manifest_lines

        manifest_section_groups = [
            self._prediction_artifact_summary_lines(),
            self._design_space_summary_lines(),
            self._hls_artifacts_summary_lines(),
            self._fit_policy_gate_summary_lines(),
            self._vivado_bridge_summary_lines(),
            self._runtime_package_summary_lines(),
            self._pipeline_stage_summary_lines(),
        ]

        for section_lines in manifest_section_groups:
            if section_lines:
                lines.insert(-1, "---------------------------------------------------")
                lines[-1:-1] = section_lines

        hardware_knob_contract_lines = self._hardware_knob_contract_summary_lines()
        if hardware_knob_contract_lines:
            lines.insert(-1, "---------------------------------------------------")
            lines[-1:-1] = hardware_knob_contract_lines

        return "\n".join(lines)
