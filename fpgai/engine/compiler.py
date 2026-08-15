from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time
import numpy as np

from fpgai.config.access import get_path
from fpgai.config.loader import FPGAIConfig
from fpgai.config.contract import build_config_contract_report, render_config_contract_markdown
from fpgai.capabilities.architecture_capabilities import (
    validate_architecture_capabilities,
)
from fpgai.engine.analysis import analyze_graph
from fpgai.engine.communication import make_communication_plan
from fpgai.engine.memory import make_memory_plan
from fpgai.engine.planner import make_compile_plan
from fpgai.engine.result import CompileResult
from fpgai.engine.partition import single_device_plan
from fpgai.engine.layerwise_precision import resolve_layerwise_precision
from fpgai.engine.training import (
    build_training_plan,
    emit_training_artifacts,
    resolve_training_execution_schedule,
)
from fpgai.analysis.model_inspection import inspect_config, write_model_inspection_report
from fpgai.analysis.model_compatibility import emit_model_compatibility_reports
from fpgai.analysis.resource_estimator import estimate_resources_from_descriptors
from fpgai.analysis.performance_estimator import estimate_performance
try:
    from fpgai.analysis.quantization_report import run_quantization_report
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in lightweight test envs
    _QUANTIZATION_REPORT_IMPORT_ERROR = exc

    def run_quantization_report(*args, **kwargs):
        raise _QUANTIZATION_REPORT_IMPORT_ERROR
try:
    from fpgai.analysis.precision_sweep import run_precision_sweep
except ModuleNotFoundError as exc:  # pragma: no cover
    _PRECISION_SWEEP_IMPORT_ERROR = exc

    def run_precision_sweep(*args, **kwargs):
        raise _PRECISION_SWEEP_IMPORT_ERROR
try:
    from fpgai.analysis.design_space_report import run_design_space_report
except ModuleNotFoundError as exc:  # pragma: no cover
    _DESIGN_SPACE_IMPORT_ERROR = exc

    def run_design_space_report(*args, **kwargs):
        raise _DESIGN_SPACE_IMPORT_ERROR
from fpgai.analysis.post_synthesis import run_post_synthesis_analysis
from fpgai.analysis.hls_estimate_compare import parse_hls_csynth_report
from fpgai.analysis.training_resource_estimate import run_training_resource_estimate
from fpgai.benchmark.training_reference import run_training_reference_step
from fpgai.validation.capture_adapters import orchestrate_training_numeric_equivalence
from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference
from fpgai.benchmark.training_compare import compare_training_artifacts, build_dataset_training_comparison, build_training_semantic_trace_report, build_training_per_sample_gradient_trace_report, build_training_gradient_layer_role_reports
from fpgai.util.fs import ensure_clean_dir, write_text
from fpgai.numerics.precision_policy import (
    build_precision_layout,
    precision_layout_markdown,
)
from fpgai.analysis.hls_schedule_report import write_hls_schedule_summary
from fpgai.analysis.hls_ii_comparison import write_requested_achieved_ii_summary
from fpgai.analysis.hls_artifact_metadata import emit_hls_artifact_metadata
from fpgai.analysis.hls_calibration_runner import run_hls_calibration
from fpgai.util.binio import write_f32_bin
from fpgai.runtime.package import emit_runtime_package
from fpgai.validation.numeric import emit_numeric_validation_report
from fpgai.validation.dataset import (
    emit_dataset_artifacts,
    emit_dataset_model_contract,
    emit_training_validation_dataset_artifacts,
    emit_training_validation_split_contract,
)
from fpgai.benchmark.verification import emit_validation_summary_artifacts
from fpgai.benchmark.experiment_artifacts import emit_experiment_artifact_reports
from fpgai.backends.vivado.boards import get_board
from fpgai.backends.vivado.vivado_bridge import emit_vivado_project_handoff, emit_vivado_validation_reports
from fpgai.backends.vivado.run_bridge import run_vivado_bridge_flow
from fpgai.reporting.hardware_feasibility import emit_board_fit_report
from fpgai.reporting.hls_explanation import emit_generated_hls_explanation_reports
from fpgai.reporting.hls_validation import emit_hls_validation_reports
from fpgai.reporting.precision_effect import emit_precision_effect_reports
from fpgai.reporting.parallel_pipeline_effect import emit_parallel_pipeline_effect_reports
from fpgai.reporting.data_movement import emit_data_movement_reports, emit_movement_contract_validation, movement_contract_validation_summary


_cfg_get = get_path

from fpgai.engine.build_stages import (
    BUILD_STAGE_KEYS as _BUILD_STAGE_KEYS,
    build_stage_summary as _build_stage_summary,
    cfg_has_path as _cfg_has_path,
    resolve_build_stages as _resolve_build_stages,
)



from fpgai.engine.inference_reference import (
    _emit_inference_reference_artifacts,
    _normalise_onnx_shape,
)
from fpgai.engine.training_contracts import (
    _CODEGEN_READABILITY,
    _OPTIMIZER_STATE_STORAGE,
    _RUNTIME_COMMANDS,
    _TRAINING_LOSS_TYPES,
    _TRAINING_OPTIMIZER_TYPES,
    _movement_cfg,
    _resolve_codegen_readability,
    _resolve_gradient_export_mode,
    _resolve_optimizer_state_movement,
    _resolve_runtime_sequence,
    _resolve_stream_tiled_io_contract,
    _resolve_training_batch_accumulation_contract,
    _resolve_training_io_movement,
    _resolve_training_optimizer_loss_contract,
    _runtime_io_summary_from_plan,
    _runtime_support_from_semantics,
    _scan_hls_top_ports,
    _write_execution_semantics_reports,
    _write_feature_validation_reports,
    _write_runtime_sequence_report,
    _write_training_movement_reports,
    _write_training_optimizer_loss_reports,
    _write_vivado_bd_contract_reports,
)
from fpgai.engine.compiler_reports import (
    _emit_resolved_config_reports,
    _plan_notes,
    _resolved_toolchain_summary,
)
from fpgai.engine.vivado_pipeline import (
    _existing_hls_ip_component_exists,
    _read_json_file,
    _runtime_package_manifest_summary,
    _update_manifest_after_vivado_bridge,
    _vivado_bridge_timeout_sec,
    _yaml_requested_vivado_bridge,
)


from fpgai.engine.memory_semantics import MemorySemanticsMixin
from fpgai.engine.hls_project_generation import HLSProjectGenerationMixin


def _run_yaml_requested_vivado_bridge(
    out_dir: Path,
    raw: Dict[str, Any],
    build_stages: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Execute YAML-requested Vivado stages through the existing bridge backend."""
    if not _yaml_requested_vivado_bridge(build_stages):
        return None

    board = str(
        _cfg_get(
            raw,
            "targets.platform.board",
            _cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "pynq_z2")),
        )
        or "pynq_z2"
    )
    run_impl_requested = bool(
        build_stages.get("vivado_implementation") or build_stages.get("bitstream")
    )
    run_bitstream_requested = bool(build_stages.get("bitstream"))

    existing_hls_ip = bool(_cfg_get(raw, "build.existing_hls_ip", False))
    hls_synthesis_requested = bool(build_stages.get("hls_synthesis"))
    concrete_existing_ip = _existing_hls_ip_component_exists(out_dir)

    export_hls_ip = bool(
        (build_stages.get("vivado_project") or run_impl_requested)
        and hls_synthesis_requested
        and not existing_hls_ip
    )
    run_impl = run_impl_requested
    run_bitstream = run_bitstream_requested

    if existing_hls_ip and run_impl_requested and not concrete_existing_ip:
        run_impl = False
        run_bitstream = False

    payload = run_vivado_bridge_flow(
        out_dir,
        board=board,
        export_hls_ip=export_hls_ip,
        run_vivado_synth=False,
        run_vivado_impl=run_impl,
        run_bitstream=run_bitstream,
        timeout_sec=_vivado_bridge_timeout_sec(raw),
    )
    _update_manifest_after_vivado_bridge(out_dir, payload)

    if build_stages.get("runtime_package"):
        emit_runtime_package(out_dir)

    emit_experiment_artifact_reports(out_dir)

    failed_rows = payload.get("failed_rows") or []
    if failed_rows:
        reason = "; ".join(f"{design}: {why}" for design, why in failed_rows)
        raise RuntimeError(f"YAML-requested Vivado bridge flow failed: {reason}")

    return payload

@dataclass
class Compiler(HLSProjectGenerationMixin, MemorySemanticsMixin):
    cfg: FPGAIConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Compiler":
        from fpgai.config.loader import load_config
        return cls(load_config(path))

    def compile(self) -> CompileResult:
        mode = str(self.cfg.pipeline.mode).lower()
        raw = self.cfg.raw
        ecosystem_cfg = _cfg_get(raw, "ecosystem", None)
        if isinstance(ecosystem_cfg, dict) and bool(ecosystem_cfg.get("enabled", False)):
            from fpgai.ecosystem import compile_external_hls_if_configured

            out_dir = self._prepare_out_dir(raw)
            build_stages = _resolve_build_stages(raw)
            external = compile_external_hls_if_configured(
                self,
                out_dir=out_dir,
                build_stages=build_stages,
            )
            if external.handled:
                hls_run = external.hls_run
                return CompileResult(
                    out_dir=out_dir,
                    graph=external.graph,
                    hls_project_dir=external.hls_dir,
                    hls_ran=hls_run is not None,
                    hls_ok=(None if hls_run is None else hls_run.ok),
                    hls_returncode=(None if hls_run is None else hls_run.returncode),
                    hls_stdout_log=(None if hls_run is None else hls_run.stdout_log),
                    hls_stderr_log=(None if hls_run is None else hls_run.stderr_log),
                    hls_csynth_report=(None if hls_run is None else hls_run.csynth_report),
                )
        if mode == "inference":
            return self._compile_inference()
        if mode == "training_on_device":
            return self._compile_training()
        raise RuntimeError(f"Unsupported pipeline mode: {self.cfg.pipeline.mode}")

    def _compile_inference(self) -> CompileResult:
        raw = self.cfg.raw
        t0 = time.time()
        out_dir = self._prepare_out_dir(raw)
        top_name = str(_cfg_get(raw, "pipeline.outputs.top_kernel_name", "deeplearn"))
        verbose = bool(_cfg_get(raw, "debug.verbose", False))
        emit_manifest = bool(_cfg_get(raw, "project.reproducibility.emit_manifest", True))
        build_stages = _resolve_build_stages(raw)
        enable_hls = bool(build_stages.get("cpp", False))
        enable_host = bool(build_stages.get("host_cpp", False))
        enable_reports = bool(build_stages.get("reports", True))
        enable_runtime_package = bool(build_stages.get("runtime_package", True))
        enable_quant_report = bool(_cfg_get(raw, "analysis.quantization_report.enabled", False))
        enable_precision_sweep = bool(_cfg_get(raw, "analysis.precision_sweep.enabled", False))
        enable_design_space = bool(_cfg_get(raw, "analysis.design_space.enabled", False))
        act_kind, act_alpha, act_except_last = self._read_activation_insert_cfg(raw)
        self._reject_unsupported_training_weight_storage(raw)
        weights_mode = self._resolve_hls_weights_mode(raw)

        g = self._import_and_prepare_graph(
            act_kind=act_kind,
            act_alpha=act_alpha,
            act_except_last=act_except_last,
        )
        resolve_layerwise_precision(g, raw)

        descriptors = analyze_graph(g)
        compile_plan = make_compile_plan(self.cfg, descriptors)
        memory_plan = make_memory_plan(g, descriptors, compile_plan)
        self._annotate_memory_movement_semantics(compile_plan, memory_plan, raw)
        communication_plan = make_communication_plan(self.cfg, memory_plan)
        runtime_sequence = _resolve_runtime_sequence(
            raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            memory_semantics_mode=str(memory_plan.notes.get("memory_semantics_mode", weights_mode)),
        )
        resolved_config_artifacts = _emit_resolved_config_reports(
            out_dir,
            raw,
            build_stages=build_stages,
            runtime_sequence=runtime_sequence,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            weights_mode=weights_mode,
        )
        data_movement_artifacts = emit_data_movement_reports(
            out_dir,
            raw_config=raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            weights_mode=str(memory_plan.notes.get("memory_semantics_mode", weights_mode)),
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            runtime_sequence=runtime_sequence,
        )
        _write_runtime_sequence_report(out_dir, runtime_sequence)
        training_movement_artifacts = _write_training_movement_reports(out_dir, raw)
        training_optimizer_loss_artifacts = _write_training_optimizer_loss_reports(out_dir, raw)
        training_movement_artifacts.update(training_optimizer_loss_artifacts)
        capability_report = self._validate_architecture(
            out_dir,
            compile_plan,
            memory_plan,
        )

        self._emit_ir_artifacts(out_dir, g, descriptors, compile_plan, memory_plan, communication_plan)
        prediction_artifacts = self._emit_prediction_artifacts(
            out_dir,
            descriptors,
            compile_plan,
        )
        execution_semantics_artifacts = _write_execution_semantics_reports(
            out_dir,
            raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            prediction_artifacts=prediction_artifacts,
        ) if enable_reports else None
        self._emit_dummy_input(out_dir, g)

        quant_result = run_quantization_report(
            model_path=self.cfg.model.path, raw_cfg=raw, out_dir=out_dir
        ) if enable_quant_report else None
        sweep_result = run_precision_sweep(
            model_path=self.cfg.model.path, raw_cfg=raw, out_dir=out_dir
        ) if enable_precision_sweep else None
        design_result = run_design_space_report(
            graph=g, model_path=self.cfg.model.path, raw_cfg=raw, out_dir=out_dir
        ) if enable_design_space else None
        if design_result is not None and bool(_cfg_get(raw, "analysis.design_space.print_terminal_summary", True)):
            print("\n" + design_result.terminal_summary + "\n")

        hls_dir: Optional[Path] = self._emit_hls(
            out_dir,
            g,
            top_name=top_name,
            weights_mode=weights_mode,
            compile_plan=compile_plan,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            build_stages=build_stages,
        ) if enable_hls else None
        host_dir: Optional[Path] = self._emit_hostcpp(out_dir, g, top_name=top_name) if enable_host else None
        hls_run = self._maybe_run_vitis_hls(hls_dir, build_stages=build_stages) if enable_hls and hls_dir is not None else None
        hls_calibration_result = run_hls_calibration(
            out_dir=out_dir,
            raw_cfg=raw,
            compile_plan=compile_plan,
            hls_report_dir=(hls_dir if hls_dir is not None else out_dir),
            clock_mhz=float(getattr(compile_plan, "clock_mhz", _cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200.0))),
            verbose=verbose,
        ) if enable_reports else None
        if enable_reports:
            prediction_artifacts = self._refresh_board_fit_from_hls(
                out_dir=out_dir,
                compile_plan=compile_plan,
                hls_run=hls_run,
                prediction_artifacts=prediction_artifacts,
                build_stages=build_stages,
            )
            execution_semantics_artifacts = _write_execution_semantics_reports(
                out_dir,
                raw,
                pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
                memory_plan=memory_plan,
                communication_plan=communication_plan,
                prediction_artifacts=prediction_artifacts,
            )

        estimate_vs_hls_result = None
        hls_module_breakdown_result = None
        if design_result is not None:
            best = None
            try:
                ds_payload = json.loads(design_result.results_json.read_text(encoding="utf-8"))
                best = (
                    ds_payload.get("recommended_balanced")
                    or ds_payload.get("recommended_smallest_valid")
                    or ds_payload.get("recommended_best_accuracy")
                )
            except Exception:
                best = None
            if best is not None:
                post_synthesis_result = run_post_synthesis_analysis(
                    out_dir=out_dir,
                    design_space_summary=best,
                    csynth_report_path=(hls_run.csynth_report if hls_run is not None else None),
                    clock_mhz=float(getattr(compile_plan, "clock_mhz", _cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200.0))),
                    top_name=top_name,
                )
                estimate_vs_hls_result = post_synthesis_result.estimate_comparison
                hls_module_breakdown_result = post_synthesis_result.module_breakdown

        if enable_reports:
            hls_schedule_summary = self._emit_hls_schedule_summary(out_dir)
            hls_artifact_metadata = emit_hls_artifact_metadata(
                out_dir,
                compile_plan,
                schedule_summary=hls_schedule_summary,
            )
            hls_ii_comparison = write_requested_achieved_ii_summary(
                out_dir,
                compile_plan,
            )
        else:
            hls_schedule_summary = None
            hls_artifact_metadata = None
            hls_ii_comparison = None
        runtime_package = emit_runtime_package(
            out_dir,
            board=str(_cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "")) or ""),
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            top_name=top_name,
            weights_mode=memory_plan.notes.get("memory_semantics_mode", weights_mode),
            communication_plan=communication_plan,
            memory_plan=memory_plan,
            build_stages=build_stages,
            runtime_sequence=runtime_sequence,
            hls_artifacts=self._hls_artifacts_manifest_payload(
                out_dir=out_dir,
                hls_run=hls_run,
                hls_schedule_summary=hls_schedule_summary,
                hls_artifact_metadata=hls_artifact_metadata,
                hls_ii_comparison=hls_ii_comparison,
            ),
        ) if enable_runtime_package else None
        vivado_bd_contract_artifacts = _write_vivado_bd_contract_reports(
            out_dir,
            raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            build_stages=build_stages,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            hls_dir=hls_dir,
        ) if enable_reports else None
        vivado_handoff_artifacts = emit_vivado_project_handoff(
            out_dir,
            raw_config=raw,
            build_stages=build_stages,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            top_name=top_name,
            hls_dir=hls_dir,
            runtime_sequence=runtime_sequence,
            source_ports=_scan_hls_top_ports(hls_dir),
        ) if enable_reports else None
        vivado_validation_artifacts = emit_vivado_validation_reports(
            out_dir,
            raw_config=raw,
            build_stages=build_stages,
            vivado_handoff_artifacts=vivado_handoff_artifacts,
            board_fit_artifacts=(prediction_artifacts.get("board_fit") if isinstance(prediction_artifacts, dict) else None),
        ) if enable_reports else None
        hls_validation_artifacts = emit_hls_validation_reports(
            out_dir=out_dir,
            hls_dir=hls_dir,
            build_stages=build_stages,
            hls_run=hls_run,
            design_result=design_result,
            clock_mhz=float(getattr(compile_plan, "clock_mhz", _cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200.0))),
        ) if enable_reports else None
        precision_layout_artifacts_for_effect = self._emit_precision_layout_reports(
            out_dir=out_dir,
            graph=g,
            descriptors=descriptors,
            compile_plan=compile_plan,
        ) if enable_reports else None
        precision_effect_artifacts = emit_precision_effect_reports(
            out_dir=out_dir,
            raw_config=raw,
            hls_dir=hls_dir,
            precision_layout_artifacts=precision_layout_artifacts_for_effect,
            quant_result=quant_result,
            sweep_result=sweep_result,
            hls_validation_artifacts=hls_validation_artifacts,
        ) if enable_reports else None
        parallel_pipeline_effect_artifacts = emit_parallel_pipeline_effect_reports(
            out_dir=out_dir,
            raw_config=raw,
            hls_dir=hls_dir,
            hls_validation_artifacts=hls_validation_artifacts,
        ) if enable_reports else None

        inference_reference_artifacts = _emit_inference_reference_artifacts(
            out_dir,
            model_path=getattr(self.cfg.model, "path", None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            raw_config=raw,
        ) if enable_reports else None

        numeric_validation_artifacts = emit_numeric_validation_report(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            source_generated=(hls_dir is not None),
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            hls_csynth_report=(hls_run.csynth_report if hls_run is not None else None),
            inference_reference_artifacts=inference_reference_artifacts,
            raw_config=raw,
            runtime_sequence=runtime_sequence,
        ) if enable_reports else None
        generated_hls_explanation_artifacts = emit_generated_hls_explanation_reports(
            out_dir,
            raw_config=raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            top_name=top_name,
            hls_dir=hls_dir,
            build_stages=build_stages,
            runtime_sequence=runtime_sequence,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            numeric_validation_artifacts=numeric_validation_artifacts,
        ) if enable_reports else None
        movement_contract_validation_artifacts = emit_movement_contract_validation(
            out_dir,
            data_movement_artifacts=data_movement_artifacts,
        ) if enable_reports else None
        if data_movement_artifacts is not None and movement_contract_validation_artifacts is not None:
            data_movement_artifacts = dict(data_movement_artifacts)
            data_movement_artifacts.update(movement_contract_validation_artifacts)

        validation_summary_artifacts = emit_validation_summary_artifacts(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            source_generated=(hls_dir is not None),
            numeric_validation_json=(
                numeric_validation_artifacts.get("numeric_validation_json")
                if numeric_validation_artifacts is not None
                else None
            ),
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            build_stages=build_stages,
        ) if enable_reports else None
        feature_validation_artifacts = _write_feature_validation_reports(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            build_stages=build_stages,
            hls_dir=hls_dir,
            hls_run=hls_run,
            runtime_package=runtime_package,
            numeric_validation_artifacts=numeric_validation_artifacts,
            validation_summary_artifacts=validation_summary_artifacts,
            vivado_bd_contract_artifacts=vivado_bd_contract_artifacts,
            vivado_handoff_artifacts=vivado_handoff_artifacts,
        ) if enable_reports else None


        if emit_manifest:
            self._emit_manifest(
                hls_schedule_summary=hls_schedule_summary,
                hls_artifact_metadata=hls_artifact_metadata,
                hls_ii_comparison=hls_ii_comparison,
                runtime_package=runtime_package,
                build_stages=build_stages,
                runtime_sequence=runtime_sequence,
                resolved_config_artifacts=resolved_config_artifacts,
                numeric_validation_artifacts=numeric_validation_artifacts,
                generated_hls_explanation_artifacts=generated_hls_explanation_artifacts,
                precision_effect_artifacts=precision_effect_artifacts,
                parallel_pipeline_effect_artifacts=parallel_pipeline_effect_artifacts,
                data_movement_artifacts=data_movement_artifacts,
                validation_summary_artifacts=validation_summary_artifacts,
                vivado_bd_contract_artifacts=vivado_bd_contract_artifacts,
                vivado_handoff_artifacts=vivado_handoff_artifacts,
                vivado_validation_artifacts=vivado_validation_artifacts,
                hls_validation_artifacts=hls_validation_artifacts,
                feature_validation_artifacts=feature_validation_artifacts,
                out_dir=out_dir,
                top_name=top_name,
                weights_mode=weights_mode,
                graph=g,
                descriptors=descriptors,
                compile_plan=compile_plan,
                memory_plan=memory_plan,
                communication_plan=communication_plan,
                capability_report=capability_report,
                hls_run=hls_run,
                quant_result=quant_result,
                sweep_result=sweep_result,
                design_result=design_result,
                estimate_vs_hls_result=estimate_vs_hls_result,
                hls_module_breakdown_result=hls_module_breakdown_result,
                prediction_artifacts=prediction_artifacts,
                training_plan=None,
                training_reference_result=None,
                training_compare_result=None,
                training_estimate_result=None,
                seconds=time.time() - t0,
            )

        if emit_manifest:
            emit_experiment_artifact_reports(out_dir)
            _run_yaml_requested_vivado_bridge(out_dir, raw, build_stages)

        if verbose:
            if quant_result is not None:
                print("[FPGAI] quant_report:", quant_result.summary_txt)
            if sweep_result is not None:
                print("[FPGAI] precision_sweep:", sweep_result.summary_txt)
            if design_result is not None:
                print("[FPGAI] design_space:", design_result.summary_txt)
            if estimate_vs_hls_result is not None:
                print("[FPGAI] estimate_vs_hls:", estimate_vs_hls_result.summary_txt)
            if hls_module_breakdown_result is not None:
                print(
                    "[FPGAI] hls_module_breakdown:",
                    hls_module_breakdown_result.summary_txt,
                )

        return CompileResult(
            out_dir=out_dir,
            graph=g,
            hls_project_dir=hls_dir,
            host_project_dir=host_dir,
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            hls_returncode=(hls_run.returncode if hls_run is not None else None),
            hls_stdout_log=(hls_run.stdout_log if hls_run is not None else None),
            hls_stderr_log=(hls_run.stderr_log if hls_run is not None else None),
            hls_csynth_report=(hls_run.csynth_report if hls_run is not None else None),
            quant_report_dir=(quant_result.out_dir if quant_result is not None else None),
            quant_metrics_json=(quant_result.metrics_json if quant_result is not None else None),
            quant_summary_txt=(quant_result.summary_txt if quant_result is not None else None),
            quant_layerwise_csv=(quant_result.layerwise_csv if quant_result is not None else None),
            precision_sweep_dir=(sweep_result.out_dir if sweep_result is not None else None),
            precision_sweep_results_json=(sweep_result.results_json if sweep_result is not None else None),
            precision_sweep_summary_txt=(sweep_result.summary_txt if sweep_result is not None else None),
            precision_sweep_results_csv=(sweep_result.results_csv if sweep_result is not None else None),
            design_space_dir=(design_result.out_dir if design_result is not None else None),
            design_space_results_json=(design_result.results_json if design_result is not None else None),
            design_space_summary_txt=(design_result.summary_txt if design_result is not None else None),
            design_space_results_csv=(design_result.results_csv if design_result is not None else None),
            design_space_layer_breakdown_csv=(
                design_result.out_dir / "layer_breakdown.csv"
                if design_result is not None
                else None
            ),
            design_space_terminal_summary=(design_result.terminal_summary if design_result is not None else None),
            estimate_vs_hls_dir=(
                estimate_vs_hls_result.out_dir
                if estimate_vs_hls_result is not None
                else None
            ),
            estimate_vs_hls_results_json=(
                estimate_vs_hls_result.results_json
                if estimate_vs_hls_result is not None
                else None
            ),
            estimate_vs_hls_summary_txt=(
                estimate_vs_hls_result.summary_txt
                if estimate_vs_hls_result is not None
                else None
            ),
            hls_module_breakdown_dir=(
                hls_module_breakdown_result.out_dir
                if hls_module_breakdown_result is not None
                else None
            ),
            hls_module_breakdown_json=(
                hls_module_breakdown_result.results_json
                if hls_module_breakdown_result is not None
                else None
            ),
            hls_module_breakdown_csv=(
                hls_module_breakdown_result.results_csv
                if hls_module_breakdown_result is not None
                else None
            ),
            hls_module_breakdown_summary_txt=(
                hls_module_breakdown_result.summary_txt
                if hls_module_breakdown_result is not None
                else None
            ),
            training_plan_json=None,
            training_summary_txt=None,
        )

    def _compile_training(self) -> CompileResult:
        raw = self.cfg.raw
        t0 = time.time()
        out_dir = self._prepare_out_dir(raw)
        top_name = str(_cfg_get(raw, "pipeline.outputs.top_kernel_name", "deeplearn"))
        verbose = bool(_cfg_get(raw, "debug.verbose", False))
        emit_manifest = bool(_cfg_get(raw, "project.reproducibility.emit_manifest", True))
        build_stages = _resolve_build_stages(raw)
        enable_hls = bool(build_stages.get("cpp", False))
        enable_host = bool(build_stages.get("host_cpp", False))
        enable_reports = bool(build_stages.get("reports", True))
        enable_runtime_package = bool(build_stages.get("runtime_package", True))
        act_kind, act_alpha, act_except_last = self._read_activation_insert_cfg(raw)
        self._reject_unsupported_training_weight_storage(raw)
        weights_mode = self._resolve_hls_weights_mode(raw)

        g = self._import_and_prepare_graph(
            act_kind=act_kind,
            act_alpha=act_alpha,
            act_except_last=act_except_last,
        )
        resolve_layerwise_precision(g, raw)

        descriptors = analyze_graph(g)
        compile_plan = make_compile_plan(self.cfg, descriptors)
        memory_plan = make_memory_plan(g, descriptors, compile_plan)
        self._annotate_memory_movement_semantics(compile_plan, memory_plan, raw)
        communication_plan = make_communication_plan(self.cfg, memory_plan)
        runtime_sequence = _resolve_runtime_sequence(
            raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            memory_semantics_mode=str(memory_plan.notes.get("memory_semantics_mode", weights_mode)),
        )
        resolved_config_artifacts = _emit_resolved_config_reports(
            out_dir,
            raw,
            build_stages=build_stages,
            runtime_sequence=runtime_sequence,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            weights_mode=weights_mode,
        )
        data_movement_artifacts = emit_data_movement_reports(
            out_dir,
            raw_config=raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            weights_mode=str(memory_plan.notes.get("memory_semantics_mode", weights_mode)),
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            runtime_sequence=runtime_sequence,
        )
        _write_runtime_sequence_report(out_dir, runtime_sequence)
        training_movement_artifacts = _write_training_movement_reports(out_dir, raw)
        training_optimizer_loss_artifacts = _write_training_optimizer_loss_reports(out_dir, raw)
        training_movement_artifacts.update(training_optimizer_loss_artifacts)
        capability_report = self._validate_architecture(
            out_dir,
            compile_plan,
            memory_plan,
        )

        self._emit_ir_artifacts(out_dir, g, descriptors, compile_plan, memory_plan, communication_plan)
        prediction_artifacts = self._emit_prediction_artifacts(
            out_dir,
            descriptors,
            compile_plan,
        )
        execution_semantics_artifacts = _write_execution_semantics_reports(
            out_dir,
            raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            prediction_artifacts=prediction_artifacts,
        )

        training_plan = build_training_plan(
            g,
            raw,
            compile_plan=compile_plan,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
        )

        # Training kernels mutate weights during optimizer/update. The inference
        # backend supports runtime DDR/URAM weights through weights_mem/m_axi and
        # runtime-load helpers, but the generated training top does not yet expose
        # or update an external/runtime weight buffer. Reject this combination
        # instead of silently generating a runtime package that the training HLS
        # top does not consume.
        training_weight_storage = str(getattr(training_plan, "weight_storage", "") or "").strip().lower()
        training_weights_mode = str(getattr(training_plan, "weights_mode", "") or "").strip().lower()
        hls_weights_mode = str(weights_mode or "").strip().lower()
        training_semantics = str(memory_plan.notes.get("memory_semantics_mode", "") or "").strip().lower()
        unsupported_training_storage = set()
        unsupported_training_modes = {
            "runtime",
            "runtime_external",
            "external",
            "external_ddr",
            "dma_ddr",
        }
        bram_training_runtime_supported = training_semantics in {
            "bram_static",
            "bram_import_full",
            "bram_import_export_full",
        }
        if (
            training_weight_storage in unsupported_training_storage
            or training_weights_mode in unsupported_training_storage
            or (hls_weights_mode in unsupported_training_storage and not bram_training_runtime_supported)
            or training_weights_mode in unsupported_training_modes
        ):
            raise ValueError(
                "Training runtime/external weight storage is not implemented in generated HLS yet: "
                f"weight_storage={training_weight_storage!r}, "
                f"weights_mode={training_weights_mode!r}, "
                f"hls_weights_mode={hls_weights_mode!r}, "
                f"memory_semantics_mode={training_semantics!r}. "
                "Use BRAM/URAM training weights for now, or implement the training "
                "DDR tiled import-update-export backend before enabling DDR "
                "training weight storage."
            )

        # Training BRAM import/export uses explicit runtime command modes in
        # top_train_cpp. Keep the legacy backend selector for testbench/runtime
        # payload generation, but do not classify it as unsupported external DDR.

        emit_training_artifacts(out_dir, training_plan)

        training_estimate_result = None
        if bool(training_plan.estimator.get("enabled", True)):
            training_estimate_result = run_training_resource_estimate(
                graph=g, training_plan=training_plan, out_dir=out_dir
            )
            print("\n" + training_estimate_result.summary_txt.read_text(encoding="utf-8") + "\n")

        training_dataset_artifacts = emit_dataset_artifacts(out_dir, raw_config=raw)
        held_out_dataset_artifacts = emit_training_validation_dataset_artifacts(
            out_dir, raw_config=raw
        )
        training_validation_split_contract = emit_training_validation_split_contract(
            out_dir,
            training_artifacts=training_dataset_artifacts,
            validation_artifacts=held_out_dataset_artifacts,
        )
        if training_validation_split_contract.get("status") == "incompatible":
            raise ValueError(
                "Training/held-out validation dataset contract failed: "
                + str(training_validation_split_contract.get("reason") or "incompatible split")
            )
        held_out_dataset_available = held_out_dataset_artifacts.get("status") == "available"
        if held_out_dataset_available:
            self._emit_held_out_dataset_target(out_dir, g, held_out_dataset_artifacts)
        training_dataset_available = training_dataset_artifacts.get("status") == "available"
        dataset_inputs_matrix = None
        dataset_targets_matrix = None
        if training_dataset_available:
            dataset_model_contract = emit_dataset_model_contract(
                out_dir,
                graph=g,
                dataset_artifacts=training_dataset_artifacts,
                require_supervision=True,
            )
            if dataset_model_contract.get("status") != "compatible":
                raise ValueError(
                    str(dataset_model_contract.get("reason") or "training dataset/model contract is incompatible")
                    + f" See {dataset_model_contract.get('json_path')}"
                )
            input_path = Path(training_dataset_artifacts["inputs_bin"])
            target_path = self._emit_training_dataset_target(
                out_dir,
                g,
                raw,
                training_dataset_artifacts,
            )
            all_inputs = np.fromfile(input_path, dtype=np.float32)
            all_targets = np.fromfile(target_path, dtype=np.float32)
            dataset_input_shape = tuple(
                int(value) for value in (training_dataset_artifacts.get("input_shape") or ())
            )
            input_words_per_sample = int(
                training_dataset_artifacts.get("input_words_per_sample")
                or (np.prod(dataset_input_shape) if dataset_input_shape else 1)
            )
            sample_count = int(training_dataset_artifacts.get("sample_count") or 1)
            output_words_per_sample = max(1, int(all_targets.size // max(1, sample_count)))
            dataset_inputs_matrix = all_inputs.reshape(sample_count, input_words_per_sample)
            dataset_targets_matrix = all_targets.reshape(sample_count, output_words_per_sample)
            x_input = dataset_inputs_matrix[0]
            y_target = dataset_targets_matrix[0]
            training_execution_schedule = resolve_training_execution_schedule(
                raw, sample_count=sample_count
            )
            training_dataset_contract = {
                "artifact_kind": "fpgai_training_dataset_contract",
                "schema_version": 2,
                "status": "available",
                "sample_count": sample_count,
                "input_words_per_sample": input_words_per_sample,
                "target_words_per_sample": output_words_per_sample,
                "inputs_bin": str(input_path),
                "targets_bin": str(target_path),
                "reference_scope": (
                    "full_dataset_accumulated_update"
                    if int(training_execution_schedule.total_optimizer_updates or 0) == 1
                    else "deterministic_multi_epoch_accumulated_training"
                ),
                "reference_scope_reason": (
                    "The HLS testbench and software references share the same canonical "
                    "epoch, batch, deterministic-shuffle, and partial-batch schedule."
                ),
                "execution_schedule": training_execution_schedule.to_dict(),
                "dataset_model_contract": str(dataset_model_contract.get("json_path") or ""),
                "dataset_claim_scope": str(dataset_model_contract.get("claim_scope") or "unknown"),
            }
            write_text(
                out_dir / "reports" / "training_dataset_contract.json",
                json.dumps(training_dataset_contract, indent=2) + "\n",
            )
        else:
            input_path = self._emit_dummy_input(out_dir, g)
            target_path = self._emit_training_target(out_dir, g, raw)
            x_input = np.fromfile(input_path, dtype=np.float32)
            y_target = np.fromfile(target_path, dtype=np.float32)

        if training_dataset_available:
            training_reference_result = run_training_dataset_reference(
                graph=g,
                raw_cfg=raw,
                out_dir=out_dir,
                inputs=dataset_inputs_matrix,
                targets=dataset_targets_matrix,
            )
        else:
            training_reference_result = run_training_reference_step(
                graph=g, raw_cfg=raw, out_dir=out_dir, x_input=x_input, target=y_target
            )

        hls_dir: Optional[Path] = self._emit_hls(
            out_dir,
            g,
            top_name=top_name,
            weights_mode=weights_mode,
            compile_plan=compile_plan,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            build_stages=build_stages,
        ) if enable_hls else None
        host_dir: Optional[Path] = self._emit_hostcpp(out_dir, g, top_name=top_name) if enable_host else None
        hls_run = self._maybe_run_vitis_hls(hls_dir, build_stages=build_stages) if enable_hls and hls_dir is not None else None
        self._emit_training_dataset_execution_report(
            out_dir=out_dir,
            hls_dir=hls_dir,
            training_dataset_artifacts=training_dataset_artifacts,
        )
        self._emit_training_validation_execution_report(
            out_dir=out_dir,
            hls_dir=hls_dir,
            held_out_dataset_artifacts=held_out_dataset_artifacts,
        )
        training_numeric_equivalence_artifacts = orchestrate_training_numeric_equivalence(
            graph=g,
            training_reference_result=training_reference_result,
            hls_artifact_dir=hls_dir,
            training_dir=out_dir / "training",
            optimizer_type=str(getattr(training_reference_result, "optimizer_type", "sgd")),
            raw_config=raw,
        )
        hls_calibration_result = run_hls_calibration(
            out_dir=out_dir,
            raw_cfg=raw,
            compile_plan=compile_plan,
            hls_report_dir=(hls_dir if hls_dir is not None else out_dir),
            clock_mhz=float(getattr(compile_plan, "clock_mhz", _cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200.0))),
            verbose=verbose,
        ) if enable_reports else None
        if enable_reports:
            prediction_artifacts = self._refresh_board_fit_from_hls(
                out_dir=out_dir,
                compile_plan=compile_plan,
                hls_run=hls_run,
                prediction_artifacts=prediction_artifacts,
                build_stages=build_stages,
            )
            execution_semantics_artifacts = _write_execution_semantics_reports(
                out_dir,
                raw,
                pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
                memory_plan=memory_plan,
                communication_plan=communication_plan,
                prediction_artifacts=prediction_artifacts,
            )

        training_compare_result = None
        hardware_training_compare_result = None
        if hls_run is not None and hls_dir is not None:
            hls_grads = self._find_file_recursive(hls_dir, "grads.bin")
            hls_w_before = self._find_file_recursive(hls_dir, "weights_before.bin")
            hls_w_after = self._find_file_recursive(hls_dir, "weights_after.bin")
            if hls_grads is not None and hls_w_before is not None and hls_w_after is not None:
                training_compare_result = compare_training_artifacts(
                    out_dir=out_dir,
                    ref_grads_bin=training_reference_result.grads_flat_path,
                    ref_weights_before_bin=training_reference_result.weights_before_flat_path,
                    ref_weights_after_bin=training_reference_result.weights_after_flat_path,
                    hls_grads_bin=hls_grads,
                    hls_weights_before_bin=hls_w_before,
                    hls_weights_after_bin=hls_w_after,
                )
                print("\n" + training_compare_result.summary_txt.read_text(encoding="utf-8") + "\n")
                if training_dataset_available:
                    reference_preview = json.loads(training_reference_result.summary_json.read_text(encoding="utf-8"))
                    hardware_reference = reference_preview.get("hardware_domain_reference") or {}
                    hardware_grads = Path(str(hardware_reference.get("grads_ref_bin", "")))
                    hardware_before = Path(str(hardware_reference.get("weights_before_ref_bin", "")))
                    hardware_after = Path(str(hardware_reference.get("weights_after_ref_bin", "")))
                    if hardware_grads.exists() and hardware_before.exists() and hardware_after.exists():
                        hardware_training_compare_result = compare_training_artifacts(
                            out_dir=out_dir / "hardware_domain",
                            ref_grads_bin=hardware_grads,
                            ref_weights_before_bin=hardware_before,
                            ref_weights_after_bin=hardware_after,
                            hls_grads_bin=hls_grads,
                            hls_weights_before_bin=hls_w_before,
                            hls_weights_after_bin=hls_w_after,
                        )

        if training_dataset_available:
            reference_payload = json.loads(training_reference_result.summary_json.read_text(encoding="utf-8"))
            execution_path = out_dir / "reports" / "training_dataset_execution.json"
            execution_payload = (
                json.loads(execution_path.read_text(encoding="utf-8"))
                if execution_path.exists()
                else None
            )
            compare_payload = build_dataset_training_comparison(
                training_compare_result=hardware_training_compare_result,
                float_training_compare_result=training_compare_result,
                execution_payload=execution_payload,
                reference_payload=reference_payload,
            )
            write_text(
                out_dir / "reports" / "training_dataset_comparison.json",
                json.dumps(compare_payload, indent=2) + "\n",
            )
            hardware_reference = reference_payload.get("hardware_domain_reference") or {}
            semantic_trace_payload = build_training_semantic_trace_report(
                hls_gradient_accumulated=(self._find_file_recursive(hls_dir, "gradient_accumulated_pre_reduce.bin") if hls_dir is not None else None),
                hls_gradient_reduced=(self._find_file_recursive(hls_dir, "gradient_reduced_export.bin") if hls_dir is not None else None),
                ref_gradient_accumulated=Path(str(hardware_reference.get("gradient_accumulated_pre_reduce_ref_bin", ""))) if hardware_reference.get("gradient_accumulated_pre_reduce_ref_bin") else None,
                ref_gradient_reduced=Path(str(hardware_reference.get("gradient_reduced_ref_bin", ""))) if hardware_reference.get("gradient_reduced_ref_bin") else None,
                hls_weights_before=(self._find_file_recursive(hls_dir, "weights_before.bin") if hls_dir is not None else None),
                hls_weights_after=(self._find_file_recursive(hls_dir, "weights_after.bin") if hls_dir is not None else None),
            )
            write_text(
                out_dir / "reports" / "training_gradient_semantics.json",
                json.dumps(semantic_trace_payload, indent=2) + "\n",
            )
            hls_trace_root = None
            if hls_dir is not None:
                first_sample_trace = self._find_file_recursive(hls_dir, "per_sample_gradient_0000.bin")
                hls_trace_root = first_sample_trace.parent if first_sample_trace is not None else None
            per_sample_trace_payload = build_training_per_sample_gradient_trace_report(
                hls_trace_root=hls_trace_root,
                ref_per_sample_paths=[Path(str(value)) for value in (hardware_reference.get("per_sample_gradient_ref_bins") or [])],
                ref_accumulator_paths=[Path(str(value)) for value in (hardware_reference.get("accumulator_after_ref_bins") or [])],
                parameter_layer_map_path=(Path(str(hardware_reference.get("parameter_layer_map_json"))) if hardware_reference.get("parameter_layer_map_json") else None),
            )
            write_text(
                out_dir / "reports" / "training_per_sample_gradient_trace.json",
                json.dumps(per_sample_trace_payload, indent=2) + "\n",
            )
            by_layer_payload, by_role_payload = build_training_gradient_layer_role_reports(
                hls_trace_root=hls_trace_root,
                ref_per_sample_paths=[Path(str(value)) for value in (hardware_reference.get("per_sample_gradient_ref_bins") or [])],
                ref_accumulator_paths=[Path(str(value)) for value in (hardware_reference.get("accumulator_after_ref_bins") or [])],
                parameter_layer_map_path=(Path(str(hardware_reference.get("parameter_layer_map_json"))) if hardware_reference.get("parameter_layer_map_json") else None),
                batch_size=resolve_training_execution_schedule(
                    raw,
                    sample_count=int(training_dataset_artifacts.get("sample_count") or 1),
                ).batch_size,
            )
            write_text(out_dir / "reports" / "training_gradient_by_layer.json", json.dumps(by_layer_payload, indent=2) + "\n")
            write_text(out_dir / "reports" / "training_gradient_by_role.json", json.dumps(by_role_payload, indent=2) + "\n")
            float_initial_loss = float(training_reference_result.loss_before)
            float_final_loss = float(training_reference_result.loss_after)
            hardware_initial_loss = float(hardware_reference.get("initial_dataset_loss", float_initial_loss))
            hardware_final_loss = float(hardware_reference.get("final_dataset_loss", float_final_loss))
            hls_initial_loss = execution_payload.get("initial_loss") if execution_payload else None
            hls_final_loss = execution_payload.get("final_loss") if execution_payload else None
            hls_initial_accuracy = execution_payload.get("initial_accuracy") if execution_payload else None
            hls_final_accuracy = execution_payload.get("final_accuracy") if execution_payload else None
            hardware_initial_accuracy = hardware_reference.get("initial_accuracy")
            hardware_final_accuracy = hardware_reference.get("final_accuracy")
            float_initial_accuracy = reference_payload.get("initial_accuracy")
            float_final_accuracy = reference_payload.get("final_accuracy")
            headline_initial_loss = hardware_initial_loss
            headline_final_loss = hardware_final_loss
            loss_change = headline_final_loss - headline_initial_loss
            loss_direction = "decreased" if loss_change < 0 else ("increased" if loss_change > 0 else "unchanged")
            accuracy_change = None
            accuracy_direction = "not_available"
            if hardware_initial_accuracy is not None and hardware_final_accuracy is not None:
                accuracy_change = float(hardware_final_accuracy) - float(hardware_initial_accuracy)
                accuracy_direction = "increased" if accuracy_change > 0 else ("decreased" if accuracy_change < 0 else "unchanged")
            learning_payload = {
                "artifact_kind": "fpgai_training_learning_behavior",
                "schema_version": 2,
                "execution_status": "passed" if (out_dir / "reports" / "training_dataset_execution.json").exists() else "not_available",
                "numeric_validation_status": str(compare_payload.get("status", "pending_comparison")),
                "numeric_validation_reference_domain": str(compare_payload.get("decision_reference_domain", "hardware_fixed_point")),
                "headline_domain": "hardware_fixed_point",
                "sample_count": int(training_dataset_artifacts.get("sample_count") or 0),
                "optimizer_updates": int(reference_payload.get("optimizer_updates") or 0),
                "epochs_completed": int(reference_payload.get("epochs_completed") or 0),
                "records_consumed": int(reference_payload.get("records_consumed") or 0),
                "execution_schedule": reference_payload.get("execution_schedule"),
                "initial_dataset_loss": headline_initial_loss,
                "final_dataset_loss": headline_final_loss,
                "loss_change": loss_change,
                "loss_reduction": headline_initial_loss - headline_final_loss,
                "loss_direction": loss_direction,
                "learning_observed": bool(headline_final_loss < headline_initial_loss),
                "initial_accuracy": hardware_initial_accuracy,
                "final_accuracy": hardware_final_accuracy,
                "accuracy_change": accuracy_change,
                "accuracy_direction": accuracy_direction,
                "domains": {
                    "hls_csim": {
                        "initial_loss": hls_initial_loss,
                        "final_loss": hls_final_loss,
                        "initial_accuracy": hls_initial_accuracy,
                        "final_accuracy": hls_final_accuracy,
                    },
                    "hardware_fixed_point": {
                        "initial_loss": hardware_initial_loss,
                        "final_loss": hardware_final_loss,
                        "initial_accuracy": hardware_initial_accuracy,
                        "final_accuracy": hardware_final_accuracy,
                    },
                    "float_reference": {
                        "initial_loss": float_initial_loss,
                        "final_loss": float_final_loss,
                        "initial_accuracy": float_initial_accuracy,
                        "final_accuracy": float_final_accuracy,
                    },
                },
                "convergence_claim": "not_evaluated",
                "gradient_l1_norm": hardware_reference.get("gradient_l1_norm", reference_payload.get("gradient_l1_norm")),
                "gradient_l2_norm": hardware_reference.get("gradient_l2_norm", reference_payload.get("gradient_l2_norm")),
                "gradient_max_abs": hardware_reference.get("gradient_max_abs", reference_payload.get("gradient_max_abs")),
                "weight_update_l2_norm": hardware_reference.get("weight_update_l2_norm", reference_payload.get("weight_update_l2_norm")),
                "interpretation": (
                    ("The HLS training operation satisfies the dataset-wide hardware-domain comparison checks. " if compare_payload.get("passed") else "Learning behavior is reported, but HLS/hardware-domain numerical equivalence is not validated. ")
                    + "Headline loss and accuracy use the hardware-fixed-point decision domain; float-reference values are diagnostic. "
                    + "Loss or accuracy direction alone does not establish convergence or generalization."
                ),
            }
            learning_json = out_dir / "reports" / "training_learning_behavior.json"
            learning_md = out_dir / "reports" / "training_learning_behavior.md"
            write_text(learning_json, json.dumps(learning_payload, indent=2) + "\n")
            write_text(learning_md, "\n".join([
                "# FPGAI training learning behavior",
                "",
                f"- Headline domain: {learning_payload['headline_domain']}",
                f"- Samples processed: {learning_payload['sample_count']}",
                f"- Epochs completed: {learning_payload['epochs_completed']}",
                f"- Optimizer updates: {learning_payload['optimizer_updates']}",
                f"- Records consumed: {learning_payload['records_consumed']}",
                f"- Initial dataset loss: {headline_initial_loss:.9g}",
                f"- Final dataset loss: {headline_final_loss:.9g}",
                f"- Loss direction: {loss_direction}",
                f"- Initial accuracy: {hardware_initial_accuracy}",
                f"- Final accuracy: {hardware_final_accuracy}",
                f"- Accuracy direction: {accuracy_direction}",
                f"- Gradient L2 norm: {learning_payload['gradient_l2_norm']}",
                f"- Weight-update L2 norm: {learning_payload['weight_update_l2_norm']}",
                f"- Numerical validation: {learning_payload['numeric_validation_status']}",
                "- Convergence claim: not evaluated",
                "",
                learning_payload["interpretation"],
                "",
            ]))

        if enable_reports:
            hls_schedule_summary = self._emit_hls_schedule_summary(out_dir)
            hls_artifact_metadata = emit_hls_artifact_metadata(
                out_dir,
                compile_plan,
                schedule_summary=hls_schedule_summary,
            )
            hls_ii_comparison = write_requested_achieved_ii_summary(
                out_dir,
                compile_plan,
            )
        else:
            hls_schedule_summary = None
            hls_artifact_metadata = None
            hls_ii_comparison = None
        runtime_package = emit_runtime_package(
            out_dir,
            board=str(_cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "")) or ""),
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            top_name=top_name,
            weights_mode=memory_plan.notes.get("memory_semantics_mode", weights_mode),
            communication_plan=communication_plan,
            memory_plan=memory_plan,
            build_stages=build_stages,
            runtime_sequence=runtime_sequence,
            hls_artifacts=self._hls_artifacts_manifest_payload(
                out_dir=out_dir,
                hls_run=hls_run,
                hls_schedule_summary=hls_schedule_summary,
                hls_artifact_metadata=hls_artifact_metadata,
                hls_ii_comparison=hls_ii_comparison,
            ),
        ) if enable_runtime_package else None
        vivado_bd_contract_artifacts = _write_vivado_bd_contract_reports(
            out_dir,
            raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            build_stages=build_stages,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            hls_dir=hls_dir,
        ) if enable_reports else None
        vivado_handoff_artifacts = emit_vivado_project_handoff(
            out_dir,
            raw_config=raw,
            build_stages=build_stages,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            top_name=top_name,
            hls_dir=hls_dir,
            runtime_sequence=runtime_sequence,
            source_ports=_scan_hls_top_ports(hls_dir),
        ) if enable_reports else None
        vivado_validation_artifacts = emit_vivado_validation_reports(
            out_dir,
            raw_config=raw,
            build_stages=build_stages,
            vivado_handoff_artifacts=vivado_handoff_artifacts,
            board_fit_artifacts=(prediction_artifacts.get("board_fit") if isinstance(prediction_artifacts, dict) else None),
        ) if enable_reports else None
        hls_validation_artifacts = emit_hls_validation_reports(
            out_dir=out_dir,
            hls_dir=hls_dir,
            build_stages=build_stages,
            hls_run=hls_run,
            design_result=None,
            clock_mhz=float(getattr(compile_plan, "clock_mhz", _cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200.0))),
        ) if enable_reports else None

        _gradient_export_cfg = _cfg_get(raw, "data_movement.gradients.export", {}) or {}
        _gradient_export_requested = isinstance(_gradient_export_cfg, dict) and str(_gradient_export_cfg.get("interface", "")).lower().replace("-", "_") == "m_axi" and str(_gradient_export_cfg.get("policy", "")).lower().replace("-", "_") in {"full", "tiled"}

        def _first_existing_gradient_file(*candidates: Path) -> Path | None:
            for candidate in candidates:
                try:
                    if candidate.exists():
                        return candidate
                except OSError:
                    continue
            return None

        _preserved_validation_dir = out_dir / ".fpgai_preserved_validation"
        _gradient_ref_file = _first_existing_gradient_file(
            _preserved_validation_dir / "training_reference" / "grads_ref.bin",
            _preserved_validation_dir / "training_reference" / "gradients_after_ref.bin",
            _preserved_validation_dir / "gradients_after_ref.bin",
            _preserved_validation_dir / "reference" / "gradients_after_ref.bin",
            _preserved_validation_dir / "runtime_package" / "reference" / "grads_ref.bin",
            _preserved_validation_dir / "runtime_package" / "reference" / "gradients_after_ref.bin",
            out_dir / "training_reference" / "grads_ref.bin",
            out_dir / "training_reference" / "gradients_after_ref.bin",
            out_dir / "gradients_after_ref.bin",
            out_dir / "reference" / "gradients_after_ref.bin",
            out_dir / "runtime_package" / "reference" / "grads_ref.bin",
            out_dir / "runtime_package" / "reference" / "gradients_after_ref.bin",
        )
        _gradient_got_file = _first_existing_gradient_file(
            _preserved_validation_dir / "gradients_after.bin",
            _preserved_validation_dir / "gradients_export.bin",
            _preserved_validation_dir / "gradients_mem_after.bin",
            _preserved_validation_dir / "hls" / "gradients_after.bin",
            _preserved_validation_dir / "runtime_package" / "outputs" / "gradients_after.bin",
            _preserved_validation_dir / "runtime_package" / "outputs" / "gradients_export.bin",
            out_dir / "gradients_after.bin",
            out_dir / "gradients_export.bin",
            out_dir / "gradients_mem_after.bin",
            out_dir / "hls" / "gradients_after.bin",
            out_dir / "runtime_package" / "outputs" / "gradients_after.bin",
            out_dir / "runtime_package" / "outputs" / "gradients_export.bin",
        )
        _gradient_export_comparisons = {}
        # Only request a dedicated export comparison when both sides exist.
        # A generated training reference alone is not a failed export capture; it
        # simply means the HLS/runtime export mode has not been captured yet.
        if _gradient_ref_file is not None and _gradient_got_file is not None:
            _gradient_export_comparisons["flattened_gradients_export"] = {
                "ref": _gradient_ref_file,
                "got": _gradient_got_file,
            }

        _gradient_export_artifacts = {
            "requested": bool(_gradient_export_requested),
            "policy": (str(_gradient_export_cfg.get("policy", "none")).lower().replace("-", "_") if isinstance(_gradient_export_cfg, dict) else "none"),
            "status": ("covered_by_training_gradient_compare" if (_gradient_export_requested and training_compare_result is not None) else ("generated_not_captured_by_testbench" if _gradient_export_requested else "not_requested")),
            "export_capture_mode": 7 if _gradient_export_requested else None,
            "note": "Gradient export HLS mode is generated; dedicated gradients_mem capture is compared when gradients_after.bin/gradients_export.bin and a reference gradient file exist." if _gradient_export_requested else "Gradient export was not requested.",
            "comparisons": _gradient_export_comparisons,
            "capture_files": {
                "gradients_ref_bin": str(_gradient_ref_file) if _gradient_ref_file is not None else None,
                "gradients_after_bin": str(_gradient_got_file) if _gradient_got_file is not None else None,
            },
        }
        _optimizer_contract = _resolve_training_optimizer_loss_contract(raw)
        _optimizer_type = str(_optimizer_contract.get("optimizer", {}).get("type", "sgd")).lower().replace("-", "_")
        _optimizer_cfg_raw = (raw.get("training", {}) or {}).get("optimizer", {}) if isinstance(raw.get("training", {}), dict) else {}
        _optimizer_bias_correction = bool(_optimizer_cfg_raw.get("bias_correction", False)) if isinstance(_optimizer_cfg_raw, dict) else False
        _optimizer_state = _optimizer_contract.get("optimizer_state", {}) or {}
        _optimizer_state_required = bool(_optimizer_state.get("required", False))
        _optimizer_state_export = _optimizer_state.get("export", {}) if isinstance(_optimizer_state.get("export", {}), dict) else {}
        _optimizer_state_requested = bool(_optimizer_state_required or _optimizer_state_export.get("supported", False))
        _state_names = []
        if _optimizer_type == "momentum":
            _state_names = ["velocity"]
        elif _optimizer_type == "adam":
            _state_names = ["first_moment", "second_moment", "optimizer_step"]

        def _first_existing_optimizer_state_file(*candidates: Path) -> Path | None:
            for candidate in candidates:
                try:
                    if candidate.exists():
                        return candidate
                except OSError:
                    continue
            return None

        _optimizer_state_before_ref_file = _first_existing_optimizer_state_file(
            _preserved_validation_dir / "training_reference" / "optimizer_state_before_ref.bin",
            _preserved_validation_dir / "optimizer_state_before_ref.bin",
            _preserved_validation_dir / "reference" / "optimizer_state_before_ref.bin",
            _preserved_validation_dir / "runtime_package" / "reference" / "optimizer_state_before_ref.bin",
            out_dir / "training_dataset_reference" / "hardware_domain" / "optimizer_state_before_ref.bin",
            out_dir / "hardware_domain" / "optimizer_state_before_ref.bin",
            out_dir / "training_reference" / "optimizer_state_before_ref.bin",
            out_dir / "optimizer_state_before_ref.bin",
            out_dir / "reference" / "optimizer_state_before_ref.bin",
            out_dir / "runtime_package" / "reference" / "optimizer_state_before_ref.bin",
        )
        _optimizer_state_ref_file = _first_existing_optimizer_state_file(
            _preserved_validation_dir / "training_reference" / "optimizer_state_after_ref.bin",
            _preserved_validation_dir / "optimizer_state_after_ref.bin",
            _preserved_validation_dir / "reference" / "optimizer_state_after_ref.bin",
            _preserved_validation_dir / "runtime_package" / "reference" / "optimizer_state_after_ref.bin",
            out_dir / "training_dataset_reference" / "hardware_domain" / "optimizer_state_after_ref.bin",
            out_dir / "hardware_domain" / "optimizer_state_after_ref.bin",
            out_dir / "training_reference" / "optimizer_state_after_ref.bin",
            out_dir / "optimizer_state_after_ref.bin",
            out_dir / "reference" / "optimizer_state_after_ref.bin",
            out_dir / "runtime_package" / "reference" / "optimizer_state_after_ref.bin",
        )
        _optimizer_state_before_got_file = _first_existing_optimizer_state_file(
            _preserved_validation_dir / "optimizer_state_before.bin",
            _preserved_validation_dir / "hls" / "optimizer_state_before.bin",
            _preserved_validation_dir / "runtime_package" / "outputs" / "optimizer_state_before.bin",
            out_dir / "optimizer_state_before.bin",
            out_dir / "hls" / "optimizer_state_before.bin",
            out_dir / "runtime_package" / "outputs" / "optimizer_state_before.bin",
        )
        _optimizer_state_got_file = _first_existing_optimizer_state_file(
            _preserved_validation_dir / "optimizer_state_after.bin",
            _preserved_validation_dir / "hls" / "optimizer_state_after.bin",
            _preserved_validation_dir / "runtime_package" / "outputs" / "optimizer_state_after.bin",
            out_dir / "optimizer_state_after.bin",
            out_dir / "hls" / "optimizer_state_after.bin",
            out_dir / "hls" / "fpgai_hls_proj" / "sol1" / "csim" / "build" / "optimizer_state_after.bin",
            out_dir / "runtime_package" / "outputs" / "optimizer_state_after.bin",
        )
        _optimizer_state_comparisons = {}
        if _optimizer_state_before_got_file is not None:
            _optimizer_state_comparisons["packed_optimizer_state_before"] = {
                "ref": _optimizer_state_before_ref_file,
                "got": _optimizer_state_before_got_file,
            }
        if _optimizer_state_ref_file is not None or _optimizer_state_got_file is not None:
            _optimizer_state_comparisons["packed_optimizer_state_after"] = {
                "ref": _optimizer_state_ref_file,
                "got": _optimizer_state_got_file,
            }

        _optimizer_state_artifacts = {
            "requested": _optimizer_state_requested,
            "optimizer": _optimizer_type,
            "storage": _optimizer_state.get("storage", "none"),
            "expected_tensors": _state_names,
            "layout": ("m_then_v_then_step_canonical_parameter_order" if _optimizer_type == "adam" else "canonical_parameter_order"),
            "layout_version": 2 if _optimizer_type == "adam" else 1,
            "reference_domain": "hardware_domain_fixed_point",
            "claim_scope": "hls_csim_optimizer_state_numeric_validation",
            "bias_correction": _optimizer_bias_correction if _optimizer_type == "adam" else False,
            "bias_correction_status": ("enabled" if (_optimizer_type == "adam" and _optimizer_bias_correction) else ("disabled" if _optimizer_type == "adam" else "not_applicable")),
            "status": (
                "generated_export_capture_supported"
                if (_optimizer_state_requested and _optimizer_state_export.get("supported", False))
                else ("generated_not_captured_by_testbench" if _optimizer_state_requested else "not_requested")
            ),
            "export_capture_mode": 9 if (_optimizer_state_requested and _optimizer_state_export.get("supported", False)) else None,
            "note": (
                "Persistent optimizer-state tensors are generated and export_optimizer_state mode 9 can write them to optimizer_state_mem; runtime/testbench must capture got/ref files for numeric proof."
                if (_optimizer_state_requested and _optimizer_state_export.get("supported", False))
                else (
                    "Persistent optimizer-state tensors are generated in HLS; runtime/testbench capture must provide ref/got files for numeric proof."
                    if _optimizer_state_requested
                    else "Optimizer does not require persistent state."
                )
            ),
            "comparisons": _optimizer_state_comparisons,
            "capture_files": {
                "optimizer_state_before_ref_bin": str(_optimizer_state_before_ref_file) if _optimizer_state_before_ref_file is not None else None,
                "optimizer_state_before_bin": str(_optimizer_state_before_got_file) if _optimizer_state_before_got_file is not None else None,
                "optimizer_state_after_ref_bin": str(_optimizer_state_ref_file) if _optimizer_state_ref_file is not None else None,
                "optimizer_state_after_bin": str(_optimizer_state_got_file) if _optimizer_state_got_file is not None else None,
            },
        }
        _parameter_update_ref_file = _first_existing_optimizer_state_file(
            out_dir / "training_dataset_reference" / "hardware_domain" / "weights_after_ref.bin",
            out_dir / "hardware_domain" / "weights_after_ref.bin",
            out_dir / "training_reference" / "weights_after_ref.bin",
        )
        _parameter_update_got_file = (
            self._find_file_recursive(hls_dir, "weights_after.bin")
            if hls_dir is not None
            else None
        )
        _parameter_update_artifacts = {
            "requested": True,
            "reference_domain": "hardware_domain_fixed_point",
            "claim_scope": "hls_csim_parameter_update_numeric_validation",
            "ref": _parameter_update_ref_file,
            "got": _parameter_update_got_file,
        }
        numeric_validation_artifacts = emit_numeric_validation_report(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            source_generated=(hls_dir is not None),
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            hls_csynth_report=(hls_run.csynth_report if hls_run is not None else None),
            training_reference_result=training_reference_result,
            training_compare_result=training_compare_result,
            gradient_export_artifacts=_gradient_export_artifacts,
            optimizer_state_artifacts=_optimizer_state_artifacts,
            parameter_update_artifacts=_parameter_update_artifacts,
            raw_config=raw,
            runtime_sequence=runtime_sequence,
        ) if enable_reports else None
        generated_hls_explanation_artifacts = emit_generated_hls_explanation_reports(
            out_dir,
            raw_config=raw,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            top_name=top_name,
            hls_dir=hls_dir,
            build_stages=build_stages,
            runtime_sequence=runtime_sequence,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
            numeric_validation_artifacts=numeric_validation_artifacts,
        ) if enable_reports else None
        precision_layout_artifacts_for_effect = self._emit_precision_layout_reports(
            out_dir=out_dir,
            graph=g,
            descriptors=descriptors,
            compile_plan=compile_plan,
        ) if enable_reports else None
        precision_effect_artifacts = emit_precision_effect_reports(
            out_dir=out_dir,
            raw_config=raw,
            hls_dir=hls_dir,
            precision_layout_artifacts=precision_layout_artifacts_for_effect,
            quant_result=None,
            sweep_result=None,
            hls_validation_artifacts=hls_validation_artifacts,
        ) if enable_reports else None
        parallel_pipeline_effect_artifacts = emit_parallel_pipeline_effect_reports(
            out_dir=out_dir,
            raw_config=raw,
            hls_dir=hls_dir,
            hls_validation_artifacts=hls_validation_artifacts,
        ) if enable_reports else None
        movement_contract_validation_artifacts = emit_movement_contract_validation(
            out_dir,
            data_movement_artifacts=data_movement_artifacts,
        ) if enable_reports else None
        if data_movement_artifacts is not None and movement_contract_validation_artifacts is not None:
            data_movement_artifacts = dict(data_movement_artifacts)
            data_movement_artifacts.update(movement_contract_validation_artifacts)

        validation_summary_artifacts = emit_validation_summary_artifacts(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            source_generated=(hls_dir is not None),
            numeric_validation_json=(
                numeric_validation_artifacts.get("numeric_validation_json")
                if numeric_validation_artifacts is not None
                else None
            ),
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            build_stages=build_stages,
        ) if enable_reports else None
        feature_validation_artifacts = _write_feature_validation_reports(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            build_stages=build_stages,
            hls_dir=hls_dir,
            hls_run=hls_run,
            runtime_package=runtime_package,
            numeric_validation_artifacts=numeric_validation_artifacts,
            validation_summary_artifacts=validation_summary_artifacts,
            vivado_bd_contract_artifacts=vivado_bd_contract_artifacts,
        ) if enable_reports else None

        if emit_manifest:
            self._emit_manifest(
                hls_schedule_summary=hls_schedule_summary,
                hls_artifact_metadata=hls_artifact_metadata,
                hls_ii_comparison=hls_ii_comparison,
                runtime_package=runtime_package,
                build_stages=build_stages,
                runtime_sequence=runtime_sequence,
                resolved_config_artifacts=resolved_config_artifacts,
                numeric_validation_artifacts=numeric_validation_artifacts,
                generated_hls_explanation_artifacts=generated_hls_explanation_artifacts,
                precision_effect_artifacts=precision_effect_artifacts,
                parallel_pipeline_effect_artifacts=parallel_pipeline_effect_artifacts,
                data_movement_artifacts=data_movement_artifacts,
                validation_summary_artifacts=validation_summary_artifacts,
                vivado_bd_contract_artifacts=vivado_bd_contract_artifacts,
                vivado_handoff_artifacts=vivado_handoff_artifacts,
                vivado_validation_artifacts=vivado_validation_artifacts,
                hls_validation_artifacts=hls_validation_artifacts,
                feature_validation_artifacts=feature_validation_artifacts,
                out_dir=out_dir,
                top_name=top_name,
                weights_mode=weights_mode,
                graph=g,
                descriptors=descriptors,
                compile_plan=compile_plan,
                memory_plan=memory_plan,
                communication_plan=communication_plan,
                capability_report=capability_report,
                hls_run=hls_run,
                quant_result=None,
                sweep_result=None,
                design_result=None,
                estimate_vs_hls_result=None,
                hls_module_breakdown_result=None,
                prediction_artifacts=prediction_artifacts,
                training_plan=training_plan,
                training_reference_result=training_reference_result,
                training_compare_result=training_compare_result,
                training_estimate_result=training_estimate_result,
                seconds=time.time() - t0,
            )

        if emit_manifest:
            emit_experiment_artifact_reports(out_dir)
            _run_yaml_requested_vivado_bridge(out_dir, raw, build_stages)

        if verbose:
            print("[FPGAI] training mode enabled")
            print(f"[FPGAI] training optimizer: {training_plan.optimizer_type}")
            print(f"[FPGAI] training loss: {training_plan.loss_type}")
            print(f"[FPGAI] training weights_mode: {training_plan.weights_mode}")
            print(f"[FPGAI] training parallel policy: {training_plan.planner_policy.get('parallel_policy')}")
            print(f"[FPGAI] training reference: {training_reference_result.summary_txt}")
            if training_estimate_result is not None:
                print(f"[FPGAI] training estimate: {training_estimate_result.summary_txt}")
            if training_compare_result is not None:
                print(f"[FPGAI] training compare: {training_compare_result.summary_txt}")

        return CompileResult(
            out_dir=out_dir,
            graph=g,
            hls_project_dir=hls_dir,
            host_project_dir=host_dir,
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            hls_returncode=(hls_run.returncode if hls_run is not None else None),
            hls_stdout_log=(hls_run.stdout_log if hls_run is not None else None),
            hls_stderr_log=(hls_run.stderr_log if hls_run is not None else None),
            hls_csynth_report=(hls_run.csynth_report if hls_run is not None else None),
            quant_report_dir=None,
            quant_metrics_json=None,
            quant_summary_txt=None,
            quant_layerwise_csv=None,
            precision_sweep_dir=None,
            precision_sweep_results_json=None,
            precision_sweep_summary_txt=None,
            precision_sweep_results_csv=None,
            design_space_dir=None,
            design_space_results_json=None,
            design_space_summary_txt=None,
            design_space_results_csv=None,
            design_space_layer_breakdown_csv=None,
            design_space_terminal_summary=None,
            estimate_vs_hls_dir=None,
            estimate_vs_hls_results_json=None,
            estimate_vs_hls_summary_txt=None,
            hls_module_breakdown_dir=None,
            hls_module_breakdown_json=None,
            hls_module_breakdown_csv=None,
            hls_module_breakdown_summary_txt=None,
            training_plan_json=(out_dir / "training" / "training_plan.json"),
            training_summary_txt=(out_dir / "training" / "summary.txt"),
        )

    def _prepare_out_dir(self, raw: Dict[str, Any]) -> Path:
        out_dir = Path(_cfg_get(raw, "project.out_dir", "build/fpgai")).resolve()
        clean = bool(_cfg_get(raw, "project.clean", True))

        # Preserve externally captured validation artifacts across the normal clean
        # compile.  Some CSim/runtime steps write payloads before a follow-up
        # compile/audit pass.  Cleaning the project directory must not silently
        # discard those ref/got files before numeric_validation.json can compare
        # them.  Keep this list intentionally narrow: only externally produced
        # runtime/testbench capture validation files are restored.
        preserve_relpaths = [
            Path("gradients_after.bin"),
            Path("gradients_export.bin"),
            Path("gradients_mem_after.bin"),
            Path("gradients_after_ref.bin"),
            Path("training_reference") / "grads_ref.bin",
            Path("training_reference") / "gradients_after_ref.bin",
            Path("reference") / "gradients_after_ref.bin",
            Path("runtime_package") / "outputs" / "gradients_after.bin",
            Path("runtime_package") / "outputs" / "gradients_export.bin",
            Path("runtime_package") / "reference" / "grads_ref.bin",
            Path("runtime_package") / "reference" / "gradients_after_ref.bin",
            Path("optimizer_state_before.bin"),
            Path("optimizer_state_before_ref.bin"),
            Path("optimizer_state_after.bin"),
            Path("optimizer_state_after_ref.bin"),
            Path("training_reference") / "optimizer_state_before_ref.bin",
            Path("training_reference") / "optimizer_state_after_ref.bin",
            Path("reference") / "optimizer_state_after_ref.bin",
            Path("runtime_package") / "outputs" / "optimizer_state_before.bin",
            Path("runtime_package") / "outputs" / "optimizer_state_after.bin",
            Path("runtime_package") / "reference" / "optimizer_state_before_ref.bin",
            Path("runtime_package") / "reference" / "optimizer_state_after_ref.bin",
        ]
        preserved: list[tuple[Path, bytes]] = []
        if clean and out_dir.exists():
            for rel in preserve_relpaths:
                candidate = out_dir / rel
                try:
                    if candidate.is_file():
                        preserved.append((rel, candidate.read_bytes()))
                except OSError:
                    # Preserve is best-effort; the compiler should not fail just
                    # because a stale capture disappeared while starting a clean
                    # build.
                    continue

        ensure_clean_dir(out_dir, clean=clean)

        for rel, data in preserved:
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            preserved_target = out_dir / ".fpgai_preserved_validation" / rel
            preserved_target.parent.mkdir(parents=True, exist_ok=True)
            preserved_target.write_bytes(data)
        return out_dir

    def _read_activation_insert_cfg(self, raw: Dict[str, Any]) -> tuple[str, float, bool]:
        act_cfg = _cfg_get(raw, "operators.defaults.activation_insert", {}) or {}
        return (
            str(act_cfg.get("kind", "none")).lower(),
            float(act_cfg.get("alpha", 0.1)),
            bool(act_cfg.get("except_last", True)),
        )

    def _import_and_prepare_graph(self, *, act_kind: str, act_alpha: float, act_except_last: bool):
        from fpgai.frontend.onnx import import_onnx
        from fpgai.ir.passes import validate_allowlist, assign_stable_names, insert_activations

        g = import_onnx(self.cfg.model.path, canonicalize=True, infer_shapes=True)
        if act_kind != "none":
            g = insert_activations(g, kind=act_kind, alpha=act_alpha, except_last=act_except_last)
        g = assign_stable_names(g)
        validate_allowlist(g, self.cfg.operators.supported)
        return g

    def _emit_prediction_artifacts(
        self,
        out_dir: Path,
        descriptors,
        compile_plan,
    ) -> dict[str, str]:
        """Write pre-HLS model/resource/timing prediction artifacts."""
        raw = self.cfg.raw
        reports_dir = out_dir / "reports"

        inspection = inspect_config(self.cfg)

        resource_prediction = estimate_resources_from_descriptors(
            descriptors,
            raw,
            compile_plan=compile_plan,
        )
        timing_prediction = estimate_performance(
            resource_estimate=resource_prediction,
            raw_cfg=raw,
        )

        resource_prediction = dict(resource_prediction)
        timing_prediction = dict(timing_prediction)

        resource_prediction["prediction_kind"] = "pre_hls_resource_estimate"
        resource_prediction["prediction_status"] = "estimate"
        resource_prediction["model_path"] = str(self.cfg.model.path)
        resource_prediction["descriptor_count"] = len(descriptors)
        resource_prediction["architecture_signature"] = getattr(
            compile_plan,
            "architecture_signature",
            None,
        )

        timing_prediction["prediction_kind"] = "pre_hls_timing_estimate"
        timing_prediction["prediction_status"] = "estimate"
        timing_prediction["model_path"] = str(self.cfg.model.path)
        timing_prediction["descriptor_count"] = len(descriptors)
        timing_prediction["architecture_signature"] = getattr(
            compile_plan,
            "architecture_signature",
            None,
        )

        prediction_artifacts = write_model_inspection_report(
            inspection,
            reports_dir,
            resource_prediction=resource_prediction,
            timing_prediction=timing_prediction,
        )

        compatibility_artifacts = emit_model_compatibility_reports(
            reports_dir,
            inspection,
            raw_cfg=raw,
        )
        prediction_artifacts.update(
            {key: str(value) for key, value in compatibility_artifacts.items()}
        )

        board = str(
            _cfg_get(
                raw,
                "targets.platform.board",
                _cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "")),
            )
            or ""
        )
        part = str(_cfg_get(raw, "targets.platform.part", "") or "")
        target_clock_mhz = getattr(compile_plan, "clock_mhz", _cfg_get(raw, "targets.platform.clocks.0.target_mhz", None))

        board_fit_artifacts = emit_board_fit_report(
            reports_dir,
            resource_data=resource_prediction,
            timing_data=timing_prediction,
            board=board,
            part=part,
            target_clock_mhz=target_clock_mhz,
            source="prediction",
            raw_config=raw,
            build_stages=_resolve_build_stages(raw),
        )
        prediction_artifacts["board_fit"] = board_fit_artifacts
        prediction_artifacts["board_fit_json"] = board_fit_artifacts.get("json")
        prediction_artifacts["board_fit_markdown"] = board_fit_artifacts.get("markdown")

        return prediction_artifacts

    def _refresh_board_fit_from_hls(
        self,
        *,
        out_dir: Path,
        compile_plan,
        hls_run: Any | None,
        prediction_artifacts: dict[str, Any],
        build_stages: dict[str, bool],
    ) -> dict[str, Any]:
        """Promote successful C-synthesis resources to the active board-fit source.

        The pre-HLS prediction fit is preserved as a separate artifact. The
        canonical reports/board_fit.json remains the active fit used by stage
        gating and manifests, so later stages do not continue to act on stale
        estimator values after csynth.rpt is available.
        """
        csynth_report = getattr(hls_run, "csynth_report", None) if hls_run is not None else None
        if not (hls_run is not None and getattr(hls_run, "ok", False) and csynth_report):
            return prediction_artifacts
        csynth_path = Path(csynth_report)
        if not csynth_path.is_file():
            return prediction_artifacts

        reports_dir = out_dir / "reports"
        active_json = reports_dir / "board_fit.json"
        active_md = reports_dir / "board_fit.md"
        prediction_payload: dict[str, Any] = {}
        if active_json.is_file():
            try:
                prediction_payload = json.loads(active_json.read_text(encoding="utf-8"))
            except Exception:
                prediction_payload = {}
        prediction_json = reports_dir / "board_fit_prediction.json"
        prediction_md = reports_dir / "board_fit_prediction.md"
        if prediction_payload:
            prediction_json.write_text(json.dumps(prediction_payload, indent=2, sort_keys=True), encoding="utf-8")
            if active_md.is_file():
                prediction_md.write_text(active_md.read_text(encoding="utf-8"), encoding="utf-8")

        actual = parse_hls_csynth_report(csynth_path)
        board = str(
            _cfg_get(
                self.cfg.raw,
                "targets.platform.board",
                _cfg_get(self.cfg.raw, "targets.board", _cfg_get(self.cfg.raw, "project.board", "")),
            )
            or ""
        )
        part = str(_cfg_get(self.cfg.raw, "targets.platform.part", "") or "")
        target_clock_mhz = getattr(
            compile_plan,
            "clock_mhz",
            _cfg_get(self.cfg.raw, "targets.platform.clocks.0.target_mhz", None),
        )
        hls_summary = emit_board_fit_report(
            reports_dir,
            resource_data=actual,
            timing_data={},
            board=board,
            part=part,
            target_clock_mhz=target_clock_mhz,
            source="hls_synthesis",
            raw_config=self.cfg.raw,
            build_stages=build_stages,
        )
        hls_payload = json.loads(active_json.read_text(encoding="utf-8"))
        hls_snapshot = dict(hls_payload)
        hls_json = reports_dir / "board_fit_hls_synthesis.json"
        hls_md = reports_dir / "board_fit_hls_synthesis.md"
        hls_json.write_text(json.dumps(hls_snapshot, indent=2, sort_keys=True), encoding="utf-8")
        if active_md.is_file():
            hls_md.write_text(active_md.read_text(encoding="utf-8"), encoding="utf-8")

        hls_payload["format"] = "fpgai.board_fit.v2"
        hls_payload["active_fit_source"] = "hls_synthesis"
        hls_payload["available_fit_sources"] = [
            source for source, present in (("prediction", bool(prediction_payload)), ("hls_synthesis", True)) if present
        ]
        hls_payload["prediction_fit"] = prediction_payload or None
        hls_payload["hls_synthesis_fit"] = hls_snapshot
        hls_payload["csynth_report"] = str(csynth_path)
        active_json.write_text(json.dumps(hls_payload, indent=2, sort_keys=True), encoding="utf-8")

        result = dict(prediction_artifacts)
        result["board_fit"] = {
            **hls_summary,
            "source": "hls_synthesis",
            "active_fit_source": "hls_synthesis",
        }
        result["board_fit_json"] = str(active_json)
        result["board_fit_markdown"] = str(active_md)
        result["board_fit_prediction_json"] = str(prediction_json) if prediction_payload else None
        result["board_fit_prediction_markdown"] = str(prediction_md) if prediction_payload else None
        result["board_fit_hls_synthesis_json"] = str(hls_json)
        result["board_fit_hls_synthesis_markdown"] = str(hls_md)
        return result

    def _emit_ir_artifacts(self, out_dir: Path, g, descriptors, compile_plan, memory_plan, communication_plan) -> None:
        write_text(out_dir / "ir_summary.txt", g.summary())
        part_plan = single_device_plan(g, device_id="fpga0")
        write_text(out_dir / "partition_plan.json", json.dumps(part_plan.to_dict(), indent=2))

        ir_dir = out_dir / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        write_text(ir_dir / "descriptors.json", json.dumps([d.to_dict() for d in descriptors], indent=2))
        write_text(ir_dir / "compile_plan.json", json.dumps(compile_plan.to_dict(), indent=2))
        write_text(ir_dir / "memory_plan.json", json.dumps(memory_plan.to_dict(), indent=2))
        write_text(ir_dir / "comm_plan.json", json.dumps(communication_plan.to_dict(), indent=2))

        prec_dump = []
        for idx, op in enumerate(g.ops):
            prec_dump.append(
                {
                    "index": idx,
                    "name": op.name,
                    "op_type": op.op_type,
                    "precision": op.attrs.get("precision"),
                    "precision_tag": op.attrs.get("precision_tag"),
                }
            )
        write_text(ir_dir / "layerwise_precision.json", json.dumps(prec_dump, indent=2))

    def _emit_dummy_input(self, out_dir: Path, g) -> Path:
        p = out_dir / "input.bin"
        if p.exists():
            return p
        x_name = g.inputs[0]
        x_spec = g.get_tensor(x_name)
        in_words = int(np.prod(tuple(int(d) for d in x_spec.shape))) if x_spec and x_spec.shape else 1
        x = (np.arange(in_words, dtype=np.float32) + 1.0) * 0.1
        write_f32_bin(p, x)
        return p

    def _emit_training_target(self, out_dir: Path, g, raw: Dict[str, Any]) -> Path:
        p = out_dir / "target.bin"
        if p.exists():
            return p
        y_name = g.outputs[0]
        y_spec = g.get_tensor(y_name)
        out_words = 1
        if y_spec is not None and getattr(y_spec, "shape", None):
            shape = tuple(int(x) for x in y_spec.shape)
            if len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            out_words = int(np.prod(shape)) if shape else 1
        target = np.zeros((out_words,), dtype=np.float32)
        if out_words > 0:
            target[0] = 1.0
        write_f32_bin(p, target)
        return p

    def _emit_training_dataset_target(
        self,
        out_dir: Path,
        g,
        raw: Dict[str, Any],
        dataset_artifacts: Dict[str, Any],
    ) -> Path:
        """Materialize dataset labels/targets as the training target record stream.

        Classification labels are lowered to one-hot float32 records matching the
        model output width. Regression targets are preserved as float32 records.
        """
        dataset_dir = out_dir / "validation" / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target_path = dataset_dir / "training_targets.bin"

        sample_count = int(dataset_artifacts.get("sample_count") or 1)
        labels_path = dataset_artifacts.get("labels_path")
        targets_path = dataset_artifacts.get("targets_path")

        if targets_path:
            targets = np.asarray(np.load(targets_path), dtype=np.float32)
            if targets.shape[0] != sample_count:
                raise ValueError(
                    f"training dataset target count {targets.shape[0]} does not match sample_count {sample_count}"
                )
            write_f32_bin(target_path, targets.reshape(-1))
            return target_path

        if not labels_path:
            raise ValueError(
                "training dataset requires labels or targets; no labels_path/targets_path was emitted"
            )

        labels = np.asarray(np.load(labels_path), dtype=np.int64).reshape(-1)
        if labels.size != sample_count:
            raise ValueError(
                f"training dataset label count {labels.size} does not match sample_count {sample_count}"
            )

        y_name = g.outputs[0]
        y_spec = g.get_tensor(y_name)
        out_words = 1
        if y_spec is not None and getattr(y_spec, "shape", None):
            shape = tuple(int(x) for x in y_spec.shape)
            if len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            out_words = int(np.prod(shape)) if shape else 1
        if out_words <= 1:
            raise ValueError(
                "classification training dataset requires a model output width greater than one"
            )
        if np.any(labels < 0) or np.any(labels >= out_words):
            raise ValueError(
                f"training dataset labels must be in [0, {out_words - 1}]"
            )

        targets = np.zeros((sample_count, out_words), dtype=np.float32)
        targets[np.arange(sample_count), labels] = 1.0
        write_f32_bin(target_path, targets.reshape(-1))
        return target_path


    def _emit_held_out_dataset_target(
        self,
        out_dir: Path,
        g,
        dataset_artifacts: Dict[str, Any],
    ) -> Path:
        """Materialize held-out supervision independently from training targets."""
        dataset_dir = out_dir / "validation" / "held_out_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target_path = dataset_dir / "validation_targets.bin"
        sample_count = int(dataset_artifacts.get("sample_count") or 0)
        targets_path = dataset_artifacts.get("targets_path")
        labels_path = dataset_artifacts.get("labels_path")
        if targets_path:
            targets = np.asarray(np.load(targets_path), dtype=np.float32)
            if targets.shape[0] != sample_count:
                raise ValueError("held-out target count does not match sample_count")
            write_f32_bin(target_path, targets.reshape(-1))
            return target_path
        if not labels_path:
            raise ValueError("held-out dataset requires labels or targets")
        labels = np.asarray(np.load(labels_path), dtype=np.int64).reshape(-1)
        if labels.size != sample_count:
            raise ValueError("held-out label count does not match sample_count")
        y_spec = g.get_tensor(g.outputs[0])
        shape = tuple(int(x) for x in getattr(y_spec, "shape", ()) or ())
        if len(shape) > 1 and shape[0] == 1:
            shape = shape[1:]
        out_words = int(np.prod(shape)) if shape else 1
        if out_words <= 1 or np.any(labels < 0) or np.any(labels >= out_words):
            raise ValueError("held-out labels are incompatible with model output width")
        targets = np.zeros((sample_count, out_words), dtype=np.float32)
        targets[np.arange(sample_count), labels] = 1.0
        write_f32_bin(target_path, targets.reshape(-1))
        return target_path

    def _emit_training_validation_execution_report(
        self,
        *,
        out_dir: Path,
        hls_dir: Optional[Path],
        held_out_dataset_artifacts: Dict[str, Any],
    ) -> Optional[Path]:
        if hls_dir is None or held_out_dataset_artifacts.get("status") != "available":
            return None
        source = self._find_file_recursive(hls_dir, "held_out_validation_summary.json")
        if source is None:
            return None
        report_dir = out_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["artifact_kind"] = "fpgai_training_validation_execution"
        payload["dataset_sample_count"] = int(held_out_dataset_artifacts.get("sample_count") or 0)
        payload["statistical_generalization_claim"] = False
        for filename, public_name, key in (
            ("held_out_curve.csv", "training_validation_curve.csv", "curve_csv"),
            ("held_out_predictions_before.csv", "training_validation_predictions_before.csv", "predictions_before_csv"),
            ("held_out_predictions_after.csv", "training_validation_predictions_after.csv", "predictions_after_csv"),
        ):
            artifact = self._find_file_recursive(hls_dir, filename)
            if artifact is not None:
                destination = report_dir / public_name
                destination.write_bytes(artifact.read_bytes())
                payload[key] = str(destination)
        before = payload.get("before") or {}
        after = payload.get("after") or {}
        payload["loss_change"] = float(after.get("average_loss", 0.0)) - float(before.get("average_loss", 0.0))
        payload["accuracy_change"] = float(after.get("accuracy", 0.0)) - float(before.get("accuracy", 0.0))
        report_path = report_dir / "training_validation_execution.json"
        write_text(report_path, json.dumps(payload, indent=2) + "\n")
        write_text(
            report_dir / "training_validation_summary.md",
            "# FPGAI held-out HLS validation summary\n\n"
            f"- Samples: `{payload['dataset_sample_count']}`\n"
            f"- Before loss: `{before.get('average_loss')}`\n"
            f"- After loss: `{after.get('average_loss')}`\n"
            f"- Before accuracy: `{before.get('accuracy')}`\n"
            f"- After accuracy: `{after.get('accuracy')}`\n"
            f"- Claim scope: `{payload.get('claim_scope')}`\n"
            "- Statistical generalization claim: `False`\n",
        )
        return report_path

    def _emit_training_dataset_execution_report(
        self,
        *,
        out_dir: Path,
        hls_dir: Optional[Path],
        training_dataset_artifacts: dict[str, Any],
    ) -> Optional[Path]:
        """Publish the canonical dataset-training execution report.

        This helper is intentionally owned by the training compile path. Inference
        compilation must not reference dataset-training locals or emit training
        execution artifacts.
        """
        if hls_dir is None or training_dataset_artifacts.get("status") != "available":
            return None

        multistep_summary = self._find_file_recursive(
            hls_dir,
            "training_multistep_summary.json",
        )
        if multistep_summary is None:
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        canonical_summary = out_dir / "training_multistep_summary.json"
        canonical_summary.write_bytes(multistep_summary.read_bytes())

        execution_payload = json.loads(
            multistep_summary.read_text(encoding="utf-8")
        )
        sample_count = int(training_dataset_artifacts.get("sample_count") or 0)
        record_visits = int(
            execution_payload.get(
                "dataset_records_consumed",
                execution_payload.get("total_train_calls", 0),
            )
        )

        report_dir = out_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        copied_artifacts: dict[str, Any] = {}
        for filename, public_name, key in (
            ("training_epoch_curve.csv", "training_epoch_curve_hls.csv", "training_epoch_curve_csv"),
            ("training_batch_curve.csv", "training_batch_curve_hls.csv", "training_batch_curve_csv"),
        ):
            source = self._find_file_recursive(hls_dir, filename)
            if source is not None:
                destination = report_dir / public_name
                destination.write_bytes(source.read_bytes())
                copied_artifacts[key] = str(destination)

        checkpoint_files = sorted(
            {
                str(path)
                for path in hls_dir.rglob("epoch_*_weights.bin")
                if path.is_file()
            }
        )
        execution_payload.update(
            {
                "artifact_kind": "fpgai_training_dataset_execution",
                "schema_version": 2,
                # Compatibility fields retained for existing consumers.
                "sample_count_requested": sample_count,
                "sample_count_executed": record_visits,
                "reference_samples_executed": sample_count,
                # Canonical multi-epoch execution fields.
                "dataset_sample_count": sample_count,
                "unique_records_executed": sample_count,
                "record_visits_executed": record_visits,
                "forward_backward_calls": int(execution_payload.get("total_forward_backward_calls", execution_payload.get("total_train_calls", 0))),
                "optimizer_updates": int(execution_payload.get("optimizer_update_calls", 0)),
                "batches_completed": int(execution_payload.get("optimizer_update_calls", 0)),
                # Deprecated compatibility alias retained through the schema transition.
                "unique_dataset_records": sample_count,
                "checkpoint_count": len(checkpoint_files),
                "checkpoint_files": checkpoint_files,
                **copied_artifacts,
            }
        )
        report_path = report_dir / "training_dataset_execution.json"
        write_text(report_path, json.dumps(execution_payload, indent=2) + "\n")
        return report_path

    def _find_file_recursive(self, root: Path, filename: str) -> Optional[Path]:
        for p in root.rglob(filename):
            return p
        return None

    def _maybe_run_vitis_hls(self, hls_dir: Path, *, build_stages: Optional[Dict[str, bool]] = None):
        raw = self.cfg.raw
        if build_stages is None:
            run_enabled = bool(_cfg_get(raw, "toolchain.vitis_hls.enabled", False))
        else:
            run_enabled = bool(build_stages.get("hls_synthesis", False))
        if not run_enabled:
            return None
        vitis_exe = str(
            _cfg_get(
                raw,
                "backends.hls.vitis.exe",
                _cfg_get(
                    raw,
                    "toolchain.vitis_hls.exe",
                    _cfg_get(
                        raw,
                        "toolchain.vitis_hls.executable",
                        _cfg_get(raw, "toolchain.vitis_hls.path", "vitis_hls"),
                    ),
                ),
            )
        )
        settings64 = _cfg_get(
            raw,
            "toolchain.vitis_hls.settings64",
            _cfg_get(raw, "toolchain.vitis_hls.settings", None),
        )
        from fpgai.backends.hls.runner import HLSRunResult, run_vitis_hls
        try:
            return run_vitis_hls(hls_dir=hls_dir, vitis_hls_exe=vitis_exe, settings64=settings64)
        except FileNotFoundError as exc:
            logs_dir = hls_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = logs_dir / "vitis_hls_stdout.log"
            stderr_log = logs_dir / "vitis_hls_stderr.log"
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text(f"{exc}\n", encoding="utf-8")
            return HLSRunResult(
                ok=False,
                returncode=127,
                command=str(vitis_exe),
                workdir=str(hls_dir),
                stdout_log=str(stdout_log.resolve()),
                stderr_log=str(stderr_log.resolve()),
                csynth_report=None,
            )

    def _emit_hostcpp(self, out_dir: Path, g, *, top_name: str) -> Path:
        pipeline_mode = str(getattr(self.cfg.pipeline, "mode", "inference")).lower()
        if pipeline_mode == "training_on_device":
            from fpgai.backends.hostcpp.emit_host_train import emit_hostcpp_project_train
            return emit_hostcpp_project_train(g, out_dir, top_name=top_name, raw_cfg=self.cfg.raw)
        from fpgai.backends.hostcpp.emit_host_model import emit_hostcpp_project
        return emit_hostcpp_project(g, out_dir, top_name=top_name)

    def _validate_architecture(
        self,
        out_dir: Path,
        compile_plan,
        memory_plan,
    ):
        strict = bool(
            _cfg_get(
                self.cfg.raw,
                "optimization.capabilities.strict",
                False,
            )
        )
        report = validate_architecture_capabilities(
            compile_plan,
            memory_plan=memory_plan,
            pipeline_mode=self.cfg.pipeline.mode,
            strict=False,
        )
        analysis_dir = out_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        write_text(
            analysis_dir / "architecture_capabilities.json",
            json.dumps(report.to_dict(), indent=2),
        )
        write_text(
            analysis_dir / "architecture_capabilities.txt",
            report.summary(),
        )

        if strict:
            return validate_architecture_capabilities(
                compile_plan,
                memory_plan=memory_plan,
                pipeline_mode=self.cfg.pipeline.mode,
                strict=True,
            )

        return report

    def _emit_hls_schedule_summary(self, out_dir: Path) -> dict[str, Any] | None:
        """Discover HLS schedule reports and write one normalized summary.

        This is intentionally best-effort: normal compilation should not fail
        just because no HLS report exists yet, or because a vendor report has
        an unexpected format.
        """
        summary_path = out_dir / "hls_schedule_summary.json"

        try:
            write_hls_schedule_summary(
                out_dir,
                summary_path,
            )
        except Exception as exc:
            write_text(
                out_dir / "hls_schedule_summary_error.txt",
                f"{type(exc).__name__}: {exc}\n",
            )
            return None

        if not summary_path.exists():
            return None

        try:
            data = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            write_text(
                out_dir / "hls_schedule_summary_error.txt",
                f"Failed to read generated schedule summary: "
                f"{type(exc).__name__}: {exc}\n",
            )
            return None

        if not isinstance(data, dict):
            return None

        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        reports = data.get("reports", [])
        if not isinstance(reports, list):
            reports = []

        if not summary:
            loop_count = 0
            for report in reports:
                if not isinstance(report, dict):
                    continue

                report_summary = report.get("summary", {})
                if isinstance(report_summary, dict):
                    try:
                        loop_count += int(report_summary.get("loop_count", 0))
                        continue
                    except Exception:
                        pass

                loops = report.get("loops", [])
                if isinstance(loops, list):
                    loop_count += len(loops)

            summary = {
                "report_count": len(reports),
                "loop_count": loop_count,
            }
            data["summary"] = summary
            summary_path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )

        report_count_raw = summary.get("report_count", len(reports))
        try:
            report_count = int(report_count_raw)
        except Exception:
            report_count = len(reports)

        if report_count <= 0:
            try:
                summary_path.unlink()
            except FileNotFoundError:
                pass
            return None

        return {
            "path": str(summary_path.relative_to(out_dir)),
            "summary": summary,
        }

    @staticmethod
    def _pipeline_stage(
        name: str,
        status: str,
        *,
        detail: str = "",
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "name": name,
            "status": status,
        }
        if detail:
            row["detail"] = detail
        if artifacts:
            row["artifacts"] = artifacts
        return row

    def _design_space_manifest_payload(self, design_result) -> Dict[str, Any] | None:
        if design_result is None:
            return None

        payload: Dict[str, Any] = {
            "prediction_status": "estimate",
            "out_dir": str(design_result.out_dir),
            "results_json": str(design_result.results_json),
            "summary_txt": str(design_result.summary_txt),
            "results_csv": str(design_result.results_csv),
            "layer_breakdown_csv": str(design_result.out_dir / "layer_breakdown.csv"),
        }

        try:
            data = json.loads(design_result.results_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        if isinstance(data, dict):
            for key in (
                "format",
                "analytical_models",
                "recommendation_policy",
                "recommendation_scope",
                "search_enabled",
                "recommendation_kind",
                "dse_validation",
                "recommended_smallest_valid",
                "recommended_balanced",
                "recommended_best_accuracy",
            ):
                if key in data:
                    payload[key] = data[key]

        return payload

    def _hls_artifacts_manifest_payload(
        self,
        *,
        out_dir: Path,
        hls_run,
        hls_schedule_summary,
        hls_artifact_metadata,
        hls_ii_comparison,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "hls_ran": hls_run is not None,
            "hls_ok": hls_run.ok if hls_run is not None else None,
            "hls_returncode": hls_run.returncode if hls_run is not None else None,
            "hls_csim_ran": getattr(hls_run, "csim_ran", None) if hls_run is not None else None,
            "hls_csim_ok": getattr(hls_run, "csim_ok", None) if hls_run is not None else None,
            "hls_csynth_ran": getattr(hls_run, "csynth_ran", None) if hls_run is not None else None,
            "hls_csynth_ok": getattr(hls_run, "csynth_ok", None) if hls_run is not None else None,
            "hls_failure_stage": getattr(hls_run, "failure_stage", None) if hls_run is not None else None,
            "hls_failure_reason": getattr(hls_run, "failure_reason", None) if hls_run is not None else None,
            "hls_project_dir": str(out_dir / "hls"),
            "stdout_log": (
                str(hls_run.stdout_log)
                if hls_run is not None and hls_run.stdout_log is not None
                else None
            ),
            "stderr_log": (
                str(hls_run.stderr_log)
                if hls_run is not None and hls_run.stderr_log is not None
                else None
            ),
            "csynth_report": (
                str(hls_run.csynth_report)
                if hls_run is not None and hls_run.csynth_report is not None
                else None
            ),
            "schedule_summary": hls_schedule_summary,
            "artifact_metadata": hls_artifact_metadata,
            "ii_comparison": hls_ii_comparison,
        }

        return payload

    def _build_pipeline_stages(
        self,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Describe the effective compile pipeline in the existing manifest.

        This is traceability metadata only. It does not create a new pipeline
        orchestrator and does not claim that Vivado/runtime stages ran through
        the main compile command.
        """
        raw = self.cfg.raw
        graph = kwargs.get("graph")
        compile_plan = kwargs.get("compile_plan")
        memory_plan = kwargs.get("memory_plan")
        communication_plan = kwargs.get("communication_plan")
        hls_run = kwargs.get("hls_run")
        training_plan = kwargs.get("training_plan")

        build_stages = kwargs.get("build_stages") or _resolve_build_stages(raw)
        hls_enabled = bool(build_stages.get("cpp", False))
        hls_project_enabled = bool(build_stages.get("hls_project", False))
        host_cpp_enabled = bool(build_stages.get("host_cpp", False))

        stages: List[Dict[str, Any]] = [
            self._pipeline_stage(
                "load_config",
                "done",
                detail="YAML configuration loaded and normalized.",
            ),
            self._pipeline_stage(
                "import_model",
                "done",
                detail="Model imported into FPGAI IR.",
                artifacts={
                    "num_ops": len(getattr(graph, "ops", []) or []),
                    "num_params": len(getattr(graph, "params", {}) or {}),
                },
            ),
            self._pipeline_stage(
                "analyze_model",
                "done",
                detail="Graph descriptors, capability report, memory plan, and communication plan generated.",
                artifacts={
                    "num_descriptors": len(kwargs.get("descriptors", []) or []),
                    "num_memory_placements": len(getattr(memory_plan, "placements", []) or []),
                    "num_communication_edges": len(getattr(communication_plan, "edges", []) or []),
                },
            ),
            self._pipeline_stage(
                "plan_architecture",
                "done",
                detail="Compile plan generated.",
                artifacts={
                    "num_layer_plans": len(getattr(compile_plan, "layer_plans", []) or []),
                    "architecture_signature": getattr(compile_plan, "architecture_signature", None),
                },
            ),
        ]

        optional_results = [
            ("quantization_report", kwargs.get("quant_result"), "Optional quantization report."),
            ("precision_sweep", kwargs.get("sweep_result"), "Optional precision sweep."),
            ("design_space", kwargs.get("design_result"), "Optional design-space report."),
            ("estimate_vs_hls", kwargs.get("estimate_vs_hls_result"), "Optional estimate-vs-HLS report."),
            ("hls_module_breakdown", kwargs.get("hls_module_breakdown_result"), "Optional HLS module breakdown report."),
        ]

        for name, result, detail in optional_results:
            if result is None:
                stages.append(
                    self._pipeline_stage(
                        name,
                        "skipped",
                        detail=f"{detail} Not requested or unavailable.",
                    )
                )
                continue

            artifacts = {
                key: str(value)
                for key, value in {
                    "out_dir": getattr(result, "out_dir", None),
                    "summary_txt": getattr(result, "summary_txt", None),
                    "results_json": getattr(result, "results_json", None),
                }.items()
                if value is not None
            }
            stages.append(
                self._pipeline_stage(
                    name,
                    "done",
                    detail=detail,
                    artifacts=artifacts,
                )
            )

        stages.append(
            self._pipeline_stage(
                "generate_host_cpp",
                "done" if host_cpp_enabled else "skipped",
                detail=(
                    "Host C++ reference artifacts requested."
                    if host_cpp_enabled
                    else "Host C++ backend disabled in config."
                ),
            )
        )

        stages.append(
            self._pipeline_stage(
                "generate_cpp",
                "done" if hls_enabled else "skipped",
                detail=(
                    "Generated HLS-compatible C++ source/include artifacts."
                    if hls_enabled
                    else "C++ generation disabled by build stages."
                ),
            )
        )

        stages.append(
            self._pipeline_stage(
                "generate_hls_project",
                "done" if hls_project_enabled else "skipped",
                detail=(
                    "HLS project/run script artifacts requested."
                    if hls_project_enabled
                    else "C++-only mode: HLS project/run script artifacts not requested."
                ),
            )
        )

        if not hls_enabled:
            hls_status = "skipped"
            hls_detail = "HLS backend disabled in config."
        elif hls_run is None:
            hls_status = "skipped"
            hls_detail = "Vitis HLS run was not requested or not reached."
        elif hls_run.ok:
            hls_status = "done"
            hls_detail = "Vitis HLS run completed successfully."
        else:
            hls_status = "failed"
            hls_detail = "Vitis HLS run failed; inspect HLS logs."

        stages.append(
            self._pipeline_stage(
                "run_hls",
                hls_status,
                detail=hls_detail,
                artifacts=(
                    {
                        key: str(value)
                        for key, value in {
                            "returncode": getattr(hls_run, "returncode", None),
                            "stdout_log": getattr(hls_run, "stdout_log", None),
                            "stderr_log": getattr(hls_run, "stderr_log", None),
                            "csynth_report": getattr(hls_run, "csynth_report", None),
                            "csim_ran": getattr(hls_run, "csim_ran", None),
                            "csim_ok": getattr(hls_run, "csim_ok", None),
                            "csynth_ran": getattr(hls_run, "csynth_ran", None),
                            "csynth_ok": getattr(hls_run, "csynth_ok", None),
                            "failure_stage": getattr(hls_run, "failure_stage", None),
                            "failure_reason": getattr(hls_run, "failure_reason", None),
                        }.items()
                        if value is not None
                    }
                    if hls_run is not None
                    else None
                ),
            )
        )

        stages.append(
            self._pipeline_stage(
                "training_artifacts",
                "done" if training_plan is not None else "skipped",
                detail=(
                    "Training plan and reference artifacts generated."
                    if training_plan is not None
                    else "Pipeline mode is not training_on_device."
                ),
            )
        )

        stages.append(
            self._pipeline_stage(
                "vivado_project",
                "not_requested" if not build_stages.get("vivado_project") else "requested_external_flow",
                detail=(
                    "Vivado project was requested in build stages; execution remains in the Vivado bridge flow."
                    if build_stages.get("vivado_project")
                    else "Vivado project is not requested by build stages."
                ),
            )
        )

        stages.append(
            self._pipeline_stage(
                "bitstream",
                "not_requested" if not build_stages.get("bitstream") else "requested_external_flow",
                detail=(
                    "Bitstream was requested in build stages; execution remains in the Vivado bridge flow."
                    if build_stages.get("bitstream")
                    else "Bitstream is not requested by build stages."
                ),
            )
        )

        stages.append(
            self._pipeline_stage(
                "runtime_package",
                "done" if kwargs.get("runtime_package") is not None else "skipped",
                detail=(
                    "Runtime package manifest emitted under runtime_package/."
                    if kwargs.get("runtime_package") is not None
                    else "Runtime package was not emitted."
                ),
                artifacts=kwargs.get("runtime_package"),
            )
        )

        return stages


    @staticmethod
    def _shape_element_count(shape) -> int:
        if shape in (None, "", []):
            return 0
        try:
            total = 1
            seen = False
            for value in shape:
                if value in (None, "", "?"):
                    return 0
                ivalue = int(value)
                if ivalue <= 0:
                    return 0
                total *= ivalue
                seen = True
            return int(total) if seen else 0
        except Exception:
            return 0

    @classmethod
    def _tensor_element_count(cls, value) -> int:
        if value is None:
            return 0

        size = getattr(value, "size", None)
        if size is not None:
            try:
                return int(size)
            except Exception:
                pass

        shape = getattr(value, "shape", None)
        count = cls._shape_element_count(shape)
        if count:
            return count

        if isinstance(value, dict):
            for key in ("shape", "dims"):
                count = cls._shape_element_count(value.get(key))
                if count:
                    return count
            for key in ("values", "data", "array"):
                if key in value:
                    count = cls._tensor_element_count(value[key])
                    if count:
                        return count

        if isinstance(value, (list, tuple)):
            if not value:
                return 0
            if all(not isinstance(x, (list, tuple, dict)) for x in value):
                return len(value)
            return sum(cls._tensor_element_count(x) for x in value)

        return 0

    @classmethod
    def _graph_parameter_counts(cls, graph) -> dict:
        counts = {
            "weight_elements": 0,
            "bias_elements": 0,
            "parameter_elements": 0,
        }

        params = getattr(graph, "params", {}) or {}

        if isinstance(params, dict):
            items = params.items()
        else:
            try:
                items = enumerate(params)
            except Exception:
                items = []

        for name, value in items:
            n = cls._tensor_element_count(value)
            if n <= 0:
                continue

            lname = str(name).lower()
            counts["parameter_elements"] += n

            if (
                lname.startswith("b")
                or "bias" in lname
                or lname.endswith(".b")
                or lname.endswith("_b")
            ):
                counts["bias_elements"] += n
            else:
                counts["weight_elements"] += n

        return counts

    @classmethod
    def _object_to_builtin(cls, value):
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {str(k): cls._object_to_builtin(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [cls._object_to_builtin(v) for v in value]

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return cls._object_to_builtin(to_dict())
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return {
                    str(k): cls._object_to_builtin(v)
                    for k, v in vars(value).items()
                    if not str(k).startswith("_")
                }
            except Exception:
                pass

        return None

    @classmethod
    def _shape_candidates_from_object(cls, value):
        data = cls._object_to_builtin(value)
        out = []

        def walk(node, path):
            if isinstance(node, dict):
                for key, child in node.items():
                    key_s = str(key).lower()
                    next_path = path + [key_s]

                    if key_s in {
                        "shape",
                        "dims",
                        "dim",
                        "input_shape",
                        "input_shapes",
                        "inputs_shape",
                        "output_shape",
                        "output_shapes",
                        "outputs_shape",
                        "activation_shape",
                        "activation_shapes",
                        "buffer_shape",
                    }:
                        count = cls._shape_element_count(child)
                        if count:
                            out.append((next_path, count))

                    walk(child, next_path)

            elif isinstance(node, list):
                # A plain integer list can itself be a shape.
                if node and all(isinstance(x, int) for x in node):
                    count = cls._shape_element_count(node)
                    if count:
                        out.append((path, count))
                else:
                    for i, child in enumerate(node):
                        walk(child, path + [str(i)])

        walk(data, [])
        return out

    @classmethod
    def _classify_shape_count(cls, path) -> str:
        joined = ".".join(path).lower()

        if any(tok in joined for tok in ("weight", "param", "kernel", "bias")):
            return "ignore"

        if any(tok in joined for tok in ("input", "in_shape", "inputs")):
            return "input"

        if any(tok in joined for tok in ("output", "out_shape", "outputs", "result")):
            return "output"

        if any(tok in joined for tok in ("activation", "buffer", "tensor")):
            return "activation"

        return "activation"

    @classmethod
    def _graph_io_activation_counts(cls, graph, descriptors=None, compile_plan=None) -> dict:
        counts = {
            "input_elements": 0,
            "output_elements": 0,
            "activation_buffer_elements": 0,
        }

        # Source 1: graph op attrs.
        ops = list(getattr(graph, "ops", []) or [])
        for index, op in enumerate(ops):
            attrs = getattr(op, "attrs", {}) or {}
            if not isinstance(attrs, dict):
                continue

            input_shape = (
                attrs.get("input_shape")
                or attrs.get("input_shapes")
                or attrs.get("in_shape")
                or attrs.get("shape_in")
            )
            output_shape = (
                attrs.get("output_shape")
                or attrs.get("output_shapes")
                or attrs.get("out_shape")
                or attrs.get("shape_out")
                or attrs.get("shape")
            )

            in_count = cls._shape_element_count(input_shape)
            out_count = cls._shape_element_count(output_shape)

            if index == 0 and in_count:
                counts["input_elements"] = in_count
            if out_count:
                counts["output_elements"] = out_count
                counts["activation_buffer_elements"] += out_count

        # Source 2: descriptors and compile-plan layer plans. This is usually
        # where normalized imported-model tensor metadata is available.
        objects = []
        objects.extend(list(descriptors or []))
        if compile_plan is not None:
            objects.append(compile_plan)
            objects.extend(list(getattr(compile_plan, "layer_plans", []) or []))

        first_input_seen = False
        output_candidates = []
        activation_sum = 0

        for obj in objects:
            for path, count in cls._shape_candidates_from_object(obj):
                role = cls._classify_shape_count(path)
                if role == "ignore" or count <= 0:
                    continue

                if role == "input":
                    if not first_input_seen:
                        counts["input_elements"] = counts["input_elements"] or count
                        first_input_seen = True
                    activation_sum += count
                elif role == "output":
                    output_candidates.append(count)
                    activation_sum += count
                elif role == "activation":
                    activation_sum += count

        if output_candidates:
            counts["output_elements"] = counts["output_elements"] or output_candidates[-1]

        if activation_sum:
            counts["activation_buffer_elements"] = counts["activation_buffer_elements"] or activation_sum

        # Source 3: graph-level shape fields.
        for key, target in (
            ("input_shape", "input_elements"),
            ("inputs_shape", "input_elements"),
            ("output_shape", "output_elements"),
            ("outputs_shape", "output_elements"),
        ):
            if counts[target]:
                continue
            shape = getattr(graph, key, None)
            n = cls._shape_element_count(shape)
            if n:
                counts[target] = n

        return counts


    def _emit_precision_layout_reports(self, **kwargs) -> dict:
        out_dir = kwargs["out_dir"]
        reports_dir = out_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        graph = kwargs["graph"]
        param_counts = self._graph_parameter_counts(graph)
        io_counts = self._graph_io_activation_counts(
            graph,
            descriptors=kwargs.get("descriptors"),
            compile_plan=kwargs.get("compile_plan"),
        )

        layout = build_precision_layout(
            self.cfg.raw,
            input_elements=io_counts["input_elements"],
            output_elements=io_counts["output_elements"],
            weight_elements=param_counts["weight_elements"],
            bias_elements=param_counts["bias_elements"],
            activation_buffer_elements=io_counts["activation_buffer_elements"],
        )

        json_path = reports_dir / "precision_layout.json"
        md_path = reports_dir / "precision_layout.md"

        write_text(json_path, json.dumps(layout, indent=2))
        write_text(md_path, precision_layout_markdown(layout))

        return {
            "json": str(json_path),
            "markdown": str(md_path),
            "precision_mode": layout.get("precision_mode"),
            "bits": layout.get("bits"),
            "pack_factors": layout.get("pack_factors"),
            "raw_bytes": layout.get("raw_bytes"),
            "packed_transfer_bytes": layout.get("packed_transfer_bytes"),
        }

    @staticmethod
    def _raw_has_path(raw: Dict[str, Any], path: str) -> bool:
        cur: Any = raw
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
                continue
            if isinstance(cur, list) and part.isdigit():
                idx = int(part)
                if 0 <= idx < len(cur):
                    cur = cur[idx]
                    continue
            return False
        return True

    @staticmethod
    def _raw_get_path(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
        cur: Any = raw
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
                continue
            if isinstance(cur, list) and part.isdigit():
                idx = int(part)
                if 0 <= idx < len(cur):
                    cur = cur[idx]
                    continue
            return default
        return cur

    @staticmethod
    def _layer_plan_dicts(compile_plan) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for lp in getattr(compile_plan, "layer_plans", []) or []:
            if hasattr(lp, "to_dict"):
                try:
                    out.append(lp.to_dict())
                    continue
                except Exception:
                    pass
            if isinstance(lp, dict):
                out.append(lp)
        return out

    @staticmethod
    def _contract_status(requested: Any, effective: Any, *, manual: bool) -> str:
        if requested is None and effective is None:
            return "unknown"

        if effective is None:
            return "not_requested"

        if manual:
            try:
                if isinstance(requested, (int, float)) and isinstance(effective, (int, float)):
                    return "applied" if float(requested) == float(effective) else "changed_or_clamped"
            except Exception:
                pass

            return "applied" if str(requested) == str(effective) else "changed_or_clamped"

        return "applied"

    def _emit_hardware_knob_contract_reports(self, **kwargs) -> dict[str, Any]:
        """Write user-facing traceability for YAML hardware decisions.

        This is intentionally conservative: it reports what the compiler can
        prove from the YAML, compile plan, and layer plans. It must not claim a
        knob affects HLS/Vivado unless the generated artifacts expose that path.
        """
        out_dir = kwargs["out_dir"]
        reports_dir = out_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        raw = self.cfg.raw
        compile_plan = kwargs["compile_plan"]
        notes = getattr(compile_plan, "notes", {}) or {}
        layer_plans = self._layer_plan_dicts(compile_plan)

        policy_resource_awareness = notes.get("policy_resource_awareness", {}) or {}
        board_aware_changed = {
            "optimization.parallel.pe": "pe",
            "optimization.parallel.simd": "simd",
            "optimization.parallel.unroll_factor": "unroll_factor",
            "optimization.parallel.partition_factor": "partition_factor",
            "targets.platform.clocks.0.target_mhz": "target_clock_mhz",
        }

        def source_for(path: str) -> str:
            if self._raw_has_path(raw, path):
                return "manual_yaml"
            changes = policy_resource_awareness.get("changes", {})
            changed_key = board_aware_changed.get(path)
            if changed_key and changed_key in changes:
                return "board_aware_policy"
            if "policy" in path or path.startswith("optimization."):
                return "policy_preset"
            return "compiler_default"

        def requested(path: str) -> Any:
            return self._raw_get_path(raw, path, None)

        def _dict_path_value(obj: dict[str, Any], path: str) -> tuple[bool, Any]:
            cur: Any = obj
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return False, None
            return True, cur

        def first_layer_value(*paths: str) -> Any:
            if not layer_plans:
                return None
            for path in paths:
                ok, value = _dict_path_value(layer_plans[0], path)
                if ok:
                    return value
            return None

        def first_layer_of_type_value(op_type: str, *paths: str) -> Any:
            wanted = str(op_type).lower()
            for lp in layer_plans:
                actual = str(lp.get("op_type", "")).lower()
                if actual != wanted:
                    continue
                for path in paths:
                    ok, value = _dict_path_value(lp, path)
                    if ok:
                        return value
            return None

        def has_layer_type(op_type: str) -> bool:
            wanted = str(op_type).lower()
            return any(str(lp.get("op_type", "")).lower() == wanted for lp in layer_plans)

        contract: list[dict[str, Any]] = []

        def add(
            path: str,
            effective: Any,
            *,
            applied_to: list[str],
            note: str = "",
            status: str | None = None,
        ) -> None:
            req = requested(path)
            manual = self._raw_has_path(raw, path)
            contract.append(
                {
                    "path": path,
                    "source": source_for(path),
                    "requested": req,
                    "effective": effective,
                    "status": status or self._contract_status(req, effective, manual=manual),
                    "applied_to": applied_to,
                    "note": note,
                }
            )

        add(
            "optimization.parallel_policy",
            notes.get("parallel_policy"),
            applied_to=["planner.policy", "compile_plan.notes.parallel_policy"],
            note="Policy is a preset only. Manual YAML overrides below have priority.",
        )
        add(
            "optimization.parallel.pe",
            notes.get("parallel_pe", first_layer_value("architecture.parallelism.pe")),
            applied_to=[
                "planner.policy.pe",
                "layer_plan.architecture.parallelism.pe",
                "Dense output unroll / Conv output-channel unroll",
                "generated HLS template args and artifact comments",
            ],
        )
        add(
            "optimization.parallel.simd",
            notes.get("parallel_simd", first_layer_value("architecture.parallelism.simd")),
            applied_to=[
                "planner.policy.simd",
                "layer_plan.architecture.parallelism.simd",
                "Dense input unroll / Conv input-channel unroll",
                "generated HLS template args and artifact comments",
            ],
        )
        add(
            "optimization.parallel.unroll_factor",
            notes.get("parallel_unroll_factor", first_layer_value("architecture.parallelism.unroll.element")),
            applied_to=[
                "planner.policy.unroll_factor",
                "elementwise activation unroll",
                "FPGAI_ACT_UNROLL macro",
            ],
        )
        add(
            "optimization.parallel.partition_factor",
            notes.get("parallel_partition_factor", first_layer_value("architecture.partitioning.factor")),
            applied_to=[
                "planner.policy.partition_factor",
                "layer_plan.architecture.partitioning.factor",
                "input/output/weight/gradient partition targets",
                "generated HLS template args and ARRAY_PARTITION factors",
            ],
        )
        add(
            "optimization.parallel.array_partition_mode",
            notes.get("parallel_array_partition_mode", first_layer_value("architecture.partitioning.mode")),
            applied_to=[
                "planner.policy.array_partition_mode",
                "layer_plan.architecture.partitioning.mode",
                "HLS ARRAY_PARTITION mode where supported",
            ],
        )
        add(
            "training.gradients.materialization",
            notes.get("gradient_materialization", "full"),
            applied_to=[
                "generated training gradient export structure",
                "OUT_grad scratch-array materialization",
                "training_resource_ownership",
            ],
            note="full preserves per-layer scratch arrays; tiled uses bounded tiles; streamed writes directly.",
        )
        add(
            "training.gradients.tile_size",
            notes.get("gradient_materialization_tile_size", 256),
            applied_to=[
                "generated gradient export tile dimensions",
                "HLS loop bounds for tiled materialization",
            ],
            status=("not_applicable" if notes.get("gradient_materialization", "full") != "tiled" else None),
        )
        add(
            "training.memory_lifetime.policy",
            notes.get("training_memory_lifetime_policy", "separate"),
            applied_to=[
                "gradient export tile physical ownership",
                "training_resource_ownership reuse groups",
            ],
            note="phase_shared reuses one physical export tile; separate preserves per-layer ownership.",
        )

        add(
            "optimization.pipeline.style",
            first_layer_value("architecture.pipeline.style", "pipeline_style"),
            applied_to=[
                "planner.pipeline_style",
                "layer_plan.architecture.pipeline.style",
                "pipeline II lowering",
                "generated HLS artifact comments",
            ],
        )
        add(
            "optimization.pipeline.ii",
            first_layer_value("architecture.pipeline.ii", "pipeline_ii"),
            applied_to=[
                "planner.pipeline_ii",
                "layer_plan.pipeline_ii",
                "FPGAI_PIPELINE_II macro / HLS template args",
            ],
            note="Manual II overrides policy-derived pipeline style.",
        )
        dense_tiling_effective = first_layer_of_type_value(
            "Dense",
            "architecture.tiling.sizes",
            "tile",
        )
        add(
            "optimization.tiling.dense",
            dense_tiling_effective,
            applied_to=[
                "planner dense tile selection",
                "layer_plan.architecture.tiling",
                "dense_tiling_codegen rewrite when Dense layers are present",
            ],
            note="Layer-specific tiling can override global dense tiling.",
            status=(
                None
                if has_layer_type("Dense")
                else "not_applicable"
            ),
        )

        conv_tiling_effective = first_layer_of_type_value(
            "Conv",
            "architecture.tiling.sizes",
            "tile",
        )
        add(
            "optimization.tiling.conv",
            conv_tiling_effective,
            applied_to=[
                "planner conv tile selection",
                "layer_plan.architecture.tiling",
                "conv_tiling_codegen rewrite when Conv layers are present",
            ],
            note="Layer-specific tiling can override global conv tiling.",
            status=(
                None
                if has_layer_type("Conv")
                else "not_applicable"
            ),
        )
        add(
            "optimization.tiling.layers",
            self._raw_get_path(raw, "optimization.tiling.layers", None),
            applied_to=[
                "planner layer-specific tile selection",
                "layer_plan.architecture.tiling for matching layer names",
            ],
            note="Manual layer entries have priority over global tiling defaults.",
            status="applied" if self._raw_has_path(raw, "optimization.tiling.layers") else "not_requested",
        )
        add(
            "memory.weight_storage",
            notes.get("weight_storage", self._raw_get_path(raw, "memory.weight_storage", None)),
            applied_to=[
                "memory plan",
                "weight storage pragmas",
                "embedded/stream/runtime weight path selection",
            ],
        )
        add(
            "memory.weight_region_preference",
            notes.get("weight_region_preference", None),
            applied_to=["planner memory policy", "layer_plan.memory.weight_region"],
        )
        add(
            "memory.activation_region_preference",
            notes.get("activation_region_preference", None),
            applied_to=["planner memory policy", "layer_plan.memory.activation_region"],
        )
        add(
            "memory.allow_double_buffer",
            notes.get("allow_double_buffer", None),
            applied_to=["planner buffering policy", "layer_plan.memory.double_buffer"],
        )
        add(
            "targets.platform.board",
            self._raw_get_path(raw, "targets.platform.board", self._raw_get_path(raw, "targets.board", None)),
            applied_to=[
                "board registry",
                "board_fit.json",
                "Vivado bridge board selection when requested",
            ],
        )
        add(
            "targets.platform.clocks.0.target_mhz",
            getattr(compile_plan, "clock_mhz", None),
            applied_to=[
                "compile_plan.clock_mhz",
                "timing_prediction.json",
                "board_fit.json clock classification",
                "HLS/Vivado clock when backend is enabled",
            ],
        )
        fit_policy, fit_policy_source, requested_fit_policy = self._resolved_fit_policy()
        add(
            fit_policy_source,
            fit_policy,
            applied_to=[
                "board-fit reporting now",
                "fit_policy_gate manifest decision",
                "Vivado/bitstream gating decision",
            ],
            status="changed_or_clamped" if requested_fit_policy != fit_policy else "applied",
            note=(
                f"requested_fit_policy={requested_fit_policy!r}; "
                f"effective_fit_policy={fit_policy!r}. "
                "fit_policy is enforced through fit_policy_gate. Supported paths are "
                "normalized through the shared compiler resolver before Vivado/bitstream gating."
            ),
        )

        # Normalized PS/PL data_movement schema reporting.
        # Legacy ps_pl/pl_ps rows remain supported; these rows expose the
        # normalized schema when it is present in YAML.
        normalized_data_movement_rows = [
            (
                "data_movement.input.load",
                "data_movement.input.load",
                [
                    "communication planner input edge",
                    "PS-to-PL load interface selection",
                    "runtime/HLS input transfer metadata",
                ],
            ),
            (
                "data_movement.output.store",
                "data_movement.output.store",
                [
                    "communication planner output edge",
                    "PL-to-PS store interface selection",
                    "runtime/HLS output transfer metadata",
                ],
            ),
            (
                "data_movement.weights.load.interface",
                "data_movement.weights.load.interface",
                [
                    "compiler weight-mode resolver",
                    "HLS weight import/storage path selection",
                    "runtime package weight payload requirement",
                ],
            ),
            (
                "data_movement.weights.store.interface",
                "data_movement.weights.store.interface",
                [
                    "weight export/store schema",
                    "training/runtime weight movement metadata when enabled",
                ],
            ),
        ]
        for report_path, value_path, applied_to in normalized_data_movement_rows:
            if self._raw_has_path(raw, value_path):
                add(
                    report_path,
                    self._raw_get_path(raw, value_path, None),
                    applied_to=applied_to,
                    status="applied",
                    note="Normalized data_movement schema path. Legacy ps_pl/pl_ps paths remain supported.",
                )

        payload = {
            "format": "fpgai.hardware_knob_contract.v1",
            "board_fit_status": "not_evaluated",
            "board_fit_reason": (
                "Capacity/resource feasibility is recorded as a contract placeholder here; "
                "full board-fit enforcement is handled by the dedicated board-fit validation/report path."
            ),
            "precedence": [
                "manual_yaml_override",
                "board_aware_policy_scaling",
                "policy_preset",
                "compiler_default",
            ],
            "validation_boundary": {
                "planner_trace": True,
                "hls_trace": "through generated macros/comments/template args where available",
                "vivado_trace": "requires Vivado report/bitstream stages",
                "runtime_trace": "requires real board runtime artifacts",
            },
            "knobs": contract,
        }

        json_path = reports_dir / "hardware_knob_contract.json"
        md_path = reports_dir / "hardware_knob_contract.md"

        write_text(json_path, json.dumps(payload, indent=2, sort_keys=True))

        lines = [
            "# FPGAI hardware knob contract",
            "",
            "Precedence:",
            "1. manual YAML override",
            "2. policy preset",
            "3. compiler default",
            "",
            "| YAML path | source | requested | effective | status | applied to |",
            "|---|---|---|---|---|---|",
        ]
        for item in contract:
            applied = "<br>".join(str(x) for x in item.get("applied_to", []))
            lines.append(
                "| {path} | {source} | `{requested}` | `{effective}` | {status} | {applied} |".format(
                    path=item.get("path"),
                    source=item.get("source"),
                    requested=item.get("requested"),
                    effective=item.get("effective"),
                    status=item.get("status"),
                    applied=applied,
                )
            )
            if item.get("note"):
                lines.append(f"|  |  |  |  | note | {item['note']} |")

        lines.extend(
            [
                "",
                "## Validation boundary",
                "",
                "- This report proves YAML-to-planner traceability.",
                "- HLS traceability is proven where generated macros, comments, template arguments, or pragmas expose the knob.",
                "- Vivado and runtime validation require real Vivado reports, bitstreams, and board execution artifacts.",
                "- If a manual YAML knob appears as `unknown`, `not_requested`, `changed_or_clamped`, or `report_only`, it must not be claimed as fully implemented until a later implementation change fixes or validates it.",
                "",
            ]
        )
        write_text(md_path, "\n".join(lines))

        return {
            "json": str(json_path),
            "markdown": str(md_path),
            "knob_count": len(contract),
            "manual_yaml_count": sum(1 for x in contract if x.get("source") == "manual_yaml"),
            "changed_or_clamped_count": sum(1 for x in contract if x.get("status") == "changed_or_clamped"),
            "report_only_count": sum(1 for x in contract if x.get("status") == "report_only"),
        }


    def _resolved_fit_policy(self) -> tuple[str, str, Any]:
        raw = self.cfg.raw
        paths = (
            "targets.platform.fit_policy",
            "hardware.fit_policy",
            "build.fit_policy",
        )
        requested: Any = "report_only"
        source = "compiler_default"
        for path in paths:
            if self._raw_has_path(raw, path):
                requested = self._raw_get_path(raw, path, None)
                source = path
                break

        policy = str(requested or "report_only").strip().lower()
        aliases = {"block_over_limit": "enforce"}
        policy = aliases.get(policy, policy)
        if policy not in {"report_only", "warn", "enforce"}:
            policy = "report_only"
            source = "invalid_fallback"
        return policy, source, requested

    def _fit_policy_gate(self, prediction_artifacts: dict[str, Any] | None) -> dict[str, Any]:
        policy, policy_source, requested_policy = self._resolved_fit_policy()

        board_fit = {}
        if isinstance(prediction_artifacts, dict):
            board_fit = prediction_artifacts.get("board_fit") or {}
        if not isinstance(board_fit, dict):
            board_fit = {}

        status = board_fit.get("status", "unknown")
        vivado_allowed = board_fit.get("vivado_allowed")
        over_limit = bool(status == "over_limit" or vivado_allowed is False)

        blocked = bool(policy == "enforce" and over_limit)
        warning = bool(policy == "warn" and over_limit)

        blocked_stages = []
        if blocked:
            blocked_stages = [
                "vivado_impl",
                "bitstream",
                "deployable_runtime_overlay",
            ]

        if blocked:
            reason = "Board fit status is over_limit under fit_policy=enforce."
            severity = "error"
        elif warning:
            reason = "Board fit status is over_limit under fit_policy=warn."
            severity = "warning"
        elif over_limit:
            reason = "Board fit status is over_limit but fit_policy=report_only does not block."
            severity = "info"
        else:
            reason = "Board fit gate passed or board fit status is not over_limit."
            severity = "info"

        return {
            "format": "fpgai.fit_policy_gate.v1",
            "policy": policy,
            "policy_source": policy_source,
            "requested_policy": requested_policy,
            "board_fit_status": status,
            "board_fit_limiting_dimension": board_fit.get("limiting_dimension"),
            "vivado_allowed_by_board_fit": vivado_allowed,
            "over_limit": over_limit,
            "blocked": blocked,
            "warning": warning,
            "severity": severity,
            "blocked_stages": blocked_stages,
            "reason": reason,
        }

    def _emit_manifest(self, **kwargs) -> None:
        out_dir = kwargs["out_dir"]
        precision_layout_artifacts = self._emit_precision_layout_reports(**kwargs)
        hardware_knob_contract = self._emit_hardware_knob_contract_reports(**kwargs)
        fit_policy_gate = self._fit_policy_gate(kwargs.get("prediction_artifacts"))
        manifest = {
            "version": self.cfg.version,
            "model_path": self.cfg.model.path,
            "pipeline_mode": self.cfg.pipeline.mode,
            "top_kernel_name": kwargs["top_name"],
            "weights_mode": kwargs["weights_mode"],
            "configuration": {
                "requested": {
                    "clock_mhz": _cfg_get(
                        self.cfg.raw,
                        "targets.platform.clocks.0.target_mhz",
                        None,
                    ),
                    "parallel_policy": _cfg_get(
                        self.cfg.raw,
                        "optimization.parallel_policy",
                        _cfg_get(
                            self.cfg.raw,
                            "analysis.design_space.policy_name",
                            "Balanced",
                        ),
                    ),
                    "weights_mode": _cfg_get(
                        self.cfg.raw,
                        "data_movement.weights.load.interface",
                        _cfg_get(self.cfg.raw, "data_movement.ps_pl.weights.mode", "embedded"),
                    ),
                    "top_kernel_name": _cfg_get(
                        self.cfg.raw,
                        "pipeline.outputs.top_kernel_name",
                        "deeplearn",
                    ),
                    "hls_enabled": _cfg_get(
                        self.cfg.raw,
                        "backends.hls.enabled",
                        True,
                    ),
                    "host_cpp_enabled": _cfg_get(
                        self.cfg.raw,
                        "backends.host_cpp.enabled",
                        True,
                    ),
                    "build_stages": _cfg_get(self.cfg.raw, "build.stages", None),
                },
                "effective": {
                    "clock_mhz": kwargs["compile_plan"].clock_mhz,
                    "parallel_policy": kwargs["compile_plan"].notes.get(
                        "parallel_policy"
                    ),
                    "weights_mode": kwargs["weights_mode"],
                    "top_kernel_name": kwargs["top_name"],
                    "build_stages": _build_stage_summary(kwargs.get("build_stages") or _resolve_build_stages(self.cfg.raw)),
                },
            },
            "out_dir": str(out_dir),
            "num_ops": len(kwargs["graph"].ops),
            "num_params": len(kwargs["graph"].params),
            "num_descriptors": len(kwargs["descriptors"]),
            "num_layer_plans": len(kwargs["compile_plan"].layer_plans),
            "architecture_signature": (
                kwargs["compile_plan"].architecture_signature
            ),
            "architecture_capabilities": (
                kwargs["capability_report"].to_dict()
            ),
            "prediction_artifacts": kwargs.get("prediction_artifacts"),
            "precision_layout_artifacts": precision_layout_artifacts,
            "hardware_knob_contract": hardware_knob_contract,
            "fit_policy_gate": fit_policy_gate,
            "num_memory_placements": len(kwargs["memory_plan"].placements),
            "num_communication_edges": len(kwargs["communication_plan"].edges),
            "memory_totals": kwargs["memory_plan"].total_bytes_by_region,
            "ops": [
                {
                    "name": op.name,
                    "type": op.op_type,
                    "precision": op.attrs.get("precision"),
                    "precision_tag": op.attrs.get("precision_tag"),
                }
                for op in kwargs["graph"].ops
            ],
            "training_plan": (None if kwargs["training_plan"] is None else kwargs["training_plan"].to_dict()),
            "training_reference": None if kwargs["training_reference_result"] is None else {
                "loss_before": kwargs["training_reference_result"].loss_before,
                "loss_after": kwargs["training_reference_result"].loss_after,
                "grads_ref_bin": str(kwargs["training_reference_result"].grads_flat_path),
                "weights_before_ref_bin": str(kwargs["training_reference_result"].weights_before_flat_path),
                "weights_after_ref_bin": str(kwargs["training_reference_result"].weights_after_flat_path),
                "optimizer_type": getattr(kwargs["training_reference_result"], "optimizer_type", "sgd"),
                "optimizer_bias_correction": getattr(kwargs["training_reference_result"], "optimizer_bias_correction", False),
                "optimizer_state_before_ref_bin": (str(kwargs["training_reference_result"].optimizer_state_before_flat_path) if getattr(kwargs["training_reference_result"], "optimizer_state_before_flat_path", None) is not None else None),
                "optimizer_state_after_ref_bin": (str(kwargs["training_reference_result"].optimizer_state_after_flat_path) if getattr(kwargs["training_reference_result"], "optimizer_state_after_flat_path", None) is not None else None),
                "summary_json": str(kwargs["training_reference_result"].summary_json),
                "summary_txt": str(kwargs["training_reference_result"].summary_txt),
            },
            "training_compare": None if kwargs["training_compare_result"] is None else {
                "out_dir": str(kwargs["training_compare_result"].out_dir),
                "results_json": str(kwargs["training_compare_result"].results_json),
                "summary_txt": str(kwargs["training_compare_result"].summary_txt),
                "grad_cosine": kwargs["training_compare_result"].grad_cosine,
                "weight_after_cosine": kwargs["training_compare_result"].weight_after_cosine,
                "weight_delta_cosine": kwargs["training_compare_result"].weight_delta_cosine,
                "grad_mae": kwargs["training_compare_result"].grad_mae,
                "grad_max_abs": kwargs["training_compare_result"].grad_max_abs,
                "weight_after_mae": kwargs["training_compare_result"].weight_after_mae,
                "weight_after_max_abs": kwargs["training_compare_result"].weight_after_max_abs,
            },
            "training_estimate": None if kwargs["training_estimate_result"] is None else {
                "out_dir": str(kwargs["training_estimate_result"].out_dir),
                "results_json": str(kwargs["training_estimate_result"].results_json),
                "summary_txt": str(kwargs["training_estimate_result"].summary_txt),
                "total_param_bytes": kwargs["training_estimate_result"].total_param_bytes,
                "total_activation_cache_bytes": kwargs["training_estimate_result"].total_activation_cache_bytes,
                "total_gradient_bytes": kwargs["training_estimate_result"].total_gradient_bytes,
                "total_optimizer_state_bytes": kwargs["training_estimate_result"].total_optimizer_state_bytes,
            },
            "quant_report": None if kwargs["quant_result"] is None else {
                "out_dir": str(kwargs["quant_result"].out_dir),
                "metrics_json": str(kwargs["quant_result"].metrics_json),
                "summary_txt": str(kwargs["quant_result"].summary_txt),
                "layerwise_csv": str(kwargs["quant_result"].layerwise_csv),
            },
            "precision_sweep": None if kwargs["sweep_result"] is None else {
                "out_dir": str(kwargs["sweep_result"].out_dir),
                "results_json": str(kwargs["sweep_result"].results_json),
                "summary_txt": str(kwargs["sweep_result"].summary_txt),
                "results_csv": str(kwargs["sweep_result"].results_csv),
            },
            "design_space": self._design_space_manifest_payload(
                kwargs["design_result"]
            ),
            "estimate_vs_hls": None if kwargs["estimate_vs_hls_result"] is None else {
                "out_dir": str(kwargs["estimate_vs_hls_result"].out_dir),
                "results_json": str(kwargs["estimate_vs_hls_result"].results_json),
                "summary_txt": str(kwargs["estimate_vs_hls_result"].summary_txt),
            },
            "hls_module_breakdown": (
                None
                if kwargs["hls_module_breakdown_result"] is None
                else {
                    "available": kwargs["hls_module_breakdown_result"].available,
                    "out_dir": str(kwargs["hls_module_breakdown_result"].out_dir),
                    "results_json": str(
                        kwargs["hls_module_breakdown_result"].results_json
                    ),
                    "results_csv": str(
                        kwargs["hls_module_breakdown_result"].results_csv
                    ),
                    "summary_txt": str(
                        kwargs["hls_module_breakdown_result"].summary_txt
                    ),
                }
            ),
            "hls_ran": kwargs["hls_run"] is not None,
            "hls_ok": (kwargs["hls_run"].ok if kwargs["hls_run"] is not None else None),
            "hls_returncode": (kwargs["hls_run"].returncode if kwargs["hls_run"] is not None else None),
            "hls_stdout_log": (
                str(kwargs["hls_run"].stdout_log)
                if kwargs["hls_run"] is not None and kwargs["hls_run"].stdout_log is not None
                else None
            ),
            "hls_stderr_log": (
                str(kwargs["hls_run"].stderr_log)
                if kwargs["hls_run"] is not None and kwargs["hls_run"].stderr_log is not None
                else None
            ),
            "hls_csynth_report": (
                str(kwargs["hls_run"].csynth_report)
                if kwargs["hls_run"] is not None and kwargs["hls_run"].csynth_report is not None
                else None
            ),
            "hls_artifacts": self._hls_artifacts_manifest_payload(
                out_dir=out_dir,
                hls_run=kwargs["hls_run"],
                hls_schedule_summary=kwargs.get("hls_schedule_summary"),
                hls_artifact_metadata=kwargs.get("hls_artifact_metadata"),
                hls_ii_comparison=kwargs.get("hls_ii_comparison"),
            ),
            "build_stages": _build_stage_summary(kwargs.get("build_stages") or _resolve_build_stages(self.cfg.raw)),
            "runtime_sequence": kwargs.get("runtime_sequence"),
            "toolchain": _resolved_toolchain_summary(self.cfg.raw),
            "resolved_config_artifacts": kwargs.get("resolved_config_artifacts"),
            "numeric_validation_artifacts": None if kwargs.get("numeric_validation_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("numeric_validation_artifacts", {}).items()
            },
            "generated_hls_explanation_artifacts": None if kwargs.get("generated_hls_explanation_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("generated_hls_explanation_artifacts", {}).items()
            },
            "validation_summary_artifacts": None if kwargs.get("validation_summary_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("validation_summary_artifacts", {}).items()
            },
            "vivado_bd_contract_artifacts": None if kwargs.get("vivado_bd_contract_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("vivado_bd_contract_artifacts", {}).items()
            },
            "vivado_handoff_artifacts": None if kwargs.get("vivado_handoff_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("vivado_handoff_artifacts", {}).items()
            },
            "vivado_validation_artifacts": None if kwargs.get("vivado_validation_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("vivado_validation_artifacts", {}).items()
            },
            "hls_validation_artifacts": None if kwargs.get("hls_validation_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("hls_validation_artifacts", {}).items()
            },
            "precision_effect_artifacts": None if kwargs.get("precision_effect_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("precision_effect_artifacts", {}).items()
            },
            "parallel_pipeline_effect_artifacts": None if kwargs.get("parallel_pipeline_effect_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("parallel_pipeline_effect_artifacts", {}).items()
            },
            "data_movement_artifacts": None if kwargs.get("data_movement_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("data_movement_artifacts", {}).items()
            },
            "movement_contract_validation": movement_contract_validation_summary(out_dir),
            "feature_validation_artifacts": None if kwargs.get("feature_validation_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("feature_validation_artifacts", {}).items()
            },
            "runtime_package": kwargs.get("runtime_package"),
            "pipeline_stages": self._build_pipeline_stages(**kwargs),
            "seconds": round(float(kwargs["seconds"]), 6),
        }
        hls_schedule_summary = kwargs.get("hls_schedule_summary")
        if hls_schedule_summary is not None:
            manifest["hls_schedule_summary"] = hls_schedule_summary

        hls_artifact_metadata = kwargs.get("hls_artifact_metadata")
        if hls_artifact_metadata is not None:
            manifest["hls_artifact_metadata"] = hls_artifact_metadata

        hls_ii_comparison = kwargs.get("hls_ii_comparison")
        if hls_ii_comparison is not None:
            manifest["hls_ii_comparison"] = hls_ii_comparison

        write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2))
