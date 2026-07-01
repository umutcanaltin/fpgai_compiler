from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time
import numpy as np

from fpgai.config.access import get_path
from fpgai.config.loader import FPGAIConfig
from fpgai.compiler.architecture_capabilities import (
    validate_architecture_capabilities,
)
from fpgai.engine.analysis import analyze_graph
from fpgai.engine.communication import make_communication_plan
from fpgai.engine.memory import make_memory_plan
from fpgai.engine.planner import make_compile_plan
from fpgai.engine.result import CompileResult
from fpgai.engine.partition import single_device_plan
from fpgai.engine.layerwise_precision import resolve_layerwise_precision
from fpgai.engine.training import build_training_plan, emit_training_artifacts
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
from fpgai.analysis.training_resource_estimate import run_training_resource_estimate
from fpgai.benchmark.training_reference import run_training_reference_step
from fpgai.benchmark.training_compare import compare_training_artifacts
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
from fpgai.paper.verification import emit_paper_verification_artifacts
from fpgai.backends.vivado.boards import get_board
from fpgai.reporting.hardware_feasibility import emit_board_fit_report
from fpgai.reporting.hls_explanation import emit_generated_hls_explanation_reports

from fpgai.backends.hls.emit.types_h import emit_types_h
from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp
_emit_top_train_cpp = emit_top_train_cpp
from fpgai.backends.hls.emit.layers_dense import emit_dense_h, emit_dense_cpp
from fpgai.backends.hls.emit.layers_conv import emit_conv_h, emit_conv_cpp
from fpgai.backends.hls.emit.layers_pool import emit_pool_h, emit_pool_cpp
from fpgai.backends.hls.emit.layers_activations import emit_activations_h, emit_activations_cpp
from fpgai.backends.hls.emit.layers_batchnorm import emit_batchnorm_h, emit_batchnorm_cpp
from fpgai.backends.hls.emit.params_h import emit_params_h, _conv_sizes, _dense_sizes
from fpgai.backends.hls.emit.params_cpp import emit_params_cpp
from fpgai.backends.hls.emit.weights_runtime_h import emit_weights_runtime_h
from fpgai.backends.hls.emit.weights_runtime_cpp import emit_weights_runtime_cpp
from fpgai.backends.hls.emit.csim_tcl import emit_csim_tcl
from fpgai.backends.hls.emit.csim_train_tcl import emit_csim_train_tcl
from fpgai.backends.hls.testbench import emit_tb_cpp
from fpgai.backends.hls.testbench_train import emit_tb_train_cpp


_cfg_get = get_path


def _is_runtime_weight_mode(weights_mode: str) -> bool:
    return str(weights_mode).strip().lower() in {
        "stream",
        "streamed",
        "ddr",
        "dma_ddr",
        "uram",
        "bram_import_full",
        "bram_import_export_full",
        "uram_import_full",
        "uram_import_export_full",
        "ddr_tiled",
        "ddr_tiled_mutable",
    }


def _runtime_weight_word_count(graph) -> int:
    total = 0
    for op in graph.ops:
        if op.op_type == "Conv":
            weight_count, bias_count = _conv_sizes(graph, op)
            total += int(weight_count) + int(bias_count)
        elif op.op_type == "Dense":
            weight_count, bias_count = _dense_sizes(graph, op)
            total += int(weight_count) + int(bias_count)
    return int(total)


_BUILD_STAGE_KEYS = (
    "cpp",
    "testbench",
    "hls_project",
    "hls_synthesis",
    "vivado_project",
    "vivado_implementation",
    "bitstream",
    "runtime_package",
    "reports",
    "host_cpp",
)


def _cfg_has_path(raw: Dict[str, Any], path: str) -> bool:
    cur: Any = raw
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _resolve_build_stages(raw: Dict[str, Any]) -> Dict[str, bool]:
    """Resolve user-facing build.stages into an explicit stage contract.

    Legacy configs without build.stages keep the old behavior: HLS source/project
    generation follows backends.hls.enabled, Vitis execution follows
    toolchain.vitis_hls.enabled, host C++ follows backends.host_cpp.enabled, and
    runtime package/report metadata are emitted.
    """
    explicit = _cfg_has_path(raw, "build.stages")
    requested = _cfg_get(raw, "build.stages", {}) or {}
    if requested is not None and not isinstance(requested, dict):
        raise ValueError("build.stages must be a mapping of stage names to booleans.")

    legacy_hls = _as_bool(_cfg_get(raw, "backends.hls.enabled", True))
    legacy_host = _as_bool(_cfg_get(raw, "backends.host_cpp.enabled", True))
    legacy_hls_run = _as_bool(_cfg_get(raw, "toolchain.vitis_hls.enabled", False))

    if explicit:
        stages: Dict[str, bool] = {
            "cpp": True,
            "testbench": True,
            "hls_project": False,
            "hls_synthesis": False,
            "vivado_project": False,
            "vivado_implementation": False,
            "bitstream": False,
            "runtime_package": True,
            "reports": True,
            "host_cpp": legacy_host,
        }
        unknown = sorted(set(requested) - set(_BUILD_STAGE_KEYS) - {"existing_hls_ip"})
        if unknown:
            raise ValueError(
                "Unsupported build.stages keys: " + ", ".join(unknown) + ". "
                "Supported keys are: " + ", ".join(_BUILD_STAGE_KEYS) + "."
            )
        for key, value in requested.items():
            if key in stages:
                stages[key] = _as_bool(value)
    else:
        stages = {
            "cpp": legacy_hls,
            "testbench": legacy_hls,
            "hls_project": legacy_hls,
            "hls_synthesis": legacy_hls and legacy_hls_run,
            "vivado_project": False,
            "vivado_implementation": False,
            "bitstream": False,
            "runtime_package": True,
            "reports": True,
            "host_cpp": legacy_host,
        }

    _validate_build_stage_dependencies(raw, stages)
    return stages


def _validate_build_stage_dependencies(raw: Dict[str, Any], stages: Dict[str, bool]) -> None:
    if stages.get("testbench") and not stages.get("cpp"):
        raise ValueError("build.stages.testbench=true requires build.stages.cpp=true.")
    if stages.get("hls_project") and not stages.get("cpp"):
        raise ValueError("build.stages.hls_project=true requires build.stages.cpp=true.")
    if stages.get("hls_synthesis") and not stages.get("hls_project"):
        raise ValueError("build.stages.hls_synthesis=true requires build.stages.hls_project=true.")

    existing_hls_ip = _as_bool(_cfg_get(raw, "build.existing_hls_ip", False))
    if stages.get("vivado_project") and not (stages.get("hls_synthesis") or existing_hls_ip):
        raise ValueError(
            "build.stages.vivado_project=true requires build.stages.hls_synthesis=true "
            "or build.existing_hls_ip=true."
        )
    if stages.get("vivado_implementation") and not stages.get("vivado_project"):
        raise ValueError("build.stages.vivado_implementation=true requires build.stages.vivado_project=true.")
    if stages.get("bitstream") and not (stages.get("vivado_project") and stages.get("vivado_implementation")):
        raise ValueError(
            "build.stages.bitstream=true requires build.stages.vivado_project=true "
            "and build.stages.vivado_implementation=true."
        )


def _build_stage_summary(stages: Dict[str, bool]) -> Dict[str, Any]:
    return {key: bool(stages.get(key, False)) for key in _BUILD_STAGE_KEYS}


_RUNTIME_COMMANDS = {
    "import_weights",
    "run_inference",
    "run_training",
    "export_weights",
    "export_gradients",
    "reset_accumulators",
    "accumulate_gradients",
    "apply_accumulated_gradients",
}

_CODEGEN_READABILITY = {"compact", "normal", "high", "debug"}


def _movement_cfg(raw: Dict[str, Any], tensor: str, direction: str) -> Dict[str, str]:
    cfg = _cfg_get(raw, f"data_movement.{tensor}.{direction}", {}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "interface": str(cfg.get("interface", "")).strip().lower().replace("-", "_"),
        "transport": str(cfg.get("transport", "")).strip().lower().replace("-", "_"),
        "policy": str(cfg.get("policy", "")).strip().lower().replace("-", "_"),
    }


def _resolve_training_io_movement(raw: Dict[str, Any]) -> Dict[str, Any]:
    def _one(tensor: str, direction: str) -> Dict[str, Any]:
        mv = _movement_cfg(raw, tensor, direction)
        interface = mv["interface"] or ("axi_stream" if tensor in {"inputs", "labels", "outputs"} else "none")
        transport = mv["transport"] or ("ps_runtime" if interface == "m_axi" else ("dma" if interface == "axi_stream" else "none"))
        policy = mv["policy"] or ("full" if interface != "none" else "none")
        if interface == "m_axi" and policy == "tiled":
            resolved = f"m_axi_{direction}_tiled"
        elif interface == "m_axi" and policy == "full":
            resolved = f"m_axi_{direction}_full"
        elif interface == "axi_stream" and policy == "tiled":
            resolved = f"axi_stream_{direction}_tiled"
        elif interface == "axi_stream":
            resolved = f"axi_stream_{direction}_full"
        else:
            resolved = "none"
        return {"interface": interface, "transport": transport, "policy": policy, "resolved": resolved}
    return {
        "inputs": {"import": _one("inputs", "import")},
        "labels": {"import": _one("labels", "import")},
        "outputs": {"export": _one("outputs", "export")},
    }


def _resolve_gradient_export_mode(raw: Dict[str, Any]) -> Dict[str, Any]:
    mv = _movement_cfg(raw, "gradients", "export")
    interface = mv["interface"] or "none"
    transport = mv["transport"] or ("ps_runtime" if interface == "m_axi" else "none")
    policy = mv["policy"] or "none"
    if interface == "none" or policy == "none":
        resolved = "none"
        supported = False
    elif interface == "m_axi" and policy == "full":
        resolved = "m_axi_export_full"
        supported = True
    elif interface == "m_axi" and policy == "tiled":
        resolved = "m_axi_export_tiled"
        supported = True
    else:
        raise ValueError(
            "data_movement.gradients.export currently supports interface=m_axi, transport=ps_runtime, policy=full; "
            f"got interface={interface!r}, transport={transport!r}, policy={policy!r}."
        )
    return {"interface": interface, "transport": transport, "policy": policy, "resolved": resolved, "supported": supported}


def _write_training_movement_reports(out_dir: Path, raw: Dict[str, Any]) -> Dict[str, str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    io = _resolve_training_io_movement(raw)
    grad = _resolve_gradient_export_mode(raw)
    (reports_dir / "training_io_movement.json").write_text(json.dumps(io, indent=2, sort_keys=True), encoding="utf-8")
    (reports_dir / "gradient_export.json").write_text(json.dumps(grad, indent=2, sort_keys=True), encoding="utf-8")
    (reports_dir / "training_io_movement.md").write_text(
        "# Training I/O movement\n\n"
        f"- inputs.import: `{io['inputs']['import']['resolved']}`\n"
        f"- labels.import: `{io['labels']['import']['resolved']}`\n"
        f"- outputs.export: `{io['outputs']['export']['resolved']}`\n",
        encoding="utf-8",
    )
    (reports_dir / "gradient_export.md").write_text(
        "# Gradient export\n\n"
        f"- resolved: `{grad['resolved']}`\n"
        f"- supported: `{grad['supported']}`\n",
        encoding="utf-8",
    )
    return {
        "training_io_movement_json": str(reports_dir / "training_io_movement.json"),
        "gradient_export_json": str(reports_dir / "gradient_export.json"),
    }


def _as_positive_int(value: Any, default: int = 1) -> int:
    try:
        ivalue = int(value)
    except Exception:
        ivalue = int(default)
    return max(1, ivalue)


def _resolve_training_batch_accumulation_contract(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer the new public shorthand training.batch_size when present, while
    # preserving the older training.execution.batch_size path for legacy configs.
    # Do not use nested _cfg_get(...) calls as default arguments here: Python
    # evaluates defaults eagerly and training.execution.batch_size=1 would mask
    # a user-specified training.batch_size=2.
    batch_raw = _cfg_get(raw, "training.batch_size", None)
    if batch_raw is None:
        batch_raw = _cfg_get(raw, "training.execution.batch_size", 1)
    batch_size = _as_positive_int(batch_raw, 1)

    steps_raw = _cfg_get(raw, "training.gradient_accumulation.steps", None)
    if steps_raw is None:
        steps_raw = _cfg_get(raw, "training.accumulation.steps", 1)
    steps = _as_positive_int(steps_raw, 1)

    mode_raw = _cfg_get(raw, "training.gradient_accumulation.mode", None)
    if mode_raw is None:
        mode_raw = _cfg_get(raw, "training.accumulation.mode", "none")
    mode = str(mode_raw or "none").strip().lower().replace("-", "_")
    if steps <= 1 and mode in {"", "none", "false"}:
        mode = "none"
    supported_modes = {"none", "native", "testbench", "native_accumulated", "testbench_accumulated"}
    if mode not in supported_modes:
        raise ValueError(
            "training.gradient_accumulation.mode must be one of none, native, testbench, "
            f"native_accumulated, or testbench_accumulated; got {mode!r}."
        )
    native = mode in {"native", "native_accumulated"}
    testbench = mode in {"testbench", "testbench_accumulated"}
    active = steps > 1 or batch_size > 1 or native or testbench
    return {
        "batch_size": batch_size,
        "accumulation_steps": steps,
        "mode": mode,
        "active": bool(active),
        "native_update_boundary": bool(native),
        "testbench_accumulation": bool(testbench),
        "generated_hls_status": "implemented" if mode in {"none", "native", "native_accumulated"} else "testbench_only",
        "numeric_validation_status": "requires_training_compare_artifacts" if active else "not_required_for_batch1",
        "hls_modes": {
            "accumulate_gradients": 3,
            "apply_accumulated_gradients": 4,
            "reset_accumulators": 5,
        } if native else {},
        "runtime_commands": [
            "reset_accumulators",
            "accumulate_gradients",
            "apply_accumulated_gradients",
        ] if native else [],
        "truth_boundary": (
            "Native batch/gradient accumulation generates explicit HLS modes 3/4/5 for "
            "reset, accumulate, and apply/update. Paper-safe correctness still requires the "
            "training numeric comparison artifacts to pass for the selected model/config."
        ),
    }


def _resolve_stream_tiled_io_contract(raw: Dict[str, Any], *, pipeline_mode: str) -> Dict[str, Any]:
    def _one(tensor: str, direction: str) -> Dict[str, Any]:
        mv = _movement_cfg(raw, tensor, direction)
        interface = mv["interface"] or "axi_stream"
        transport = mv["transport"] or ("dma" if interface == "axi_stream" else "ps_runtime" if interface == "m_axi" else "none")
        policy = mv["policy"] or "full"
        requested = interface == "axi_stream" and policy == "tiled"
        if requested and pipeline_mode == "training_on_device":
            status = "generated_interface_supported"
            reason = "training AXI-stream tiled I/O is implemented by generated tile buffers, stream tile readers/writers, and TLAST-aware output emission"
        elif requested:
            status = "generated_interface_supported"
            reason = "inference AXI-stream tiled I/O is supported by the HLS top interface contract"
        else:
            status = "not_requested"
            reason = "full/default I/O path selected"
        return {
            "interface": interface,
            "transport": transport,
            "policy": policy,
            "requested": bool(requested),
            "status": status,
            "reason": reason,
        }
    return {
        "pipeline_mode": pipeline_mode,
        "inputs": {"import": _one("inputs", "import")},
        "outputs": {"export": _one("outputs", "export")},
    }


def _write_execution_semantics_reports(
    out_dir: Path,
    raw: Dict[str, Any],
    *,
    pipeline_mode: str,
    memory_plan=None,
    communication_plan=None,
    prediction_artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    batch = _resolve_training_batch_accumulation_contract(raw)
    stream_io = _resolve_stream_tiled_io_contract(raw, pipeline_mode=pipeline_mode)
    board_fit = {}
    if isinstance(prediction_artifacts, dict):
        board_fit = prediction_artifacts.get("board_fit") or {}
        if not isinstance(board_fit, dict):
            board_fit = {}
    memory_notes = dict(getattr(memory_plan, "notes", {}) or {}) if memory_plan is not None else {}
    comm_notes = dict(getattr(communication_plan, "notes", {}) or {}) if communication_plan is not None else {}
    hardware_contract = {
        "board": str(_cfg_get(raw, "targets.platform.board", _cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", ""))) or ""),
        "memory_semantics_mode": memory_notes.get("memory_semantics_mode"),
        "activation_storage": memory_notes.get("resolved_activation_storage"),
        "weight_storage": memory_notes.get("resolved_weight_storage"),
        "communication_scope": comm_notes.get("scope"),
        "board_fit_status": board_fit.get("status", "unknown"),
        "board_fit_limiting_dimension": board_fit.get("limiting_dimension"),
        "vivado_allowed_by_board_fit": board_fit.get("vivado_allowed"),
        "enforcement_status": "report_generated",
        "truth_boundary": (
            "This contract records compiler-side feasibility and selected knobs. "
            "Vivado/bitstream enforcement is paper-safe only when Vivado artifacts are present."
        ),
    }

    paths = {
        "training_batch_accumulation_json": reports_dir / "training_batch_accumulation.json",
        "stream_tiled_io_json": reports_dir / "stream_tiled_io.json",
        "hardware_knob_contract_json": reports_dir / "hardware_knob_contract.json",
    }
    paths["training_batch_accumulation_json"].write_text(json.dumps(batch, indent=2, sort_keys=True), encoding="utf-8")
    paths["stream_tiled_io_json"].write_text(json.dumps(stream_io, indent=2, sort_keys=True), encoding="utf-8")
    paths["hardware_knob_contract_json"].write_text(json.dumps(hardware_contract, indent=2, sort_keys=True), encoding="utf-8")

    (reports_dir / "training_batch_accumulation.md").write_text(
        "# Training batch and gradient accumulation\n\n"
        f"- batch_size: `{batch['batch_size']}`\n"
        f"- accumulation_steps: `{batch['accumulation_steps']}`\n"
        f"- mode: `{batch['mode']}`\n"
        f"- generated_hls_status: `{batch['generated_hls_status']}`\n",
        encoding="utf-8",
    )
    (reports_dir / "stream_tiled_io.md").write_text(
        "# AXI-stream tiled I/O contract\n\n"
        f"- pipeline_mode: `{pipeline_mode}`\n"
        f"- inputs.import: `{stream_io['inputs']['import']['status']}`\n"
        f"- outputs.export: `{stream_io['outputs']['export']['status']}`\n",
        encoding="utf-8",
    )
    (reports_dir / "hardware_knob_contract.md").write_text(
        "# Hardware knob contract\n\n"
        f"- board_fit_status: `{hardware_contract['board_fit_status']}`\n"
        f"- limiting_dimension: `{hardware_contract['board_fit_limiting_dimension']}`\n"
        f"- vivado_allowed_by_board_fit: `{hardware_contract['vivado_allowed_by_board_fit']}`\n",
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def _runtime_io_summary_from_plan(communication_plan: Any | None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "inputs": {"import": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_import_full"}},
        "outputs": {"export": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_export_full"}},
    }
    edges = getattr(communication_plan, "edges", []) or []
    for edge in edges:
        notes = getattr(edge, "notes", {}) or {}
        kind = str(notes.get("kind", "")).strip().lower()
        interface = str(notes.get("interface") or "").strip().lower().replace("-", "_")
        transport = str(notes.get("transport") or "").strip().lower().replace("-", "_")
        policy = str(notes.get("policy") or "").strip().lower().replace("-", "_")
        if not interface:
            mode = str(notes.get("mode") or "").strip().lower().replace("-", "_")
            interface = "m_axi" if mode in {"m_axi", "maxi", "ddr"} else "axi_stream"
        if not transport:
            transport = "ps_runtime" if interface == "m_axi" else "dma"
        if not policy:
            policy = "full"
        if kind in {"input", "inputs", "activation_in"}:
            if interface == "m_axi" and policy == "tiled":
                resolved = "m_axi_import_tiled"
            elif interface == "m_axi":
                resolved = "m_axi_import_full"
            elif interface == "axi_stream" and policy == "tiled":
                resolved = "dma_stream_import_tiled"
            else:
                resolved = "dma_stream_import_full"
            summary["inputs"] = {"import": {"interface": interface, "transport": transport, "policy": policy, "resolved": resolved}}
        elif kind in {"output", "outputs", "activation_out"}:
            if interface == "m_axi" and policy == "tiled":
                resolved = "m_axi_export_tiled"
            elif interface == "m_axi":
                resolved = "m_axi_export_full"
            elif interface == "axi_stream" and policy == "tiled":
                resolved = "dma_stream_export_tiled"
            else:
                resolved = "dma_stream_export_full"
            summary["outputs"] = {"export": {"interface": interface, "transport": transport, "policy": policy, "resolved": resolved}}
    return summary


def _scan_hls_top_ports(hls_dir: Optional[Path]) -> Dict[str, Any]:
    source = None
    if hls_dir is not None:
        candidate = Path(hls_dir) / "src" / "deeplearn.cpp"
        if candidate.exists():
            source = candidate.read_text(encoding="utf-8", errors="replace")
    if not source:
        return {"source_present": False, "m_axi_ports": [], "axis_ports": [], "dma_required": None}
    m_axi_ports = sorted(set(__import__('re').findall(r"#pragma\s+HLS\s+INTERFACE\s+m_axi\s+port\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", source)))
    axis_ports = sorted(set(__import__('re').findall(r"#pragma\s+HLS\s+INTERFACE\s+axis\s+port\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", source)))
    return {
        "source_present": True,
        "m_axi_ports": m_axi_ports,
        "axis_ports": axis_ports,
        "dma_required": bool(axis_ports),
    }


def _write_vivado_bd_contract_reports(
    out_dir: Path,
    raw: Dict[str, Any],
    *,
    pipeline_mode: str,
    build_stages: Dict[str, bool],
    memory_plan=None,
    communication_plan=None,
    hls_dir: Optional[Path] = None,
) -> Dict[str, str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    io_summary = _runtime_io_summary_from_plan(communication_plan)
    source_ports = _scan_hls_top_ports(hls_dir)
    notes = dict(getattr(memory_plan, "notes", {}) or {}) if memory_plan is not None else {}
    m_axi_ports = list(source_ports.get("m_axi_ports") or [])
    axis_ports = list(source_ports.get("axis_ports") or [])
    required_blocks = ["axi_lite_control", "ps_memory_port"]
    if axis_ports:
        required_blocks.append("axi_dma")
    if m_axi_ports:
        required_blocks.append("axi_interconnect_or_smartconnect")
    if pipeline_mode == "training_on_device":
        required_blocks.append("training_aux_buffers")
    status = "not_requested"
    if build_stages.get("vivado_project"):
        status = "contract_generated_waiting_for_vivado_bridge"
    payload = {
        "format": "fpgai.vivado_bd_contract.v1",
        "pipeline_mode": pipeline_mode,
        "board": str(_cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "")) or ""),
        "status": status,
        "build_stages": {str(k): bool(v) for k, v in build_stages.items()},
        "memory_semantics_mode": notes.get("memory_semantics_mode"),
        "input_movement": io_summary["inputs"]["import"],
        "output_movement": io_summary["outputs"]["export"],
        "source_ports": source_ports,
        "required_blocks": required_blocks,
        "wiring_contract": {
            "axi_lite_control": "PS control master -> AXI-Lite interconnect -> HLS s_axi_control",
            "stream_dma": "PS DDR -> AXI DMA MM2S -> HLS AXIS input; HLS AXIS output -> AXI DMA S2MM -> PS DDR" if axis_ports else "not_required_by_generated_ports",
            "m_axi_memory": "HLS m_axi bundles -> AXI interconnect/smartconnect -> PS DDR HP/HPC port" if m_axi_ports else "not_required_by_generated_ports",
        },
        "truth_boundary": "This report records the expected Vivado block-design wiring from generated HLS interfaces. It is not proof that Vivado implemented the design; Vivado reports/bitstream are required for that claim.",
    }
    json_path = reports_dir / "vivado_bd_contract.json"
    md_path = reports_dir / "vivado_bd_contract.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "# Vivado BD wiring contract\n\n"
        f"- status: `{payload['status']}`\n"
        f"- pipeline_mode: `{pipeline_mode}`\n"
        f"- m_axi_ports: `{', '.join(m_axi_ports) if m_axi_ports else 'none'}`\n"
        f"- axis_ports: `{', '.join(axis_ports) if axis_ports else 'none'}`\n"
        f"- required_blocks: `{', '.join(required_blocks)}`\n\n"
        "This is a contract derived from generated HLS interfaces, not a Vivado implementation result.\n",
        encoding="utf-8",
    )
    return {"vivado_bd_contract_json": str(json_path), "vivado_bd_contract_md": str(md_path)}


def _write_feature_truth_reports(
    out_dir: Path,
    *,
    pipeline_mode: str,
    build_stages: Dict[str, bool],
    hls_dir: Optional[Path],
    hls_run: Any | None,
    runtime_package: Any | None,
    numeric_validation_artifacts: Optional[Dict[str, Any]],
    paper_verification_artifacts: Optional[Dict[str, Any]],
    vivado_bd_contract_artifacts: Optional[Dict[str, str]],
) -> Dict[str, str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    source_generated = bool(hls_dir is not None and (Path(hls_dir) / "src" / "deeplearn.cpp").exists())
    hls_synthesized = bool(hls_run is not None and getattr(hls_run, "ok", False))
    runtime_manifest = out_dir / "runtime_package" / "package_manifest.json"
    runtime_packaged = bool(runtime_manifest.exists())
    numeric_validated = False
    if numeric_validation_artifacts:
        nv_path = numeric_validation_artifacts.get("numeric_validation_json")
        try:
            nv = json.loads(Path(nv_path).read_text(encoding="utf-8")) if nv_path else {}
            numeric_validated = bool(nv.get("numeric_validated") or nv.get("passed"))
        except Exception:
            numeric_validated = False
    paper_safe = False
    if paper_verification_artifacts:
        pv_path = paper_verification_artifacts.get("paper_verification_json") or paper_verification_artifacts.get("paper_row_json")
        try:
            pv = json.loads(Path(pv_path).read_text(encoding="utf-8")) if pv_path else {}
            paper_safe = bool(pv.get("paper_safe"))
        except Exception:
            paper_safe = False
    features = [
        {"feature": "source_generation", "status": "validated_by_generated_source" if source_generated else "not_generated", "paper_safe_for": ["source-exists claims"] if source_generated else []},
        {"feature": "numeric_correctness", "status": "validated" if numeric_validated else "not_validated", "paper_safe_for": ["correctness claims"] if numeric_validated else []},
        {"feature": "hls_resource_timing", "status": "validated_by_hls" if hls_synthesized else "not_validated", "paper_safe_for": ["HLS resource/timing claims"] if hls_synthesized else []},
        {"feature": "vivado_bd_wiring", "status": "contract_generated" if vivado_bd_contract_artifacts else "not_generated", "paper_safe_for": ["BD contract claims"] if vivado_bd_contract_artifacts else []},
        {"feature": "runtime_package", "status": "packaged" if runtime_packaged else "not_packaged", "paper_safe_for": ["runtime package existence claims"] if runtime_packaged else []},
        {"feature": "fpga_execution", "status": "not_validated", "paper_safe_for": []},
    ]
    payload = {
        "format": "fpgai.feature_contract.v1",
        "pipeline_mode": pipeline_mode,
        "build_stages": {str(k): bool(v) for k, v in build_stages.items()},
        "source_generated": source_generated,
        "numeric_validated": numeric_validated,
        "hls_synthesized": hls_synthesized,
        "runtime_packaged": runtime_packaged,
        "paper_safe": paper_safe,
        "features": features,
        "truth_boundary": "Each feature is paper-safe only at the verification level recorded here. Contract/report generation is not the same as HLS, Vivado, bitstream, or FPGA validation.",
    }
    json_path = reports_dir / "feature_contract.json"
    audit_path = reports_dir / "claim_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# FPGAI claim audit", "", f"- pipeline_mode: `{pipeline_mode}`", f"- source_generated: `{source_generated}`", f"- numeric_validated: `{numeric_validated}`", f"- hls_synthesized: `{hls_synthesized}`", f"- runtime_packaged: `{runtime_packaged}`", f"- paper_safe: `{paper_safe}`", "", "| Feature | Status | Paper-safe for |", "|---|---|---|"]
    for item in features:
        lines.append(f"| {item['feature']} | {item['status']} | {', '.join(item['paper_safe_for']) if item['paper_safe_for'] else 'none'} |")
    lines.extend(["", "This audit must be used by paper table/plot generation to avoid intention-only claims.", ""])
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    return {"feature_contract_json": str(json_path), "claim_audit_md": str(audit_path)}


_TRAINING_OPTIMIZER_TYPES = {"sgd", "momentum", "adam"}
_TRAINING_LOSS_TYPES = {"mse", "cross_entropy"}
_OPTIMIZER_STATE_STORAGE = {"none", "bram", "uram", "ddr"}


def _resolve_optimizer_state_movement(raw: Dict[str, Any], direction: str) -> Dict[str, Any]:
    mv = _movement_cfg(raw, "optimizer_state", direction)
    interface = mv["interface"] or "none"
    transport = mv["transport"] or ("ps_runtime" if interface == "m_axi" else "none")
    policy = mv["policy"] or "none"
    if interface == "none" or policy == "none":
        resolved = "none"
        supported = False
    elif interface == "m_axi" and policy == "full":
        resolved = f"m_axi_{direction}_full"
        supported = True
    elif interface == "m_axi" and policy == "tiled":
        resolved = f"m_axi_{direction}_tiled"
        supported = True
    else:
        raise ValueError(
            f"data_movement.optimizer_state.{direction} currently supports "
            "interface=m_axi, transport=ps_runtime, policy=full; "
            f"got interface={interface!r}, transport={transport!r}, policy={policy!r}."
        )
    return {"interface": interface, "transport": transport, "policy": policy, "resolved": resolved, "supported": supported}


def _resolve_training_optimizer_loss_contract(raw: Dict[str, Any]) -> Dict[str, Any]:
    optimizer_type = str(_cfg_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace("-", "_")
    if optimizer_type not in _TRAINING_OPTIMIZER_TYPES:
        raise ValueError(
            "training.optimizer.type must be one of "
            + ", ".join(sorted(_TRAINING_OPTIMIZER_TYPES))
            + f"; got {optimizer_type!r}."
        )
    learning_rate = float(_cfg_get(raw, "training.optimizer.learning_rate", 0.01))
    if learning_rate <= 0.0:
        raise ValueError("training.optimizer.learning_rate must be positive.")

    loss_type = str(_cfg_get(raw, "training.loss.type", "mse") or "mse").strip().lower().replace("-", "_")
    if loss_type not in _TRAINING_LOSS_TYPES:
        raise ValueError(
            "training.loss.type must be one of "
            + ", ".join(sorted(_TRAINING_LOSS_TYPES))
            + f"; got {loss_type!r}."
        )

    storage_raw = _cfg_get(raw, "training.storage.optimizer_state", None)
    if storage_raw is None:
        # SGD has no persistent optimizer state. Momentum/Adam will require it once
        # their generated update kernels are implemented.
        storage = "none" if optimizer_type == "sgd" else "bram"
        explicit_storage = False
    else:
        storage = str(storage_raw or "none").strip().lower().replace("-", "_")
        explicit_storage = True
    if storage not in _OPTIMIZER_STATE_STORAGE:
        raise ValueError(
            "training.storage.optimizer_state must be one of none, bram, uram, ddr; "
            f"got {storage!r}."
        )

    import_movement = _resolve_optimizer_state_movement(raw, "import")
    export_movement = _resolve_optimizer_state_movement(raw, "export")

    state_required = optimizer_type in {"momentum", "adam"}
    hls_update_status = "implemented" if optimizer_type in {"sgd", "momentum", "adam"} else "not_implemented"
    loss_hls_status = "implemented" if loss_type in {"mse", "cross_entropy"} else "not_implemented"
    loss_numeric_status = (
        "implemented"
        if loss_type in {"mse", "cross_entropy"}
        else "not_implemented"
    )

    generated_state = storage in {"bram", "uram", "ddr"} or import_movement["supported"] or export_movement["supported"]
    return {
        "schema_version": 1,
        "optimizer": {
            "type": optimizer_type,
            "learning_rate": learning_rate,
            "momentum": float(_cfg_get(raw, "training.optimizer.momentum", 0.9) or 0.9),
            "beta1": float(_cfg_get(raw, "training.optimizer.beta1", 0.9) or 0.9),
            "beta2": float(_cfg_get(raw, "training.optimizer.beta2", 0.999) or 0.999),
            "epsilon": float(_cfg_get(raw, "training.optimizer.epsilon", 1.0e-8) or 1.0e-8),
            "hls_update_status": hls_update_status,
            "numeric_validation_status": "implemented" if optimizer_type in {"sgd", "momentum", "adam"} else "not_implemented",
        },
        "optimizer_state": {
            "required": state_required,
            "explicit_storage": explicit_storage,
            "storage": storage,
            "storage_supported": storage in {"none", "bram", "uram", "ddr"},
            "generated_interface": bool(generated_state),
            "export_capture_mode": 9 if (state_required and export_movement["supported"]) else None,
            "export_capture_status": "generated_hls_mode" if (state_required and export_movement["supported"]) else "not_requested",
            "export_capture_words_known_after_codegen": bool(state_required and export_movement["supported"]),
            "import": import_movement,
            "export": export_movement,
        },
        "loss": {
            "type": loss_type,
            "hls_status": loss_hls_status,
            "numeric_validation_status": loss_numeric_status,
        },
    }


def _write_training_optimizer_loss_reports(out_dir: Path, raw: Dict[str, Any]) -> Dict[str, str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    contract = _resolve_training_optimizer_loss_contract(raw)
    opt_path = reports_dir / "training_optimizer_state.json"
    loss_path = reports_dir / "training_loss_contract.json"
    opt_payload = {
        "schema_version": 1,
        "optimizer": contract["optimizer"],
        "optimizer_state": contract["optimizer_state"],
    }
    loss_payload = {
        "schema_version": 1,
        "loss": contract["loss"],
        "labels_movement": _resolve_training_io_movement(raw).get("labels", {}),
    }
    opt_path.write_text(json.dumps(opt_payload, indent=2, sort_keys=True), encoding="utf-8")
    loss_path.write_text(json.dumps(loss_payload, indent=2, sort_keys=True), encoding="utf-8")
    (reports_dir / "training_optimizer_state.md").write_text(
        "# Training optimizer/state contract\n\n"
        f"- optimizer: `{contract['optimizer']['type']}`\n"
        f"- learning_rate: `{contract['optimizer']['learning_rate']}`\n"
        f"- optimizer_state.storage: `{contract['optimizer_state']['storage']}`\n"
        f"- optimizer_state.import: `{contract['optimizer_state']['import']['resolved']}`\n"
        f"- optimizer_state.export: `{contract['optimizer_state']['export']['resolved']}`\n"
        f"- hls_update_status: `{contract['optimizer']['hls_update_status']}`\n",
        encoding="utf-8",
    )
    (reports_dir / "training_loss_contract.md").write_text(
        "# Training loss contract\n\n"
        f"- loss: `{contract['loss']['type']}`\n"
        f"- hls_status: `{contract['loss']['hls_status']}`\n",
        encoding="utf-8",
    )
    return {
        "training_optimizer_state_json": str(opt_path),
        "training_loss_contract_json": str(loss_path),
    }


def _resolve_codegen_readability(raw: Dict[str, Any]) -> str:
    value = str(_cfg_get(raw, "codegen.readability", "high") or "high").strip().lower().replace("-", "_")
    if value not in _CODEGEN_READABILITY:
        raise ValueError(
            "codegen.readability must be one of "
            + ", ".join(sorted(_CODEGEN_READABILITY))
            + f"; got {value!r}."
        )
    return value


def _runtime_support_from_semantics(memory_semantics_mode: str, *, pipeline_mode: str, raw: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    mode = str(memory_semantics_mode or "").strip().lower()
    is_training = str(pipeline_mode).strip().lower() == "training_on_device"
    raw_cfg = raw or {}
    gradient_export = _resolve_gradient_export_mode(raw_cfg).get("resolved") in {"m_axi_export_full", "m_axi_export_tiled"}
    batch_contract = _resolve_training_batch_accumulation_contract(raw_cfg) if is_training else {}
    native_accumulation = bool(batch_contract.get("native_update_boundary"))
    return {
        "import_weights": mode in {
            "bram_import_full",
            "uram_import_full",
            "bram_import_export_full",
            "uram_import_export_full",
            "ddr_tiled",
            "ddr_tiled_mutable",
        },
        "export_weights": mode in {
            "bram_import_export_full",
            "uram_import_export_full",
            "ddr_tiled_mutable",
        },
        "run_inference": not is_training,
        "run_training": is_training,
        "export_gradients": bool(is_training and gradient_export),
        "reset_accumulators": bool(is_training and native_accumulation),
        "accumulate_gradients": bool(is_training and native_accumulation),
        "apply_accumulated_gradients": bool(is_training and native_accumulation),
    }


def _normalize_runtime_sequence_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, str):
        command = entry.strip().lower().replace("-", "_")
        args: Dict[str, Any] = {}
    elif isinstance(entry, dict) and len(entry) == 1:
        command, args_raw = next(iter(entry.items()))
        command = str(command).strip().lower().replace("-", "_")
        args = dict(args_raw or {}) if isinstance(args_raw, dict) else {}
    else:
        raise ValueError("Each runtime.sequence entry must be a command string or a single-key mapping.")

    if command not in _RUNTIME_COMMANDS:
        raise ValueError(
            f"Unsupported runtime sequence command {command!r}. "
            "Supported commands are: " + ", ".join(sorted(_RUNTIME_COMMANDS)) + "."
        )

    if command == "run_inference":
        repeat = int(args.get("repeat", 1))
        if repeat < 1:
            raise ValueError("runtime.sequence run_inference.repeat must be a positive integer.")
        args["repeat"] = repeat
    if command == "run_training":
        steps = int(args.get("steps", 1))
        if steps < 1:
            raise ValueError("runtime.sequence run_training.steps must be a positive integer.")
        args["steps"] = steps
    if command == "accumulate_gradients":
        steps = int(args.get("steps", args.get("micro_batches", 1)))
        if steps < 1:
            raise ValueError("runtime.sequence accumulate_gradients.steps must be a positive integer.")
        args["steps"] = steps
    return {"command": command, "args": args}


def _resolve_runtime_sequence(
    raw: Dict[str, Any],
    *,
    pipeline_mode: str,
    memory_semantics_mode: str,
) -> Dict[str, Any]:
    explicit = _cfg_has_path(raw, "runtime.sequence")
    raw_sequence = _cfg_get(raw, "runtime.sequence", None)
    if raw_sequence is None:
        raw_sequence = ["run_training"] if str(pipeline_mode).lower() == "training_on_device" else ["run_inference"]
    if not isinstance(raw_sequence, list) or not raw_sequence:
        raise ValueError("runtime.sequence must be a non-empty list of runtime commands.")

    sequence = [_normalize_runtime_sequence_entry(entry) for entry in raw_sequence]
    support = _runtime_support_from_semantics(memory_semantics_mode, pipeline_mode=pipeline_mode, raw=raw)
    unsupported = [item["command"] for item in sequence if not support.get(item["command"], False)]
    if unsupported:
        raise ValueError(
            "runtime.sequence requests command(s) not supported by generated artifacts: "
            + ", ".join(unsupported)
            + f". memory_semantics_mode={memory_semantics_mode!r}, pipeline_mode={pipeline_mode!r}."
        )
    return {
        "schema_version": 1,
        "explicit": explicit,
        "pipeline_mode": str(pipeline_mode),
        "memory_semantics_mode": str(memory_semantics_mode),
        "supported_commands": support,
        "sequence": sequence,
    }


def _write_runtime_sequence_report(out_dir: Path, runtime_sequence: Dict[str, Any]) -> Path:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "runtime_sequence.json"
    path.write_text(json.dumps(runtime_sequence, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Runtime sequence",
        "",
        f"- pipeline_mode: `{runtime_sequence.get('pipeline_mode')}`",
        f"- memory_semantics_mode: `{runtime_sequence.get('memory_semantics_mode')}`",
        f"- explicit: `{runtime_sequence.get('explicit')}`",
        "",
        "## Commands",
    ]
    for item in runtime_sequence.get("sequence", []):
        md.append(f"- `{item.get('command')}` args=`{json.dumps(item.get('args', {}), sort_keys=True)}`")
    (reports_dir / "runtime_sequence.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def _emit_resolved_config_reports(
    out_dir: Path,
    raw: Dict[str, Any],
    *,
    build_stages: Dict[str, bool],
    runtime_sequence: Dict[str, Any],
    memory_plan: Any,
    communication_plan: Any,
    weights_mode: str,
) -> Dict[str, str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "schema_version": 1,
        "pipeline_mode": str(_cfg_get(raw, "pipeline.mode", "inference")),
        "top_kernel_name": str(_cfg_get(raw, "pipeline.outputs.top_kernel_name", "deeplearn")),
        "weights_mode": str(weights_mode),
        "memory_semantics_mode": str(_plan_notes(memory_plan).get("memory_semantics_mode", weights_mode)),
        "build_stages": _build_stage_summary(build_stages),
        "runtime_sequence": runtime_sequence,
        "codegen": {"readability": _resolve_codegen_readability(raw)},
        "memory_plan_notes": _plan_notes(memory_plan),
        "communication_edges": [getattr(edge, "to_dict", lambda: dict(edge))() if not isinstance(edge, dict) else edge for edge in getattr(communication_plan, "edges", [])],
    }
    json_path = reports_dir / "resolved_config.json"
    json_path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    yml_path = reports_dir / "resolved_config.yml"
    try:
        import yaml  # type: ignore
        yml_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    except Exception:
        yml_path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    contract = {
        "schema_version": 1,
        "top_level_sections": sorted(list(getattr(__import__("fpgai.config.loader", fromlist=["TOP_LEVEL_SECTIONS_V1"]), "TOP_LEVEL_SECTIONS_V1"))),
        "build_stage_keys": list(_BUILD_STAGE_KEYS),
        "runtime_commands": sorted(_RUNTIME_COMMANDS),
        "codegen_readability": sorted(_CODEGEN_READABILITY),
        "training_optimizer_types": sorted(_TRAINING_OPTIMIZER_TYPES),
        "training_loss_types": sorted(_TRAINING_LOSS_TYPES),
        "optimizer_state_storage": sorted(_OPTIMIZER_STATE_STORAGE),
        "priority_rules": [
            "manual data_movement overrides user-facing modes",
            "user-facing modes override policy/defaults",
            "unsupported runtime commands reject clearly",
        ],
    }
    contract_path = reports_dir / "config_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "resolved_config_json": str(json_path),
        "resolved_config_yml": str(yml_path),
        "config_contract_json": str(contract_path),
    }


def _plan_notes(plan: Any) -> Dict[str, Any]:
    if plan is None:
        return {}
    notes = getattr(plan, "notes", None)
    if isinstance(notes, dict):
        return dict(notes)
    if isinstance(plan, dict):
        notes = plan.get("notes", plan)
        return dict(notes) if isinstance(notes, dict) else {}
    return {}


@dataclass
class Compiler:
    cfg: FPGAIConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Compiler":
        from fpgai.config.loader import load_config
        return cls(load_config(path))

    def compile(self) -> CompileResult:
        mode = str(self.cfg.pipeline.mode).lower()
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

        numeric_validation_artifacts = emit_numeric_validation_report(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            source_generated=(hls_dir is not None),
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
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
        paper_verification_artifacts = emit_paper_verification_artifacts(
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
        feature_truth_artifacts = _write_feature_truth_reports(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "inference")),
            build_stages=build_stages,
            hls_dir=hls_dir,
            hls_run=hls_run,
            runtime_package=runtime_package,
            numeric_validation_artifacts=numeric_validation_artifacts,
            paper_verification_artifacts=paper_verification_artifacts,
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
                paper_verification_artifacts=paper_verification_artifacts,
                vivado_bd_contract_artifacts=vivado_bd_contract_artifacts,
                feature_truth_artifacts=feature_truth_artifacts,
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

        input_path = self._emit_dummy_input(out_dir, g)
        target_path = self._emit_training_target(out_dir, g, raw)
        x_input = np.fromfile(input_path, dtype=np.float32)
        y_target = np.fromfile(target_path, dtype=np.float32)

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
        hls_calibration_result = run_hls_calibration(
            out_dir=out_dir,
            raw_cfg=raw,
            compile_plan=compile_plan,
            hls_report_dir=(hls_dir if hls_dir is not None else out_dir),
            clock_mhz=float(getattr(compile_plan, "clock_mhz", _cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200.0))),
            verbose=verbose,
        ) if enable_reports else None

        training_compare_result = None
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

        _gradient_export_cfg = _cfg_get(raw, "data_movement.gradients.export", {}) or {}
        _gradient_export_requested = isinstance(_gradient_export_cfg, dict) and str(_gradient_export_cfg.get("interface", "")).lower().replace("-", "_") == "m_axi" and str(_gradient_export_cfg.get("policy", "")).lower().replace("-", "_") in {"full", "tiled"}
        _gradient_export_artifacts = {
            "requested": bool(_gradient_export_requested),
            "policy": (str(_gradient_export_cfg.get("policy", "none")).lower().replace("-", "_") if isinstance(_gradient_export_cfg, dict) else "none"),
            "status": ("covered_by_training_gradient_compare" if (_gradient_export_requested and training_compare_result is not None) else ("generated_not_captured_by_testbench" if _gradient_export_requested else "not_requested")),
            "note": "Gradient export HLS mode is generated; dedicated gradients_mem capture is required for export-specific numeric proof." if _gradient_export_requested else "Gradient export was not requested.",
        }
        _optimizer_contract = _resolve_training_optimizer_loss_contract(raw)
        _optimizer_type = str(_optimizer_contract.get("optimizer", {}).get("type", "sgd"))
        _optimizer_state = _optimizer_contract.get("optimizer_state", {}) or {}
        _optimizer_state_required = bool(_optimizer_state.get("required", False))
        _optimizer_state_export = _optimizer_state.get("export", {}) if isinstance(_optimizer_state.get("export", {}), dict) else {}
        _optimizer_state_requested = bool(_optimizer_state_required or _optimizer_state_export.get("supported", False))
        _state_names = []
        if _optimizer_type == "momentum":
            _state_names = ["velocity"]
        elif _optimizer_type == "adam":
            _state_names = ["first_moment", "second_moment"]
        _optimizer_state_artifacts = {
            "requested": _optimizer_state_requested,
            "optimizer": _optimizer_type,
            "storage": _optimizer_state.get("storage", "none"),
            "expected_tensors": _state_names,
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
            "comparisons": {},
        }
        numeric_validation_artifacts = emit_numeric_validation_report(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            source_generated=(hls_dir is not None),
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            training_reference_result=training_reference_result,
            training_compare_result=training_compare_result,
            gradient_export_artifacts=_gradient_export_artifacts,
            optimizer_state_artifacts=_optimizer_state_artifacts,
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
        paper_verification_artifacts = emit_paper_verification_artifacts(
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
        feature_truth_artifacts = _write_feature_truth_reports(
            out_dir,
            pipeline_mode=str(getattr(self.cfg.pipeline, "mode", "training_on_device")),
            build_stages=build_stages,
            hls_dir=hls_dir,
            hls_run=hls_run,
            runtime_package=runtime_package,
            numeric_validation_artifacts=numeric_validation_artifacts,
            paper_verification_artifacts=paper_verification_artifacts,
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
                paper_verification_artifacts=paper_verification_artifacts,
                vivado_bd_contract_artifacts=vivado_bd_contract_artifacts,
                feature_truth_artifacts=feature_truth_artifacts,
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
        ensure_clean_dir(out_dir, clean=clean)
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
        )
        prediction_artifacts["board_fit"] = board_fit_artifacts
        prediction_artifacts["board_fit_json"] = board_fit_artifacts.get("json")
        prediction_artifacts["board_fit_markdown"] = board_fit_artifacts.get("markdown")

        return prediction_artifacts

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

    def _find_file_recursive(self, root: Path, filename: str) -> Optional[Path]:
        for p in root.rglob(filename):
            return p
        return None

    def _hls_array_partition_mode(self, compile_plan=None) -> str:
        raw_mode = _cfg_get(
            self.cfg.raw,
            "optimization.parallel.array_partition_mode",
            None,
        )
        if raw_mode is None and compile_plan is not None:
            try:
                raw_mode = compile_plan.notes.get("array_partition_mode")
            except Exception:
                raw_mode = None

        mode = str(raw_mode or "cyclic").strip().lower()
        if mode not in {"cyclic", "block"}:
            mode = "cyclic"
        return mode

    def _apply_hls_array_partition_mode(self, source: str, mode: str) -> str:
        mode = str(mode or "cyclic").strip().lower()
        if mode not in {"cyclic", "block"}:
            mode = "cyclic"

        if mode == "cyclic":
            return source

        return source.replace(
            "#pragma HLS ARRAY_PARTITION variable=",
            f"// FPGAI array partition mode: {mode}\\n#pragma HLS ARRAY_PARTITION variable=",
        ).replace(
            " cyclic factor=",
            f" {mode} factor=",
        )

    @staticmethod
    def _normalise_weight_storage(value: Any) -> str:
        raw_value = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "embedded": "bram",
            "on_chip": "bram",
            "onchip": "bram",
            "block": "bram",
            "block_ram": "bram",
            "bram": "bram",
            "uram": "uram",
            "ultra": "uram",
            "ultra_ram": "uram",
            "ddr": "ddr",
            "external": "ddr",
            "external_ddr": "ddr",
            "dma_ddr": "ddr",
        }
        return aliases.get(raw_value, raw_value)

    @staticmethod
    def _normalise_movement_interface(value: Any) -> str:
        raw_value = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "compile": "compile_time",
            "compiletime": "compile_time",
            "compiled": "compile_time",
            "static": "compile_time",
            "embedded": "compile_time",
            "const": "compile_time",
            "bram": "compile_time",
            "uram": "compile_time",
            "axi_dma": "axi_stream",
            "axis": "axi_stream",
            "stream": "axi_stream",
            "streamed": "axi_stream",
            "dma": "axi_stream",
            "ddr": "m_axi",
            "dma_ddr": "m_axi",
            "external": "m_axi",
            "external_ddr": "m_axi",
            "m_axi": "m_axi",
            "maxi": "m_axi",
            "none": "none",
            "off": "none",
        }
        return aliases.get(raw_value, raw_value)

    @staticmethod
    def _normalise_movement_policy(value: Any) -> str:
        raw_value = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "static": "static",
            "compile_time": "static",
            "compiletime": "static",
            "full": "full",
            "preload": "full",
            "preload_full": "full",
            "import_full": "full",
            "load_full": "full",
            "tiled": "tiled",
            "tile": "tiled",
            "stream": "tiled",
            "streaming": "tiled",
            "none": "none",
            "off": "none",
        }
        return aliases.get(raw_value, raw_value)

    @staticmethod
    def _normalise_transport(value: Any, interface: str) -> str:
        raw_value = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "axi_dma": "dma",
            "dma": "dma",
            "ps": "ps_runtime",
            "ps_ddr": "ps_runtime",
            "runtime": "ps_runtime",
            "host": "ps_runtime",
            "none": "none",
            "off": "none",
        }
        if raw_value:
            return aliases.get(raw_value, raw_value)
        if interface == "axi_stream":
            return "dma"
        if interface == "m_axi":
            return "ps_runtime"
        return "none"

    @staticmethod
    def _has_explicit_weight_data_movement(raw: Dict[str, Any]) -> bool:
        """Return True when the user provided detailed weight import/export movement.

        weights.mode is an intent shortcut.  Detailed data_movement entries remain
        higher priority and must not be overwritten by the shortcut expansion.
        Legacy data_movement.ps_pl.weights.mode is treated as a legacy/default
        shortcut, so an explicit top-level weights.mode may override it.
        """
        explicit_paths = (
            "data_movement.weights.import",
            "data_movement.weights.load",
            "data_movement.weights.export",
            "data_movement.weights.store",
        )
        for path in explicit_paths:
            value = _cfg_get(raw, path, None)
            if isinstance(value, dict) and value:
                return True
        scalar_paths = (
            "data_movement.weights.import.interface",
            "data_movement.weights.load.interface",
            "data_movement.weights.export.interface",
            "data_movement.weights.store.interface",
        )
        return any(_cfg_get(raw, path, None) is not None for path in scalar_paths)

    @staticmethod
    def _normalise_user_weight_mode(value: Any) -> str:
        mode = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "": "",
            "static": "embedded",
            "compile_time": "embedded",
            "compiletime": "embedded",
            "const": "embedded",
            "on_chip": "embedded",
            "onchip": "embedded",
            "embedded": "embedded",
            "runtime_import": "import",
            "runtime_loaded": "import",
            "load": "import",
            "import": "import",
            "load_store": "import_export",
            "import_store": "import_export",
            "import_export": "import_export",
            "export_import": "import_export",
            "ddr": "tiled",
            "ddr_tiled": "tiled",
            "tile": "tiled",
            "tiled": "tiled",
            "mutable_tiled": "tiled_mutable",
            "ddr_tiled_mutable": "tiled_mutable",
            "tiled_mutable": "tiled_mutable",
        }
        return aliases.get(mode, mode)

    def _expand_user_weight_mode_to_movement(
        self,
        raw: Dict[str, Any],
        *,
        storage: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any], str | None]:
        """Expand weights.mode / training.weight_initialization.mode to import/export cfg.

        Returns (import_cfg, export_cfg, source_mode).  source_mode is the
        normalized high-level mode that produced the expansion, or None when no
        user-facing shortcut was present.
        """
        mode_value = _cfg_get(raw, "weights.mode", None)
        source_path = "weights.mode"
        if mode_value is None:
            init_mode = _cfg_get(raw, "training.weight_initialization.mode", None)
            if init_mode is not None:
                init = str(init_mode or "").strip().lower().replace("-", "_")
                init_aliases = {
                    "compile_time": "embedded",
                    "compiletime": "embedded",
                    "static": "embedded",
                    "embedded": "embedded",
                    "const": "embedded",
                    "import": "import",
                    "runtime_import": "import",
                    "load": "import",
                }
                if init in {"zero", "xavier", "he", "random", "random_seeded", "seeded_random"}:
                    raise ValueError(
                        "training.weight_initialization.mode={!r} is not implemented yet. "
                        "Supported modes are compile_time and import.".format(init_mode)
                    )
                mode_value = init_aliases.get(init, init)
                source_path = "training.weight_initialization.mode"
        mode = self._normalise_user_weight_mode(mode_value)
        if not mode:
            return {}, {}, None

        pipeline_mode = str(_cfg_get(raw, "pipeline.mode", "inference") or "inference").strip().lower()
        if mode == "embedded":
            if storage == "ddr":
                raise ValueError(
                    f"{source_path}=embedded is only valid with BRAM/URAM weight storage. "
                    "DDR storage means tiled external-memory execution; use weights.mode=tiled."
                )
            return (
                {"interface": "compile_time", "transport": "none", "policy": "static"},
                {"interface": "none", "transport": "none", "policy": "none"},
                mode,
            )
        if mode == "import":
            if storage == "ddr":
                raise ValueError(
                    f"{source_path}=import is only valid with BRAM/URAM weight storage. "
                    "Full import into local memory is not DDR storage; use BRAM/URAM or weights.mode=tiled."
                )
            return (
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                {"interface": "none", "transport": "none", "policy": "none"},
                mode,
            )
        if mode == "import_export":
            if storage == "ddr":
                raise ValueError(
                    f"{source_path}=import_export is only valid with BRAM/URAM weight storage. "
                    "DDR mutable weights must use weights.mode=tiled_mutable."
                )
            return (
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                mode,
            )
        if mode == "tiled":
            if storage != "ddr":
                raise ValueError(
                    f"{source_path}=tiled requires memory.storage.weights=ddr. "
                    f"Got memory.storage.weights={storage!r}."
                )
            return (
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                {"interface": "none", "transport": "none", "policy": "none"},
                mode,
            )
        if mode == "tiled_mutable":
            if storage != "ddr":
                raise ValueError(
                    f"{source_path}=tiled_mutable requires memory.storage.weights=ddr. "
                    f"Got memory.storage.weights={storage!r}."
                )
            if pipeline_mode != "training_on_device":
                raise ValueError(
                    f"{source_path}=tiled_mutable is a training weight mode. "
                    f"Got pipeline.mode={pipeline_mode!r}."
                )
            return (
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                mode,
            )
        raise ValueError(
            f"Unsupported {source_path}={mode_value!r}. Supported values are "
            "embedded, import, import_export, tiled, and tiled_mutable."
        )

    def _reject_unsupported_training_weight_storage(self, raw: Dict[str, Any]) -> None:
        pipeline_mode = str(_cfg_get(raw, "pipeline.mode", "") or "").strip().lower()
        if pipeline_mode != "training_on_device":
            return
        training_weight_storage = str(
            _cfg_get(
                raw,
                "training.storage.weights",
                _cfg_get(raw, "memory.storage.weights", _cfg_get(raw, "memory.weight_storage", "bram")),
            )
            or "bram"
        ).strip().lower().replace("-", "_")
        weight_storage = str(
            _cfg_get(raw, "memory.storage.weights", _cfg_get(raw, "memory.weight_storage", training_weight_storage))
            or training_weight_storage
        ).strip().lower().replace("-", "_")
        aliases = {"ultra": "uram", "ultra_ram": "uram", "external": "ddr", "external_ddr": "ddr", "dma_ddr": "ddr"}
        training_weight_storage = aliases.get(training_weight_storage, training_weight_storage)
        weight_storage = aliases.get(weight_storage, weight_storage)
        # DDR training now maps to the tiled mutable backend for Dense/Conv graphs.
        # Unsupported layer types are rejected by top_train_cpp so the compiler
        # no longer rejects the storage choice before codegen.
        if training_weight_storage == "uram" or weight_storage == "uram":
            board_name = str(_cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "")) or "").strip()
            if board_name:
                try:
                    board = get_board(board_name)
                    if int(board.uram or 0) <= 0:
                        raise ValueError(
                            f"memory.storage.weights=uram requires URAM, but selected board {board_name!r} has 0 URAM blocks."
                        )
                except KeyError:
                    pass

    def _resolve_weight_movement_semantics(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        requested_storage = _cfg_get(
            raw,
            "memory.storage.weights",
            _cfg_get(raw, "memory.weight_storage", None),
        )

        # Legacy configs sometimes specified only data_movement.ps_pl.weights.mode.
        # Treat that as a legacy/default shortcut, not as detailed movement;
        # top-level weights.mode is the newer user-facing intent and should win.
        legacy_mode = None if _cfg_get(raw, "weights.mode", None) is not None else _cfg_get(raw, "data_movement.ps_pl.weights.mode", None)
        storage = self._normalise_weight_storage(requested_storage or "bram")

        explicit_weight_movement = self._has_explicit_weight_data_movement(raw)
        expanded_user_mode: str | None = None
        if explicit_weight_movement:
            import_cfg = _cfg_get(raw, "data_movement.weights.import", None)
            if not isinstance(import_cfg, dict):
                import_cfg = _cfg_get(raw, "data_movement.weights.load", None)
            if not isinstance(import_cfg, dict):
                import_cfg = {}
            export_cfg = _cfg_get(raw, "data_movement.weights.export", None)
            if not isinstance(export_cfg, dict):
                export_cfg = _cfg_get(raw, "data_movement.weights.store", None)
            if not isinstance(export_cfg, dict):
                export_cfg = {}
        else:
            import_cfg, export_cfg, expanded_user_mode = self._expand_user_weight_mode_to_movement(
                raw,
                storage=storage,
            )

        legacy_interface = _cfg_get(raw, "data_movement.weights.load.interface", legacy_mode)
        import_interface = self._normalise_movement_interface(
            import_cfg.get("interface", legacy_interface)
            or ("m_axi" if storage == "ddr" else "compile_time")
        )
        import_policy = self._normalise_movement_policy(
            import_cfg.get("policy", None)
            or ("tiled" if storage == "ddr" else "static")
        )
        import_transport = self._normalise_transport(import_cfg.get("transport", None), import_interface)

        export_interface = self._normalise_movement_interface(export_cfg.get("interface", "none"))
        export_policy = self._normalise_movement_policy(export_cfg.get("policy", "none"))
        export_transport = self._normalise_transport(export_cfg.get("transport", None), export_interface)

        if storage not in {"bram", "uram", "ddr"}:
            raise ValueError(
                "memory.storage.weights must be one of bram, uram, or ddr; "
                f"got {requested_storage!r}. Use data_movement.weights.import/export "
                "for transfer behavior."
            )

        if storage == "bram":
            if import_interface == "compile_time" and import_policy == "static":
                resolved = "bram_static"
                hls_mode = "embedded"
                payload_required = False
            elif import_interface == "m_axi" and import_policy == "full":
                resolved = "bram_import_export_full" if export_interface == "m_axi" and export_policy == "full" else "bram_import_full"
                hls_mode = "ddr"
                payload_required = True
            else:
                raise ValueError(
                    "BRAM weight storage supports import compile_time/static or m_axi/full. "
                    f"Got import {import_interface}/{import_policy}."
                )
        elif storage == "uram":
            if import_interface == "compile_time" and import_policy == "static":
                resolved = "uram_static"
                hls_mode = "embedded"
                payload_required = False
            elif import_interface == "m_axi" and import_policy == "full":
                resolved = "uram_import_export_full" if export_interface == "m_axi" and export_policy == "full" else "uram_import_full"
                hls_mode = "uram"
                payload_required = True
            else:
                raise ValueError(
                    "URAM weight storage supports import compile_time/static or m_axi/full. "
                    f"Got import {import_interface}/{import_policy}."
                )
        else:
            if import_interface == "m_axi" and import_policy == "tiled":
                if export_interface not in {"none", "m_axi"} or (export_interface == "m_axi" and export_policy != "tiled"):
                    raise ValueError(
                        "DDR weight storage supports export none/none for inference or m_axi/tiled for future mutable training. "
                        f"Got export {export_interface}/{export_policy}."
                    )
                resolved = "ddr_tiled_mutable" if export_interface == "m_axi" and export_policy == "tiled" else "ddr_tiled"
                hls_mode = resolved
                payload_required = True
            else:
                raise ValueError(
                    "DDR weight storage means weights stay in DDR and must use m_axi/tiled import. "
                    f"Got import {import_interface}/{import_policy}. Full import belongs to BRAM/URAM storage."
                )

        if storage != "ddr" and export_interface != "none" and not (export_interface == "m_axi" and export_policy == "full"):
            raise ValueError(
                "Weight export currently supports none/none or m_axi/full only for BRAM/URAM storage; "
                f"got export {export_interface}/{export_policy}."
            )

        return {
            "requested_weight_storage": str(requested_storage or storage),
            "resolved_weight_storage": storage,
            "resolved_weight_semantics": resolved,
            "memory_semantics_mode": resolved,
            "hls_weights_mode": hls_mode,
            "runtime_weight_payload_required": payload_required,
            "full_local_weight_replica": storage in {"bram", "uram"},
            "tile_weight_buffer": storage == "ddr",
            "scalable_external_weight_execution": storage == "ddr",
            "weight_import_interface": import_interface,
            "weight_import_transport": import_transport,
            "weight_import_policy": import_policy,
            "weight_export_interface": export_interface,
            "weight_export_transport": export_transport,
            "weight_export_policy": export_policy,
            "user_weight_mode": expanded_user_mode,
            "weight_movement_source": "data_movement" if explicit_weight_movement else ("weights.mode" if expanded_user_mode else "compiler_default"),
            "runtime_commands_supported": [
                *(["import_weights"] if payload_required else []),
                "run_inference",
                *(["export_weights"] if export_interface == "m_axi" and export_policy == "full" else []),
            ],
            "reload_before_each_compute": False,
        }

    def _resolve_activation_storage_semantics(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        requested = _cfg_get(raw, "memory.storage.activations", _cfg_get(raw, "memory.activation_storage", "bram"))
        value = str(requested or "bram").strip().lower().replace("-", "_")
        aliases = {
            "block": "bram",
            "block_ram": "bram",
            "bram": "bram",
            "ultra": "uram",
            "ultra_ram": "uram",
            "uram": "uram",
        }
        storage = aliases.get(value)
        if storage not in {"bram", "uram"}:
            raise ValueError(
                "memory.storage.activations must be bram or uram for this backend; "
                f"got {requested!r}."
            )
        board_name = str(_cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "")) or "").strip()
        if storage == "uram" and board_name:
            try:
                board = get_board(board_name)
                if int(board.uram or 0) <= 0:
                    raise ValueError(
                        f"memory.storage.activations=uram requires URAM, but selected board {board_name!r} has 0 URAM blocks."
                    )
            except KeyError:
                # Unknown-board validation is handled by the board/backend path.
                pass
        return {
            "requested_activation_storage": str(requested or storage),
            "resolved_activation_storage": storage,
            "activation_storage_semantics": f"activation_{storage}",
            "activation_local_buffers": True,
        }

    def _resolve_hls_weights_mode(self, raw: Dict[str, Any]) -> str:
        """Resolve storage + import/export movement into the existing HLS backend mode."""
        return str(self._resolve_weight_movement_semantics(raw)["hls_weights_mode"])

    def _annotate_memory_movement_semantics(self, compile_plan, memory_plan, raw: Dict[str, Any]) -> Dict[str, Any]:
        semantics = self._resolve_weight_movement_semantics(raw)
        activation_semantics = self._resolve_activation_storage_semantics(raw)
        semantics.update(activation_semantics)
        gradient_storage = str(_cfg_get(raw, "training.storage.gradients", _cfg_get(raw, "memory.storage.gradients", "bram")) or "bram").strip().lower().replace("-", "_")
        gradient_aliases = {"block": "bram", "block_ram": "bram", "bram": "bram", "ultra": "uram", "ultra_ram": "uram", "uram": "uram"}
        gradient_storage = gradient_aliases.get(gradient_storage, gradient_storage)
        if gradient_storage not in {"bram", "uram"}:
            raise ValueError(f"training.storage.gradients must be bram or uram for this backend; got {gradient_storage!r}.")
        semantics.update({
            "requested_gradient_storage": gradient_storage,
            "resolved_gradient_storage": gradient_storage,
            "gradient_storage_semantics": f"gradient_{gradient_storage}",
        })
        compile_plan.notes.update(semantics)
        memory_plan.notes.update(semantics)
        return semantics

    def _hls_weight_storage_impl(self, memory_plan=None) -> str:
        raw = self.cfg.raw
        requested = _cfg_get(
            raw,
            "memory.weight_storage",
            _cfg_get(
                raw,
                "memory.storage.weights",
                _cfg_get(raw, "training.storage.weights", "bram"),
            ),
        )
        requested = str(requested or "bram").strip().lower()

        aliases = {
            "embedded": "bram",
            "on_chip": "bram",
            "onchip": "bram",
            "block": "bram",
            "block_ram": "bram",
            "bram": "bram",
            "uram": "uram",
            "ultra": "uram",
            "ultra_ram": "uram",
            "lutram": "lutram",
            "lut_ram": "lutram",
            "distributed": "lutram",
            "ddr": "ddr",
            "external": "ddr",
            "external_ddr": "ddr",
            "dma_ddr": "ddr",
            "stream": "stream",
            "streaming": "stream",
        }
        return aliases.get(requested, "bram")

    def _emit_hls(
        self,
        out_dir: Path,
        g,
        *,
        top_name: str,
        weights_mode: str,
        compile_plan=None,
        memory_plan=None,
        communication_plan=None,
        build_stages: Optional[Dict[str, bool]] = None,
    ) -> Path:
        from fpgai.backends.hls.codegen import emit_hls_stub

        raw = self.cfg.raw
        part = str(_cfg_get(raw, "targets.platform.part", "xck26-sfvc784-2LV-c"))
        clk_mhz = float(getattr(compile_plan, "clock_mhz", _cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200)))
        pipeline_mode = str(self.cfg.pipeline.mode).lower()
        training_cfg = (_cfg_get(raw, "training", {}) or {})

        intermediate_dump = bool(_cfg_get(raw, "benchmark.intermediate.enabled", False))
        stages = build_stages or _resolve_build_stages(raw)
        emit_hls_project = bool(stages.get("hls_project", True))
        emit_testbench = bool(stages.get("testbench", True))
        if pipeline_mode == "training_on_device":
            intermediate_dump = bool(_cfg_get(raw, "training.debug.dump_intermediates", False))

        proj = emit_hls_stub(
            graph=g,
            out_dir=out_dir,
            top_name=top_name,
            hls_options={
                "weights_mode": weights_mode,
                "part": part,
                "clk_mhz": int(clk_mhz),
                "proj_name": "fpgai_hls_proj",
                "intermediate_dump": intermediate_dump,
                "pipeline_mode": pipeline_mode,
                "training_cfg": training_cfg,
                "raw_cfg": raw,
            },
            compile_plan=compile_plan,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
        )
        hls_dir = proj.hls_dir
        inc_dir = hls_dir / "include"
        layers_inc_dir = inc_dir / "layers"
        src_dir = hls_dir / "src"
        array_partition_mode = self._hls_array_partition_mode(compile_plan)
        layers_src_dir = src_dir / "layers"

        inc_dir.mkdir(parents=True, exist_ok=True)
        layers_inc_dir.mkdir(parents=True, exist_ok=True)
        src_dir.mkdir(parents=True, exist_ok=True)
        layers_src_dir.mkdir(parents=True, exist_ok=True)

        write_text(
            inc_dir / "fpgai_types.h",
            emit_types_h(g, top_name=top_name, raw_cfg=raw, compile_plan=compile_plan),
        )
        write_text(layers_inc_dir / "dense.h", emit_dense_h())
        write_text(
            layers_src_dir / "dense.cpp",
            self._apply_hls_array_partition_mode(
                emit_dense_cpp(),
                array_partition_mode,
            ),
        )
        write_text(layers_inc_dir / "conv.h", emit_conv_h())
        write_text(
            layers_src_dir / "conv.cpp",
            self._apply_hls_array_partition_mode(
                emit_conv_cpp(),
                array_partition_mode,
            ),
        )
        write_text(layers_inc_dir / "pool.h", emit_pool_h())
        write_text(
            layers_src_dir / "pool.cpp",
            self._apply_hls_array_partition_mode(
                emit_pool_cpp(),
                array_partition_mode,
            ),
        )
        write_text(layers_inc_dir / "activations.h", emit_activations_h())
        write_text(layers_src_dir / "activations.cpp", emit_activations_cpp())
        write_text(layers_inc_dir / "batchnorm.h", emit_batchnorm_h())
        write_text(layers_src_dir / "batchnorm.cpp", emit_batchnorm_cpp())

        normalized_weights_mode = str(weights_mode).strip().lower()
        runtime_weight_mode = _is_runtime_weight_mode(normalized_weights_mode)
        storage_impl = self._hls_weight_storage_impl(memory_plan)

        if runtime_weight_mode:
            write_text(inc_dir / "weights_runtime.h", emit_weights_runtime_h(g))
            write_text(src_dir / "weights_runtime.cpp", emit_weights_runtime_cpp(g))
            write_text(
                inc_dir / "fpgai_params.h",
                emit_params_h(g, weights_mode=normalized_weights_mode),
            )
            write_text(
                src_dir / "fpgai_params.cpp",
                emit_params_cpp(
                    g,
                    weights_mode=normalized_weights_mode,
                    storage_impl=storage_impl,
                ),
            )
        else:
            write_text(inc_dir / "fpgai_params.h", emit_params_h(g, weights_mode="embedded"))
            write_text(
                src_dir / "fpgai_params.cpp",
                emit_params_cpp(
                    g,
                    weights_mode="embedded",
                    storage_impl=storage_impl,
                ),
            )

        input_bin = str((out_dir / "input.bin").resolve())

        if pipeline_mode == "training_on_device":
            write_text(
                src_dir / f"{top_name}.cpp",
                _emit_top_train_cpp(
                    graph=g,
                    top_name=top_name,
                    weights_mode=weights_mode,
                    training_cfg=training_cfg,
                    compile_plan=compile_plan,
                    memory_plan=memory_plan,
                    communication_plan=communication_plan,
                    raw_cfg=self.cfg.raw,
                ),
            )

            target_bin = str((out_dir / "target.bin").resolve())

            def _try_numel_tensor(name: str) -> int:
                if not name:
                    return 0

                try:
                    t = g.get_tensor(name)
                except Exception:
                    t = None

                if t is not None and getattr(t, "shape", None):
                    return int(np.prod(tuple(int(v) for v in t.shape)))

                if hasattr(g, "constants") and name in getattr(g, "constants", {}):
                    return int(np.asarray(g.constants[name]).size)

                if hasattr(g, "params") and name in getattr(g, "params", {}):
                    return int(np.asarray(g.params[name]).size)

                if t is not None:
                    data = getattr(t, "data", None)
                    if data is not None:
                        return int(np.asarray(data).size)
                    for attr_name in ("initializer", "value", "values"):
                        if hasattr(t, attr_name):
                            arr = getattr(t, attr_name)
                            if arr is not None:
                                return int(np.asarray(arr).size)

                return 0

            def _attr_numel(op, *keys: str) -> int:
                attrs = getattr(op, "attrs", {}) or {}
                for k in keys:
                    if k not in attrs:
                        continue
                    v = attrs[k]
                    if isinstance(v, str):
                        n = _try_numel_tensor(v)
                        if n > 0:
                            return n
                    else:
                        try:
                            arr = np.asarray(v)
                            if arr.dtype.kind not in ("U", "S", "O"):
                                return int(arr.size)
                        except Exception:
                            pass
                return 0

            total_param_words = 0
            for op in g.ops:
                if op.op_type == "Dense":
                    in_f = int(op.attrs.get("in_features") or 0)
                    out_f = int(op.attrs.get("out_features") or 0)

                    if in_f > 0 and out_f > 0:
                        total_param_words += out_f * in_f
                        total_param_words += out_f
                    else:
                        w_n = 0
                        b_n = 0
                        if len(op.inputs) > 1:
                            w_n = _try_numel_tensor(op.inputs[1])
                        if len(op.inputs) > 2:
                            b_n = _try_numel_tensor(op.inputs[2])
                        if w_n == 0:
                            w_n = _attr_numel(op, "weights", "weight", "W", "kernel", "weights_name", "weight_name")
                        if b_n == 0:
                            b_n = _attr_numel(op, "bias", "biases", "B", "bias_name")
                        total_param_words += w_n + b_n

                elif op.op_type == "Conv":
                    w_n = 0
                    b_n = 0

                    if len(op.inputs) > 1:
                        w_n = _try_numel_tensor(op.inputs[1])
                    if len(op.inputs) > 2:
                        b_n = _try_numel_tensor(op.inputs[2])

                    if w_n == 0:
                        w_n = _attr_numel(op, "weights", "weight", "W", "kernel", "weights_name", "weight_name")
                    if b_n == 0:
                        b_n = _attr_numel(op, "bias", "biases", "B", "bias_name")

                    total_param_words += w_n + b_n

                elif op.op_type == "BatchNormalization":
                    xshape = None
                    try:
                        xshape = g.get_tensor(op.inputs[0])
                    except Exception:
                        xshape = None

                    c = None
                    if xshape is not None and getattr(xshape, "shape", None):
                        shp = tuple(int(v) for v in xshape.shape)
                        if len(shp) > 1 and shp[0] == 1:
                            shp = shp[1:]
                        if len(shp) == 3:
                            c = int(shp[0])

                    if c is None:
                        gamma_n = 0
                        if len(op.inputs) > 1:
                            gamma_n = _try_numel_tensor(op.inputs[1])
                        if gamma_n == 0:
                            gamma_n = _attr_numel(op, "scale", "gamma", "scale_name", "gamma_name")
                        c = gamma_n if gamma_n > 0 else None

                    if c is not None:
                        total_param_words += 2 * c

            if emit_testbench:
                emit_tb_train_cpp(
                    src_dir,
                    graph=g,
                    top_name=top_name,
                    in_words=int(np.fromfile(input_bin, dtype=np.float32).size),
                    out_words=0,
                    weights_mode=weights_mode,
                    weight_words=total_param_words,
                    preload_weights=[],
                    training_cfg=training_cfg,
                )

            if emit_hls_project:
                write_text(
                    hls_dir / "run_hls.tcl",
                    emit_csim_train_tcl(
                        top_name=top_name,
                        part=part,
                        input_bin_path=input_bin,
                        target_bin_path=target_bin,
                        weights_mode=weights_mode,
                        intermediate_dump=intermediate_dump,
                    ),
                )
        else:
            write_text(
                src_dir / f"{top_name}.cpp",
                emit_top_cpp(
                    g,
                    top_name=top_name,
                    weights_mode=weights_mode,
                    compile_plan=compile_plan,
                    memory_plan=memory_plan,
                    communication_plan=communication_plan,
                raw_cfg=self.cfg.raw,
                ),
            )

            x_name = g.inputs[0]
            y_name = g.outputs[0]
            x_spec = g.get_tensor(x_name)
            y_spec = g.get_tensor(y_name)

            in_shape = tuple(int(d) for d in x_spec.shape) if x_spec and x_spec.shape else (1,)
            out_shape = tuple(int(d) for d in y_spec.shape) if y_spec and y_spec.shape else (1,)

            in_words = int(np.prod(in_shape)) if in_shape else 1
            if len(out_shape) > 1 and out_shape[0] == 1:
                out_shape = out_shape[1:]
            out_words = int(np.prod(out_shape)) if out_shape else 1

            runtime_weight_words = (
                _runtime_weight_word_count(g)
                if runtime_weight_mode
                else 0
            )

            if emit_testbench:
                emit_tb_cpp(
                    src_dir,
                    top_name=top_name,
                    in_words=in_words,
                    out_words=out_words,
                    weights_mode=weights_mode,
                    weight_words=runtime_weight_words,
                
                    raw_cfg=self.cfg.raw,)

            if emit_hls_project:
                write_text(
                    hls_dir / "run_hls.tcl",
                    emit_csim_tcl(
                        top_name=top_name,
                        part=part,
                        clk_period_ns=(1000.0 / clk_mhz),
                        input_bin_path=input_bin,
                        output_bin_path=str((out_dir / "output.bin").resolve()),
                        weights_mode=weights_mode,
                        intermediate_dump=intermediate_dump,
                    ),
                )

        if not emit_hls_project:
            # C++-only builds must not leave an HLS project driver behind.
            # Some lower-level emit helpers may create a default run_hls.tcl as
            # part of their legacy project scaffold; remove it after source
            # emission so build.stages.hls_project=false has a strict artifact
            # contract.
            run_hls_tcl = hls_dir / "run_hls.tcl"
            if run_hls_tcl.exists():
                run_hls_tcl.unlink()

        return hls_dir

    def _maybe_run_vitis_hls(self, hls_dir: Path, *, build_stages: Optional[Dict[str, bool]] = None):
        raw = self.cfg.raw
        if build_stages is None:
            run_enabled = bool(_cfg_get(raw, "toolchain.vitis_hls.enabled", False))
        else:
            run_enabled = bool(build_stages.get("hls_synthesis", False))
        if not run_enabled:
            return None
        vitis_exe = str(_cfg_get(raw, "backends.hls.vitis.exe", _cfg_get(raw, "toolchain.vitis_hls.exe", "vitis_hls")))
        settings64 = _cfg_get(raw, "toolchain.vitis_hls.settings64", None)
        from fpgai.backends.hls.runner import run_vitis_hls
        return run_vitis_hls(hls_dir=hls_dir, vitis_hls_exe=vitis_exe, settings64=settings64)

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
                "dse_truth",
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
        add(
            "targets.platform.fit_policy",
            self._raw_get_path(raw, "targets.platform.fit_policy", self._raw_get_path(raw, "hardware.fit_policy", "report_only")),
            applied_to=[
                "board-fit reporting now",
                "fit_policy_gate manifest decision",
                "Vivado/bitstream gating decision",
            ],
            status="report_only",
            note="fit_policy is enforced through fit_policy_gate. Main compile records the gate; the Vivado bridge flow must honor blocked=true before implementation/bitstream.",
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
                "full board-fit enforcement is handled by the dedicated board-fit sprint/report."
            ),
            "precedence": [
                "manual_yaml_override",
                "board_aware_policy_scaling",
                "policy_preset",
                "compiler_default",
            ],
            "truth_boundary": {
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
                "## Truth boundary",
                "",
                "- This report proves YAML-to-planner traceability.",
                "- HLS traceability is proven where generated macros, comments, template arguments, or pragmas expose the knob.",
                "- Vivado and runtime truth require real Vivado reports, bitstreams, and board execution artifacts.",
                "- If a manual YAML knob appears as `unknown`, `not_requested`, `changed_or_clamped`, or `report_only`, it must not be claimed as fully implemented until a later sprint fixes or validates it.",
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


    def _fit_policy_gate(self, prediction_artifacts: dict[str, Any] | None) -> dict[str, Any]:
        raw = self.cfg.raw
        policy = str(
            self._raw_get_path(
                raw,
                "targets.platform.fit_policy",
                self._raw_get_path(raw, "hardware.fit_policy", "report_only"),
            )
            or "report_only"
        ).strip().lower()

        if policy not in {"report_only", "warn", "enforce"}:
            policy = "report_only"

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
            "resolved_config_artifacts": kwargs.get("resolved_config_artifacts"),
            "numeric_validation_artifacts": None if kwargs.get("numeric_validation_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("numeric_validation_artifacts", {}).items()
            },
            "generated_hls_explanation_artifacts": None if kwargs.get("generated_hls_explanation_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("generated_hls_explanation_artifacts", {}).items()
            },
            "paper_verification_artifacts": None if kwargs.get("paper_verification_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("paper_verification_artifacts", {}).items()
            },
            "vivado_bd_contract_artifacts": None if kwargs.get("vivado_bd_contract_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("vivado_bd_contract_artifacts", {}).items()
            },
            "feature_truth_artifacts": None if kwargs.get("feature_truth_artifacts") is None else {
                key: str(value) for key, value in kwargs.get("feature_truth_artifacts", {}).items()
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
