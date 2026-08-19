from __future__ import annotations

"""Training, movement, runtime-sequence, and feature contracts.

These functions resolve user configuration into explicit compiler contracts and
write the corresponding reports. They do not orchestrate compilation stages.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fpgai.config.access import get_path
from fpgai.engine.build_stages import cfg_has_path as _cfg_has_path

_cfg_get = get_path

_RUNTIME_COMMANDS = {
    "import_weights",
    "run_inference",
    "run_training",
    "export_weights",
    "export_gradients",
    "export_optimizer_state",
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
    # The public shorthand is an explicit user override and must win over
    # legacy/default training.batch.size values already present in example
    # configs.  Otherwise setting training.batch_size=2 can be silently masked
    # by an inherited training.batch.size=1.
    batch_raw = _cfg_get(raw, "training.batch_size", None)
    if batch_raw is None:
        batch_raw = _cfg_get(raw, "training.batch.size", None)
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
        "validation_boundary": (
            "Native batch/gradient accumulation generates explicit HLS modes 3/4/5 for "
            "reset, accumulate, and apply/update. Validation-qualified correctness still requires the "
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
        "validation_boundary": (
            "This contract records compiler-side feasibility and selected knobs. "
            "Vivado/bitstream enforcement is validation-qualified only when Vivado artifacts are present."
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
        "validation_boundary": "This report records the expected Vivado block-design wiring from generated HLS interfaces. It is not proof that Vivado implemented the design; Vivado reports/bitstream are required for that claim.",
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


def _write_feature_validation_reports(
    out_dir: Path,
    *,
    pipeline_mode: str,
    build_stages: Dict[str, bool],
    hls_dir: Optional[Path],
    hls_run: Any | None,
    runtime_package: Any | None,
    numeric_validation_artifacts: Optional[Dict[str, Any]],
    validation_summary_artifacts: Optional[Dict[str, Any]],
    vivado_bd_contract_artifacts: Optional[Dict[str, str]],
    vivado_handoff_artifacts: Optional[Dict[str, str]] = None,
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
    validation_ready = False
    if validation_summary_artifacts:
        pv_path = validation_summary_artifacts.get("validation_summary_json") or validation_summary_artifacts.get("benchmark_row_json")
        try:
            pv = json.loads(Path(pv_path).read_text(encoding="utf-8")) if pv_path else {}
            validation_ready = bool(pv.get("validation_ready"))
        except Exception:
            validation_ready = False
    features = [
        {"feature": "source_generation", "status": "validated_by_generated_source" if source_generated else "not_generated", "validation_ready_for": ["source-exists records"] if source_generated else []},
        {"feature": "numeric_correctness", "status": "validated" if numeric_validated else "not_validated", "validation_ready_for": ["correctness records"] if numeric_validated else []},
        {"feature": "hls_resource_timing", "status": "validated_by_hls" if hls_synthesized else "not_validated", "validation_ready_for": ["HLS resource/timing records"] if hls_synthesized else []},
        {"feature": "vivado_bd_wiring", "status": "tcl_generated" if vivado_handoff_artifacts else ("contract_generated" if vivado_bd_contract_artifacts else "not_generated"), "validation_ready_for": ["BD Tcl generation records"] if vivado_handoff_artifacts else (["BD contract records"] if vivado_bd_contract_artifacts else [])},
        {"feature": "runtime_package", "status": "packaged" if runtime_packaged else "not_packaged", "validation_ready_for": ["runtime package existence records"] if runtime_packaged else []},
        {"feature": "fpga_execution", "status": "not_validated", "validation_ready_for": []},
    ]
    payload = {
        "format": "fpgai.feature_contract.v1",
        "pipeline_mode": pipeline_mode,
        "build_stages": {str(k): bool(v) for k, v in build_stages.items()},
        "source_generated": source_generated,
        "numeric_validated": numeric_validated,
        "hls_synthesized": hls_synthesized,
        "runtime_packaged": runtime_packaged,
        "validation_ready": validation_ready,
        "features": features,
        "validation_boundary": "Each feature is validation-qualified only at the verification level recorded here. Contract/report generation is not the same as HLS, Vivado, bitstream, or FPGA validation.",
    }
    json_path = reports_dir / "feature_contract.json"
    audit_path = reports_dir / "claim_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# FPGAI claim audit", "", f"- pipeline_mode: `{pipeline_mode}`", f"- source_generated: `{source_generated}`", f"- numeric_validated: `{numeric_validated}`", f"- hls_synthesized: `{hls_synthesized}`", f"- runtime_packaged: `{runtime_packaged}`", f"- validation_ready: `{validation_ready}`", "", "| Feature | Status | Validation-qualified for |", "|---|---|---|"]
    for item in features:
        lines.append(f"| {item['feature']} | {item['status']} | {', '.join(item['validation_ready_for']) if item['validation_ready_for'] else 'none'} |")
    lines.extend(["", "This audit must be used by benchmark table/plot generation to avoid intention-only records.", ""])
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

    storage_raw = _cfg_get(raw, "memory.optimizer_state_storage", None)
    if storage_raw is None:
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
            "memory.optimizer_state_storage must be one of none, bram, uram, ddr; "
            f"got {storage!r}."
        )

    import_movement = _resolve_optimizer_state_movement(raw, "import")
    export_movement = _resolve_optimizer_state_movement(raw, "export")

    state_required = optimizer_type in {"momentum", "adam"}
    if state_required and storage == "none":
        raise ValueError(
            f"training.optimizer.type={optimizer_type!r} requires persistent optimizer state; "
            "memory.optimizer_state_storage must be bram, uram, or ddr."
        )

    momentum_value = float(_cfg_get(raw, "training.optimizer.momentum", 0.9) or 0.9)
    beta1_value = float(_cfg_get(raw, "training.optimizer.beta1", 0.9) or 0.9)
    beta2_value = float(_cfg_get(raw, "training.optimizer.beta2", 0.999) or 0.999)
    epsilon_value = float(_cfg_get(raw, "training.optimizer.epsilon", 1.0e-8) or 1.0e-8)
    if optimizer_type == "momentum" and not (0.0 <= momentum_value < 1.0):
        raise ValueError("training.optimizer.momentum must satisfy 0 <= momentum < 1.")
    if optimizer_type == "adam":
        if not (0.0 <= beta1_value < 1.0):
            raise ValueError("training.optimizer.beta1 must satisfy 0 <= beta1 < 1.")
        if not (0.0 <= beta2_value < 1.0):
            raise ValueError("training.optimizer.beta2 must satisfy 0 <= beta2 < 1.")
        if epsilon_value <= 0.0:
            raise ValueError("training.optimizer.epsilon must be positive.")

    # Support is reported by execution path. SGD, Momentum, and Adam have
    # dataset-wide multi-epoch software and hardware-domain references. Adam HLS
    # is implemented at CSim level with persistent step, bias correction,
    # canonical m/v export, parameter-update validation, and propagated
    # quantization alignment. Board runtime remains separately not validated.
    dataset_multi_epoch_status = (
        "implemented" if optimizer_type in {"sgd", "momentum", "adam"} else "not_implemented"
    )
    dataset_multi_epoch_hls_status = "implemented" if optimizer_type in {"sgd", "momentum", "adam"} else "not_implemented"
    end_to_end_multi_epoch_status = "implemented" if optimizer_type in {"sgd", "momentum", "adam"} else "not_implemented"
    hls_update_status = "implemented"
    loss_hls_status = "implemented" if loss_type in {"mse", "cross_entropy"} else "not_implemented"
    loss_numeric_status = (
        "implemented"
        if loss_type in {"mse", "cross_entropy"}
        else "not_implemented"
    )

    generated_state = storage in {"bram", "uram", "ddr"} or import_movement["supported"] or export_movement["supported"]
    return {
        "schema_version": 2,
        "optimizer": {
            "type": optimizer_type,
            "learning_rate": learning_rate,
            "momentum": momentum_value,
            "beta1": beta1_value,
            "beta2": beta2_value,
            "epsilon": epsilon_value,
            "hls_update_status": hls_update_status,
            "numeric_validation_status": "implemented",
            "support_status": {
                "generated_hls_update": "implemented",
                "single_step_reference": "implemented",
                "single_step_numeric_validation": "implemented",
                "dataset_multi_epoch_reference": dataset_multi_epoch_status,
                "dataset_multi_epoch_hls": dataset_multi_epoch_hls_status,
                "end_to_end_multi_epoch_validation": end_to_end_multi_epoch_status,
                "board_runtime_validation": "not_validated",
            },
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
        "schema_version": 2,
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
        f"- hls_update_status: `{contract['optimizer']['hls_update_status']}`\n"
        f"- dataset_multi_epoch_reference: `{contract['optimizer']['support_status']['dataset_multi_epoch_reference']}`\n"
        f"- end_to_end_multi_epoch_validation: `{contract['optimizer']['support_status']['end_to_end_multi_epoch_validation']}`\n"
        f"- board_runtime_validation: `{contract['optimizer']['support_status']['board_runtime_validation']}`\n",
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
    optimizer_contract = _resolve_training_optimizer_loss_contract(raw_cfg) if is_training else {}
    optimizer_state = optimizer_contract.get("optimizer_state", {}) if isinstance(optimizer_contract, dict) else {}
    optimizer_state_export = False
    if isinstance(optimizer_state, dict):
        optimizer_export = optimizer_state.get("export", {})
        optimizer_state_export = bool(isinstance(optimizer_export, dict) and optimizer_export.get("supported"))
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
        "export_optimizer_state": bool(is_training and optimizer_state_export),
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
        if str(pipeline_mode).lower() == "training_on_device":
            raw_sequence = ["run_training"]
        else:
            import_required = memory_semantics_mode in {
                "bram_import_export_full",
                "uram_import_export_full",
                "ddr_tiled_mutable",
                "ddr",
                "dma_ddr",
                "ddr_tiled",
                "runtime_ddr",
                "m_axi",
                "external_ddr",
                "uram",
            }
            raw_sequence = (["import_weights"] if import_required else []) + ["run_inference"]
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
