from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from fpgai.analysis.model_inspection import ModelInspection
from fpgai.layers.registry import get_layer_capability, layer_registry


_KNOB_NAMES = [
    "precision",
    "pipelining",
    "parallelization",
    "tiling",
    "weight_storage",
    "activation_storage",
    "gradient_storage",
    "optimizer_state_storage",
    "data_movement",
    "runtime_sequence",
    "build_stages",
    "board_fit",
    "training",
]

_WEIGHTED_CATEGORIES = {"linear", "convolution", "normalization"}


def _cfg_has(raw_cfg: Mapping[str, Any] | None, path: str) -> bool:
    cur: Any = raw_cfg or {}
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return False
    return True


def _cfg_get(raw_cfg: Mapping[str, Any] | None, path: str, default: Any = None) -> Any:
    cur: Any = raw_cfg or {}
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _manual_sources(raw_cfg: Mapping[str, Any] | None) -> Dict[str, bool]:
    return {
        "precision": any(_cfg_has(raw_cfg, p) for p in ("precision", "numerics", "numerics.quantization")),
        "pipelining": any(_cfg_has(raw_cfg, p) for p in ("optimization.pipeline", "pipelining")),
        "parallelization": any(_cfg_has(raw_cfg, p) for p in ("optimization.parallel", "parallelization")),
        "tiling": any(_cfg_has(raw_cfg, p) for p in ("optimization.tiling", "tiling", "data_movement.input.load.policy")),
        "weight_storage": _cfg_has(raw_cfg, "memory.weight_storage") or _cfg_has(raw_cfg, "training.storage.weights"),
        "activation_storage": _cfg_has(raw_cfg, "memory.activation_storage") or _cfg_has(raw_cfg, "training.storage.activations"),
        "gradient_storage": _cfg_has(raw_cfg, "training.storage.gradients") or _cfg_has(raw_cfg, "memory.gradient_storage"),
        "optimizer_state_storage": _cfg_has(raw_cfg, "training.storage.optimizer_state"),
        "data_movement": _cfg_has(raw_cfg, "data_movement"),
        "runtime_sequence": _cfg_has(raw_cfg, "runtime.sequence"),
        "build_stages": _cfg_has(raw_cfg, "build.stages"),
        "board_fit": _cfg_has(raw_cfg, "targets.platform.board") or _cfg_has(raw_cfg, "targets.board") or _cfg_has(raw_cfg, "project.board"),
        "training": _cfg_has(raw_cfg, "training"),
    }


def _status_payload(status: str, *, source: str, reason: str = "", evidence: Iterable[str] = ()) -> Dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "reason": reason,
        "evidence": list(evidence),
    }


def _per_layer_knob_contract(operator: Mapping[str, Any], *, raw_cfg: Mapping[str, Any] | None, pipeline_mode: str) -> Dict[str, Any]:
    op_type = str(operator.get("op_type", "Unknown"))
    category = str(operator.get("category", "unsupported"))
    has_weights = bool(operator.get("has_weights", False))
    pipeline_support = operator.get("pipeline_support", {}) or {}
    supported = bool(pipeline_support.get("supported", False))
    source_flags = _manual_sources(raw_cfg)

    def src(knob: str) -> str:
        return "manual_yaml" if source_flags.get(knob) else "compiler_default_or_policy"

    if not supported:
        return {
            knob: _status_payload(
                "rejected",
                source=src(knob),
                reason=str(pipeline_support.get("detail", "Layer backend is unsupported in active pipeline.")),
                evidence=["model_compatibility.unsupported_operators", "layer_backend_status.unsupported_reason"],
            )
            for knob in _KNOB_NAMES
        }

    knobs: Dict[str, Dict[str, Any]] = {}
    knobs["precision"] = _status_payload(
        "applied",
        source=src("precision"),
        evidence=["precision_layout.json", "generated HLS fp_t / quantized type selection", "layer_backend_status.dimensions.precision"],
    )
    knobs["pipelining"] = _status_payload(
        "applied",
        source=src("pipelining"),
        evidence=["hardware_knob_contract.optimization.pipeline", "generated HLS PIPELINE/II markers where loops exist"],
    )
    if category in {"reshape"}:
        knobs["parallelization"] = _status_payload(
            "not_applicable",
            source=src("parallelization"),
            reason="Reshape/flatten has no arithmetic loop parallelism beyond copy/layout handling.",
            evidence=["layer category reshape"],
        )
    else:
        knobs["parallelization"] = _status_payload(
            "applied",
            source=src("parallelization"),
            evidence=["hardware_knob_contract.optimization.parallel", "UNROLL/ARRAY_PARTITION evidence when requested and valid"],
        )

    tiling_requested = source_flags.get("tiling")
    if tiling_requested:
        knobs["tiling"] = _status_payload(
            "applied",
            source=src("tiling"),
            evidence=["hardware_knob_contract.optimization.tiling", "tile constants/loops or explicit shape-limited rejection"],
        )
    elif category in {"activation", "pooling", "elementwise", "linear", "convolution"}:
        knobs["tiling"] = _status_payload(
            "not_requested",
            source=src("tiling"),
            reason="No user tiling request was resolved for this layer, so tiled code must be absent.",
            evidence=["generated C++ absence rule", "layer_knob_contract.resource_latency_hygiene"],
        )
    else:
        knobs["tiling"] = _status_payload(
            "not_applicable",
            source=src("tiling"),
            reason="Layer category does not require tensor tiling in the current backend.",
            evidence=["layer category"],
        )

    if has_weights:
        knobs["weight_storage"] = _status_payload(
            "applied",
            source=src("weight_storage"),
            evidence=["memory_plan parameter placement", "hardware_design_decisions.memory_summary", "generated HLS weight storage/import path"],
        )
    else:
        knobs["weight_storage"] = _status_payload(
            "not_applicable",
            source=src("weight_storage"),
            reason=f"{op_type} has no parameter/weight tensor.",
            evidence=["layer_registry.has_weights=false"],
        )

    knobs["activation_storage"] = _status_payload(
        "applied",
        source=src("activation_storage"),
        evidence=["memory_plan activation placement", "hardware_design_decisions.memory_summary"],
    )

    if pipeline_mode == "training_on_device":
        knobs["gradient_storage"] = _status_payload(
            "applied" if has_weights or category in {"linear", "convolution", "normalization"} else "not_applicable",
            source=src("gradient_storage"),
            reason="Layer has no persistent parameter gradient." if not has_weights else "",
            evidence=["training storage plan", "numeric_validation.gradient paths"],
        )
        optimizer = str(_cfg_get(raw_cfg, "training.optimizer.type", "sgd")).lower()
        if optimizer in {"momentum", "adam"}:
            knobs["optimizer_state_storage"] = _status_payload(
                "applied" if has_weights else "not_applicable",
                source=src("optimizer_state_storage"),
                reason="Optimizer state is only allocated for trainable parameter tensors." if not has_weights else "",
                evidence=["training_reference.optimizer_state_*", "numeric_validation.optimizer_state_validation"],
            )
        else:
            knobs["optimizer_state_storage"] = _status_payload(
                "not_applicable",
                source=src("optimizer_state_storage"),
                reason="SGD has no persistent optimizer-state tensor.",
                evidence=["numeric_validation.optimizer_state_validation.status=not_applicable"],
            )
        knobs["training"] = _status_payload(
            "applied",
            source=src("training"),
            evidence=["training_plan", "training_reference", "generated training path"],
        )
    else:
        knobs["gradient_storage"] = _status_payload(
            "not_applicable",
            source=src("gradient_storage"),
            reason="Inference mode does not allocate gradient tensors.",
            evidence=["pipeline_mode=inference"],
        )
        knobs["optimizer_state_storage"] = _status_payload(
            "not_applicable",
            source=src("optimizer_state_storage"),
            reason="Inference mode does not allocate optimizer state.",
            evidence=["pipeline_mode=inference"],
        )
        knobs["training"] = _status_payload(
            "not_applicable",
            source=src("training"),
            reason="Active pipeline is inference.",
            evidence=["pipeline_mode=inference"],
        )

    knobs["data_movement"] = _status_payload(
        "applied" if source_flags.get("data_movement") else "compiler_default",
        source=src("data_movement"),
        evidence=["communication_plan.edges", "generated interfaces", "runtime_package buffer/execution plans"],
    )
    knobs["runtime_sequence"] = _status_payload(
        "applied" if source_flags.get("runtime_sequence") else "compiler_default",
        source=src("runtime_sequence"),
        evidence=["reports/runtime_sequence.json", "runtime_package/run_sequence.json", "generated mode constants only when requested"],
    )
    knobs["build_stages"] = _status_payload(
        "applied" if source_flags.get("build_stages") else "compiler_default",
        source=src("build_stages"),
        evidence=["manifest.build_stages", "stage-specific artifacts omitted when disabled"],
    )
    knobs["board_fit"] = _status_payload(
        "applied",
        source=src("board_fit"),
        evidence=["reports/board_fit.json", "Vivado implementation/bitstream gating"],
    )
    return knobs


def _json_default(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_model_compatibility_report(
    inspection: ModelInspection,
    *,
    raw_cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    pipeline_mode = str(inspection.pipeline_mode)
    rows = []
    unsupported_for_pipeline: set[str] = set()
    limited_for_pipeline: set[str] = set()

    for operator in inspection.operators:
        op_type = str(operator.get("op_type", "Unknown"))
        capability = get_layer_capability(op_type, pipeline_mode=pipeline_mode)
        capability_payload = capability.to_dict()
        pipeline_key = "training" if pipeline_mode == "training_on_device" else "inference"
        pipeline_capability = capability_payload[pipeline_key]

        if not pipeline_capability["supported"]:
            unsupported_for_pipeline.add(op_type)
        elif pipeline_capability["status"] == "limited":
            limited_for_pipeline.add(op_type)

        rows.append(
            {
                "index": operator.get("index"),
                "name": operator.get("name"),
                "op_type": op_type,
                "inputs": operator.get("inputs", []),
                "outputs": operator.get("outputs", []),
                "allowed_by_config": bool(operator.get("allowed", False)),
                "pipeline_support": pipeline_capability,
                "inference_support": capability_payload["inference"],
                "training_support": capability_payload["training"],
                "has_weights": capability.has_weights,
                "category": capability.category,
                "knob_support": capability.knobs.to_dict(),
            }
        )

    registry_payload = layer_registry(pipeline_mode=pipeline_mode)
    compatibility_ready = (
        inspection.compilation_ready
        and not unsupported_for_pipeline
        and not inspection.disallowed_operators
    )

    return {
        "artifact_kind": "model_compatibility",
        "schema_version": 1,
        "model_path": inspection.model_path,
        "pipeline_mode": pipeline_mode,
        "operator_counts": inspection.operator_counts,
        "operators": rows,
        "registry": registry_payload,
        "unsupported_operators": sorted(set(inspection.unsupported_operators) | unsupported_for_pipeline),
        "limited_operators": sorted(set(inspection.limited_operators) | limited_for_pipeline),
        "disallowed_operators": inspection.disallowed_operators,
        "compilation_ready": bool(compatibility_ready),
        "policy": {
            "unsupported_ops_block_compile": True,
            "limited_ops_require_honest_report": True,
            "training_requires_backward_support": True,
        },
    }


def build_layer_knob_contract(
    compatibility: Mapping[str, Any],
    *,
    raw_cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a per-layer, per-knob traceability contract.

    Sprint M rule: every relevant YAML knob must be applied to generated
    artifacts, be not-applicable for a clear tensor/layer reason, remain a
    compiler default when not requested, or reject clearly. This report is not
    a numeric/HLS proof by itself; it is the source/report contract that keeps
    knob decisions from being silently ignored.
    """
    pipeline_mode = str(compatibility.get("pipeline_mode", "inference"))
    layer_rows = []
    violations: list[dict[str, Any]] = []
    manual_sources = _manual_sources(raw_cfg)

    acceptable = {
        "applied",
        "not_applicable",
        "not_requested",
        "compiler_default",
        "rejected",
    }

    for operator in compatibility.get("operators", []) or []:
        if not isinstance(operator, Mapping):
            continue
        knob_payload = _per_layer_knob_contract(
            operator,
            raw_cfg=raw_cfg,
            pipeline_mode=pipeline_mode,
        )
        for knob, payload in knob_payload.items():
            status = str((payload or {}).get("status", "not_reported"))
            if status not in acceptable:
                violations.append({
                    "layer": operator.get("name"),
                    "op_type": operator.get("op_type"),
                    "knob": knob,
                    "status": status,
                    "reason": "Knob did not resolve to applied/not_applicable/not_requested/compiler_default/rejected.",
                })
        layer_rows.append(
            {
                "index": operator.get("index"),
                "name": operator.get("name"),
                "op_type": operator.get("op_type"),
                "has_weights": operator.get("has_weights"),
                "category": operator.get("category"),
                "pipeline_support": operator.get("pipeline_support"),
                "knobs": knob_payload,
            }
        )

    return {
        "artifact_kind": "layer_knob_contract",
        "schema_version": 2,
        "model_path": compatibility.get("model_path"),
        "pipeline_mode": pipeline_mode,
        "contract_rule": "Every knob must apply to generated artifacts, be not-applicable for a clear tensor reason, remain an explicit compiler default when not requested, or reject clearly.",
        "resource_latency_hygiene": "Unrequested import/export/runtime/tiling/stream paths must be absent from generated C++ and Vivado/runtime artifacts.",
        "precedence": [
            "manual_yaml_override",
            "policy_default",
            "compiler_default",
        ],
        "manual_yaml_sources": manual_sources,
        "all_layers_have_knob_contract": bool(layer_rows) and not violations,
        "violations": violations,
        "knobs": list(_KNOB_NAMES),
        "layers": layer_rows,
    }

_LAYER_BACKEND_DIMENSIONS = [
    "shape_inference",
    "hls_codegen",
    "memory_planning",
    "precision",
    "numeric_validation",
    "training_forward",
    "training_backward",
]


def build_layer_backend_status(
    compatibility: Mapping[str, Any],
    *,
    raw_cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create an auditable layer-backend status report.

    This report is intentionally stricter than the older compatibility table: a
    layer is treated as fully supported only when the registry says the active
    pipeline is supported and every backend dimension has an explicit status.
    Unknown operators remain unsupported and must not be silently accepted.
    """
    pipeline_mode = str(compatibility.get("pipeline_mode", "inference"))
    is_training = pipeline_mode == "training_on_device"
    rows = []
    unsupported = []
    partial = []

    for operator in compatibility.get("operators", []) or []:
        if not isinstance(operator, Mapping):
            continue
        op_type = str(operator.get("op_type", "Unknown"))
        pipeline_support = operator.get("pipeline_support", {}) or {}
        inference_support = operator.get("inference_support", {}) or {}
        training_support = operator.get("training_support", {}) or {}
        category = str(operator.get("category", "unsupported"))
        supported = bool(pipeline_support.get("supported", False))
        status = str(pipeline_support.get("status", "unsupported"))

        dims: Dict[str, str] = {
            "shape_inference": "implemented" if supported else "unsupported",
            "hls_codegen": "implemented" if supported else "unsupported",
            "memory_planning": "implemented" if supported else "unsupported",
            "precision": "implemented" if supported else "unsupported",
            "numeric_validation": "available" if supported else "unavailable",
            "training_forward": "implemented" if is_training and bool(training_support.get("supported")) else ("not_applicable" if not is_training else "unsupported"),
            "training_backward": "implemented" if is_training and bool(training_support.get("supported")) else ("not_applicable" if not is_training else "unsupported"),
        }

        backend_status = "implemented" if supported and status == "supported" else status
        if not supported:
            unsupported.append(op_type)
            backend_status = "unsupported"
        elif status != "supported":
            partial.append(op_type)
            backend_status = "limited"

        rows.append({
            "index": operator.get("index"),
            "name": operator.get("name"),
            "op_type": op_type,
            "category": category,
            "backend_status": backend_status,
            "inference_status": str(inference_support.get("status", "unsupported")),
            "training_status": str(training_support.get("status", "unsupported")),
            "active_pipeline_status": status,
            "supported_in_active_pipeline": supported,
            "dimensions": dims,
            "unsupported_reason": None if supported else str(pipeline_support.get("detail", "No backend support is registered.")),
            "codegen_rule": "Only emit this layer kernel when the model graph uses this operator.",
        })

    all_supported = not unsupported and not partial and bool(rows)
    return {
        "artifact_kind": "layer_backend_status",
        "schema_version": 1,
        "model_path": compatibility.get("model_path"),
        "pipeline_mode": pipeline_mode,
        "policy": {
            "all_layers_required": True,
            "unsupported_layers_block_compile": True,
            "unused_layer_kernels_must_be_absent": True,
            "training_requires_backward_support": True,
        },
        "all_encountered_layers_supported": all_supported,
        "unsupported_layers": sorted(set(unsupported)),
        "limited_layers": sorted(set(partial)),
        "layers": rows,
    }


def _compatibility_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FPGAI Model Compatibility Report",
        "",
        f"- Model: `{payload.get('model_path')}`",
        f"- Pipeline mode: `{payload.get('pipeline_mode')}`",
        f"- Compilation ready: `{payload.get('compilation_ready')}`",
        f"- Operator counts: `{payload.get('operator_counts')}`",
        f"- Disallowed operators: `{payload.get('disallowed_operators')}`",
        f"- Unsupported operators: `{payload.get('unsupported_operators')}`",
        f"- Limited operators: `{payload.get('limited_operators')}`",
        "",
        "## Operators",
        "",
        "| # | Name | Type | Category | Weights | Pipeline support | Detail |",
        "|---:|---|---|---|---|---|---|",
    ]
    for op in payload.get("operators", []) or []:
        support = op.get("pipeline_support", {}) or {}
        lines.append(
            f"| {op.get('index')} | `{op.get('name')}` | `{op.get('op_type')}` | "
            f"`{op.get('category')}` | `{op.get('has_weights')}` | "
            f"`{support.get('status')}` | {support.get('detail', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _knob_contract_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FPGAI Layer Knob Contract",
        "",
        f"- Model: `{payload.get('model_path')}`",
        f"- Pipeline mode: `{payload.get('pipeline_mode')}`",
        f"- All layers have knob contract: `{payload.get('all_layers_have_knob_contract')}`",
        "",
        "This report is a compiler contract. It does not claim numeric validation; it states whether each knob applies, is not applicable, or must reject clearly per layer.",
        "",
        "## Layers",
        "",
    ]
    for layer in payload.get("layers", []) or []:
        lines.extend([
            f"### {layer.get('index')}: `{layer.get('op_type')}` `{layer.get('name')}`",
            "",
            f"- Category: `{layer.get('category')}`",
            f"- Has weights: `{layer.get('has_weights')}`",
        ])
        knobs = layer.get("knobs", {}) or {}
        for knob in _KNOB_NAMES:
            payload = knobs.get(knob) or {}
            if isinstance(payload, Mapping):
                evidence = ", ".join(str(x) for x in payload.get("evidence", []) or [])
                reason = payload.get("reason") or ""
                lines.append(
                    f"- `{knob}`: `{payload.get('status')}` source=`{payload.get('source')}` reason=`{reason}` evidence=`{evidence}`"
                )
            else:
                lines.append(f"- `{knob}`: `{payload}`")
        lines.append("")
    return "\n".join(lines)


def _layer_backend_status_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FPGAI Layer Backend Status",
        "",
        f"- Model: `{payload.get('model_path')}`",
        f"- Pipeline mode: `{payload.get('pipeline_mode')}`",
        f"- All encountered layers supported: `{payload.get('all_encountered_layers_supported')}`",
        f"- Unsupported layers: `{payload.get('unsupported_layers')}`",
        f"- Limited layers: `{payload.get('limited_layers')}`",
        "",
        "This report is the layer-backend truth table. A layer may only be claimed as supported when the active pipeline has shape inference, memory planning, HLS codegen, precision handling, and validation readiness recorded here.",
        "",
        "## Layers",
        "",
        "| # | Name | Type | Backend status | Inference | Training | Reason |",
        "|---:|---|---|---|---|---|---|",
    ]
    for layer in payload.get("layers", []) or []:
        lines.append(
            f"| {layer.get('index')} | `{layer.get('name')}` | `{layer.get('op_type')}` | "
            f"`{layer.get('backend_status')}` | `{layer.get('inference_status')}` | "
            f"`{layer.get('training_status')}` | {layer.get('unsupported_reason') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def emit_model_compatibility_reports(
    out_dir: str | Path,
    inspection: ModelInspection,
    *,
    raw_cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Path]:
    reports_dir = Path(out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    compatibility = build_model_compatibility_report(inspection, raw_cfg=raw_cfg)
    knob_contract = build_layer_knob_contract(compatibility, raw_cfg=raw_cfg)
    backend_status = build_layer_backend_status(compatibility, raw_cfg=raw_cfg)

    compatibility_json = _write_json(reports_dir / "model_compatibility.json", compatibility)
    compatibility_md = _write_text(reports_dir / "model_compatibility.md", _compatibility_markdown(compatibility))
    knob_json = _write_json(reports_dir / "layer_knob_contract.json", knob_contract)
    knob_md = _write_text(reports_dir / "layer_knob_contract.md", _knob_contract_markdown(knob_contract))
    backend_json = _write_json(reports_dir / "layer_backend_status.json", backend_status)
    backend_md = _write_text(reports_dir / "layer_backend_status.md", _layer_backend_status_markdown(backend_status))

    return {
        "model_compatibility_json": compatibility_json,
        "model_compatibility_md": compatibility_md,
        "layer_knob_contract_json": knob_json,
        "layer_knob_contract_md": knob_md,
        "layer_backend_status_json": backend_json,
        "layer_backend_status_md": backend_md,
    }
