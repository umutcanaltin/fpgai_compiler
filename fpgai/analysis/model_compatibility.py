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
    "data_movement",
    "training",
]


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
    layer_rows = []
    all_apply_or_reject = True

    for operator in compatibility.get("operators", []) or []:
        if not isinstance(operator, Mapping):
            continue
        knob_support = operator.get("knob_support", {}) or {}
        knob_rows = {}
        for knob in _KNOB_NAMES:
            status = str(knob_support.get(knob, "not_reported"))
            knob_rows[knob] = status
            if status == "not_reported":
                all_apply_or_reject = False
        layer_rows.append(
            {
                "index": operator.get("index"),
                "name": operator.get("name"),
                "op_type": operator.get("op_type"),
                "has_weights": operator.get("has_weights"),
                "category": operator.get("category"),
                "knobs": knob_rows,
            }
        )

    return {
        "artifact_kind": "layer_knob_contract",
        "schema_version": 1,
        "model_path": compatibility.get("model_path"),
        "pipeline_mode": compatibility.get("pipeline_mode"),
        "contract_rule": "Every knob must apply to generated artifacts, be not-applicable for a clear tensor reason, or reject clearly.",
        "all_layers_have_knob_contract": all_apply_or_reject,
        "knobs": list(_KNOB_NAMES),
        "layers": layer_rows,
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
            lines.append(f"- `{knob}`: `{knobs.get(knob)}`")
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

    compatibility_json = _write_json(reports_dir / "model_compatibility.json", compatibility)
    compatibility_md = _write_text(reports_dir / "model_compatibility.md", _compatibility_markdown(compatibility))
    knob_json = _write_json(reports_dir / "layer_knob_contract.json", knob_contract)
    knob_md = _write_text(reports_dir / "layer_knob_contract.md", _knob_contract_markdown(knob_contract))

    return {
        "model_compatibility_json": compatibility_json,
        "model_compatibility_md": compatibility_md,
        "layer_knob_contract_json": knob_json,
        "layer_knob_contract_md": knob_md,
    }
