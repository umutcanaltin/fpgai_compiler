from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from fpgai.analysis.hls_schedule_report import (
    compare_requested_achieved_ii,
    discover_hls_schedule_reports,
    parse_hls_schedule_report,
)


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, dict):
            return result

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _layer_name(layer: Any, index: int) -> str:
    layer_dict = _to_dict(layer)

    name = layer_dict.get("name")
    if name:
        return str(name)

    return f"layer_{index}"


def _layer_requested_ii(layer: Any) -> int | None:
    layer_dict = _to_dict(layer)

    architecture = layer_dict.get("architecture", {})
    if isinstance(architecture, Mapping):
        pipeline = architecture.get("pipeline", {})
        if isinstance(pipeline, Mapping) and pipeline.get("ii") is not None:
            return int(pipeline["ii"])

    pipeline = layer_dict.get("pipeline", {})
    if isinstance(pipeline, Mapping) and pipeline.get("ii") is not None:
        return int(pipeline["ii"])

    if layer_dict.get("pipeline_ii") is not None:
        return int(layer_dict["pipeline_ii"])

    if hasattr(layer, "pipeline_ii"):
        value = getattr(layer, "pipeline_ii")
        if value is not None:
            return int(value)

    return None


def requested_ii_by_layer(compile_plan: Any) -> dict[str, int]:
    plan_dict = _to_dict(compile_plan)

    layers = getattr(compile_plan, "layer_plans", None)
    if layers is None:
        layers = plan_dict.get("layer_plans", [])

    requested: dict[str, int] = {}

    for index, layer in enumerate(layers or []):
        requested_ii = _layer_requested_ii(layer)
        if requested_ii is None:
            continue

        requested[_layer_name(layer, index)] = int(requested_ii)

    return requested


def compare_requested_achieved_ii_for_directory(
    out_dir: Path | str,
    compile_plan: Any,
) -> dict[str, Any] | None:
    root = Path(out_dir)
    requested = requested_ii_by_layer(compile_plan)

    if not requested:
        return None

    report_paths = discover_hls_schedule_reports(root)
    if not report_paths:
        return None

    reports: list[dict[str, Any]] = []

    for report_path in report_paths:
        parsed = parse_hls_schedule_report(report_path)
        comparison = compare_requested_achieved_ii(
            requested,
            parsed,
        )

        comparison["source"] = str(comparison.get("source", report_path))

        reports.append(comparison)

    matched_layer_count = max(
        (int(item.get("matched_layer_count", 0)) for item in reports),
        default=0,
    )
    failed_layer_count = max(
        (int(item.get("failed_layer_count", 0)) for item in reports),
        default=0,
    )

    all_layers = [
        layer
        for report in reports
        for layer in report.get("layers", [])
        if isinstance(layer, dict)
    ]

    return {
        "schema_version": 1,
        "requested_by_layer": requested,
        "summary": {
            "report_count": len(reports),
            "layer_count": len(requested),
            "matched_layer_count": matched_layer_count,
            "failed_layer_count": failed_layer_count,
            "all_requested_layers_matched": matched_layer_count == len(requested),
            "all_matched_layers_met_ii": failed_layer_count == 0,
        },
        "reports": reports,
        "layers": all_layers,
    }


def write_requested_achieved_ii_summary(
    out_dir: Path | str,
    compile_plan: Any,
    output_path: Path | str | None = None,
) -> dict[str, Any] | None:
    root = Path(out_dir)
    result = compare_requested_achieved_ii_for_directory(
        root,
        compile_plan,
    )

    if result is None:
        return None

    target = Path(output_path) if output_path is not None else root / "hls_ii_comparison.json"
    target.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return {
        "path": str(target.relative_to(root)),
        "summary": result["summary"],
    }
