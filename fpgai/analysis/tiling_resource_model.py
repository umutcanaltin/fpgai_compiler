from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from fpgai.analysis.tiling_analysis import analyze_tiling


DEFAULT_ACTIVATION_BITS = 16
DEFAULT_WEIGHT_BITS = 16
DEFAULT_ACCUMULATOR_BITS = 32
BRAM18_BITS = 18_432
BRAM36_BITS = 36_864


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return int(math.ceil(a / b))


def _precision_for_layer(
    layer_name: str,
    precision_by_layer: Mapping[str, Any] | None,
) -> dict[str, int]:
    precision: dict[str, int] = {
        "activation_bits": DEFAULT_ACTIVATION_BITS,
        "weight_bits": DEFAULT_WEIGHT_BITS,
        "accumulator_bits": DEFAULT_ACCUMULATOR_BITS,
    }

    if not precision_by_layer:
        return precision

    raw = precision_by_layer.get(layer_name, {})
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()

    if not isinstance(raw, Mapping):
        return precision

    precision["activation_bits"] = _safe_int(
        raw.get("activation_bits", raw.get("act_bits", raw.get("bits", DEFAULT_ACTIVATION_BITS))),
        DEFAULT_ACTIVATION_BITS,
    )
    precision["weight_bits"] = _safe_int(
        raw.get("weight_bits", raw.get("w_bits", raw.get("bits", DEFAULT_WEIGHT_BITS))),
        DEFAULT_WEIGHT_BITS,
    )
    precision["accumulator_bits"] = _safe_int(
        raw.get("accumulator_bits", raw.get("acc_bits", DEFAULT_ACCUMULATOR_BITS)),
        DEFAULT_ACCUMULATOR_BITS,
    )
    return precision


def estimate_layer_tile_buffer_bits(
    layer_report: Mapping[str, Any],
    *,
    precision_by_layer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate tile-buffer storage from a tiling analysis layer report."""
    layer_name = str(layer_report.get("layer_name", "unknown"))
    op_type = str(layer_report.get("op_type", "unknown"))
    local_buffers = layer_report.get("local_buffers", {})

    if not isinstance(local_buffers, Mapping):
        local_buffers = {}

    precision = _precision_for_layer(
        layer_name,
        precision_by_layer,
    )

    input_elements = _safe_int(
        local_buffers.get("input_tile_elements"),
    )
    weight_elements = _safe_int(
        local_buffers.get("weight_tile_elements"),
    )
    accumulator_elements = _safe_int(
        local_buffers.get("accumulator_tile_elements"),
    )

    activation_bits = input_elements * precision["activation_bits"]
    weight_bits = weight_elements * precision["weight_bits"]
    accumulator_bits = accumulator_elements * precision["accumulator_bits"]
    total_bits = activation_bits + weight_bits + accumulator_bits

    return {
        "layer_name": layer_name,
        "op_type": op_type,
        "precision": precision,
        "tile_buffer_elements": {
            "input": input_elements,
            "weight": weight_elements,
            "accumulator": accumulator_elements,
            "total": input_elements + weight_elements + accumulator_elements,
        },
        "tile_buffer_bits": {
            "activation": activation_bits,
            "weight": weight_bits,
            "accumulator": accumulator_bits,
            "total": total_bits,
        },
        "estimated_bram18": _ceil_div(total_bits, BRAM18_BITS),
        "estimated_bram36": _ceil_div(total_bits, BRAM36_BITS),
    }


def estimate_tiling_resource_overhead(
    tiling_report: Mapping[str, Any],
    *,
    precision_by_layer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate additional local storage required by tiled kernels."""
    layers = [
        estimate_layer_tile_buffer_bits(
            layer,
            precision_by_layer=precision_by_layer,
        )
        for layer in tiling_report.get("layers", [])
        if isinstance(layer, Mapping)
        and layer.get("op_type") in {"Dense", "Conv"}
        and isinstance(layer.get("local_buffers"), Mapping)
    ]

    total_bits = sum(
        _safe_int(layer["tile_buffer_bits"]["total"])
        for layer in layers
    )
    total_elements = sum(
        _safe_int(layer["tile_buffer_elements"]["total"])
        for layer in layers
    )

    return {
        "format": "fpgai.tiling_resource_model.v1",
        "assumptions": {
            "bram18_bits": BRAM18_BITS,
            "bram36_bits": BRAM36_BITS,
            "default_activation_bits": DEFAULT_ACTIVATION_BITS,
            "default_weight_bits": DEFAULT_WEIGHT_BITS,
            "default_accumulator_bits": DEFAULT_ACCUMULATOR_BITS,
            "note": (
                "This estimates local tile buffer storage only. It does not "
                "replace full HLS resource reports."
            ),
        },
        "totals": {
            "tiled_layer_count": len(layers),
            "tile_buffer_elements": total_elements,
            "tile_buffer_bits": total_bits,
            "estimated_bram18": _ceil_div(total_bits, BRAM18_BITS),
            "estimated_bram36": _ceil_div(total_bits, BRAM36_BITS),
        },
        "layers": layers,
    }


def analyze_tiling_resource_overhead(
    compile_plan: Any,
    *,
    graph: Any | None = None,
    precision_by_layer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tiling_report = analyze_tiling(
        compile_plan,
        graph=graph,
    )
    return estimate_tiling_resource_overhead(
        tiling_report,
        precision_by_layer=precision_by_layer,
    )


def write_tiling_resource_estimate_json(
    path: str | Path,
    tiling_report_or_compile_plan: Any,
    *,
    graph: Any | None = None,
    precision_by_layer: Mapping[str, Any] | None = None,
    input_is_tiling_report: bool = False,
) -> dict[str, Any]:
    if input_is_tiling_report:
        report = estimate_tiling_resource_overhead(
            tiling_report_or_compile_plan,
            precision_by_layer=precision_by_layer,
        )
    else:
        report = analyze_tiling_resource_overhead(
            tiling_report_or_compile_plan,
            graph=graph,
            precision_by_layer=precision_by_layer,
        )

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def attach_tiling_resource_estimate_to_manifest(
    manifest: MutableMapping[str, Any],
    resource_report: Mapping[str, Any],
    *,
    path: str = "tiling_resource_estimate.json",
) -> MutableMapping[str, Any]:
    totals = resource_report.get("totals", {})
    if not isinstance(totals, Mapping):
        totals = {}

    manifest["tiling_resource_estimate"] = {
        "format": resource_report.get(
            "format",
            "fpgai.tiling_resource_model.v1",
        ),
        "path": path,
        "tiled_layer_count": _safe_int(
            totals.get("tiled_layer_count"),
        ),
        "tile_buffer_elements": _safe_int(
            totals.get("tile_buffer_elements"),
        ),
        "tile_buffer_bits": _safe_int(
            totals.get("tile_buffer_bits"),
        ),
        "estimated_bram18": _safe_int(
            totals.get("estimated_bram18"),
        ),
        "estimated_bram36": _safe_int(
            totals.get("estimated_bram36"),
        ),
    }
    return manifest
