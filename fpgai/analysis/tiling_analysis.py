from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, dict):
            return result

    if isinstance(value, Mapping):
        return dict(value)

    result: dict[str, Any] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if callable(item):
            continue
        result[key] = item
    return result


def _nested_dict(value: Any, *keys: str) -> dict[str, Any]:
    current = _to_dict(value)
    for key in keys:
        current = _to_dict(current.get(key))
    return current


def _layers(compile_plan: Any) -> list[Any]:
    if compile_plan is None:
        return []

    if isinstance(compile_plan, Mapping):
        return list(compile_plan.get("layer_plans", []))

    return list(getattr(compile_plan, "layer_plans", []) or [])


def _ceil_div(a: int | None, b: int | None) -> int | None:
    if a is None or b is None or b <= 0:
        return None
    return int(math.ceil(a / b))


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except Exception:
        return None
    return ivalue if ivalue > 0 else None


def _first_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _positive_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _shape_value(layer: Mapping[str, Any], *keys: str) -> int | None:
    for container_name in (
        "shape",
        "shapes",
        "dims",
        "dimensions",
        "metadata",
        "attributes",
        "params",
    ):
        container = _to_dict(layer.get(container_name))
        value = _first_int(container, *keys)
        if value is not None:
            return value

    return _first_int(layer, *keys)


def _tiling_sizes(layer: Mapping[str, Any]) -> dict[str, Any]:
    architecture = _to_dict(layer.get("architecture"))
    tiling = _to_dict(architecture.get("tiling"))

    sizes = _to_dict(tiling.get("sizes"))
    if sizes:
        return sizes

    if tiling:
        return tiling

    return _to_dict(layer.get("tile"))


def _dense_dimensions(layer: Mapping[str, Any]) -> tuple[int | None, int | None]:
    input_features = _shape_value(
        layer,
        "in",
        "input",
        "input_features",
        "input_size",
        "in_features",
    )
    output_features = _shape_value(
        layer,
        "out",
        "output",
        "output_features",
        "output_size",
        "out_features",
    )
    return input_features, output_features


def _conv_dimensions(
    layer: Mapping[str, Any],
) -> dict[str, int | None]:
    kernel = _shape_value(
        layer,
        "kernel",
        "kernel_size",
        "k",
        "kh",
        "kw",
    )
    return {
        "input_height": _shape_value(layer, "input_height", "in_h", "ih"),
        "input_width": _shape_value(layer, "input_width", "in_w", "iw"),
        "input_channels": _shape_value(layer, "input_channels", "in_c", "ic"),
        "output_height": _shape_value(layer, "output_height", "out_h", "oh"),
        "output_width": _shape_value(layer, "output_width", "out_w", "ow"),
        "output_channels": _shape_value(layer, "output_channels", "out_c", "oc"),
        "kernel": kernel,
        "stride": _shape_value(layer, "stride") or 1,
        "padding": _shape_value(layer, "padding", "pad") or 0,
    }


def _dense_tile(sizes: Mapping[str, Any]) -> tuple[int, int]:
    tile_in = _first_int(
        sizes,
        "in",
        "input",
        "input_features",
        "input_tile",
    ) or 1
    tile_out = _first_int(
        sizes,
        "out",
        "output",
        "output_features",
        "output_tile",
    ) or 1
    return tile_in, tile_out


def _conv_tile(sizes: Mapping[str, Any]) -> tuple[int, int, int, int]:
    tile_oc = _first_int(
        sizes,
        "oc",
        "out_channels",
        "output_channels",
        "channel",
        "channels",
    ) or 1
    tile_oh = _first_int(
        sizes,
        "oh",
        "output_height",
        "spatial_height",
        "height",
    ) or 1
    tile_ow = _first_int(
        sizes,
        "ow",
        "output_width",
        "spatial_width",
        "width",
    ) or tile_oh
    tile_ic = _first_int(
        sizes,
        "ic",
        "in_channels",
        "input_channels",
        "input_channel",
    ) or 1
    return tile_oc, tile_oh, tile_ow, tile_ic


def _maybe_mul(*values: int | None) -> int | None:
    result = 1
    for value in values:
        if value is None:
            return None
        result *= value
    return result


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def analyze_dense_tiling(
    layer_name: str,
    layer: Mapping[str, Any],
    sizes: Mapping[str, Any],
) -> dict[str, Any]:
    input_features, output_features = _dense_dimensions(layer)
    tile_in, tile_out = _dense_tile(sizes)

    input_tile_count = _ceil_div(input_features, tile_in)
    output_tile_count = _ceil_div(output_features, tile_out)
    total_tile_count = _maybe_mul(input_tile_count, output_tile_count)

    dense_macs = _maybe_mul(input_features, output_features)
    activation_reads = (
        input_features * output_tile_count
        if input_features is not None and output_tile_count is not None
        else None
    )
    weight_reads = dense_macs
    output_writes = output_features

    return {
        "layer_name": layer_name,
        "op_type": "Dense",
        "tile": {
            "input_features": tile_in,
            "output_features": tile_out,
        },
        "dimensions": {
            "input_features": input_features,
            "output_features": output_features,
        },
        "tile_counts": {
            "input_tiles": input_tile_count,
            "output_tiles": output_tile_count,
            "total_tiles": total_tile_count,
        },
        "local_buffers": {
            "input_tile_elements": tile_in,
            "weight_tile_elements": tile_in * tile_out,
            "accumulator_tile_elements": tile_out,
            "total_tile_elements": tile_in + tile_in * tile_out + tile_out,
        },
        "estimated_traffic": {
            "activation_reads": activation_reads,
            "weight_reads": weight_reads,
            "output_writes": output_writes,
            "macs": dense_macs,
        },
        "reuse": {
            "input_reuse_within_tile": tile_out,
            "weight_reuse_within_tile": 1,
            "accumulator_reuse_within_tile": tile_in,
            "activation_read_reduction_vs_untiled_output_loop": _round(
                output_features / output_tile_count
                if output_features and output_tile_count
                else None
            ),
        },
    }


def analyze_conv_tiling(
    layer_name: str,
    layer: Mapping[str, Any],
    sizes: Mapping[str, Any],
) -> dict[str, Any]:
    dims = _conv_dimensions(layer)
    tile_oc, tile_oh, tile_ow, tile_ic = _conv_tile(sizes)

    oc_tiles = _ceil_div(dims["output_channels"], tile_oc)
    oh_tiles = _ceil_div(dims["output_height"], tile_oh)
    ow_tiles = _ceil_div(dims["output_width"], tile_ow)
    ic_tiles = _ceil_div(dims["input_channels"], tile_ic)
    spatial_tiles = _maybe_mul(oh_tiles, ow_tiles)
    total_tiles = _maybe_mul(oc_tiles, oh_tiles, ow_tiles, ic_tiles)

    k = dims["kernel"]
    stride = dims["stride"] or 1
    input_tile_h = tile_oh * stride + (k or 1)
    input_tile_w = tile_ow * stride + (k or 1)

    activation_reads_per_ic_tile = tile_ic * input_tile_h * input_tile_w
    weight_reads_per_tile = tile_oc * tile_ic * (k or 1) * (k or 1)
    acc_tile_elements = tile_oc * tile_oh * tile_ow

    output_elements = _maybe_mul(
        dims["output_channels"],
        dims["output_height"],
        dims["output_width"],
    )
    macs = _maybe_mul(
        dims["output_channels"],
        dims["output_height"],
        dims["output_width"],
        dims["input_channels"],
        k,
        k,
    )
    activation_reads = (
        activation_reads_per_ic_tile * (total_tiles or 0)
        if total_tiles is not None
        else None
    )
    weight_reads = (
        weight_reads_per_tile * (total_tiles or 0)
        if total_tiles is not None
        else None
    )

    return {
        "layer_name": layer_name,
        "op_type": "Conv",
        "tile": {
            "output_channels": tile_oc,
            "output_height": tile_oh,
            "output_width": tile_ow,
            "input_channels": tile_ic,
        },
        "dimensions": dims,
        "tile_counts": {
            "output_channel_tiles": oc_tiles,
            "output_height_tiles": oh_tiles,
            "output_width_tiles": ow_tiles,
            "spatial_tiles": spatial_tiles,
            "input_channel_tiles": ic_tiles,
            "total_tiles": total_tiles,
        },
        "local_buffers": {
            "input_tile_elements": activation_reads_per_ic_tile,
            "weight_tile_elements": weight_reads_per_tile,
            "accumulator_tile_elements": acc_tile_elements,
            "total_tile_elements": (
                activation_reads_per_ic_tile
                + weight_reads_per_tile
                + acc_tile_elements
            ),
        },
        "estimated_traffic": {
            "activation_reads": activation_reads,
            "weight_reads": weight_reads,
            "output_writes": output_elements,
            "macs": macs,
        },
        "reuse": {
            "input_reuse_within_tile": tile_oc * (k or 1) * (k or 1),
            "weight_reuse_within_tile": tile_oh * tile_ow,
            "accumulator_reuse_within_tile": tile_ic * (k or 1) * (k or 1),
            "weight_read_reduction_vs_per_output_reload": _round(
                (tile_oh * tile_ow)
            ),
        },
    }


def analyze_layer_tiling(layer: Any) -> dict[str, Any] | None:
    layer_dict = _to_dict(layer)
    op_type = str(layer_dict.get("op_type", ""))
    layer_name = str(
        layer_dict.get("node_name")
        or layer_dict.get("name")
        or "unknown"
    )
    sizes = _tiling_sizes(layer_dict)

    if not sizes:
        return None

    if op_type == "Dense":
        return analyze_dense_tiling(layer_name, layer_dict, sizes)

    if op_type == "Conv":
        return analyze_conv_tiling(layer_name, layer_dict, sizes)

    return {
        "layer_name": layer_name,
        "op_type": op_type,
        "tile": dict(sizes),
        "status": "planning_only",
        "detail": "Tiling analysis is implemented for Dense and Conv layers.",
    }


def analyze_tiling(
    compile_plan: Any,
    *,
    graph: Any | None = None,
) -> dict[str, Any]:
    del graph

    layer_reports = [
        report
        for layer in _layers(compile_plan)
        for report in [analyze_layer_tiling(layer)]
        if report is not None
    ]

    implemented_reports = [
        report
        for report in layer_reports
        if report.get("op_type") in {"Dense", "Conv"}
    ]

    totals = {
        "tiled_layer_count": len(layer_reports),
        "implemented_tiled_layer_count": len(implemented_reports),
        "planning_only_tiled_layer_count": (
            len(layer_reports) - len(implemented_reports)
        ),
        "local_buffer_elements": sum(
            int(report.get("local_buffers", {}).get("total_tile_elements", 0))
            for report in implemented_reports
        ),
        "estimated_activation_reads": sum(
            int(report.get("estimated_traffic", {}).get("activation_reads") or 0)
            for report in implemented_reports
        ),
        "estimated_weight_reads": sum(
            int(report.get("estimated_traffic", {}).get("weight_reads") or 0)
            for report in implemented_reports
        ),
        "estimated_output_writes": sum(
            int(report.get("estimated_traffic", {}).get("output_writes") or 0)
            for report in implemented_reports
        ),
        "estimated_macs": sum(
            int(report.get("estimated_traffic", {}).get("macs") or 0)
            for report in implemented_reports
        ),
    }

    return {
        "format": "fpgai.tiling_analysis.v1",
        "totals": totals,
        "layers": layer_reports,
    }


def write_tiling_analysis_json(
    path: str | Path,
    compile_plan: Any,
    *,
    graph: Any | None = None,
) -> dict[str, Any]:
    report = analyze_tiling(
        compile_plan,
        graph=graph,
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
