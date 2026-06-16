from __future__ import annotations

import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from fpgai.analysis.tiling_analysis import analyze_tiling
from fpgai.analysis.tiling_performance_model import estimate_tiling_performance
from fpgai.analysis.tiling_resource_model import estimate_tiling_resource_overhead


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


def _layers(compile_plan: Any) -> list[Any]:
    if isinstance(compile_plan, Mapping):
        return list(compile_plan.get("layer_plans", []))
    return list(getattr(compile_plan, "layer_plans", []) or [])


def _layer_name(layer: Any) -> str:
    layer_dict = _to_dict(layer)
    return str(layer_dict.get("node_name") or layer_dict.get("name") or "unknown")


def _layer_op_type(layer: Any) -> str:
    return str(_to_dict(layer).get("op_type", ""))


def _set_layer_tiling(layer: Any, sizes: Mapping[str, int]) -> None:
    """Set tiling sizes on dict-like or object-like layer plans."""
    if isinstance(layer, dict):
        architecture = layer.setdefault("architecture", {})
        if hasattr(architecture, "to_dict"):
            architecture = architecture.to_dict()
            layer["architecture"] = architecture
        if isinstance(architecture, dict):
            tiling = architecture.setdefault("tiling", {})
            if hasattr(tiling, "to_dict"):
                tiling = tiling.to_dict()
                architecture["tiling"] = tiling
            if isinstance(tiling, dict):
                tiling["sizes"] = dict(sizes)
                return
        layer["tile"] = dict(sizes)
        return

    architecture = getattr(layer, "architecture", None)
    if architecture is None:
        setattr(layer, "tile", dict(sizes))
        return

    tiling = getattr(architecture, "tiling", None)
    if tiling is None:
        try:
            setattr(architecture, "tiling", {"sizes": dict(sizes)})
        except Exception:
            setattr(layer, "tile", dict(sizes))
        return

    if isinstance(tiling, dict):
        tiling["sizes"] = dict(sizes)
        return

    if hasattr(tiling, "sizes"):
        setattr(tiling, "sizes", dict(sizes))
        return

    setattr(layer, "tile", dict(sizes))


def dense_tile_candidates(
    *,
    input_features: int | None = None,
    output_features: int | None = None,
    input_tiles: Iterable[int] = (8, 16, 32, 64),
    output_tiles: Iterable[int] = (4, 8, 16, 32),
) -> list[dict[str, int]]:
    candidates: list[dict[str, int]] = []

    for tile_in, tile_out in itertools.product(input_tiles, output_tiles):
        tile_in = int(tile_in)
        tile_out = int(tile_out)

        if tile_in <= 0 or tile_out <= 0:
            continue
        if input_features is not None and tile_in > int(input_features):
            continue
        if output_features is not None and tile_out > int(output_features):
            continue

        candidates.append(
            {
                "in": tile_in,
                "out": tile_out,
            }
        )

    return candidates


def conv_tile_candidates(
    *,
    input_channels: int | None = None,
    output_channels: int | None = None,
    output_height: int | None = None,
    output_width: int | None = None,
    input_channel_tiles: Iterable[int] = (1, 2, 4, 8),
    output_channel_tiles: Iterable[int] = (2, 4, 8, 16),
    output_height_tiles: Iterable[int] = (4, 7, 8, 14),
    output_width_tiles: Iterable[int] | None = None,
) -> list[dict[str, int]]:
    if output_width_tiles is None:
        output_width_tiles = output_height_tiles

    candidates: list[dict[str, int]] = []

    for tile_ic, tile_oc, tile_oh, tile_ow in itertools.product(
        input_channel_tiles,
        output_channel_tiles,
        output_height_tiles,
        output_width_tiles,
    ):
        tile_ic = int(tile_ic)
        tile_oc = int(tile_oc)
        tile_oh = int(tile_oh)
        tile_ow = int(tile_ow)

        if min(tile_ic, tile_oc, tile_oh, tile_ow) <= 0:
            continue
        if input_channels is not None and tile_ic > int(input_channels):
            continue
        if output_channels is not None and tile_oc > int(output_channels):
            continue
        if output_height is not None and tile_oh > int(output_height):
            continue
        if output_width is not None and tile_ow > int(output_width):
            continue

        candidates.append(
            {
                "input_channels": tile_ic,
                "output_channels": tile_oc,
                "output_height": tile_oh,
                "output_width": tile_ow,
            }
        )

    return candidates


def _candidate_metric(
    *,
    performance_report: Mapping[str, Any],
    resource_report: Mapping[str, Any],
    latency_weight: float,
    bram_weight: float,
) -> float:
    perf_totals = performance_report.get("totals", {})
    res_totals = resource_report.get("totals", {})

    latency = float(perf_totals.get("overlapped_total_cycles", 0) or 0)
    bram18 = float(res_totals.get("estimated_bram18", 0) or 0)
    return latency_weight * latency + bram_weight * bram18


def score_tile_candidate(
    compile_plan: Any,
    *,
    layer_name: str,
    tile_sizes: Mapping[str, int],
    graph: Any | None = None,
    precision_by_layer: Mapping[str, Any] | None = None,
    parallelism_by_layer: Mapping[str, Any] | None = None,
    ii_by_layer: Mapping[str, Any] | None = None,
    clock_mhz: float = 200.0,
    memory_words_per_cycle: float = 1.0,
    tile_overhead_cycles: int = 4,
    latency_weight: float = 1.0,
    bram_weight: float = 1000.0,
) -> dict[str, Any]:
    candidate_plan = deepcopy(compile_plan)

    matched = False
    for layer in _layers(candidate_plan):
        if _layer_name(layer) == layer_name:
            _set_layer_tiling(layer, tile_sizes)
            matched = True
            break

    if not matched:
        raise ValueError(f"Layer not found in compile_plan: {layer_name}")

    tiling_report = analyze_tiling(
        candidate_plan,
        graph=graph,
    )
    performance_report = estimate_tiling_performance(
        tiling_report,
        parallelism_by_layer=parallelism_by_layer,
        ii_by_layer=ii_by_layer,
        clock_mhz=clock_mhz,
        memory_words_per_cycle=memory_words_per_cycle,
        tile_overhead_cycles=tile_overhead_cycles,
    )
    resource_report = estimate_tiling_resource_overhead(
        tiling_report,
        precision_by_layer=precision_by_layer,
    )
    score = _candidate_metric(
        performance_report=performance_report,
        resource_report=resource_report,
        latency_weight=latency_weight,
        bram_weight=bram_weight,
    )

    return {
        "layer_name": layer_name,
        "tile": dict(tile_sizes),
        "score": round(float(score), 6),
        "performance": performance_report["totals"],
        "resources": resource_report["totals"],
    }


def sweep_layer_tiles(
    compile_plan: Any,
    *,
    layer_name: str,
    candidates: Iterable[Mapping[str, int]],
    graph: Any | None = None,
    precision_by_layer: Mapping[str, Any] | None = None,
    parallelism_by_layer: Mapping[str, Any] | None = None,
    ii_by_layer: Mapping[str, Any] | None = None,
    clock_mhz: float = 200.0,
    memory_words_per_cycle: float = 1.0,
    tile_overhead_cycles: int = 4,
    latency_weight: float = 1.0,
    bram_weight: float = 1000.0,
    top_k: int | None = None,
) -> dict[str, Any]:
    scored = [
        score_tile_candidate(
            compile_plan,
            layer_name=layer_name,
            tile_sizes=candidate,
            graph=graph,
            precision_by_layer=precision_by_layer,
            parallelism_by_layer=parallelism_by_layer,
            ii_by_layer=ii_by_layer,
            clock_mhz=clock_mhz,
            memory_words_per_cycle=memory_words_per_cycle,
            tile_overhead_cycles=tile_overhead_cycles,
            latency_weight=latency_weight,
            bram_weight=bram_weight,
        )
        for candidate in candidates
    ]

    scored.sort(
        key=lambda item: (
            item["score"],
            item["performance"].get("overlapped_total_cycles", 0),
            item["resources"].get("estimated_bram18", 0),
        )
    )

    if top_k is not None:
        scored = scored[: max(0, int(top_k))]

    return {
        "format": "fpgai.tiling_sweep.v1",
        "layer_name": layer_name,
        "candidate_count": len(scored),
        "ranking": scored,
        "best": scored[0] if scored else None,
        "objective": {
            "latency_weight": latency_weight,
            "bram_weight": bram_weight,
            "lower_is_better": True,
        },
    }


def sweep_compile_plan_tiles(
    compile_plan: Any,
    *,
    candidates_by_layer: Mapping[str, Iterable[Mapping[str, int]]],
    graph: Any | None = None,
    precision_by_layer: Mapping[str, Any] | None = None,
    parallelism_by_layer: Mapping[str, Any] | None = None,
    ii_by_layer: Mapping[str, Any] | None = None,
    clock_mhz: float = 200.0,
    memory_words_per_cycle: float = 1.0,
    tile_overhead_cycles: int = 4,
    latency_weight: float = 1.0,
    bram_weight: float = 1000.0,
    top_k: int | None = 5,
) -> dict[str, Any]:
    layer_reports = []
    for layer_name, candidates in candidates_by_layer.items():
        layer_reports.append(
            sweep_layer_tiles(
                compile_plan,
                layer_name=layer_name,
                candidates=candidates,
                graph=graph,
                precision_by_layer=precision_by_layer,
                parallelism_by_layer=parallelism_by_layer,
                ii_by_layer=ii_by_layer,
                clock_mhz=clock_mhz,
                memory_words_per_cycle=memory_words_per_cycle,
                tile_overhead_cycles=tile_overhead_cycles,
                latency_weight=latency_weight,
                bram_weight=bram_weight,
                top_k=top_k,
            )
        )

    best_by_layer = {
        report["layer_name"]: report["best"]
        for report in layer_reports
    }

    return {
        "format": "fpgai.tiling_sweep_collection.v1",
        "layer_count": len(layer_reports),
        "best_by_layer": best_by_layer,
        "layers": layer_reports,
    }


def write_tiling_sweep_json(
    path: str | Path,
    sweep_report: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            sweep_report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dict(sweep_report)
