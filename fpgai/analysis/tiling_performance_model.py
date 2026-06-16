from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from fpgai.analysis.tiling_analysis import analyze_tiling


DEFAULT_CLOCK_MHZ = 200.0
DEFAULT_MEMORY_WORDS_PER_CYCLE = 1.0
DEFAULT_TILE_OVERHEAD_CYCLES = 4


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _ceil_div_float(a: float, b: float) -> int:
    if b <= 0:
        return 0
    return int(math.ceil(a / b))


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _layer_parallelism(
    layer_name: str,
    parallelism_by_layer: Mapping[str, Any] | None,
) -> dict[str, int]:
    result = {
        "pe": 1,
        "simd": 1,
        "parallel_macs": 1,
    }

    if not parallelism_by_layer:
        return result

    raw = parallelism_by_layer.get(layer_name, {})
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()

    if not isinstance(raw, Mapping):
        return result

    pe = _safe_int(raw.get("pe", raw.get("out", 1)), 1)
    simd = _safe_int(raw.get("simd", raw.get("in", 1)), 1)
    parallel_macs = _safe_int(
        raw.get("parallel_macs", pe * simd),
        pe * simd,
    )

    result["pe"] = max(1, pe)
    result["simd"] = max(1, simd)
    result["parallel_macs"] = max(1, parallel_macs)
    return result


def _layer_pipeline_ii(
    layer_name: str,
    ii_by_layer: Mapping[str, Any] | None,
) -> int:
    if not ii_by_layer:
        return 1

    raw = ii_by_layer.get(layer_name, 1)
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()

    if isinstance(raw, Mapping):
        raw = raw.get("ii", 1)

    return max(1, _safe_int(raw, 1))


def estimate_layer_tiling_performance(
    layer_report: Mapping[str, Any],
    *,
    parallelism_by_layer: Mapping[str, Any] | None = None,
    ii_by_layer: Mapping[str, Any] | None = None,
    clock_mhz: float = DEFAULT_CLOCK_MHZ,
    memory_words_per_cycle: float = DEFAULT_MEMORY_WORDS_PER_CYCLE,
    tile_overhead_cycles: int = DEFAULT_TILE_OVERHEAD_CYCLES,
) -> dict[str, Any]:
    """Estimate cycle-level effects of a tiled layer.

    This is a deterministic analytical estimate. It is intended for design-space
    comparisons and manifest summaries; real achieved II still comes from HLS.
    """
    layer_name = str(layer_report.get("layer_name", "unknown"))
    op_type = str(layer_report.get("op_type", "unknown"))

    traffic = layer_report.get("estimated_traffic", {})
    if not isinstance(traffic, Mapping):
        traffic = {}

    tile_counts = layer_report.get("tile_counts", {})
    if not isinstance(tile_counts, Mapping):
        tile_counts = {}

    macs = _safe_int(traffic.get("macs"))
    activation_reads = _safe_int(traffic.get("activation_reads"))
    weight_reads = _safe_int(traffic.get("weight_reads"))
    output_writes = _safe_int(traffic.get("output_writes"))
    memory_words = activation_reads + weight_reads + output_writes

    total_tiles = _safe_int(tile_counts.get("total_tiles"), 1)
    total_tiles = max(1, total_tiles)

    parallelism = _layer_parallelism(
        layer_name,
        parallelism_by_layer,
    )
    ii = _layer_pipeline_ii(
        layer_name,
        ii_by_layer,
    )

    compute_cycles = _ceil_div_float(
        macs * ii,
        parallelism["parallel_macs"],
    )
    memory_cycles = _ceil_div_float(
        memory_words,
        max(1e-9, float(memory_words_per_cycle)),
    )
    overhead_cycles = total_tiles * max(0, int(tile_overhead_cycles))

    overlapped_cycles = max(
        compute_cycles,
        memory_cycles,
    ) + overhead_cycles
    non_overlapped_cycles = (
        compute_cycles
        + memory_cycles
        + overhead_cycles
    )

    clock_mhz = max(1e-9, float(clock_mhz))
    estimated_latency_us = overlapped_cycles / clock_mhz

    if compute_cycles > memory_cycles:
        bottleneck = "compute"
    elif memory_cycles > compute_cycles:
        bottleneck = "memory"
    else:
        bottleneck = "balanced"

    return {
        "layer_name": layer_name,
        "op_type": op_type,
        "assumptions": {
            "clock_mhz": _round(clock_mhz),
            "memory_words_per_cycle": _round(memory_words_per_cycle),
            "tile_overhead_cycles": max(0, int(tile_overhead_cycles)),
            "pipeline_ii": ii,
            "parallelism": parallelism,
        },
        "cycle_estimate": {
            "compute_cycles": compute_cycles,
            "memory_cycles": memory_cycles,
            "tile_overhead_cycles": overhead_cycles,
            "overlapped_total_cycles": overlapped_cycles,
            "non_overlapped_total_cycles": non_overlapped_cycles,
        },
        "latency_estimate": {
            "overlapped_latency_us": _round(estimated_latency_us),
        },
        "traffic": {
            "activation_reads": activation_reads,
            "weight_reads": weight_reads,
            "output_writes": output_writes,
            "total_words": memory_words,
            "macs": macs,
        },
        "bottleneck": bottleneck,
    }


def estimate_tiling_performance(
    tiling_report: Mapping[str, Any],
    *,
    parallelism_by_layer: Mapping[str, Any] | None = None,
    ii_by_layer: Mapping[str, Any] | None = None,
    clock_mhz: float = DEFAULT_CLOCK_MHZ,
    memory_words_per_cycle: float = DEFAULT_MEMORY_WORDS_PER_CYCLE,
    tile_overhead_cycles: int = DEFAULT_TILE_OVERHEAD_CYCLES,
) -> dict[str, Any]:
    layers = [
        estimate_layer_tiling_performance(
            layer,
            parallelism_by_layer=parallelism_by_layer,
            ii_by_layer=ii_by_layer,
            clock_mhz=clock_mhz,
            memory_words_per_cycle=memory_words_per_cycle,
            tile_overhead_cycles=tile_overhead_cycles,
        )
        for layer in tiling_report.get("layers", [])
        if isinstance(layer, Mapping)
        and layer.get("op_type") in {"Dense", "Conv"}
        and isinstance(layer.get("estimated_traffic"), Mapping)
    ]

    total_overlapped = sum(
        _safe_int(layer["cycle_estimate"]["overlapped_total_cycles"])
        for layer in layers
    )
    total_non_overlapped = sum(
        _safe_int(layer["cycle_estimate"]["non_overlapped_total_cycles"])
        for layer in layers
    )
    total_compute = sum(
        _safe_int(layer["cycle_estimate"]["compute_cycles"])
        for layer in layers
    )
    total_memory = sum(
        _safe_int(layer["cycle_estimate"]["memory_cycles"])
        for layer in layers
    )
    total_macs = sum(
        _safe_int(layer["traffic"]["macs"])
        for layer in layers
    )
    total_words = sum(
        _safe_int(layer["traffic"]["total_words"])
        for layer in layers
    )

    clock_mhz = max(1e-9, float(clock_mhz))

    return {
        "format": "fpgai.tiling_performance_model.v1",
        "assumptions": {
            "clock_mhz": _round(clock_mhz),
            "memory_words_per_cycle": _round(memory_words_per_cycle),
            "tile_overhead_cycles": max(0, int(tile_overhead_cycles)),
            "note": (
                "This is an analytical estimate for comparing tile choices. "
                "Final achieved II and latency should be taken from HLS reports."
            ),
        },
        "totals": {
            "tiled_layer_count": len(layers),
            "compute_cycles": total_compute,
            "memory_cycles": total_memory,
            "overlapped_total_cycles": total_overlapped,
            "non_overlapped_total_cycles": total_non_overlapped,
            "overlapped_latency_us": _round(total_overlapped / clock_mhz),
            "estimated_macs": total_macs,
            "estimated_memory_words": total_words,
        },
        "layers": layers,
    }


def analyze_tiling_performance(
    compile_plan: Any,
    *,
    graph: Any | None = None,
    parallelism_by_layer: Mapping[str, Any] | None = None,
    ii_by_layer: Mapping[str, Any] | None = None,
    clock_mhz: float = DEFAULT_CLOCK_MHZ,
    memory_words_per_cycle: float = DEFAULT_MEMORY_WORDS_PER_CYCLE,
    tile_overhead_cycles: int = DEFAULT_TILE_OVERHEAD_CYCLES,
) -> dict[str, Any]:
    tiling_report = analyze_tiling(
        compile_plan,
        graph=graph,
    )
    return estimate_tiling_performance(
        tiling_report,
        parallelism_by_layer=parallelism_by_layer,
        ii_by_layer=ii_by_layer,
        clock_mhz=clock_mhz,
        memory_words_per_cycle=memory_words_per_cycle,
        tile_overhead_cycles=tile_overhead_cycles,
    )


def write_tiling_performance_estimate_json(
    path: str | Path,
    tiling_report_or_compile_plan: Any,
    *,
    graph: Any | None = None,
    parallelism_by_layer: Mapping[str, Any] | None = None,
    ii_by_layer: Mapping[str, Any] | None = None,
    clock_mhz: float = DEFAULT_CLOCK_MHZ,
    memory_words_per_cycle: float = DEFAULT_MEMORY_WORDS_PER_CYCLE,
    tile_overhead_cycles: int = DEFAULT_TILE_OVERHEAD_CYCLES,
    input_is_tiling_report: bool = False,
) -> dict[str, Any]:
    if input_is_tiling_report:
        report = estimate_tiling_performance(
            tiling_report_or_compile_plan,
            parallelism_by_layer=parallelism_by_layer,
            ii_by_layer=ii_by_layer,
            clock_mhz=clock_mhz,
            memory_words_per_cycle=memory_words_per_cycle,
            tile_overhead_cycles=tile_overhead_cycles,
        )
    else:
        report = analyze_tiling_performance(
            tiling_report_or_compile_plan,
            graph=graph,
            parallelism_by_layer=parallelism_by_layer,
            ii_by_layer=ii_by_layer,
            clock_mhz=clock_mhz,
            memory_words_per_cycle=memory_words_per_cycle,
            tile_overhead_cycles=tile_overhead_cycles,
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


def attach_tiling_performance_estimate_to_manifest(
    manifest: MutableMapping[str, Any],
    performance_report: Mapping[str, Any],
    *,
    path: str = "tiling_performance_estimate.json",
) -> MutableMapping[str, Any]:
    totals = performance_report.get("totals", {})
    if not isinstance(totals, Mapping):
        totals = {}

    manifest["tiling_performance_estimate"] = {
        "format": performance_report.get(
            "format",
            "fpgai.tiling_performance_model.v1",
        ),
        "path": path,
        "tiled_layer_count": _safe_int(
            totals.get("tiled_layer_count"),
        ),
        "compute_cycles": _safe_int(
            totals.get("compute_cycles"),
        ),
        "memory_cycles": _safe_int(
            totals.get("memory_cycles"),
        ),
        "overlapped_total_cycles": _safe_int(
            totals.get("overlapped_total_cycles"),
        ),
        "non_overlapped_total_cycles": _safe_int(
            totals.get("non_overlapped_total_cycles"),
        ),
        "overlapped_latency_us": _safe_float(
            totals.get("overlapped_latency_us"),
        ),
        "estimated_macs": _safe_int(
            totals.get("estimated_macs"),
        ),
        "estimated_memory_words": _safe_int(
            totals.get("estimated_memory_words"),
        ),
    }
    return manifest
