from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

from fpgai.analysis.tiling_analysis import write_tiling_analysis_json
from fpgai.analysis.tiling_manifest import attach_tiling_analysis_to_manifest
from fpgai.analysis.tiling_performance_model import (
    attach_tiling_performance_estimate_to_manifest,
    write_tiling_performance_estimate_json,
)
from fpgai.analysis.tiling_resource_model import (
    attach_tiling_resource_estimate_to_manifest,
    write_tiling_resource_estimate_json,
)
from fpgai.analysis.tiling_sweep import (
    sweep_compile_plan_tiles,
    write_tiling_sweep_json,
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


def collect_layer_architecture_assumptions(
    compile_plan: Any,
) -> dict[str, dict[str, Any]]:
    """Extract precision, parallelism, and pipeline assumptions from a plan.

    The analytical tiling resource/performance models accept plain dictionaries.
    This helper converts typed ArchitecturePlan objects or dict-like plans into
    those dictionaries so the compiler driver can call one unified pipeline.
    """
    precision_by_layer: dict[str, Any] = {}
    parallelism_by_layer: dict[str, Any] = {}
    ii_by_layer: dict[str, Any] = {}

    for layer in _layers(compile_plan):
        layer_dict = _to_dict(layer)
        name = _layer_name(layer)
        architecture = _to_dict(layer_dict.get("architecture"))

        precision = _to_dict(architecture.get("precision"))
        if precision:
            precision_by_layer[name] = precision

        parallelism = _to_dict(architecture.get("parallelism"))
        if parallelism:
            parallelism_by_layer[name] = parallelism

        pipeline = _to_dict(architecture.get("pipeline"))
        if pipeline:
            ii_by_layer[name] = pipeline

    return {
        "precision_by_layer": precision_by_layer,
        "parallelism_by_layer": parallelism_by_layer,
        "ii_by_layer": ii_by_layer,
    }


def tiling_sweep_manifest_entry(
    sweep_report: Mapping[str, Any],
    *,
    path: str = "tiling_sweep.json",
) -> dict[str, Any]:
    if "best_by_layer" in sweep_report:
        best_by_layer = sweep_report.get("best_by_layer", {})
        layer_count = int(sweep_report.get("layer_count", 0) or 0)
        best_layer_count = len(best_by_layer) if isinstance(best_by_layer, Mapping) else 0
        return {
            "format": sweep_report.get(
                "format",
                "fpgai.tiling_sweep_collection.v1",
            ),
            "path": path,
            "layer_count": layer_count,
            "best_layer_count": best_layer_count,
        }

    return {
        "format": sweep_report.get(
            "format",
            "fpgai.tiling_sweep.v1",
        ),
        "path": path,
        "layer_name": sweep_report.get("layer_name"),
        "candidate_count": int(sweep_report.get("candidate_count", 0) or 0),
        "has_best": sweep_report.get("best") is not None,
    }


def attach_tiling_sweep_to_manifest(
    manifest: MutableMapping[str, Any],
    sweep_report: Mapping[str, Any],
    *,
    path: str = "tiling_sweep.json",
) -> MutableMapping[str, Any]:
    manifest["tiling_sweep"] = tiling_sweep_manifest_entry(
        sweep_report,
        path=path,
    )
    return manifest


def write_tiling_report_artifacts(
    output_dir: str | Path,
    compile_plan: Any,
    *,
    graph: Any | None = None,
    manifest: MutableMapping[str, Any] | None = None,
    clock_mhz: float = 200.0,
    memory_words_per_cycle: float = 1.0,
    tile_overhead_cycles: int = 4,
    candidates_by_layer: Mapping[str, Iterable[Mapping[str, int]]] | None = None,
    sweep_top_k: int | None = 5,
) -> tuple[dict[str, Any], MutableMapping[str, Any]]:
    """Write all Sprint-4 tiling artifacts and update the compile manifest.

    Artifacts written:
      - tiling_analysis.json
      - tiling_resource_estimate.json
      - tiling_performance_estimate.json
      - tiling_sweep.json when candidates_by_layer is provided
    """
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    updated_manifest: MutableMapping[str, Any]
    if manifest is None:
        updated_manifest = {}
    else:
        updated_manifest = manifest

    assumptions = collect_layer_architecture_assumptions(
        compile_plan,
    )

    tiling_report = write_tiling_analysis_json(
        output_path / "tiling_analysis.json",
        compile_plan,
        graph=graph,
    )
    attach_tiling_analysis_to_manifest(
        updated_manifest,
        tiling_report,
        path="tiling_analysis.json",
    )

    resource_report = write_tiling_resource_estimate_json(
        output_path / "tiling_resource_estimate.json",
        tiling_report,
        precision_by_layer=assumptions["precision_by_layer"],
        input_is_tiling_report=True,
    )
    attach_tiling_resource_estimate_to_manifest(
        updated_manifest,
        resource_report,
        path="tiling_resource_estimate.json",
    )

    performance_report = write_tiling_performance_estimate_json(
        output_path / "tiling_performance_estimate.json",
        tiling_report,
        parallelism_by_layer=assumptions["parallelism_by_layer"],
        ii_by_layer=assumptions["ii_by_layer"],
        clock_mhz=clock_mhz,
        memory_words_per_cycle=memory_words_per_cycle,
        tile_overhead_cycles=tile_overhead_cycles,
        input_is_tiling_report=True,
    )
    attach_tiling_performance_estimate_to_manifest(
        updated_manifest,
        performance_report,
        path="tiling_performance_estimate.json",
    )

    reports: dict[str, Any] = {
        "tiling_analysis": tiling_report,
        "tiling_resource_estimate": resource_report,
        "tiling_performance_estimate": performance_report,
    }

    if candidates_by_layer:
        sweep_report = sweep_compile_plan_tiles(
            compile_plan,
            candidates_by_layer=candidates_by_layer,
            graph=graph,
            precision_by_layer=assumptions["precision_by_layer"],
            parallelism_by_layer=assumptions["parallelism_by_layer"],
            ii_by_layer=assumptions["ii_by_layer"],
            clock_mhz=clock_mhz,
            memory_words_per_cycle=memory_words_per_cycle,
            tile_overhead_cycles=tile_overhead_cycles,
            top_k=sweep_top_k,
        )
        write_tiling_sweep_json(
            output_path / "tiling_sweep.json",
            sweep_report,
        )
        attach_tiling_sweep_to_manifest(
            updated_manifest,
            sweep_report,
            path="tiling_sweep.json",
        )
        reports["tiling_sweep"] = sweep_report

    return reports, updated_manifest
