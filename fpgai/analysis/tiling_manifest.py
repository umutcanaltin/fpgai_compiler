from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

from fpgai.analysis.tiling_analysis import write_tiling_analysis_json


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def tiling_analysis_manifest_entry(
    report: dict[str, Any],
    *,
    path: str = "tiling_analysis.json",
) -> dict[str, Any]:
    """Create a compact manifest entry for a tiling analysis report."""
    totals = report.get("totals", {})

    return {
        "format": report.get(
            "format",
            "fpgai.tiling_analysis.v1",
        ),
        "path": path,
        "tiled_layer_count": _safe_int(
            totals.get("tiled_layer_count")
        ),
        "implemented_tiled_layer_count": _safe_int(
            totals.get("implemented_tiled_layer_count")
        ),
        "planning_only_tiled_layer_count": _safe_int(
            totals.get("planning_only_tiled_layer_count")
        ),
        "local_buffer_elements": _safe_int(
            totals.get("local_buffer_elements")
        ),
        "estimated_activation_reads": _safe_int(
            totals.get("estimated_activation_reads")
        ),
        "estimated_weight_reads": _safe_int(
            totals.get("estimated_weight_reads")
        ),
        "estimated_output_writes": _safe_int(
            totals.get("estimated_output_writes")
        ),
        "estimated_macs": _safe_int(
            totals.get("estimated_macs")
        ),
    }


def attach_tiling_analysis_to_manifest(
    manifest: MutableMapping[str, Any],
    report: dict[str, Any],
    *,
    path: str = "tiling_analysis.json",
) -> MutableMapping[str, Any]:
    """Attach a compact tiling_analysis section to an existing manifest."""
    manifest["tiling_analysis"] = tiling_analysis_manifest_entry(
        report,
        path=path,
    )
    return manifest


def write_tiling_analysis_artifact(
    output_dir: str | Path,
    compile_plan: Any,
    *,
    graph: Any | None = None,
    manifest: MutableMapping[str, Any] | None = None,
    filename: str = "tiling_analysis.json",
) -> tuple[dict[str, Any], MutableMapping[str, Any]]:
    """Write tiling_analysis.json and return an updated manifest.

    This helper is intentionally small so the compiler driver can call it
    without knowing the internals of the tiling analysis report.
    """
    output_path = Path(output_dir)
    artifact_path = output_path / filename

    report = write_tiling_analysis_json(
        artifact_path,
        compile_plan,
        graph=graph,
    )

    updated_manifest: MutableMapping[str, Any]
    if manifest is None:
        updated_manifest = {}
    else:
        updated_manifest = manifest

    attach_tiling_analysis_to_manifest(
        updated_manifest,
        report,
        path=filename,
    )

    return report, updated_manifest
