from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _collect_files(root: Path) -> list[dict[str, Any]]:
    suffixes = {
        ".cpp",
        ".h",
        ".hpp",
        ".tcl",
        ".json",
        ".xml",
        ".rpt",
    }

    files: list[dict[str, Any]] = []
    if not root.exists():
        return files

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.name == "hls_artifact_metadata.json":
            continue

        if path.suffix.lower() not in suffixes:
            continue

        files.append(
            {
                "path": _safe_relpath(path, root),
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
            }
        )

    return files


def _compile_plan_dict(compile_plan: Any) -> dict[str, Any]:
    if compile_plan is None:
        return {}

    if hasattr(compile_plan, "to_dict"):
        value = compile_plan.to_dict()
        if isinstance(value, dict):
            return value

    if isinstance(compile_plan, Mapping):
        return dict(compile_plan)

    return {}


def _layer_metadata(compile_plan: Any) -> list[dict[str, Any]]:
    layers = getattr(compile_plan, "layer_plans", None)

    if layers is None and isinstance(compile_plan, Mapping):
        layers = compile_plan.get("layer_plans")

    if not layers:
        return []

    result: list[dict[str, Any]] = []

    for index, layer in enumerate(layers):
        if hasattr(layer, "to_dict"):
            layer_dict = layer.to_dict()
        elif isinstance(layer, Mapping):
            layer_dict = dict(layer)
        else:
            layer_dict = {}

        architecture = layer_dict.get("architecture")
        if architecture is None:
            architecture = {}

        result.append(
            {
                "index": index,
                "name": layer_dict.get("name", f"layer_{index}"),
                "op_type": layer_dict.get("op_type"),
                "architecture": architecture,
                "architecture_signature": layer_dict.get(
                    "architecture_signature",
                    layer_dict.get("signature"),
                ),
            }
        )

    return result


def emit_hls_artifact_metadata(
    out_dir: Path | str,
    compile_plan: Any = None,
    *,
    schedule_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a stable metadata file describing emitted HLS artifacts.

    The goal is to make generated HLS outputs self-describing enough for
    experiment automation: which files were emitted, which per-layer
    architecture was requested/effective, and which schedule summary was
    attached later.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    plan_dict = _compile_plan_dict(compile_plan)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "artifact_root": ".",
        "architecture_signature": plan_dict.get("architecture_signature"),
        "layer_count": len(_layer_metadata(compile_plan)),
        "layers": _layer_metadata(compile_plan),
        "files": _collect_files(root),
    }

    if schedule_summary is not None:
        metadata["hls_schedule_summary"] = dict(schedule_summary)

    metadata_path = root / "hls_artifact_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "path": "hls_artifact_metadata.json",
        "schema_version": metadata["schema_version"],
        "architecture_signature": metadata["architecture_signature"],
        "layer_count": metadata["layer_count"],
        "file_count": len(metadata["files"]),
    }
