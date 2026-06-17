"""Materialize per-design FPGAI YAML configs for experiment sweeps.

This module is intentionally conservative:
- It never adds an ``experiment:`` top-level section to FPGAI YAML files.
- It writes experiment metadata to ``<design>.metadata.json`` sidecar files.
- It only applies known overrides to existing/compatible config paths.
- It can strip validator-unsafe top-level sections from generated configs for real runs.

The public API remains backwards-compatible with the earlier Sprint 6 patches:
``apply_parameter_overrides`` returns ``(config, report)`` and
``materialize_design_config`` returns an object with attributes such as
``output_config_path`` and ``metadata_path`` while also behaving like a dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import copy
import json
import re

try:  # PyYAML is already used by the project/test environment.
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


_POLICY_CANONICAL = {
    "resource_first": "ResourceFirst",
    "resource-first": "ResourceFirst",
    "resourcefirst": "ResourceFirst",
    "balanced": "Balanced",
    "latency_first": "LatencyFirst",
    "latency-first": "LatencyFirst",
    "latencyfirst": "LatencyFirst",
    "throughput_first": "ThroughputFirst",
    "throughput-first": "ThroughputFirst",
    "throughputfirst": "ThroughputFirst",
    "memory_first": "MemoryFirst",
    "memory-first": "MemoryFirst",
    "memoryfirst": "MemoryFirst",
    "calibrated_balanced": "CalibratedBalanced",
    "calibrated-balanced": "CalibratedBalanced",
    "calibratedbalanced": "CalibratedBalanced",
}


@dataclass
class MaterializationReport:
    base_config_path: str = ""
    output_config_path: str = ""
    metadata_path: str = ""
    applied: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    unapplied: dict[str, Any] = field(default_factory=dict)
    removed_top_level_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_config_path": self.base_config_path,
            "output_config_path": self.output_config_path,
            "metadata_path": self.metadata_path,
            "applied": dict(self.applied),
            "skipped": dict(self.skipped),
            "unapplied": dict(self.unapplied),
            "removed_top_level_sections": list(self.removed_top_level_sections),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()


class _EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # pragma: no cover - defensive
        if isinstance(obj, MaterializationReport):
            return obj.to_dict()
        return super().default(obj)


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required for experiment config materialization")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if text.strip() else {}
    return data if isinstance(data, dict) else {}


def _dump_yaml(path: Path, data: Mapping[str, Any]) -> None:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required for experiment config materialization")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False), encoding="utf-8")


def _normalize_policy(value: Any, *, canonicalize: bool = True) -> str:
    text = str(value)
    if not canonicalize:
        return text
    key = text.strip().lower()
    return _POLICY_CANONICAL.get(key, text)


def _set_path_if_parent_exists(data: dict[str, Any], dotted_path: str, value: Any) -> bool:
    parts = dotted_path.split(".")
    cur: Any = data
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur or not isinstance(cur[part], dict):
            return False
        cur = cur[part]
    if not isinstance(cur, dict):
        return False
    cur[parts[-1]] = value
    return True


def _existing_file(path_value: str, *, base_dir: Path | None = None, repo_root: Path | None = None) -> bool:
    p = Path(path_value)
    if p.is_absolute():
        return p.exists()
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / p)
    if repo_root is not None:
        candidates.append(repo_root / p)
    candidates.append(p)
    return any(c.exists() for c in candidates)


def _resolve_options(options: Mapping[str, Any] | None, extra: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if options:
        merged.update(dict(options))
    merged.update({k: v for k, v in extra.items() if v is not None})
    return merged


def apply_parameter_overrides(
    base_config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    base_config_path: str | Path | None = None,
    canonicalize_policy: bool = True,
) -> tuple[dict[str, Any], MaterializationReport]:
    """Apply safe parameter overrides to a config dict.

    Returns ``(updated_config, report)`` for backwards compatibility with earlier
    Sprint 6 tests.
    """

    data = copy.deepcopy(dict(base_config))
    report = MaterializationReport()
    base_dir = Path(base_config_path).resolve().parent if base_config_path else None
    root = Path(repo_root).resolve() if repo_root else None

    if "policy" in parameters:
        value = _normalize_policy(parameters["policy"], canonicalize=canonicalize_policy)
        if _set_path_if_parent_exists(data, "notes.parallel_policy", value):
            report.applied["policy"] = "notes.parallel_policy"
        elif _set_path_if_parent_exists(data, "compiler.policy", value):
            report.applied["policy"] = "compiler.policy"
        elif _set_path_if_parent_exists(data, "planner.policy", value):
            report.applied["policy"] = "planner.policy"
        else:
            report.unapplied["policy"] = parameters["policy"]
            report.skipped["policy"] = "no known existing policy key in base configuration"

    if "precision_mode" in parameters:
        value = parameters["precision_mode"]
        if _set_path_if_parent_exists(data, "notes.precision_mode", value):
            report.applied["precision_mode"] = "notes.precision_mode"
        elif _set_path_if_parent_exists(data, "quantization.precision_mode", value):
            report.applied["precision_mode"] = "quantization.precision_mode"
        elif _set_path_if_parent_exists(data, "compiler.precision_mode", value):
            report.applied["precision_mode"] = "compiler.precision_mode"
        else:
            report.unapplied["precision_mode"] = value
            report.skipped["precision_mode"] = "no known existing precision key in base configuration"

    if "model_path" in parameters:
        requested = str(parameters["model_path"])
        if _existing_file(requested, base_dir=base_dir, repo_root=root):
            if _set_path_if_parent_exists(data, "model.path", requested):
                report.applied["model_path"] = "model.path"
            else:
                report.unapplied["model_path"] = requested
                report.skipped["model_path"] = "no existing model.path key in base configuration"
        else:
            report.unapplied["model_path"] = requested
            report.skipped["model_path"] = "requested model file does not exist"

    if "board" in parameters:
        value = parameters["board"]
        if _set_path_if_parent_exists(data, "target.board", value):
            report.applied["board"] = "target.board"
        elif _set_path_if_parent_exists(data, "board.name", value):
            report.applied["board"] = "board.name"
        else:
            report.unapplied["board"] = value
            report.skipped["board"] = "no known existing board key in base configuration"

    return data, report


def _write_metadata(path: Path, report: MaterializationReport, parameters: Mapping[str, Any], design_name: str | None) -> None:
    payload = report.to_dict()
    payload["design_name"] = design_name
    payload["parameters"] = dict(parameters)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, cls=_EnhancedJSONEncoder), encoding="utf-8")


def materialize_design_config(
    base_config_path: str | Path,
    output_config_path: str | Path,
    parameters: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
    *,
    design_name: str | None = None,
    repo_root: str | Path | None = None,
    canonicalize_policy: bool | None = None,
    strip_top_level_sections: list[str] | tuple[str, ...] | None = None,
    **extra_options: Any,
) -> MaterializationReport:
    """Write a materialized FPGAI config and sidecar metadata.

    Supports both new keyword options and the previous 4th positional ``options``
    argument used by ``SweepRunner``.
    """

    opts = _resolve_options(options, extra_options)
    base_path = Path(base_config_path)
    out_path = Path(output_config_path)
    root = Path(repo_root).resolve() if repo_root else base_path.resolve().parent

    # Compatibility rule:
    # - direct legacy calls like materialize_design_config(..., {"enabled": True})
    #   keep raw policy text for old tests.
    # - SweepRunner calls include a materialization directory, so normalize policy
    #   names by default: resource_first -> ResourceFirst.
    # - Sweep YAML can still override this explicitly with canonicalize_policy.
    if canonicalize_policy is None:
        if "canonicalize_policy" in opts:
            canonicalize_policy = bool(opts["canonicalize_policy"])
        else:
            canonicalize_policy = options is None or "directory" in opts

    data = _load_yaml(base_path)
    data, report = apply_parameter_overrides(
        data,
        parameters,
        repo_root=root,
        base_config_path=base_path,
        canonicalize_policy=canonicalize_policy,
    )

    report.base_config_path = str(base_path.resolve())
    report.output_config_path = str(out_path)
    metadata_path = out_path.with_suffix(".metadata.json")
    report.metadata_path = str(metadata_path)

    sections = list(strip_top_level_sections or opts.get("strip_top_level_sections") or [])
    for section in sections:
        if section in data:
            data.pop(section, None)
            report.removed_top_level_sections.append(str(section))

    _dump_yaml(out_path, data)
    _write_metadata(metadata_path, report, parameters, design_name)
    return report


def safe_design_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "design"
