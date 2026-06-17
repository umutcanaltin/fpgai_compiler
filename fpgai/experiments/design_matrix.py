"""Experiment design-matrix expansion utilities for FPGAI.

This module is intentionally dependency-light and backwards compatible with the
first Sprint 6 automation patches.  The public API used by tests and the sweep
runner is:

- DesignPoint
- render_template
- expand_design_matrix
- load_sweep_config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

try:  # PyYAML is available in the project environment, but keep import local-safe.
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass(frozen=True)
class DesignPoint:
    """One concrete experiment point.

    `command` is kept as a first-class field for compatibility with the original
    Sprint 6 runner and tests.  The dataclass is frozen so points can be safely
    reused/resumed without accidental mutation; tests that need edits should use
    `dataclasses.replace`.
    """

    index: int
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    command: str = ""
    config_path: Optional[str] = None
    model_path: Optional[str] = None
    board: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_sweep_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML or JSON-like sweep configuration file."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load sweep configurations")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Sweep config must be a mapping: {p}")
    data.setdefault("source_path", str(p))
    return data


def render_template(template: str | None, values: Mapping[str, Any]) -> str:
    """Render `{key}` placeholders from a mapping.

    Missing keys are left unchanged instead of raising.  This is useful for
    partially specified sweep templates and keeps old tests stable.
    """

    if template is None:
        return ""

    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:  # pragma: no cover - tiny branch
            return "{" + key + "}"

    safe_values = _SafeDict({k: "" if v is None else v for k, v in values.items()})
    try:
        return str(template).format_map(safe_values)
    except Exception:
        # Last-resort fallback: do simple replacement for keys that exist.
        out = str(template)
        for k, v in values.items():
            out = out.replace("{" + str(k) + "}", str(v))
        return out


def _safe_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "design"


def _dedupe_name(base: str, seen: MutableMapping[str, int]) -> str:
    base = _safe_name(base)
    count = seen.get(base, 0)
    seen[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count:02d}"


def _normalise_parameter_values(parameters: Mapping[str, Any] | None) -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {}
    for key, value in (parameters or {}).items():
        if isinstance(value, list):
            result[key] = value
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = [value]
    return result


def _cartesian_dicts(parameters: Mapping[str, Any] | None) -> Iterable[Dict[str, Any]]:
    values = _normalise_parameter_values(parameters)
    if not values:
        yield {}
        return
    keys = list(values.keys())
    for combo in product(*(values[k] for k in keys)):
        yield dict(zip(keys, combo))


def _build_point(
    *,
    index: int,
    name: str,
    params: Dict[str, Any],
    command_template: str | None,
    sweep_name: str,
    source_path: str | None,
    explicit_command: str | None = None,
) -> DesignPoint:
    command = explicit_command if explicit_command is not None else render_template(command_template or "", params)
    return DesignPoint(
        index=index,
        name=_safe_name(name),
        parameters=dict(params),
        command=command,
        config_path=params.get("config_path") or params.get("base_config_path"),
        model_path=params.get("model_path") or params.get("model"),
        board=params.get("board") or params.get("target_board"),
        metadata={"sweep_name": sweep_name, "source_path": source_path},
    )


def expand_design_matrix(config: Mapping[str, Any], limit: int | None = None) -> List[DesignPoint]:
    """Expand a sweep configuration into concrete `DesignPoint`s.

    Supported config styles:

    1. Cartesian product style:
       defaults + parameters + command_template

    2. Explicit point style:
       design_points: [{name, command, ...}, ...]

    Optional:
       design_name_template: "policy_{policy}_{precision_mode}"

    Duplicate generated names are suffixed as `_01`, `_02`, ...
    """

    sweep_name = str(config.get("name") or "experiment")
    source_path = config.get("source_path")
    defaults = dict(config.get("defaults") or {})
    command_template = config.get("command_template")
    design_name_template = config.get("design_name_template")
    points: List[DesignPoint] = []
    seen_names: Dict[str, int] = {}

    explicit_points = config.get("design_points")
    if explicit_points:
        for raw in explicit_points:
            if not isinstance(raw, Mapping):
                continue
            params = dict(defaults)
            for k, v in raw.items():
                if k not in {"name", "command"}:
                    params[k] = v
            idx = len(points)
            if design_name_template:
                raw_name = render_template(str(design_name_template), params)
            else:
                raw_name = str(raw.get("name") or f"{sweep_name}_{idx:03d}")
            name = _dedupe_name(raw_name, seen_names)
            point = _build_point(
                index=idx,
                name=name,
                params=params,
                command_template=command_template,
                explicit_command=raw.get("command"),
                sweep_name=sweep_name,
                source_path=source_path,
            )
            points.append(point)
            if limit is not None and len(points) >= limit:
                return points
        return points

    for combo in _cartesian_dicts(config.get("parameters") or {}):
        params = dict(defaults)
        params.update(combo)
        idx = len(points)
        if design_name_template:
            raw_name = render_template(str(design_name_template), params)
        else:
            raw_name = f"{sweep_name}_{idx:03d}"
        name = _dedupe_name(raw_name, seen_names)
        points.append(
            _build_point(
                index=idx,
                name=name,
                params=params,
                command_template=command_template,
                sweep_name=sweep_name,
                source_path=source_path,
            )
        )
        if limit is not None and len(points) >= limit:
            break

    return points


__all__ = [
    "DesignPoint",
    "expand_design_matrix",
    "load_sweep_config",
    "render_template",
]
