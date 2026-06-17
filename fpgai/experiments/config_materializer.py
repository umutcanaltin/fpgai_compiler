from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import yaml

_POLICY_CANONICAL = {
    "balanced": "Balanced",
    "latency_first": "LatencyFirst",
    "resource_first": "ResourceFirst",
    "throughput_first": "ThroughputFirst",
    "Balanced": "Balanced",
    "LatencyFirst": "LatencyFirst",
    "ResourceFirst": "ResourceFirst",
    "ThroughputFirst": "ThroughputFirst",
}

# Known existing paths only. Explicit parameter_mappings can create more.
_DEFAULT_PATHS = {
    "policy": [
        "optimization.parallel_policy",
        "optimization.policy",
        "notes.parallel_policy",
    ],
    "precision_mode": [
        "precision.mode",
        "precision_mode",
        "notes.precision_mode",
        "numerics.precision_mode",
    ],
    "board": [
        "target.board",
        "targets.platform.board",
        "board",
    ],
    "model_path": [
        "model.path",
        "model_path",
    ],
}

_STRIP_DEFAULT = ["experiment", "design_parameters", "materialized_overrides"]


class MaterializationReport(dict):
    """Dict-like report with attribute access for backwards compatibility."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__ = self


def _deepcopy_mapping(data: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(data))


def _split_path(path: str) -> list[str]:
    return [p for p in str(path).split(".") if p]


def _get_path(data: Mapping[str, Any], path: str) -> Any:
    cur: Any = data
    for part in _split_path(path):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _has_path(data: Mapping[str, Any], path: str) -> bool:
    cur: Any = data
    parts = _split_path(path)
    for part in parts:
        if not isinstance(cur, Mapping) or part not in cur:
            return False
        cur = cur[part]
    return True


def _set_path(data: MutableMapping[str, Any], path: str, value: Any, *, create: bool = False) -> bool:
    parts = _split_path(path)
    if not parts:
        return False
    cur: MutableMapping[str, Any] = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            if not create:
                return False
            nxt = {}
            cur[part] = nxt
        if not isinstance(nxt, MutableMapping):
            if not create:
                return False
            nxt = {}
            cur[part] = nxt
        cur = nxt
    leaf = parts[-1]
    if not create and leaf not in cur:
        return False
    cur[leaf] = value
    return True


def _canonical_policy(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return _POLICY_CANONICAL.get(value, value)


def _compiler_policy(value: Any) -> Any:
    """Map experiment policy names to the enum values accepted by FPGAI config validation."""
    if not isinstance(value, str):
        return value
    table = {
        "balanced": "Balanced",
        "Balanced": "Balanced",
        "latency_first": "Latency-First",
        "LatencyFirst": "Latency-First",
        "Latency-First": "Latency-First",
        "throughput_first": "Throughput-First",
        "ThroughputFirst": "Throughput-First",
        "Throughput-First": "Throughput-First",
        "resource_first": "Fit-First",
        "ResourceFirst": "Fit-First",
        "FitFirst": "Fit-First",
        "Fit-First": "Fit-First",
        "bram_saver": "BRAM-Saver",
        "BRAMSaver": "BRAM-Saver",
        "BRAM-Saver": "BRAM-Saver",
        "dsp_saver": "DSP-Saver",
        "DSPSaver": "DSP-Saver",
        "DSP-Saver": "DSP-Saver",
    }
    return table.get(value, value)


def _normalise_options(options: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    opts: Dict[str, Any] = dict(options or {})
    for key, val in kwargs.items():
        if val is not None:
            opts[key] = val
    return opts


def _is_legacy_raw_options(options: Optional[Mapping[str, Any]]) -> bool:
    if not options:
        return False
    keys = set(options.keys())
    # Historical/manual fourth-argument compatibility mode.  Older callers
    # passed an options dictionary directly to materialize_design_config without
    # the sweep-runner context.  In that mode policy values were preserved as
    # raw experiment enums (for example resource_first).  Sweep YAMLs are not
    # considered legacy because they carry a directory and/or canonicalize_policy
    # option and should emit compiler-facing enum names such as ResourceFirst.
    if "canonicalize_policy" in keys:
        return False
    if "directory" in keys:
        return False
    if "parameter_mappings" in keys:
        return False
    return bool(options.get("enabled") is True or "strip_unknown_top_level_sections" in keys or "strip_top_level_sections" in keys)


def _should_canonicalize_policy(path: str, opts: Mapping[str, Any], *, legacy_raw: bool) -> bool:
    if legacy_raw and "canonicalize_policy" not in opts:
        return False
    if "canonicalize_policy" in opts:
        return bool(opts.get("canonicalize_policy"))
    # Default behavior for direct calls and normal materialization:
    # canonical compiler enum paths, but preserve raw experiment enum paths.
    if path.endswith("optimization.policy"):
        return False
    if path.endswith("parallel_policy"):
        return True
    return False


def _value_for_path(param: str, value: Any, path: str, opts: Mapping[str, Any], *, legacy_raw: bool) -> Any:
    # optimization.policy is a raw experiment/compiler enum field. Do not canonicalize
    # it even when canonicalize_policy or compiler_policy_names is enabled. Only
    # *parallel_policy fields use FPGAI's named policy enums.
    if param == "policy" and path == "optimization.policy":
        return value
    if param == "policy" and bool(opts.get("compiler_policy_names")) and path.endswith("optimization.parallel_policy"):
        return _compiler_policy(value)
    if param == "policy" and _should_canonicalize_policy(path, opts, legacy_raw=legacy_raw):
        return _canonical_policy(value)
    return value


def _iter_strip_sections(raw: Any) -> list[str]:
    if raw is None or raw is False:
        return []
    if raw is True:
        return list(_STRIP_DEFAULT)
    if isinstance(raw, str):
        return [raw]
    try:
        return [str(x) for x in raw]
    except TypeError:
        return []


def _strip_sections(data: MutableMapping[str, Any], opts: Mapping[str, Any]) -> list[str]:
    sections: list[str] = []
    sections.extend(_iter_strip_sections(opts.get("strip_top_level_sections")))
    sections.extend(_iter_strip_sections(opts.get("strip_unknown_top_level_sections")))
    # Some older sweep configs used validator_safe as a high-level request.
    if opts.get("validator_safe") is True:
        sections.extend(_STRIP_DEFAULT)
    removed: list[str] = []
    for section in dict.fromkeys(sections):
        if section in data:
            data.pop(section, None)
            removed.append(section)
    return removed


def _resolve_model_path(value: Any, repo_root: Optional[Path]) -> Tuple[bool, Any, str]:
    if value in (None, ""):
        return False, value, "empty model path"
    p = Path(str(value))
    candidates = [p]
    if not p.is_absolute() and repo_root is not None:
        candidates.insert(0, repo_root / p)
    if any(c.exists() for c in candidates):
        return True, str(value), ""
    return False, value, "requested model file does not exist"


def probe_parameter_paths(config: Mapping[str, Any]) -> Dict[str, list[str]]:
    """Return existing materializable paths for known experiment parameters."""
    out: Dict[str, list[str]] = {}
    for param, paths in _DEFAULT_PATHS.items():
        out[param] = [path for path in paths if _has_path(config, path)]
    return out


def apply_parameter_overrides(
    base_config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    options: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], MaterializationReport]:
    opts = _normalise_options(options, **kwargs)
    legacy_raw = bool(opts.pop("_legacy_raw_policy", False)) or _is_legacy_raw_options(options)
    repo_root = opts.get("repo_root")
    if repo_root is not None:
        repo_root = Path(repo_root)

    cfg = _deepcopy_mapping(base_config)
    applied: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    unapplied: Dict[str, Any] = {}

    explicit_mappings = opts.get("parameter_mappings") or {}

    for param, value in dict(parameters or {}).items():
        if param in {"config_path", "base_config_path", "metrics_json"}:
            continue

        if param == "model_path":
            ok_model, resolved_value, reason = _resolve_model_path(value, repo_root)
            if not ok_model:
                skipped[param] = reason
                unapplied[param] = value
                continue
            value_to_apply = resolved_value
        else:
            value_to_apply = value

        mapping = explicit_mappings.get(param) if isinstance(explicit_mappings, Mapping) else None
        candidate_paths: list[tuple[str, bool]] = []
        if isinstance(mapping, Mapping):
            path = mapping.get("path")
            if path:
                candidate_paths.append((str(path), bool(mapping.get("create", False))))
        elif isinstance(mapping, str):
            candidate_paths.append((mapping, False))

        if not candidate_paths:
            candidate_paths = [(p, False) for p in _DEFAULT_PATHS.get(param, [])]

        chosen = False
        existing_reason = None
        for path, create in candidate_paths:
            if create or _has_path(cfg, path):
                v = _value_for_path(param, value_to_apply, path, opts, legacy_raw=legacy_raw)
                if _set_path(cfg, path, v, create=create):
                    applied[param] = path
                    chosen = True
                    break
            else:
                existing_reason = f"no known existing {param.replace('_', ' ')} key in base configuration"

        if not chosen:
            skipped[param] = existing_reason or f"no materialization path for {param}"
            unapplied[param] = value

    removed = _strip_sections(cfg, opts)
    report = MaterializationReport(
        applied=applied,
        skipped=skipped,
        unapplied=unapplied,
        removed_top_level_sections=removed,
    )
    return cfg, report


def materialize_design_config(
    base_config_path: str | Path,
    output_config_path: str | Path,
    parameters: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
    *,
    design_name: Optional[str] = None,
    repo_root: Optional[str | Path] = None,
    canonicalize_policy: Optional[bool] = None,
    strip_top_level_sections: Optional[Any] = None,
    strip_unknown_top_level_sections: Optional[Any] = None,
    parameter_mappings: Optional[Mapping[str, Any]] = None,
) -> MaterializationReport:
    base_path = Path(base_config_path)
    out_path = Path(output_config_path)
    opts = _normalise_options(
        options,
        canonicalize_policy=canonicalize_policy,
        strip_top_level_sections=strip_top_level_sections,
        strip_unknown_top_level_sections=strip_unknown_top_level_sections,
        parameter_mappings=parameter_mappings,
        repo_root=repo_root,
    )
    if _is_legacy_raw_options(options):
        opts["_legacy_raw_policy"] = True

    with base_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        data = {}

    cfg, partial = apply_parameter_overrides(data, parameters or {}, opts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    metadata_path = out_path.with_suffix(".metadata.json")
    report = MaterializationReport(
        applied=dict(partial.get("applied", {})),
        skipped=dict(partial.get("skipped", {})),
        unapplied=dict(partial.get("unapplied", {})),
        removed_top_level_sections=list(partial.get("removed_top_level_sections", [])),
        base_config_path=str(base_path.resolve()),
        output_config_path=str(out_path.resolve()),
        metadata_path=str(metadata_path.resolve()),
        design_name=design_name,
        parameters=dict(parameters or {}),
    )
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report
