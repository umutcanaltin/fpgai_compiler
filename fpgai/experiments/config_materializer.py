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
    "weight_storage": [
        "memory.weight_storage",
        "data_movement.ps_pl.weights.mode",
    ],
    "memory_strategy": [
        "memory.strategy",
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



def _normalise_weight_storage(value: Any) -> Tuple[bool, str, str]:
    raw = str(value).strip().lower().replace("-", "_")
    table = {
        "embedded": "embedded",
        "on_chip": "embedded",
        "onchip": "embedded",
        "bram": "embedded",
        "uram": "embedded",
        "stream": "stream",
        "streaming": "stream",
        "streamed": "stream",
        "ddr": "ddr",
        "dma_ddr": "ddr",
        "external": "ddr",
        "external_ddr": "ddr",
    }
    if raw not in table:
        return False, raw, "unknown weight storage mode"
    return True, table[raw], ""


def _memory_strategy_payload(value: Any) -> Tuple[bool, Dict[str, Any], str]:
    raw = str(value).strip().lower().replace("-", "_")
    table: Dict[str, Dict[str, Any]] = {
        "on_chip": {
            "memory.strategy": "on_chip",
            "memory.weight_storage": "embedded",
            "data_movement.ps_pl.weights.mode": "embedded",
            "memory.weight_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.activation_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.allow_double_buffer": False,
        },
        "embedded": {
            "memory.strategy": "on_chip",
            "memory.weight_storage": "embedded",
            "data_movement.ps_pl.weights.mode": "embedded",
            "memory.weight_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.activation_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.allow_double_buffer": False,
        },
        "streaming": {
            "memory.strategy": "streaming",
            "memory.weight_storage": "stream",
            "data_movement.ps_pl.weights.mode": "stream",
            "memory.weight_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.activation_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.allow_double_buffer": True,
        },
        "stream": {
            "memory.strategy": "streaming",
            "memory.weight_storage": "stream",
            "data_movement.ps_pl.weights.mode": "stream",
            "memory.weight_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.activation_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.allow_double_buffer": True,
        },
        "external_ddr": {
            "memory.strategy": "external_ddr",
            "memory.weight_storage": "ddr",
            "data_movement.ps_pl.weights.mode": "ddr",
            "memory.weight_region_preference": ["DDR", "URAM", "BRAM"],
            "memory.activation_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.allow_double_buffer": True,
        },
        "ddr": {
            "memory.strategy": "external_ddr",
            "memory.weight_storage": "ddr",
            "data_movement.ps_pl.weights.mode": "ddr",
            "memory.weight_region_preference": ["DDR", "URAM", "BRAM"],
            "memory.activation_region_preference": ["BRAM", "URAM", "DDR"],
            "memory.allow_double_buffer": True,
        },
        "bram_saver": {
            "memory.strategy": "bram_saver",
            "memory.weight_storage": "stream",
            "data_movement.ps_pl.weights.mode": "stream",
            "memory.weight_region_preference": ["DDR", "URAM", "BRAM"],
            "memory.activation_region_preference": ["DDR", "URAM", "BRAM"],
            "memory.allow_double_buffer": False,
        },
        "uram_first": {
            "memory.strategy": "uram_first",
            "memory.weight_storage": "embedded",
            "data_movement.ps_pl.weights.mode": "embedded",
            "memory.weight_region_preference": ["URAM", "BRAM", "DDR"],
            "memory.activation_region_preference": ["URAM", "BRAM", "DDR"],
            "memory.allow_double_buffer": False,
        },
    }
    if raw not in table:
        return False, {}, "unknown memory strategy"
    return True, dict(table[raw]), ""


def _apply_weight_storage(cfg: MutableMapping[str, Any], value: Any) -> Tuple[bool, str]:
    ok, mode, reason = _normalise_weight_storage(value)
    if not ok:
        return False, reason
    _set_path(cfg, "memory.weight_storage", mode, create=True)
    _set_path(cfg, "data_movement.ps_pl.weights.mode", mode, create=True)
    return True, "memory.weight_storage,data_movement.ps_pl.weights.mode"


def _apply_memory_strategy(cfg: MutableMapping[str, Any], value: Any) -> Tuple[bool, str]:
    ok, payload, reason = _memory_strategy_payload(value)
    if not ok:
        return False, reason
    for key, val in payload.items():
        _set_path(cfg, key, val, create=True)
    return True, ",".join(payload.keys())


def _materialize_precision_defaults_from_candidates(
    cfg: MutableMapping[str, Any], precision_mode: Any
) -> Tuple[bool, str]:
    """Apply analysis.precision_sweep candidate defaults into numerics.defaults.

    Precision sweep values such as ``fx8_3`` are not raw config fields in
    configs/examples/default_compile.yml. They select one entry from
    analysis.precision_sweep.candidates and materialize that candidate's
    defaults into numerics.defaults so downstream compiler/codegen paths see
    the selected numeric type widths.
    """
    if precision_mode in (None, ""):
        return False, "empty precision mode"

    candidates = _dig_for_materializer(
        cfg, ("analysis", "precision_sweep", "candidates"), []
    )
    if not isinstance(candidates, list):
        return False, "analysis.precision_sweep.candidates is not a list"

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("name") != precision_mode:
            continue
        defaults = candidate.get("defaults")
        if not isinstance(defaults, Mapping):
            return False, f"precision mode {precision_mode} has no defaults"
        numerics = cfg.setdefault("numerics", {})
        if not isinstance(numerics, MutableMapping):
            cfg["numerics"] = {}
            numerics = cfg["numerics"]
        numerics["defaults"] = copy.deepcopy(dict(defaults))
        return True, "numerics.defaults"

    return False, (
        f"precision mode {precision_mode} not found in "
        "analysis.precision_sweep.candidates"
    )


def _dig_for_materializer(data: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


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

        mapping = explicit_mappings.get(param) if isinstance(explicit_mappings, Mapping) else None

        # Memory knobs are real design-space knobs.  They must be validated and
        # must create canonical compiler paths even when the base config does
        # not already contain a memory section.  Keep explicit parameter_mappings
        # above this branch so advanced users can override the default path.
        if mapping is None and param == "weight_storage":
            ok_mem, target_or_reason = _apply_weight_storage(cfg, value)
            if ok_mem:
                applied[param] = target_or_reason
            else:
                skipped[param] = target_or_reason
                unapplied[param] = value
            continue

        if mapping is None and param == "memory_strategy":
            ok_mem, target_or_reason = _apply_memory_strategy(cfg, value)
            if ok_mem:
                applied[param] = target_or_reason
            else:
                skipped[param] = target_or_reason
                unapplied[param] = value
            continue

        forced_candidate_paths: Optional[list[tuple[str, bool]]] = None

        if param == "precision_mode" and mapping is None:
            # Backwards-compatible behavior: if the base schema already has a
            # scalar precision selector, update that raw field. Only real
            # precision sweeps without such a scalar field should translate the
            # selector into numerics.defaults.
            existing_precision_paths = [
                (p, False)
                for p in _DEFAULT_PATHS.get(param, [])
                if _has_path(cfg, p)
            ]
            if existing_precision_paths:
                forced_candidate_paths = existing_precision_paths
            else:
                ok_precision, reason_or_path = _materialize_precision_defaults_from_candidates(cfg, value)
                if ok_precision:
                    applied[param] = reason_or_path
                else:
                    skipped[param] = reason_or_path
                    unapplied[param] = value
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

        candidate_paths: list[tuple[str, bool]] = []
        if isinstance(mapping, Mapping):
            path = mapping.get("path")
            if path:
                candidate_paths.append((str(path), bool(mapping.get("create", False))))
        elif isinstance(mapping, str):
            candidate_paths.append((mapping, False))

        if forced_candidate_paths is not None:
            candidate_paths = forced_candidate_paths
        elif not candidate_paths:
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
