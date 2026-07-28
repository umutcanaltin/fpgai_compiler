from __future__ import annotations

"""Resolved compiler configuration and planning report helpers."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from fpgai.config.access import get_path
from fpgai.config.contract import build_config_contract_report, render_config_contract_markdown
from fpgai.engine.build_stages import (
    BUILD_STAGE_KEYS as _BUILD_STAGE_KEYS,
    build_stage_summary as _build_stage_summary,
)
from fpgai.engine.training_contracts import (
    _CODEGEN_READABILITY,
    _OPTIMIZER_STATE_STORAGE,
    _RUNTIME_COMMANDS,
    _TRAINING_LOSS_TYPES,
    _TRAINING_OPTIMIZER_TYPES,
    _resolve_codegen_readability,
)

_cfg_get = get_path

def _resolved_toolchain_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return the YAML toolchain section in manifest-safe form.

    External bridge flows need this because they run after compile and may not
    share the user's shell environment. We preserve explicit user configuration
    without inventing success or resolving unavailable tools here.
    """
    tc = raw.get("toolchain", {}) if isinstance(raw.get("toolchain", {}), dict) else {}
    out: Dict[str, Any] = {}
    for name in ("vitis_hls", "vivado"):
        cfg = tc.get(name, {}) if isinstance(tc.get(name, {}), dict) else {}
        if not cfg:
            continue
        allowed = {
            "enabled",
            "settings64",
            "settings",
            "executable",
            "exe",
            "path",
            "version",
        }
        out[name] = {str(k): v for k, v in cfg.items() if str(k) in allowed}
    return out


def _emit_resolved_config_reports(
    out_dir: Path,
    raw: Dict[str, Any],
    *,
    build_stages: Dict[str, bool],
    runtime_sequence: Dict[str, Any],
    memory_plan: Any,
    communication_plan: Any,
    weights_mode: str,
) -> Dict[str, str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "schema_version": 1,
        "pipeline_mode": str(_cfg_get(raw, "pipeline.mode", "inference")),
        "top_kernel_name": str(_cfg_get(raw, "pipeline.outputs.top_kernel_name", "deeplearn")),
        "weights_mode": str(weights_mode),
        "memory_semantics_mode": str(_plan_notes(memory_plan).get("memory_semantics_mode", weights_mode)),
        "build_stages": _build_stage_summary(build_stages),
        "runtime_sequence": runtime_sequence,
        "toolchain": _resolved_toolchain_summary(raw),
        "codegen": {"readability": _resolve_codegen_readability(raw)},
        "memory_plan_notes": _plan_notes(memory_plan),
        "communication_edges": [getattr(edge, "to_dict", lambda: dict(edge))() if not isinstance(edge, dict) else edge for edge in getattr(communication_plan, "edges", [])],
    }
    json_path = reports_dir / "resolved_config.json"
    json_path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    yml_path = reports_dir / "resolved_config.yml"
    try:
        import yaml  # type: ignore
        yml_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    except Exception:
        yml_path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    contract = build_config_contract_report(raw)
    # Preserve the older summary fields while extending the artifact into a real W0-lite audit.
    contract.update({
        "top_level_sections": sorted(list(getattr(__import__("fpgai.config.loader", fromlist=["TOP_LEVEL_SECTIONS_V1"]), "TOP_LEVEL_SECTIONS_V1"))),
        "build_stage_keys": list(_BUILD_STAGE_KEYS),
        "runtime_commands": sorted(_RUNTIME_COMMANDS),
        "codegen_readability": sorted(_CODEGEN_READABILITY),
        "training_optimizer_types": sorted(_TRAINING_OPTIMIZER_TYPES),
        "training_loss_types": sorted(_TRAINING_LOSS_TYPES),
        "optimizer_state_storage": sorted(_OPTIMIZER_STATE_STORAGE),
        "priority_rules": [
            "manual YAML override > policy default > compiler default",
            "manual data_movement overrides user-facing modes",
            "user-facing modes override policy/defaults",
            "unsupported runtime commands reject clearly",
        ],
    })
    contract_path = reports_dir / "config_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    contract_md_path = reports_dir / "config_contract.md"
    contract_md_path.write_text(render_config_contract_markdown(contract), encoding="utf-8")
    return {
        "resolved_config_json": str(json_path),
        "resolved_config_yml": str(yml_path),
        "config_contract_json": str(contract_path),
        "config_contract_md": str(contract_md_path),
    }


def _plan_notes(plan: Any) -> Dict[str, Any]:
    if plan is None:
        return {}
    notes = getattr(plan, "notes", None)
    if isinstance(notes, dict):
        return dict(notes)
    if isinstance(plan, dict):
        notes = plan.get("notes", plan)
        return dict(notes) if isinstance(notes, dict) else {}
    return {}
