from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # Keep this module independent enough for tests/import-time use.
    from fpgai.config.loader import TOP_LEVEL_SECTIONS_V1
except Exception:  # pragma: no cover
    TOP_LEVEL_SECTIONS_V1 = set()


@dataclass(frozen=True)
class KeySpec:
    path: str
    purpose: str
    evidence: Tuple[str, ...] = ()
    prefix: bool = False


CANONICAL_KEYS: Tuple[KeySpec, ...] = (
    KeySpec("version", "Config format version", ("resolved_config",)),
    KeySpec("project.out_dir", "Compiler output directory", ("manifest.out_dir",)),
    KeySpec("project.name", "Human-readable project name", ("resolved_config", "manifest")),
    KeySpec("project.clean", "Clean output directory before compile", ("compiler execution",)),
    KeySpec("project.reproducibility.emit_manifest", "Enable manifest emission", ("manifest.json",)),
    KeySpec("pipeline.mode", "Select inference or on-device training", ("resolved_config.pipeline_mode", "manifest.pipeline_mode")),
    KeySpec("pipeline.outputs.top_kernel_name", "Generated top kernel name", ("resolved_config.top_kernel_name", "manifest.top_kernel_name")),
    KeySpec("pipeline.outputs", "Pipeline output naming and artifact controls", ("resolved_config", "manifest"), prefix=True),
    KeySpec("model.path", "Input model path", ("manifest.model_path",)),
    KeySpec("model.format", "Input model format hint", ("model loader", "model_compatibility")),
    KeySpec("targets.board", "Target board shortcut", ("board_fit", "vivado_bd_validation")),
    KeySpec("targets.platform.board", "Target board", ("board_fit", "vivado_bd_validation")),
    KeySpec("targets.platform.part", "Target FPGA part", ("board_fit", "hls/vivado scripts")),
    KeySpec("targets.platform.clocks", "Target clocks", ("resolved_config", "hls scripts"), prefix=True),
    KeySpec("build.stages", "Requested build stages", ("resolved_config.build_stages", "manifest.build_stages"), prefix=True),
    KeySpec("build.existing_hls_ip", "Allow Vivado handoff from an existing/imported HLS IP without running HLS synthesis", ("vivado_bd_validation", "vivado Tcl")),
    KeySpec("runtime.sequence", "Requested runtime commands", ("runtime_sequence", "generated C++ modes", "runtime package"), prefix=True),
    KeySpec("weights", "Weight movement and mutability controls", ("resolved_config.weights_mode", "data_movement_plan"), prefix=True),
    KeySpec("weights.mode", "Weight storage/movement mode", ("resolved_config.weights_mode", "data_movement_plan")),
    KeySpec("memory.weight_storage", "Weight storage placement", ("memory_plan", "board_fit")),
    KeySpec("memory.activation_storage", "Activation storage placement", ("memory_plan", "board_fit")),
    KeySpec("memory.gradient_storage", "Gradient storage placement", ("training movement reports", "board_fit")),
    KeySpec("memory.optimizer_state_storage", "Optimizer-state storage placement", ("training optimizer reports", "board_fit")),
    KeySpec("data_movement.inputs", "Input PS/PL movement", ("data_movement_plan", "ps_pl_transfer_plan", "movement_contract_validation"), prefix=True),
    KeySpec("data_movement.outputs", "Output PS/PL movement", ("data_movement_plan", "ps_pl_transfer_plan", "movement_contract_validation"), prefix=True),
    KeySpec("data_movement.labels", "Training label movement", ("data_movement_plan", "movement_contract_validation"), prefix=True),
    KeySpec("data_movement.weights", "Runtime weight movement", ("data_movement_plan", "movement_contract_validation"), prefix=True),
    KeySpec("data_movement.gradients", "Gradient export movement", ("data_movement_plan", "movement_contract_validation"), prefix=True),
    KeySpec("data_movement.optimizer_state", "Optimizer-state export movement", ("data_movement_plan", "movement_contract_validation"), prefix=True),
    KeySpec("optimization.pipeline", "Pipeline pragmas/II", ("parallel_pipeline_effect", "generated C++"), prefix=True),
    KeySpec("optimization.parallel", "Parallel/unroll/partition pragmas", ("parallel_pipeline_effect", "generated C++"), prefix=True),
    KeySpec("optimization.tiling", "General tiling controls", ("data_movement_plan", "generated C++"), prefix=True),
    KeySpec("codegen", "Generated code readability and explanation controls", ("generated_cpp_readability",), prefix=True),
    KeySpec("codegen.readability", "Generated C++ readability level", ("generated_cpp_readability",)),
    KeySpec("training.optimizer", "Training optimizer", ("training_optimizer_loss", "numeric_validation"), prefix=True),
    KeySpec("training.loss", "Training loss", ("training_optimizer_loss", "numeric_validation"), prefix=True),
    KeySpec("training.batch", "Training batch controls", ("numeric_validation.batch_accumulation",), prefix=True),
    KeySpec("training.accumulation", "Gradient accumulation controls", ("numeric_validation.batch_accumulation",), prefix=True),
    KeySpec("toolchain.vitis_hls", "Vitis HLS tool execution/import", ("hls_truth_artifacts",), prefix=True),
    KeySpec("toolchain.vivado", "Vivado tool execution/import", ("vivado_truth_artifacts",), prefix=True),
    KeySpec("reports", "Report switches", ("manifest",), prefix=True),
)

DEPRECATED_ALIASES: Dict[str, Dict[str, str]] = {
    "hls.pipeline_ii": {
        "replacement": "optimization.pipeline.ii",
        "reason": "Sprint O made optimization.pipeline.ii the canonical pipeline-II key.",
    },
    "optimization.pipeline_ii": {
        "replacement": "optimization.pipeline.ii",
        "reason": "Nested optimization.pipeline.ii is the canonical public key.",
    },
    "hls.unroll_factor": {
        "replacement": "optimization.parallel.unroll_factor",
        "reason": "Parallelization knobs now live under optimization.parallel.",
    },
    "optimization.parallel_policy": {
        "replacement": "optimization.parallel.policy",
        "reason": "Policy names should be grouped with other parallel controls.",
    },
    "analysis.design_space.policy_name": {
        "replacement": "optimization.parallel.policy",
        "reason": "Design-space policy is an analysis-era alias, not the canonical hardware knob path.",
    },
    "training.batch_size": {
        "replacement": "training.batch.size",
        "reason": "Batch controls should be grouped under training.batch.",
    },
    "training.execution.batch_size": {
        "replacement": "training.batch.size",
        "reason": "Batch controls should be grouped under training.batch.",
    },
    "training.gradient_accumulation.steps": {
        "replacement": "training.accumulation.steps",
        "reason": "Accumulation controls should be grouped under training.accumulation.",
    },
    "training.gradient_accumulation.mode": {
        "replacement": "training.accumulation.mode",
        "reason": "Accumulation controls should be grouped under training.accumulation.",
    },
    "training.storage.optimizer_state": {
        "replacement": "memory.optimizer_state_storage",
        "reason": "Storage placement belongs under memory.*_storage.",
    },
    "data_movement.inputs.import.interface": {
        "replacement": "data_movement.inputs.interface",
        "reason": "Sprint P direct movement style is canonical.",
    },
    "data_movement.inputs.import.transport": {
        "replacement": "data_movement.inputs.transport",
        "reason": "Sprint P direct movement style is canonical.",
    },
    "data_movement.outputs.export.interface": {
        "replacement": "data_movement.outputs.interface",
        "reason": "Sprint P direct movement style is canonical.",
    },
    "data_movement.outputs.export.transport": {
        "replacement": "data_movement.outputs.transport",
        "reason": "Sprint P direct movement style is canonical.",
    },
    "data_movement.ps_pl.weights.mode": {
        "replacement": "weights.mode",
        "reason": "Weight mode is a user-facing weights.* decision, not a nested PS/PL alias.",
    },
    "data_movement.ps_pl.input.mode": {
        "replacement": "data_movement.inputs.interface",
        "reason": "Sprint P direct input movement style is canonical; mode aliases should map to interface/transport.",
    },
    "data_movement.ps_pl.output.mode": {
        "replacement": "data_movement.outputs.interface",
        "reason": "Sprint P direct output movement style is canonical; mode aliases should map to interface/transport.",
    },
    "data_movement.ps_pl.input": {
        "replacement": "data_movement.inputs",
        "reason": "Sprint P direct movement style is canonical.",
    },
    "data_movement.pl_ps.output": {
        "replacement": "data_movement.outputs",
        "reason": "Sprint P direct movement style is canonical.",
    },
    "memory.storage.weights": {
        "replacement": "memory.weight_storage",
        "reason": "Storage placement now uses explicit memory.*_storage keys.",
    },
    "memory.storage.activations": {
        "replacement": "memory.activation_storage",
        "reason": "Storage placement now uses explicit memory.*_storage keys.",
    },
    "memory.storage.gradients": {
        "replacement": "memory.gradient_storage",
        "reason": "Storage placement now uses explicit memory.*_storage keys.",
    },
    "memory.storage.optimizer_state": {
        "replacement": "memory.optimizer_state_storage",
        "reason": "Storage placement now uses explicit memory.*_storage keys.",
    },
    "training.storage.weights": {
        "replacement": "memory.weight_storage",
        "reason": "Training storage placement should use the same memory.*_storage contract.",
    },
    "training.storage.activations": {
        "replacement": "memory.activation_storage",
        "reason": "Training storage placement should use the same memory.*_storage contract.",
    },
    "training.storage.gradients": {
        "replacement": "memory.gradient_storage",
        "reason": "Training storage placement should use the same memory.*_storage contract.",
    },
    "training.execution.epochs": {
        "replacement": "training.batch.epochs",
        "reason": "Execution-era training controls should move under canonical grouped training controls.",
    },
}

INTERNAL_OR_LEGACY_TOP_LEVEL = {
    "analysis",
    "backends",
    "benchmark",
    "communication",
    "debug",
    "metadata",
    "numerics",
    "operators",
}

# Sweep YAML files are not direct compiler configs. They intentionally use
# template-specific roots that materialize into compiler YAML later. W0-lite
# classifies them separately instead of treating them as unknown compiler keys.
SWEEP_TEMPLATE_TOP_LEVEL = {
    "name",
    "command_template",
    "defaults",
    "parameters",
    "design_points",
    "materialize_configs",
    "notes",
    "design_name_template",
    "point_name_template",
    "description",
}

PAPER_ARTIFACT_SPEC_TOP_LEVEL = {
    "claim_levels",
    "inputs",
    "limitations",
    "paper",
    "vivado",
}

CONTAINER_SECTION_PATHS = {
    "project",
    "project.reproducibility",
    "model",
    "pipeline",
    "targets",
    "targets.platform",
    "build",
    "runtime",
    "memory",
    "data_movement",
    "optimization",
    "training",
    "toolchain",
}

# Existing repo configs still contain historical knobs that are accepted by
# older flows or report generators but are not yet canonical public compiler
# keys. They are not marked unknown, so the audit queue focuses on real cleanup.
LEGACY_OR_INTERNAL_PATH_PREFIXES = (
    "data_movement.ps_pl",
    "data_movement.pl_ps",
    "memory.storage",
    "training.execution",
    "training.storage",
    "training.debug",
    "training.cache",
    "training.estimator",
    "training.phase_overrides",
    "optimization.parallel.pipeline_style",
    "toolchain",
)


def flatten_config_paths(data: Any, prefix: str = "", *, include_sequence_items: bool = False) -> List[str]:
    paths: List[str] = []
    if isinstance(data, Mapping):
        if prefix:
            paths.append(prefix)
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(flatten_config_paths(value, child, include_sequence_items=include_sequence_items))
        return paths
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        if prefix:
            paths.append(prefix)
        if include_sequence_items:
            for index, value in enumerate(data):
                paths.extend(
                    flatten_config_paths(
                        value,
                        f"{prefix}.{index}" if prefix else str(index),
                        include_sequence_items=include_sequence_items,
                    )
                )
        return paths
    if prefix:
        paths.append(prefix)
    return paths


def _canonical_match(path: str) -> Optional[KeySpec]:
    best: Optional[KeySpec] = None
    for spec in CANONICAL_KEYS:
        if path == spec.path or (spec.prefix and path.startswith(spec.path + ".")):
            if best is None or len(spec.path) > len(best.path):
                best = spec
    return best


def _alias_for(path: str) -> Optional[Tuple[str, Dict[str, str]]]:
    if path in DEPRECATED_ALIASES:
        return path, DEPRECATED_ALIASES[path]
    # Treat descendants of deprecated mapping aliases as deprecated too.
    best_key: Optional[str] = None
    for alias in DEPRECATED_ALIASES:
        if path.startswith(alias + ".") and (best_key is None or len(alias) > len(best_key)):
            best_key = alias
    if best_key is not None:
        meta = dict(DEPRECATED_ALIASES[best_key])
        suffix = path[len(best_key) :]
        meta["replacement"] = meta.get("replacement", "") + suffix
        return best_key, meta
    return None


def classify_config_path(path: str) -> Dict[str, Any]:
    alias = _alias_for(path)
    if alias is not None:
        alias_path, meta = alias
        return {
            "path": path,
            "status": "deprecated_alias",
            "alias_root": alias_path,
            "replacement": meta.get("replacement"),
            "reason": meta.get("reason"),
        }
    spec = _canonical_match(path)
    if spec is not None:
        return {
            "path": path,
            "status": "canonical",
            "canonical_root": spec.path,
            "purpose": spec.purpose,
            "evidence": list(spec.evidence),
        }
    top = path.split(".", 1)[0]
    if top in SWEEP_TEMPLATE_TOP_LEVEL:
        return {
            "path": path,
            "status": "sweep_template",
            "reason": "Sweep/materialization YAML key; not a direct compiler config key.",
        }
    if top in PAPER_ARTIFACT_SPEC_TOP_LEVEL:
        return {
            "path": path,
            "status": "paper_artifact_spec",
            "reason": "Paper/report aggregation YAML key; not a direct compiler config key.",
        }
    if path in CONTAINER_SECTION_PATHS:
        return {
            "path": path,
            "status": "section_container",
            "reason": "Container section; child keys carry the actionable compiler contract.",
        }
    if any(path == prefix or path.startswith(prefix + ".") for prefix in LEGACY_OR_INTERNAL_PATH_PREFIXES):
        return {
            "path": path,
            "status": "legacy_or_internal",
            "reason": "Historical or auxiliary config path kept for compatibility; W0/W should migrate or document it.",
        }
    if top not in TOP_LEVEL_SECTIONS_V1:
        return {
            "path": path,
            "status": "unknown_top_level",
            "reason": f"Top-level section {top!r} is not part of the accepted v1 section list.",
        }
    if top in INTERNAL_OR_LEGACY_TOP_LEVEL:
        return {
            "path": path,
            "status": "legacy_or_internal",
            "reason": "Accepted by the historical loader, but not yet selected as a W0-lite canonical public key.",
        }
    return {
        "path": path,
        "status": "unclassified_known_section",
        "reason": "Known top-level section, but this exact key still needs canonical/deprecated/rejected classification.",
    }


def build_config_contract_report(raw: Dict[str, Any]) -> Dict[str, Any]:
    paths = sorted(set(flatten_config_paths(raw)))
    classified = [classify_config_path(path) for path in paths]
    by_status: Dict[str, List[str]] = {}
    for item in classified:
        by_status.setdefault(str(item["status"]), []).append(str(item["path"]))
    deprecated = [item for item in classified if item.get("status") == "deprecated_alias"]
    unknown = [
        item
        for item in classified
        if item.get("status") in {"unknown_top_level", "unclassified_known_section"}
    ]
    canonical_sources = [item for item in classified if item.get("status") == "canonical"]
    canonical_paths = [spec.path for spec in CANONICAL_KEYS]
    return {
        "schema_version": 2,
        "artifact_kind": "config_contract",
        "status": "audit_only",
        "passed": True,
        "blocking_failure": False,
        "policy": {
            "priority_order": "manual YAML override > policy default > compiler default",
            "w0_lite_scope": "Report canonical/deprecated/unknown keys without global unknown-key rejection yet.",
            "future_w_scope": "Unknown-key rejection and legacy alias removal happen in later YAML cleanup sprints.",
        },
        "summary": {
            "total_paths": len(paths),
            "canonical_paths_used": len(canonical_sources),
            "deprecated_aliases_used": len(deprecated),
            "unknown_or_unclassified_paths": len(unknown),
            "statuses": {key: len(value) for key, value in sorted(by_status.items())},
        },
        "canonical_keys": [
            {
                "path": spec.path,
                "prefix": spec.prefix,
                "purpose": spec.purpose,
                "evidence": list(spec.evidence),
            }
            for spec in CANONICAL_KEYS
        ],
        "canonical_key_paths": canonical_paths,
        "deprecated_aliases": [
            {"path": path, **meta}
            for path, meta in sorted(DEPRECATED_ALIASES.items())
        ],
        "manual_yaml_sources": classified,
        "paths_by_status": {key: sorted(value) for key, value in sorted(by_status.items())},
        "warnings": [
            f"Deprecated alias {item['path']} should migrate to {item.get('replacement')}"
            for item in deprecated
        ]
        + [
            f"Unclassified/unknown config key {item['path']}: {item.get('reason')}"
            for item in unknown
        ],
    }


def render_config_contract_markdown(report: Dict[str, Any]) -> str:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    lines = [
        "# Config contract audit",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- scope: `{report.get('policy', {}).get('w0_lite_scope', '')}`",
        f"- total YAML paths: `{summary.get('total_paths', 0)}`",
        f"- canonical paths used: `{summary.get('canonical_paths_used', 0)}`",
        f"- deprecated aliases used: `{summary.get('deprecated_aliases_used', 0)}`",
        f"- unknown/unclassified paths: `{summary.get('unknown_or_unclassified_paths', 0)}`",
        "",
        "## Priority rule",
        "",
        f"`{report.get('policy', {}).get('priority_order', '')}`",
        "",
        "## Deprecated aliases used in this config",
        "",
    ]
    deprecated_used = [item for item in report.get("manual_yaml_sources", []) if item.get("status") == "deprecated_alias"]
    if not deprecated_used:
        lines.append("None.")
    else:
        for item in deprecated_used:
            lines.append(f"- `{item.get('path')}` → `{item.get('replacement')}` — {item.get('reason', '')}")
    lines += ["", "## Unknown or unclassified keys in this config", ""]
    unknown = [
        item
        for item in report.get("manual_yaml_sources", [])
        if item.get("status") in {"unknown_top_level", "unclassified_known_section"}
    ]
    if not unknown:
        lines.append("None.")
    else:
        for item in unknown:
            lines.append(f"- `{item.get('path')}` — {item.get('reason', '')}")
    lines += ["", "## Canonical public key roots", ""]
    for item in report.get("canonical_keys", []):
        suffix = ".*" if item.get("prefix") else ""
        evidence = ", ".join(item.get("evidence", []))
        lines.append(f"- `{item.get('path')}{suffix}` — {item.get('purpose', '')}; evidence: {evidence}")
    return "\n".join(lines) + "\n"

def build_repo_yaml_audit_report(root: str | Path, *, relative_dirs: Sequence[str] = ("configs", "examples")) -> Dict[str, Any]:
    """Audit YAML files in the repository without changing compile behavior.

    W0-lite scope: classify existing config/example YAML paths with the same
    contract used for a single compile. This is intentionally report-only;
    later W0/W sprints can turn selected findings into migration or rejection.
    """
    root_path = Path(root)
    yaml_files: List[Path] = []
    for rel in relative_dirs:
        base = root_path / rel
        if not base.exists():
            continue
        yaml_files.extend(sorted(base.rglob("*.yml")))
        yaml_files.extend(sorted(base.rglob("*.yaml")))

    file_reports: List[Dict[str, Any]] = []
    aggregate_statuses: Dict[str, int] = {}
    aggregate_paths: Dict[str, int] = {}
    unreadable: List[Dict[str, str]] = []

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - PyYAML is available in test envs.
        return {
            "schema_version": 2,
            "artifact_kind": "repo_yaml_schema_audit",
            "status": "tool_missing",
            "passed": False,
            "blocking_failure": False,
            "reason": f"PyYAML unavailable: {exc}",
            "files": [],
            "summary": {"files_scanned": 0},
        }

    for path in sorted(set(yaml_files)):
        rel_path = path.relative_to(root_path).as_posix()
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            unreadable.append({"path": rel_path, "error": str(exc)})
            continue
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            unreadable.append({"path": rel_path, "error": "Top-level YAML document is not a mapping."})
            continue
        contract = build_config_contract_report(dict(loaded))
        statuses = contract.get("summary", {}).get("statuses", {})
        for status, count in statuses.items():
            aggregate_statuses[str(status)] = aggregate_statuses.get(str(status), 0) + int(count)
        for item in contract.get("manual_yaml_sources", []):
            item_path = str(item.get("path", ""))
            if item_path:
                aggregate_paths[item_path] = aggregate_paths.get(item_path, 0) + 1
        file_reports.append({
            "path": rel_path,
            "summary": contract.get("summary", {}),
            "status_counts": statuses,
            "deprecated_aliases": [
                {
                    "path": item.get("path"),
                    "replacement": item.get("replacement"),
                    "reason": item.get("reason"),
                }
                for item in contract.get("manual_yaml_sources", [])
                if item.get("status") == "deprecated_alias"
            ],
            "unknown_or_unclassified": [
                {
                    "path": item.get("path"),
                    "status": item.get("status"),
                    "reason": item.get("reason"),
                }
                for item in contract.get("manual_yaml_sources", [])
                if item.get("status") in {"unknown_top_level", "unclassified_known_section"}
            ],
        })

    files_with_deprecated = [f["path"] for f in file_reports if f.get("deprecated_aliases")]
    files_with_unknown = [f["path"] for f in file_reports if f.get("unknown_or_unclassified")]
    most_common_paths = [
        {"path": path, "files": count}
        for path, count in sorted(aggregate_paths.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
    ]
    migration_queue: Dict[str, int] = {}
    for file_report in file_reports:
        for item in file_report.get("deprecated_aliases", []):
            replacement = str(item.get("replacement") or "")
            key = f"{item.get('path')} -> {replacement}" if replacement else str(item.get("path"))
            migration_queue[key] = migration_queue.get(key, 0) + 1
    migration_queue_items = [
        {"mapping": key, "files": count}
        for key, count in sorted(migration_queue.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
    ]
    return {
        "schema_version": 2,
        "artifact_kind": "repo_yaml_schema_audit",
        "status": "audit_only",
        "passed": True,
        "blocking_failure": False,
        "policy": {
            "scope": "W0-lite repo-level audit of configs/ and examples/ YAML files.",
            "priority_order": "manual YAML override > policy default > compiler default",
            "future_w_scope": "Unknown-key rejection and legacy alias removal happen in later YAML cleanup sprints.",
        },
        "summary": {
            "files_scanned": len(file_reports),
            "unreadable_files": len(unreadable),
            "files_with_deprecated_aliases": len(files_with_deprecated),
            "files_with_unknown_or_unclassified_keys": len(files_with_unknown),
            "aggregate_statuses": dict(sorted(aggregate_statuses.items())),
        },
        "files_with_deprecated_aliases": files_with_deprecated,
        "files_with_unknown_or_unclassified_keys": files_with_unknown,
        "unreadable": unreadable,
        "most_common_paths": most_common_paths,
        "migration_queue": migration_queue_items,
        "files": file_reports,
    }


def render_repo_yaml_audit_markdown(report: Dict[str, Any]) -> str:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    lines = [
        "# Repository YAML schema audit",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- scope: `{report.get('policy', {}).get('scope', '')}`",
        f"- files scanned: `{summary.get('files_scanned', 0)}`",
        f"- files with deprecated aliases: `{summary.get('files_with_deprecated_aliases', 0)}`",
        f"- files with unknown/unclassified keys: `{summary.get('files_with_unknown_or_unclassified_keys', 0)}`",
        f"- unreadable files: `{summary.get('unreadable_files', 0)}`",
        "",
        "## Aggregate status counts",
        "",
    ]
    statuses = summary.get("aggregate_statuses", {}) if isinstance(summary, dict) else {}
    if not statuses:
        lines.append("None.")
    else:
        for status, count in sorted(statuses.items()):
            lines.append(f"- `{status}`: `{count}`")

    lines += ["", "## Files with deprecated aliases", ""]
    deprecated_files = report.get("files_with_deprecated_aliases", [])
    if not deprecated_files:
        lines.append("None.")
    else:
        for path in deprecated_files:
            lines.append(f"- `{path}`")

    lines += ["", "## Files with unknown or unclassified keys", ""]
    unknown_files = report.get("files_with_unknown_or_unclassified_keys", [])
    if not unknown_files:
        lines.append("None.")
    else:
        for path in unknown_files:
            lines.append(f"- `{path}`")

    lines += ["", "## Migration queue", ""]
    queue = report.get("migration_queue", [])
    if not queue:
        lines.append("None.")
    else:
        for item in queue[:25]:
            lines.append(f"- `{item.get('mapping')}` — `{item.get('files')}` files")

    lines += ["", "## Per-file findings", ""]
    for file_report in report.get("files", []):
        lines.append(f"### `{file_report.get('path')}`")
        status_counts = file_report.get("status_counts", {})
        if status_counts:
            joined = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
            lines.append(f"- statuses: {joined}")
        deprecated = file_report.get("deprecated_aliases", [])
        unknown = file_report.get("unknown_or_unclassified", [])
        if deprecated:
            lines.append(f"- deprecated aliases: `{len(deprecated)}`")
            for item in deprecated[:12]:
                lines.append(f"  - `{item.get('path')}` → `{item.get('replacement')}`")
            if len(deprecated) > 12:
                lines.append(f"  - ... `{len(deprecated) - 12}` more")
        if unknown:
            lines.append(f"- unknown/unclassified: `{len(unknown)}`")
            for item in unknown[:20]:
                lines.append(f"  - `{item.get('path')}` ({item.get('status')}) — {item.get('reason', '')}")
            if len(unknown) > 20:
                lines.append(f"  - ... `{len(unknown) - 20}` more")
        if not deprecated and not unknown:
            lines.append("- no deprecated or unknown/unclassified keys detected by W0-lite.")
        lines.append("")

    lines += ["## Most common YAML paths", ""]
    for item in report.get("most_common_paths", [])[:25]:
        lines.append(f"- `{item.get('path')}` — `{item.get('files')}` files")
    return "\n".join(lines).rstrip() + "\n"
