"""Explainable generated-HLS project report helpers.

These reports are intentionally evidence-based: they summarize what the
compiler generated and point to concrete files/strings that a reviewer can
inspect.  They do not claim numeric/HLS/Vivado success unless the corresponding
artifacts exist elsewhere.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _communication_kinds(communication_plan: Any) -> dict[str, Any]:
    edges = list(_get(communication_plan, "edges", []) or [])
    result: dict[str, Any] = {"num_edges": len(edges), "edges": []}
    for edge in edges:
        result["edges"].append(
            {
                "tensor_name": _get(edge, "tensor_name"),
                "direction": _get(edge, "direction"),
                "interface": _get(edge, "interface", _get(edge, "encoding")),
                "transport": _get(edge, "transport"),
                "policy": _get(edge, "policy"),
            }
        )
    return result


def _memory_summary(memory_plan: Any) -> dict[str, Any]:
    placements = list(_get(memory_plan, "placements", []) or [])
    notes = dict(_get(memory_plan, "notes", {}) or {})
    return {
        "num_placements": len(placements),
        "total_bytes_by_region": dict(_get(memory_plan, "total_bytes_by_region", {}) or {}),
        "notes": notes,
        "placements": [
            {
                "tensor_name": _get(p, "tensor_name"),
                "region": _get(p, "region"),
                "bytes": _get(p, "nbytes", _get(p, "bytes")),
                "reason": _get(p, "reason"),
            }
            for p in placements[:50]
        ],
    }


def _source_evidence(source: str) -> dict[str, Any]:
    checks = {
        "top_function_deeplearn": "void deeplearn(" in source,
        "runtime_mode_run_inference": "FPGAI_MODE_RUN_INFERENCE" in source,
        "runtime_mode_import_weights": "FPGAI_MODE_IMPORT_WEIGHTS" in source,
        "runtime_mode_export_weights": "FPGAI_MODE_EXPORT_WEIGHTS" in source,
        "runtime_mode_run_training": "FPGAI_MODE_RUN_TRAINING" in source,
        "runtime_mode_export_gradients": "FPGAI_MODE_EXPORT_GRADIENTS" in source,
        "axi_stream_read": ".read()" in source or "in_stream.read()" in source,
        "axi_stream_write": ".write(" in source or "out_stream.write" in source,
        "m_axi_weight_port": "m_axi port=weights_mem" in source or "weights_mem" in source,
        "m_axi_gradient_port": "m_axi port=gradients_mem" in source,
        "m_axi_optimizer_state_port": "m_axi port=optimizer_state_mem" in source,
        "bram_binding": "impl=bram" in source,
        "uram_binding": "impl=uram" in source,
        "ddr_tiling": "tile_base" in source and ("weight_tile" in source or "gradient_export_tile" in source),
        "axis_tiled_tlast": "packet.last" in source and "tile_base" in source,
    }
    return {"checks": checks, "present": sorted([k for k, v in checks.items() if v])}


def _sequence_commands(runtime_sequence: Any) -> set[str]:
    seq = runtime_sequence
    if isinstance(seq, dict):
        seq = seq.get("sequence", [])
    commands: set[str] = set()
    if not isinstance(seq, list):
        return commands
    for item in seq:
        if isinstance(item, str):
            commands.add(item)
        elif isinstance(item, dict):
            for key in item.keys():
                commands.add(str(key))
    return commands


def _requested_source_features(
    *,
    raw_config: dict[str, Any] | None,
    pipeline_mode: str,
    runtime_sequence: Any,
    source: str,
) -> dict[str, bool]:
    commands = _sequence_commands(runtime_sequence)
    raw = raw_config or {}
    weights_mode = str(_get(raw.get("weights", {}), "mode", "") or "").strip().lower().replace("-", "_")
    training = raw.get("training", {}) if isinstance(raw.get("training", {}), dict) else {}
    data_movement = raw.get("data_movement", {}) if isinstance(raw.get("data_movement", {}), dict) else {}
    gradients_dm = data_movement.get("gradients", {}) if isinstance(data_movement.get("gradients", {}), dict) else {}
    optimizer_dm = data_movement.get("optimizer_state", {}) if isinstance(data_movement.get("optimizer_state", {}), dict) else {}
    optimizer = training.get("optimizer", {}) if isinstance(training.get("optimizer", {}), dict) else {}
    loss = training.get("loss", {}) if isinstance(training.get("loss", {}), dict) else {}

    import_weights = (
        "import_weights" in commands
        or weights_mode in {
            "import",
            "import_export",
            "bram_import_full",
            "bram_import_export_full",
            "uram_import_full",
            "uram_import_export_full",
            "ddr_tiled",
            "ddr_tiled_mutable",
            "tiled",
            "tiled_mutable",
        }
    )
    export_weights = (
        "export_weights" in commands
        or weights_mode in {
            "import_export",
            "bram_import_export_full",
            "uram_import_export_full",
            "ddr_tiled_mutable",
            "tiled_mutable",
        }
    )

    # A plain embedded inference top does not need a runtime mode switch.  Requiring
    # FPGAI_MODE_RUN_INFERENCE for that case would push unnecessary mode plumbing
    # into the generated C++, which is exactly what this report is meant to prevent.
    run_inference_mode = pipeline_mode != "training_on_device" and (import_weights or export_weights)

    return {
        "top_function": True,
        "run_inference_mode": run_inference_mode,
        "run_training_mode": pipeline_mode == "training_on_device",
        "import_weights_mode": import_weights,
        "export_weights_mode": export_weights,
        "export_gradients_mode": "export_gradients" in commands or bool(gradients_dm.get("export")),
        "export_optimizer_state_mode": "export_optimizer_state" in commands or bool(optimizer_dm.get("export")),
        "momentum_optimizer": str(optimizer.get("type", "")).lower() == "momentum",
        "adam_optimizer": str(optimizer.get("type", "")).lower() == "adam",
        "cross_entropy_loss": str(loss.get("type", "")).lower() == "cross_entropy" or "cross_entropy" in source,
    }


def _readability_level(raw_config: dict[str, Any] | None) -> str:
    raw = raw_config or {}
    codegen = raw.get("codegen", {}) if isinstance(raw.get("codegen", {}), dict) else {}
    level = str(codegen.get("readability", "high") or "high").strip().lower().replace("-", "_")
    return level if level in {"compact", "normal", "high", "debug"} else "high"


def _source_feature_presence(source: str) -> dict[str, bool]:
    def has_exact_mode(name: str) -> bool:
        # Do not let FPGAI_MODE_EXPORT_WEIGHTS_STREAM count as the regular
        # export_weights mode.  The readability/resource hygiene check is about
        # whether the user-requested PS/PL runtime mode is materialized.
        return re.search(rf"\b{name}\b(?!_)", source) is not None

    return {
        "top_function": "void deeplearn(" in source,
        "run_inference_mode": has_exact_mode("FPGAI_MODE_RUN_INFERENCE"),
        "run_training_mode": has_exact_mode("FPGAI_MODE_RUN_TRAINING"),
        "import_weights_mode": has_exact_mode("FPGAI_MODE_IMPORT_WEIGHTS"),
        "export_weights_mode": has_exact_mode("FPGAI_MODE_EXPORT_WEIGHTS"),
        "export_gradients_mode": has_exact_mode("FPGAI_MODE_EXPORT_GRADIENTS"),
        "export_optimizer_state_mode": has_exact_mode("FPGAI_MODE_EXPORT_OPTIMIZER_STATE"),
        "momentum_optimizer": "FPGAI Momentum optimizer update kernel" in source or "FPGAI_MOMENTUM_" in source,
        "adam_optimizer": "FPGAI Adam optimizer update kernel" in source or "FPGAI_ADAM_" in source,
        "cross_entropy_loss": "FPGAI cross_entropy loss kernel" in source or "softmax_denom" in source,
    }


def _emit_cpp_readability_and_validation_reports(
    reports: Path,
    *,
    raw_config: dict[str, Any] | None,
    pipeline_mode: str,
    runtime_sequence: Any,
    source: str,
    source_path: Path,
) -> dict[str, Path]:
    requested = _requested_source_features(
        raw_config=raw_config,
        pipeline_mode=pipeline_mode,
        runtime_sequence=runtime_sequence,
        source=source,
    )
    present = _source_feature_presence(source)
    level = _readability_level(raw_config)
    placeholder_tokens = [
        "TODO",
        "FIXME",
        "not implemented",
        "planning-only",
        "placeholder",
        "future work",
    ]
    lowered_source = source.lower()
    placeholders = [token for token in placeholder_tokens if token.lower() in lowered_source]
    comments = [line for line in source.splitlines() if line.strip().startswith("//")]
    section_markers = [line.strip() for line in comments if "====" in line or "----" in line or "FPGAI generated" in line or "Sections:" in line]

    required_missing = sorted(name for name, wanted in requested.items() if wanted and not present.get(name, False))
    unrequested_present = sorted(
        name for name, wanted in requested.items()
        if not wanted and present.get(name, False) and name not in {"top_function"}
    )
    min_comments = {"compact": 0, "normal": 1, "high": 5, "debug": 5}.get(level, 5)
    readability_checks = {
        "top_source_present": bool(source),
        "top_function_present": present["top_function"],
        "readability_level_resolved": level in {"compact", "normal", "high", "debug"},
        "comment_density_matches_level": len(comments) >= min_comments,
        "section_summary_present": level == "compact" or bool(section_markers),
        "no_placeholder_markers": not placeholders,
        "unrequested_runtime_features_absent": not unrequested_present,
        "requested_runtime_features_present": not required_missing,
    }
    readability_status = "passed" if all(readability_checks.values()) else "failed"
    readability_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "generated_cpp_readability",
        "status": readability_status,
        "source": str(source_path),
        "readability_level": level,
        "comment_line_count": len(comments),
        "section_markers": section_markers[:40],
        "checks": readability_checks,
        "placeholder_markers": placeholders,
        "requested_features": requested,
        "present_features": present,
        "unrequested_present_features": unrequested_present,
        "required_missing_features": required_missing,
        "resource_hygiene_rule": "Unrequested import/export/runtime paths must be absent from generated C++ so they cannot affect readability, latency, or HLS resources.",
    }

    validation_checks = {
        "source_structurally_present": bool(source),
        "top_function_present": present["top_function"],
        "requested_features_present": not required_missing,
        "unrequested_features_absent": not unrequested_present,
        "no_placeholder_markers": not placeholders,
    }
    validation_status = "passed" if all(validation_checks.values()) else "failed"
    validation_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "generated_cpp_validation",
        "status": validation_status,
        "validation_scope": "static_generated_source_contract",
        "tool_status": "not_applicable",
        "source": str(source_path),
        "checks": validation_checks,
        "requested_features": requested,
        "present_features": present,
        "unrequested_present_features": unrequested_present,
        "required_missing_features": required_missing,
        "note": "This report validates generated-source structure only. HLS CSim/synthesis, Vivado, and FPGA execution remain separate validation stages.",
    }

    readability_json = reports / "generated_cpp_readability.json"
    readability_md = reports / "generated_cpp_readability.md"
    validation_json = reports / "generated_cpp_validation.json"
    validation_md = reports / "generated_cpp_validation.md"

    readability_json.write_text(json.dumps(readability_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readability_md.write_text(
        "\n".join([
            "# Generated C++ readability report",
            "",
            f"Status: `{readability_status}`",
            f"Readability level: `{level}`",
            f"Comment lines: `{len(comments)}`",
            "",
            "## Checks",
            *[f"- {name}: `{str(value).lower()}`" for name, value in readability_checks.items()],
            "",
            "## Requested features",
            *[f"- {name}: `{str(value).lower()}` present=`{str(present.get(name, False)).lower()}`" for name, value in sorted(requested.items())],
            "",
            "## Resource hygiene",
            f"- Unrequested generated features: `{', '.join(unrequested_present) if unrequested_present else 'none'}`",
            f"- Missing requested features: `{', '.join(required_missing) if required_missing else 'none'}`",
        ]) + "\n",
        encoding="utf-8",
    )
    validation_json.write_text(json.dumps(validation_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_md.write_text(
        "\n".join([
            "# Generated C++ validation report",
            "",
            f"Status: `{validation_status}`",
            "Scope: `static_generated_source_contract`",
            "Tool status: `not_applicable`",
            "",
            "## Checks",
            *[f"- {name}: `{str(value).lower()}`" for name, value in validation_checks.items()],
            "",
            "## Truth boundary",
            "This report does not claim HLS synthesis, Vivado implementation, or FPGA execution success.",
        ]) + "\n",
        encoding="utf-8",
    )
    return {
        "generated_cpp_readability_json": readability_json,
        "generated_cpp_readability_md": readability_md,
        "generated_cpp_validation_json": validation_json,
        "generated_cpp_validation_md": validation_md,
    }


def emit_generated_hls_explanation_reports(
    out_dir: str | Path,
    *,
    raw_config: dict[str, Any] | None,
    pipeline_mode: str,
    top_name: str,
    hls_dir: str | Path | None,
    build_stages: dict[str, Any] | None,
    runtime_sequence: dict[str, Any] | None,
    memory_plan: Any = None,
    communication_plan: Any = None,
    numeric_validation_artifacts: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Emit explainable HLS/project review artifacts."""

    out = Path(out_dir)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    hls = Path(hls_dir) if hls_dir is not None else out / "hls"
    source_path = hls / "src" / "deeplearn.cpp"
    params_path = hls / "include" / "fpgai_params.h"
    tb_path = hls / "src" / "tb.cpp"
    source = _read_text(source_path)
    cpp_quality_artifacts = _emit_cpp_readability_and_validation_reports(
        reports,
        raw_config=raw_config,
        pipeline_mode=pipeline_mode,
        runtime_sequence=runtime_sequence,
        source=source,
        source_path=source_path,
    )

    generated_files = {
        "top_source": str(source_path),
        "top_source_exists": _path_exists(source_path),
        "params_header": str(params_path),
        "params_header_exists": _path_exists(params_path),
        "testbench": str(tb_path),
        "testbench_exists": _path_exists(tb_path),
        "run_hls_tcl": str(hls / "run_hls.tcl"),
        "run_hls_tcl_exists": _path_exists(hls / "run_hls.tcl"),
    }

    requested = raw_config or {}
    decisions = {
        "pipeline_mode": pipeline_mode,
        "top_name": top_name,
        "board": requested.get("targets", {}).get("board") or requested.get("project", {}).get("board"),
        "precision": requested.get("numerics", requested.get("precision")),
        "memory": requested.get("memory", {}),
        "weights": requested.get("weights", {}),
        "data_movement": requested.get("data_movement", {}),
        "training": requested.get("training", {}) if pipeline_mode == "training_on_device" else None,
        "build_stages": build_stages or requested.get("build", {}).get("stages", {}),
        "runtime_sequence": runtime_sequence,
    }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "generated_hls_explanation",
        "generated_files": generated_files,
        "decisions": decisions,
        "memory_summary": _memory_summary(memory_plan),
        "communication_summary": _communication_kinds(communication_plan),
        "source_evidence": _source_evidence(source),
        "verification_artifacts": {
            key: str(value) for key, value in (numeric_validation_artifacts or {}).items()
        },
        "review_status": "source_present" if generated_files["top_source_exists"] else "source_missing",
    }

    explanation_json = reports / "generated_hls_explanation.json"
    explanation_md = reports / "generated_hls_explanation.md"
    decisions_json = reports / "hardware_design_decisions.json"
    decisions_md = reports / "hardware_design_decisions.md"
    checklist_md = reports / "codegen_review_checklist.md"

    explanation_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evidence = payload["source_evidence"]["checks"]
    explanation_lines = [
        "# Generated HLS explanation",
        "",
        f"- Pipeline mode: `{pipeline_mode}`",
        f"- Top function: `{top_name}`",
        f"- Top source exists: `{str(generated_files['top_source_exists']).lower()}`",
        f"- Testbench exists: `{str(generated_files['testbench_exists']).lower()}`",
        f"- HLS project Tcl exists: `{str(generated_files['run_hls_tcl_exists']).lower()}`",
        "",
        "## Generated files",
    ]
    for key, value in generated_files.items():
        if key.endswith("_exists"):
            continue
        explanation_lines.append(f"- {key}: `{value}`")
    explanation_lines += ["", "## Source evidence"]
    for key, value in sorted(evidence.items()):
        explanation_lines.append(f"- {key}: `{str(bool(value)).lower()}`")
    explanation_lines += [
        "",
        "## Review note",
        "This report explains generated artifacts and source-level evidence. Numeric, HLS, Vivado, and FPGA claims remain governed by their dedicated validation reports.",
    ]
    explanation_md.write_text("\n".join(explanation_lines) + "\n", encoding="utf-8")

    design_payload = {
        "schema_version": 1,
        "artifact_kind": "hardware_design_decisions",
        "pipeline_mode": pipeline_mode,
        "top_name": top_name,
        "memory_summary": payload["memory_summary"],
        "communication_summary": payload["communication_summary"],
        "runtime_sequence": runtime_sequence,
        "source_evidence": payload["source_evidence"],
    }
    decisions_json.write_text(json.dumps(design_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decisions_md.write_text(
        "\n".join(
            [
                "# Hardware design decisions",
                "",
                f"- Pipeline mode: `{pipeline_mode}`",
                f"- Top function: `{top_name}`",
                f"- Memory semantics: `{payload['memory_summary']['notes'].get('memory_semantics_mode')}`",
                f"- Weight storage: `{payload['memory_summary']['notes'].get('resolved_weight_storage')}`",
                f"- Activation storage: `{payload['memory_summary']['notes'].get('resolved_activation_storage')}`",
                f"- Gradient storage: `{payload['memory_summary']['notes'].get('resolved_gradient_storage')}`",
                "",
                "## Communication edges",
                *[
                    f"- {edge.get('tensor_name')}: {edge.get('direction')} interface={edge.get('interface')} transport={edge.get('transport')} policy={edge.get('policy')}"
                    for edge in payload["communication_summary"].get("edges", [])
                ],
                "",
                "## Evidence",
                *[f"- {name}: `{str(value).lower()}`" for name, value in sorted(evidence.items())],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checklist_md.write_text(
        "\n".join(
            [
                "# Codegen review checklist",
                "",
                "- [ ] Does `hls/src/deeplearn.cpp` contain the expected top function?",
                "- [ ] Do HLS interfaces match `data_movement`?",
                "- [ ] Do BRAM/URAM pragmas match `memory.storage`?",
                "- [ ] Are runtime command constants present for the requested sequence?",
                "- [ ] Is import/export command-driven rather than automatic every compute?",
                "- [ ] Does DDR storage avoid full local weight replicas?",
                "- [ ] Are tile buffers actually used when tiled movement is requested?",
                "- [ ] Does the testbench preserve Python/reference comparison artifacts?",
                "- [ ] Did `reports/numeric_validation.json` pass for correctness claims?",
                "- [ ] Are HLS/Vivado reports present before resource/timing claims?",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "generated_hls_explanation_json": explanation_json,
        "generated_hls_explanation_md": explanation_md,
        "hardware_design_decisions_json": decisions_json,
        "hardware_design_decisions_md": decisions_md,
        "codegen_review_checklist_md": checklist_md,
        **cpp_quality_artifacts,
    }
