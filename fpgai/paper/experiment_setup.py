from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any, Iterable

import yaml


def _repo_root_from(path: Path) -> Path:
    current = path.resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "fpgai").exists() and (candidate / "examples").exists():
            return candidate
    return Path.cwd()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to read YAML file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}")
    return payload


def _get_path(obj: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = obj
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _bool_stage(cfg: dict[str, Any], stage: str) -> bool:
    return bool(_get_path(cfg, f"build.stages.{stage}", False))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe_text(row.get(key)) for key in fieldnames})


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        values = [_safe_text(row.get(field)).replace("|", "\\|") for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _artifact_bool(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key) is True or str(row.get(key)).lower() == "true")


def _current_artifact_row(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    direct = _read_json(out / "reports" / "paper_experiment_row.json")
    if direct:
        return direct
    matrix = _read_json(out / "reports" / "experiment_artifact_matrix.json")
    rows = matrix.get("rows") if isinstance(matrix, dict) else None
    if isinstance(rows, list) and rows:
        first = rows[0]
        return first if isinstance(first, dict) else {}
    manifest = _read_json(out / "manifest.json")
    vivado = manifest.get("vivado_bridge", {}) if isinstance(manifest.get("vivado_bridge"), dict) else {}
    return {
        "source_generated": bool((out / "hls" / "src" / "deeplearn.cpp").exists() or manifest.get("hls_artifacts")),
        "hls_ok": bool(manifest.get("hls_ok") is True or (isinstance(manifest.get("hls_artifacts"), dict) and manifest["hls_artifacts"].get("hls_ok") is True)),
        "vivado_implemented": bool(vivado.get("vivado_impl_requested") and not vivado.get("failed_rows") and vivado.get("ok", True)),
        "bitstream_generated": bool(vivado.get("bitstream_exists")),
        "runtime_package_created": bool((out / "runtime_package" / "package_manifest.json").exists()),
        "runtime_package_validated": bool(_read_json(out / "runtime_package" / "runtime_package_validation.json").get("deployability_ready")),
        "board_execution_passed": False,
        "claim_level": "not_generated",
    }


def validate_compile_artifacts(
    out_dir: str | Path,
    *,
    configured_stage: str,
    expected_claim_level: str | None = None,
) -> dict[str, Any]:
    """Validate that a compile row reached the artifact stage it was configured to build.

    This is intentionally stricter than process exit status.  Some compile flows
    still emit reports after a tool failure for inspection, but paper experiment
    plans must mark the row as failed if the requested HLS/Vivado/bitstream
    artifact status is not satisfied.
    """
    out = Path(out_dir)
    row = _current_artifact_row(out)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, actual: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual})

    add("manifest_or_artifact_report_present", bool(row), str(out))
    stage = configured_stage or "compiler_artifact"
    if stage in {"compiler_artifact", "hls_synthesis", "vivado_implementation", "bitstream_package"}:
        add("source_generated", _artifact_bool(row, "source_generated"), row.get("source_generated"))
    if stage in {"hls_synthesis", "vivado_implementation", "bitstream_package"}:
        add("hls_ok", _artifact_bool(row, "hls_ok"), row.get("hls_ok"))
    if stage in {"vivado_implementation", "bitstream_package"}:
        add("vivado_implemented", _artifact_bool(row, "vivado_implemented"), row.get("vivado_implemented"))
    if stage == "bitstream_package":
        add("bitstream_generated", _artifact_bool(row, "bitstream_generated"), row.get("bitstream_generated"))
        add("runtime_package_created", _artifact_bool(row, "runtime_package_created"), row.get("runtime_package_created"))
        add("runtime_package_validated", _artifact_bool(row, "runtime_package_validated"), row.get("runtime_package_validated"))

    passed = all(check["passed"] for check in checks)
    payload = {
        "status": "passed" if passed else "failed",
        "out_dir": str(out),
        "configured_stage": configured_stage,
        "expected_claim_level": expected_claim_level,
        "claim_level": row.get("claim_level"),
        "checks": checks,
        "failed_checks": [check for check in checks if not check["passed"]],
    }
    return payload

def _compile_command(config_path: str) -> str:
    # Keep command generation centralized so every markdown/csv/shell plan preserves
    # the required token boundary between `compile` and `--config`.
    return " ".join(["python", "-m", "fpgai.cli", "compile", "--config", shlex.quote(str(config_path))])


def _compile_command_valid(command: str) -> bool:
    return "python -m fpgai.cli compile --config " in command and "compile--config" not in command


def _write_command_script(path: Path, *, title: str, rows: list[dict[str, Any]], extra_lines: list[str] | None = None) -> None:
    """Write a resilient shell plan for paper experiment compilation.

    Paper matrices should not stop at the first failed design: a single Vivado
    implementation failure must be recorded as a failed row while the remaining
    experiments continue.  The script therefore avoids `set -e`, captures each
    command's exit status, writes per-row logs, and exits non-zero only after all
    selected rows have been attempted.
    """
    lines = [f"# {title}", ""]
    lines.extend([
        "set -u",
        "set -o pipefail",
        "FPGAI_EXPERIMENT_SETUP_DIR=${FPGAI_EXPERIMENT_SETUP_DIR:-paper_results/experiment_setup}",
        "FPGAI_COMPILE_LOG_DIR=\"$FPGAI_EXPERIMENT_SETUP_DIR/compile_logs\"",
        "mkdir -p \"$FPGAI_COMPILE_LOG_DIR\"",
        "FPGAI_COMPILE_STATUS_TSV=\"$FPGAI_COMPILE_LOG_DIR/compile_status.tsv\"",
        "printf 'order\\tid\\tstatus\\tlog\\n' > \"$FPGAI_COMPILE_STATUS_TSV\"",
        "FPGAI_COMPILE_FAILURES=0",
        'run_fpgai_compile() {',
        '  local order="$1"',
        '  local experiment_id="$2"',
        '  local configured_stage="$3"',
        '  local expected_claim_level="$4"',
        '  local project_out_dir="$5"',
        '  shift 5',
        '  local safe_id="${experiment_id//[^A-Za-z0-9_]/_}"',
        '  local log="$FPGAI_COMPILE_LOG_DIR/${order}_${safe_id}.log"',
        '  echo "[FPGAI][RUN] ${order} ${experiment_id}: $*"',
        '  "$@" 2>&1 | tee "$log"',
        '  local command_status=${PIPESTATUS[0]}',
        '  local artifact_status=0',
        '  if [ "$command_status" -eq 0 ]; then',
        '    python -m fpgai.paper.experiment_setup --validate-artifact "$project_out_dir" --configured-stage "$configured_stage" --expected-claim-level "$expected_claim_level" >> "$log" 2>&1',
        '    artifact_status=$?',
        '  fi',
        '  local status=$command_status',
        '  if [ "$status" -eq 0 ] && [ "$artifact_status" -ne 0 ]; then',
        '    status=$artifact_status',
        '  fi',
        '  printf \'%s\\t%s\\t%s\\t%s\\n\' "$order" "$experiment_id" "$status" "$log" >> "$FPGAI_COMPILE_STATUS_TSV"',
        '  if [ "$status" -ne 0 ]; then',
        '    echo "[FPGAI][WARN] ${experiment_id} failed with status ${status}; continuing to remaining paper rows."',
        '    FPGAI_COMPILE_FAILURES=$((FPGAI_COMPILE_FAILURES + 1))',
        '  fi',
        '}',
        "",
    ])
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    for row in rows:
        order = f"{int(row['order']):02d}"
        experiment_id = str(row["id"])
        config = str(row["config"])
        lines.append(f"# {order} {experiment_id} | {row['section']} | {row['group']} | {row['expected_claim_level']}")
        lines.append(
            "run_fpgai_compile "
            + " ".join(
                [
                    shlex.quote(order),
                    shlex.quote(experiment_id),
                    shlex.quote(str(row.get("configured_stage") or "")),
                    shlex.quote(str(row.get("expected_claim_level") or "")),
                    shlex.quote(str(row.get("project_out_dir") or "")),
                    "python",
                    "-m",
                    "fpgai.cli",
                    "compile",
                    "--config",
                    shlex.quote(config),
                ]
            )
        )
        lines.append("")
    lines.extend([
        "echo \"[FPGAI][SUMMARY] Compile status table: $FPGAI_COMPILE_STATUS_TSV\"",
        "if [ \"$FPGAI_COMPILE_FAILURES\" -ne 0 ]; then",
        "  echo \"[FPGAI][SUMMARY] $FPGAI_COMPILE_FAILURES paper compile row(s) failed. Inspect $FPGAI_COMPILE_LOG_DIR.\"",
        "  exit 1",
        "fi",
        "echo \"[FPGAI][SUMMARY] All selected paper compile rows passed.\"",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def _expected_stage_from_claim(claim_level: str) -> str:
    if claim_level == "level_4_board_execution":
        return "board_runtime"
    if claim_level == "level_3_bitstream_package":
        return "bitstream_package"
    if claim_level == "level_2_vivado_implementation":
        return "vivado_implementation"
    if claim_level == "level_1_hls_synthesis":
        return "hls_synthesis"
    return "compiler_artifact"


def _actual_stage_from_config(cfg: dict[str, Any]) -> str:
    if _bool_stage(cfg, "bitstream"):
        return "bitstream_package"
    if _bool_stage(cfg, "vivado_implementation"):
        return "vivado_implementation"
    if _bool_stage(cfg, "hls_synthesis"):
        return "hls_synthesis"
    if _bool_stage(cfg, "cpp"):
        return "compiler_artifact"
    return "unconfigured"


def _section_expected_mode(section: str) -> str | None:
    if section == "inference":
        return "inference"
    if section == "training":
        return "training_on_device"
    return None


def _row_from_experiment(
    *,
    repo_root: Path,
    row: dict[str, Any],
    index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    config = str(row.get("config") or "")
    cfg_path = repo_root / config
    cfg: dict[str, Any] = {}
    config_exists = cfg_path.exists()
    config_loadable = False
    config_valid = False
    if config_exists:
        try:
            cfg = _load_yaml(cfg_path)
            config_loadable = True
        except Exception as exc:
            issues.append({"id": row.get("id"), "severity": "error", "message": str(exc)})
        if config_loadable:
            try:
                from fpgai.config.loader import load_config

                load_config(str(cfg_path))
                config_valid = True
            except Exception as exc:
                issues.append({
                    "id": row.get("id"),
                    "severity": "error",
                    "message": f"config validation failed: {exc}",
                })
    else:
        issues.append({"id": row.get("id"), "severity": "error", "message": f"config does not exist: {config}"})

    section = str(row.get("section") or "")
    pipeline_mode = str(_get_path(cfg, "pipeline.mode", "")) if cfg else ""
    expected_mode = _section_expected_mode(section)
    if expected_mode and pipeline_mode and pipeline_mode != expected_mode:
        issues.append({
            "id": row.get("id"),
            "severity": "error",
            "message": f"section {section!r} expects pipeline.mode={expected_mode!r}, got {pipeline_mode!r}",
        })

    metadata_id = _get_path(cfg, "metadata.paper_experiment.id", None) if cfg else None
    if config_loadable and metadata_id != row.get("id"):
        issues.append({
            "id": row.get("id"),
            "severity": "warning",
            "message": "metadata.paper_experiment.id does not match matrix id",
        })

    claim_level = str(row.get("expected_claim_level") or "")
    expected_stage = _expected_stage_from_claim(claim_level)
    actual_stage = _actual_stage_from_config(cfg) if cfg else "missing"
    if expected_stage == "bitstream_package" and actual_stage != "bitstream_package":
        issues.append({"id": row.get("id"), "severity": "error", "message": "level_3 row must enable build.stages.bitstream"})
    if expected_stage == "vivado_implementation" and actual_stage not in {"vivado_implementation", "bitstream_package"}:
        issues.append({"id": row.get("id"), "severity": "error", "message": "level_2 row must enable Vivado implementation"})
    if expected_stage == "board_runtime" and actual_stage != "bitstream_package":
        issues.append({"id": row.get("id"), "severity": "error", "message": "level_4 candidate must at least generate a bitstream package before runtime"})
    if expected_stage == "board_runtime" and not row.get("runtime_measurements_required"):
        issues.append({"id": row.get("id"), "severity": "error", "message": "level_4 candidate must declare required runtime measurements"})

    command = _compile_command(config)
    result = {
        "order": index,
        "id": row.get("id"),
        "section": section,
        "group": row.get("group"),
        "knob_axis": row.get("knob_axis"),
        "purpose": row.get("purpose"),
        "config": config,
        "config_exists": config_exists,
        "config_loadable": config_loadable,
        "config_valid": config_valid,
        "project_name": _get_path(cfg, "project.name", ""),
        "project_out_dir": _get_path(cfg, "project.out_dir", ""),
        "model": _get_path(cfg, "model.path", row.get("model", "")),
        "pipeline_mode": pipeline_mode,
        "board": _get_path(cfg, "targets.platform.board", ""),
        "part": _get_path(cfg, "targets.platform.part", ""),
        "expected_claim_level": claim_level,
        "expected_stage": expected_stage,
        "configured_stage": actual_stage,
        "compile_profile": row.get("compile_profile", actual_stage),
        "hls_synthesis": _bool_stage(cfg, "hls_synthesis"),
        "vivado_implementation": _bool_stage(cfg, "vivado_implementation"),
        "bitstream": _bool_stage(cfg, "bitstream"),
        "runtime_package": _bool_stage(cfg, "runtime_package"),
        "precision_activation": _precision_label(_get_path(cfg, "numerics.defaults.activation", None)),
        "precision_weight": _precision_label(_get_path(cfg, "numerics.defaults.weight", None)),
        "pipeline_style": _get_path(cfg, "optimization.pipeline.style", ""),
        "pipeline_ii": _get_path(cfg, "optimization.pipeline.ii", ""),
        "pe": _get_path(cfg, "optimization.parallel.pe", ""),
        "simd": _get_path(cfg, "optimization.parallel.simd", ""),
        "unroll_factor": _get_path(cfg, "optimization.parallel.unroll_factor", ""),
        "partition_factor": _get_path(cfg, "optimization.parallel.partition_factor", ""),
        "weight_mode": _get_path(cfg, "weights.mode", _get_path(cfg, "data_movement.ps_pl.weights.mode", "")),
        "training_optimizer": _get_path(cfg, "training.optimizer.type", ""),
        "training_loss": _get_path(cfg, "training.loss.type", ""),
        "training_batch_size": _get_path(cfg, "training.batch.size", ""),
        "training_accumulation_steps": _get_path(cfg, "training.accumulation.steps", ""),
        "runtime_measurements_required": row.get("runtime_measurements_required", []),
        "compile_command": command,
        "issue_count": len(issues),
    }
    return result, issues


def _precision_label(spec: Any) -> str:
    if not isinstance(spec, dict):
        return ""
    typ = spec.get("type", "")
    total = spec.get("total_bits", "")
    integer = spec.get("int_bits", "")
    if typ and total and integer:
        return f"{typ}<{total},{integer}>"
    return ""


def generate_experiment_setup_artifacts(
    matrix_path: str | Path,
    *,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    matrix_path = Path(matrix_path)
    repo_root_path = Path(repo_root) if repo_root is not None else _repo_root_from(matrix_path)
    if not matrix_path.is_absolute():
        matrix_path = repo_root_path / matrix_path
    matrix = _load_yaml(matrix_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    experiments = matrix.get("experiments") or []
    if not isinstance(experiments, list):
        raise RuntimeError("paper experiment matrix must contain experiments: []")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(experiments):
        if not isinstance(item, dict):
            issues.append({"id": f"row_{index}", "severity": "error", "message": "experiment row is not a mapping"})
            continue
        rid = str(item.get("id") or "")
        if not rid:
            issues.append({"id": f"row_{index}", "severity": "error", "message": "experiment row is missing id"})
        elif rid in seen:
            issues.append({"id": rid, "severity": "error", "message": "duplicate experiment id"})
        seen.add(rid)
        row, row_issues = _row_from_experiment(repo_root=repo_root_path, row=item, index=index)
        rows.append(row)
        issues.extend(row_issues)

    table_fields = [
        "order", "id", "section", "group", "knob_axis", "expected_claim_level", "configured_stage",
        "config", "config_valid", "project_out_dir", "pipeline_mode", "precision_activation", "pipeline_style", "pe",
        "weight_mode", "training_optimizer", "training_loss", "training_batch_size", "training_accumulation_steps", "issue_count",
    ]
    _write_csv(output / "paper_experiment_setup_rows.csv", rows, table_fields)
    _write_markdown_table(output / "paper_experiment_setup_rows.md", rows, table_fields)

    command_rows = [
        {
            "order": row["order"],
            "id": row["id"],
            "section": row["section"],
            "group": row["group"],
            "configured_stage": row["configured_stage"],
            "expected_claim_level": row["expected_claim_level"],
            "config": row["config"],
            "project_out_dir": row.get("project_out_dir", ""),
            "command": row["compile_command"],
            "command_valid": _compile_command_valid(str(row["compile_command"])),
        }
        for row in rows
    ]
    for row in command_rows:
        if not row["command_valid"]:
            issues.append({"id": row["id"], "severity": "error", "message": f"invalid compile command generated: {row['command']}"})
    command_fields = ["order", "id", "section", "group", "configured_stage", "expected_claim_level", "command_valid", "command"]
    _write_csv(output / "compile_command_plan.csv", command_rows, command_fields)
    _write_markdown_table(output / "compile_command_plan.md", command_rows, command_fields)

    selected_smoke_ids = {"I0_baseline_fx16_embedded", "I1_precision_fx8_embedded", "I3_parallel_pe2", "I8_deployable_bitstream_candidate", "T0_sgd_tiled_m_axi", "T4_tile32_m_axi", "T7_deployable_training_bitstream"}
    selected_smoke_rows = [row for row in command_rows if row["id"] in selected_smoke_ids]
    inference_rows = [row for row in command_rows if row["section"] == "inference"]
    training_rows = [row for row in command_rows if row["section"] == "training"]
    bitstream_rows = [row for row in command_rows if row["configured_stage"] == "bitstream_package"]
    board_candidate_rows = [row for row in command_rows if row["expected_claim_level"] == "level_4_board_execution"]

    setup_command = f"python -m fpgai.paper.experiment_setup {matrix_path.relative_to(repo_root_path) if matrix_path.is_relative_to(repo_root_path) else matrix_path} --output-dir {output}"
    plot_command = "python -m fpgai.paper.plots build --output-dir paper_results/plots"
    common_header = [
        setup_command + " || exit 1",
        "",
        "# Compile commands are generated from paper_experiments/paper_experiment_matrix.yml.",
        "# Board-runtime rows compile deployable packages only; real board execution is a separate measured step.",
    ]
    _write_command_script(output / "compile_command_plan.sh.txt", title="FPGAI full paper experiment command plan", rows=command_rows, extra_lines=common_header + ["# Full matrix."])
    _write_command_script(output / "compile_selected_smoke.sh.txt", title="FPGAI selected paper smoke compile plan", rows=selected_smoke_rows, extra_lines=common_header + ["# Recommended first subset before the full matrix."])
    _write_command_script(output / "compile_inference_matrix.sh.txt", title="FPGAI inference paper compile plan", rows=inference_rows, extra_lines=common_header + ["# Inference section rows."])
    _write_command_script(output / "compile_training_matrix.sh.txt", title="FPGAI training paper compile plan", rows=training_rows, extra_lines=common_header + ["# Training section rows."])
    _write_command_script(output / "compile_bitstream_candidates.sh.txt", title="FPGAI deployable bitstream candidate compile plan", rows=bitstream_rows, extra_lines=common_header + ["# Rows that request bitstream/HWH/XSA/runtime package generation."])
    _write_command_script(output / "compile_board_runtime_candidates.sh.txt", title="FPGAI board-runtime candidate compile plan", rows=board_candidate_rows, extra_lines=common_header + ["# These rows prepare deployable packages. Real board reports are collected separately on the board."])
    (output / "regenerate_plots.sh.txt").write_text("\n".join(["# Regenerate paper tables and figures", "set -e", plot_command]) + "\n", encoding="utf-8")

    by_section: dict[str, int] = {}
    by_claim: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for row in rows:
        by_section[str(row.get("section"))] = by_section.get(str(row.get("section")), 0) + 1
        by_claim[str(row.get("expected_claim_level"))] = by_claim.get(str(row.get("expected_claim_level")), 0) + 1
        by_stage[str(row.get("configured_stage"))] = by_stage.get(str(row.get("configured_stage")), 0) + 1

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    manifest: dict[str, Any] = {
        "status": "ready" if error_count == 0 else "needs_attention",
        "matrix_path": str(matrix_path),
        "output_dir": str(output),
        "experiment_count": len(rows),
        "by_section": by_section,
        "by_expected_claim_level": by_claim,
        "by_configured_stage": by_stage,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "tables": {
            "setup_rows_csv": str(output / "paper_experiment_setup_rows.csv"),
            "setup_rows_md": str(output / "paper_experiment_setup_rows.md"),
            "compile_command_plan_csv": str(output / "compile_command_plan.csv"),
            "compile_command_plan_md": str(output / "compile_command_plan.md"),
            "compile_command_plan_sh_txt": str(output / "compile_command_plan.sh.txt"),
            "compile_selected_smoke_sh_txt": str(output / "compile_selected_smoke.sh.txt"),
            "compile_inference_matrix_sh_txt": str(output / "compile_inference_matrix.sh.txt"),
            "compile_training_matrix_sh_txt": str(output / "compile_training_matrix.sh.txt"),
            "compile_bitstream_candidates_sh_txt": str(output / "compile_bitstream_candidates.sh.txt"),
            "compile_board_runtime_candidates_sh_txt": str(output / "compile_board_runtime_candidates.sh.txt"),
            "regenerate_plots_sh_txt": str(output / "regenerate_plots.sh.txt"),
        },
        "plot_command": "python -m fpgai.paper.plots build --output-dir paper_results/plots",
    }
    (output / "experiment_setup_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "experiment_setup_manifest.md").write_text(_manifest_markdown(manifest), encoding="utf-8")
    return manifest


def _manifest_markdown(manifest: dict[str, Any]) -> str:
    lines: list[str] = ["# FPGAI paper experiment setup", ""]
    lines.append(f"- status: `{manifest['status']}`")
    lines.append(f"- experiment_count: `{manifest['experiment_count']}`")
    lines.append(f"- error_count: `{manifest['error_count']}`")
    lines.append(f"- warning_count: `{manifest['warning_count']}`")
    lines.append("")
    lines.append("## Sections")
    for key, value in sorted(manifest.get("by_section", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Expected claim levels")
    for key, value in sorted(manifest.get("by_expected_claim_level", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Configured stages")
    for key, value in sorted(manifest.get("by_configured_stage", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Tables")
    for key, path in manifest.get("tables", {}).items():
        lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    if manifest.get("issues"):
        lines.append("## Issues")
        for issue in manifest["issues"]:
            lines.append(f"- `{issue.get('severity')}` `{issue.get('id')}`: {issue.get('message')}")
        lines.append("")
    lines.append("## Next command")
    lines.append("")
    lines.append("```bash")
    lines.append(str(manifest.get("plot_command")))
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate FPGAI paper experiment setup reports.")
    parser.add_argument("matrix", nargs="?", help="paper experiment matrix YAML")
    parser.add_argument("--output-dir", default="paper_results/experiment_setup", help="output directory")
    parser.add_argument("--repo-root", default=None, help="repository root override")
    parser.add_argument("--validate-artifact", default=None, help="validate a compiled experiment output directory")
    parser.add_argument("--configured-stage", default="compiler_artifact", help="configured artifact stage for --validate-artifact")
    parser.add_argument("--expected-claim-level", default=None, help="expected claim level for --validate-artifact reporting")
    args = parser.parse_args(argv)
    if args.validate_artifact:
        payload = validate_compile_artifacts(
            args.validate_artifact,
            configured_stage=args.configured_stage,
            expected_claim_level=args.expected_claim_level,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if not args.matrix:
        parser.error("matrix is required unless --validate-artifact is used")
    try:
        manifest = generate_experiment_setup_artifacts(args.matrix, output_dir=args.output_dir, repo_root=args.repo_root)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({
        "status": manifest["status"],
        "experiment_count": manifest["experiment_count"],
        "error_count": manifest["error_count"],
        "warning_count": manifest["warning_count"],
        "manifest": str(Path(args.output_dir) / "experiment_setup_manifest.json"),
    }, indent=2, sort_keys=True))
    return 0 if manifest["error_count"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
