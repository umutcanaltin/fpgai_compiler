"""Experiment artifact matrix and training-curve reports.

These reports normalize per-compile artifact status into paper-facing rows.
They do not execute hardware and they do not upgrade the recorded claim level
unless the required files/reports already exist.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


_NONE = ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return _NONE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = mapping
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _project_name(manifest: Mapping[str, Any], out_dir: Path) -> str:
    for path in ("project.name", "config.project.name", "top_kernel_name"):
        value = _get(manifest, path)
        if value:
            return str(value)
    return out_dir.name


def _package_validation_payload(out_dir: Path) -> dict[str, Any]:
    return _read_json(out_dir / "runtime_package" / "runtime_package_validation.json") or _read_json(
        out_dir / "reports" / "runtime_package_validation.json"
    )


def _paper_verification_payload(out_dir: Path) -> dict[str, Any]:
    return _read_json(out_dir / "reports" / "paper_verification.json")


def _board_runtime_payload(out_dir: Path) -> dict[str, Any]:
    for candidate in (
        out_dir / "reports" / "board_runtime_report.json",
        out_dir / "runtime_package" / "board_runtime_report.json",
        out_dir / "board_runtime_report.json",
    ):
        data = _read_json(candidate)
        if data:
            return data
    return {}


def _runtime_package_payload(out_dir: Path) -> dict[str, Any]:
    return _read_json(out_dir / "runtime_package" / "package_manifest.json")


def _claim_level(row: Mapping[str, Any]) -> str:
    """Derive the strongest defensible paper claim from artifact status.

    Higher-level hardware claims must be gated by successful HLS synthesis.  A
    stale or externally copied Vivado/bitstream artifact must not upgrade a row
    if the current compile's HLS result failed.  This keeps paper tables from
    reporting deployable hardware when the generated HLS implementation did not
    synthesize successfully.
    """
    hls_ok = bool(row.get("hls_ok"))
    if hls_ok and row.get("board_execution_passed"):
        return "level_4_board_execution"
    if hls_ok and row.get("bitstream_generated") and row.get("runtime_package_validated"):
        return "level_3_bitstream_package"
    if hls_ok and row.get("vivado_implemented"):
        return "level_2_vivado_implementation"
    if hls_ok:
        return "level_1_hls_synthesis"
    if row.get("source_generated"):
        return "level_0_compiler_artifact"
    return "not_generated"


def _artifact_row(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    paper = _paper_verification_payload(out_dir)
    paper_flags = paper.get("verification_flags", {}) if isinstance(paper.get("verification_flags"), Mapping) else {}
    runtime_package = _runtime_package_payload(out_dir)
    package_validation = _package_validation_payload(out_dir)
    board_runtime = _board_runtime_payload(out_dir)
    vivado_bridge = manifest.get("vivado_bridge", {}) if isinstance(manifest.get("vivado_bridge"), Mapping) else {}

    hls_src = out_dir / "hls" / "src" / "deeplearn.cpp"
    source_generated = bool(hls_src.exists() or manifest.get("hls_project_dir") or manifest.get("hls_artifacts"))
    hls_ok = bool(manifest.get("hls_ok") is True or _get(manifest, "hls_artifacts.hls_ok") is True)
    vivado_implemented = bool(paper_flags.get("vivado_implemented") or _get(paper, "verification_flags.vivado_implemented"))
    if not vivado_implemented:
        vivado_implemented = bool(vivado_bridge.get("vivado_impl_requested") and not vivado_bridge.get("failed_rows") and vivado_bridge.get("ok", True))
    bitstream_generated = bool(paper_flags.get("bitstream_generated") or vivado_bridge.get("bitstream_exists"))
    runtime_package_validated = bool(package_validation.get("deployability_ready") and package_validation.get("failed_count", 1) == 0)
    board_execution_claimed = bool(board_runtime.get("board_execution_claimed") or board_runtime.get("board_execution_passed"))
    board_execution_passed = bool(board_runtime.get("board_execution_passed") or board_runtime.get("status") in {"passed", "success"})

    row: dict[str, Any] = {
        "experiment_name": _project_name(manifest, out_dir),
        "out_dir": out_dir.as_posix(),
        "pipeline_mode": str(manifest.get("pipeline_mode") or runtime_package.get("pipeline_mode") or ""),
        "top_kernel": str(manifest.get("top_kernel_name") or runtime_package.get("top_name") or ""),
        "board": str(runtime_package.get("board") or _get(manifest, "vivado_bridge.board") or ""),
        "source_generated": source_generated,
        "hls_ran": bool(manifest.get("hls_ran") or _get(manifest, "hls_artifacts.hls_ran")),
        "hls_ok": hls_ok,
        "hls_returncode": manifest.get("hls_returncode"),
        "vivado_project_generated": bool(vivado_bridge.get("vivado_bridge_generated") or (out_dir / "vivado" / "project.tcl").exists()),
        "vivado_implemented": vivado_implemented,
        "bitstream_requested": bool(vivado_bridge.get("bitstream_requested") or _get(manifest, "build_stages.bitstream")),
        "bitstream_generated": bitstream_generated,
        "xsa_generated": bool(vivado_bridge.get("xsa_exists") or _get(runtime_package, "hardware.xsa.present")),
        "runtime_package_created": bool((out_dir / "runtime_package" / "package_manifest.json").exists()),
        "runtime_package_validated": runtime_package_validated,
        "runtime_package_failed_count": package_validation.get("failed_count"),
        "board_execution_claimed": board_execution_claimed,
        "board_execution_passed": board_execution_passed,
        "training_curve_contract_available": False,
        "training_curve_available": False,
        "training_curve_source": "not_training_or_not_emitted",
        "training_curve_rows": 0,
        "python_reference_available": bool(manifest.get("training_reference")),
        "curve_comparison_available": bool(manifest.get("training_compare")),
        "final_loss": None,
        "final_accuracy": None,
        "best_accuracy": None,
        "total_runtime_seconds": board_runtime.get("runtime_seconds"),
        "accuracy_per_second": None,
        "claim_level": "not_generated",
    }
    row["claim_level"] = _claim_level(row)
    return row


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _training_curve_rows(out_dir: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return real board-runtime training-curve rows when they exist.

    Compile-time Python/HLS reference data is correctness evidence, but it is
    not a real FPGA training curve. Paper training curves must come from a
    board-runtime artifact such as board_training_curve.csv or
    board_runtime_report.json. Until then, the compiler emits a contract and
    marks the curve as pending board runtime.
    """
    pipeline_mode = str(manifest.get("pipeline_mode") or "")
    is_training = "train" in pipeline_mode or manifest.get("training_plan") is not None
    if not is_training:
        return [], {
            "available": False,
            "contract_available": False,
            "reason": "pipeline is not training_on_device",
            "source": "not_training",
            "row_count": 0,
            "board_execution_claimed": False,
        }

    experiment = _project_name(manifest, out_dir)
    board_runtime = _board_runtime_payload(out_dir)

    def normalize_row(raw: Mapping[str, Any], index: int) -> dict[str, Any]:
        runtime = _float_or_none(raw.get("runtime_seconds"))
        cumulative = _float_or_none(raw.get("cumulative_runtime_seconds"))
        return {
            "experiment": str(raw.get("experiment") or experiment),
            "source": str(raw.get("source") or "kv260_board_runtime"),
            "step": raw.get("step", index),
            "epoch": raw.get("epoch", 0),
            "batch": raw.get("batch", raw.get("batch_index", 0)),
            "mode": raw.get("mode", raw.get("command", "run_training")),
            "loss": _float_or_none(raw.get("loss")),
            "accuracy": _float_or_none(raw.get("accuracy")),
            "runtime_seconds": runtime,
            "cumulative_runtime_seconds": cumulative,
            "gradient_norm": _float_or_none(raw.get("gradient_norm")),
            "weight_delta_norm": _float_or_none(raw.get("weight_delta_norm")),
            "gradient_cosine_vs_reference": _float_or_none(raw.get("gradient_cosine_vs_reference")),
            "weight_after_cosine_vs_reference": _float_or_none(raw.get("weight_after_cosine_vs_reference")),
            "weight_delta_cosine_vs_reference": _float_or_none(raw.get("weight_delta_cosine_vs_reference")),
            "status": str(raw.get("status") or "board_runtime_row"),
        }

    rows: list[dict[str, Any]] = []
    curve_csv_candidates = [
        out_dir / "reports" / "board_training_curve.csv",
        out_dir / "reports" / "training_curve_board.csv",
        out_dir / "runtime_package" / "board_training_curve.csv",
        out_dir / "runtime_package" / "training_curve_board.csv",
        out_dir / "board_runtime" / "training_curve.csv",
    ]
    for candidate in curve_csv_candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open(newline="", encoding="utf-8") as f:
                for idx, raw in enumerate(csv.DictReader(f)):
                    rows.append(normalize_row(raw, idx))
        except Exception:
            rows = []
        if rows:
            break

    if not rows:
        raw_rows = board_runtime.get("training_curve") or board_runtime.get("training_curve_rows")
        if isinstance(raw_rows, list):
            for idx, raw in enumerate(raw_rows):
                if isinstance(raw, Mapping):
                    rows.append(normalize_row(raw, idx))

    if rows:
        cumulative = 0.0
        for row in rows:
            if row.get("cumulative_runtime_seconds") is None and row.get("runtime_seconds") is not None:
                cumulative += float(row["runtime_seconds"])
                row["cumulative_runtime_seconds"] = cumulative
        losses = [float(row["loss"]) for row in rows if row.get("loss") is not None]
        accuracies = [float(row["accuracy"]) for row in rows if row.get("accuracy") is not None]
        summary = {
            "available": True,
            "contract_available": True,
            "source": str(board_runtime.get("board") or "kv260_board_runtime"),
            "row_count": len(rows),
            "final_loss": losses[-1] if losses else None,
            "initial_loss": losses[0] if losses else None,
            "best_loss": min(losses) if losses else None,
            "final_accuracy": accuracies[-1] if accuracies else None,
            "best_accuracy": max(accuracies) if accuracies else None,
            "total_runtime_seconds": rows[-1].get("cumulative_runtime_seconds"),
            "board_execution_claimed": bool(board_runtime.get("board_execution_claimed") or board_runtime.get("board_execution_passed")),
            "claim_boundary": "Training curve rows come from real board-runtime artifacts.",
        }
        return rows, summary

    return [], {
        "available": False,
        "contract_available": True,
        "reason": "pending real board-runtime training curve",
        "source": "pending_board_runtime",
        "row_count": 0,
        "final_loss": None,
        "initial_loss": None,
        "best_loss": None,
        "final_accuracy": None,
        "best_accuracy": None,
        "total_runtime_seconds": None,
        "board_execution_claimed": False,
        "claim_boundary": "Compile-time artifacts provide the training-curve schema and collection contract only. Real training curves require a later board_runtime_report.json or board_training_curve.csv.",
    }

_TRAINING_CURVE_FIELDS = [
    "experiment",
    "source",
    "step",
    "epoch",
    "batch",
    "mode",
    "loss",
    "accuracy",
    "runtime_seconds",
    "cumulative_runtime_seconds",
    "gradient_norm",
    "weight_delta_norm",
    "gradient_cosine_vs_reference",
    "weight_after_cosine_vs_reference",
    "weight_delta_cosine_vs_reference",
    "status",
]

_EXPERIMENT_FIELDS = [
    "experiment_name",
    "out_dir",
    "pipeline_mode",
    "top_kernel",
    "board",
    "source_generated",
    "hls_ran",
    "hls_ok",
    "hls_returncode",
    "vivado_project_generated",
    "vivado_implemented",
    "bitstream_requested",
    "bitstream_generated",
    "xsa_generated",
    "runtime_package_created",
    "runtime_package_validated",
    "runtime_package_failed_count",
    "board_execution_claimed",
    "board_execution_passed",
    "training_curve_contract_available",
    "training_curve_available",
    "training_curve_source",
    "training_curve_rows",
    "python_reference_available",
    "curve_comparison_available",
    "final_loss",
    "final_accuracy",
    "best_accuracy",
    "total_runtime_seconds",
    "accuracy_per_second",
    "claim_level",
]


def _copy_to_package(root_path: Path, package_path: Path, package_dir: Path, package_files: dict[str, Any], key: str) -> None:
    if not root_path.exists() or not package_dir.exists():
        return
    package_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_path, package_path)
    package_files[key] = {
        "package_path": _rel(package_path, package_dir),
        "source": root_path.as_posix(),
        "bytes": package_path.stat().st_size,
        "present": True,
    }


def _update_package_manifest(out_dir: Path, artifacts: Mapping[str, Any]) -> None:
    package_dir = out_dir / "runtime_package"
    manifest_path = package_dir / "package_manifest.json"
    if not manifest_path.exists():
        return
    payload = _read_json(manifest_path)
    if not payload:
        return
    files = payload.get("files")
    if not isinstance(files, dict):
        files = {}
        payload["files"] = files

    report_paths = artifacts.get("paths", {}) if isinstance(artifacts.get("paths"), Mapping) else {}
    copy_map = {
        "experiment_artifact_matrix_json": (out_dir / "reports" / "experiment_artifact_matrix.json", package_dir / "reports" / "experiment_artifact_matrix.json"),
        "experiment_artifact_matrix_csv": (out_dir / "reports" / "experiment_artifact_matrix.csv", package_dir / "reports" / "experiment_artifact_matrix.csv"),
        "experiment_artifact_matrix_md": (out_dir / "reports" / "experiment_artifact_matrix.md", package_dir / "reports" / "experiment_artifact_matrix.md"),
        "paper_experiment_row_json": (out_dir / "reports" / "paper_experiment_row.json", package_dir / "reports" / "paper_experiment_row.json"),
        "paper_experiment_row_csv": (out_dir / "reports" / "paper_experiment_row.csv", package_dir / "reports" / "paper_experiment_row.csv"),
        "training_curve_csv": (out_dir / "training" / "training_curve.csv", package_dir / "training" / "training_curve.csv"),
        "training_curve_json": (out_dir / "training" / "training_curve.json", package_dir / "training" / "training_curve.json"),
        "training_curve_md": (out_dir / "training" / "training_curve.md", package_dir / "training" / "training_curve.md"),
        "training_curve_report_json": (out_dir / "reports" / "training_curve_report.json", package_dir / "reports" / "training_curve_report.json"),
        "training_curve_report_md": (out_dir / "reports" / "training_curve_report.md", package_dir / "reports" / "training_curve_report.md"),
        "training_curve_contract_json": (out_dir / "reports" / "training_curve_contract.json", package_dir / "reports" / "training_curve_contract.json"),
        "training_curve_contract_md": (out_dir / "reports" / "training_curve_contract.md", package_dir / "reports" / "training_curve_contract.md"),
        "paper_training_curve_csv": (out_dir / "reports" / "paper_training_curve.csv", package_dir / "reports" / "paper_training_curve.csv"),
        "paper_training_curve_summary_json": (out_dir / "reports" / "paper_training_curve_summary.json", package_dir / "reports" / "paper_training_curve_summary.json"),
    }
    for key, (src, dst) in copy_map.items():
        _copy_to_package(src, dst, package_dir, files, key)

    payload["experiment_artifacts"] = artifacts.get("summary", {})
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _update_compile_manifest(out_dir: Path, artifacts: Mapping[str, Any]) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if not manifest:
        return
    manifest["experiment_artifacts"] = artifacts.get("summary", {})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    package_manifest_copy = out_dir / "runtime_package" / "manifest.json"
    if package_manifest_copy.parent.exists():
        package_manifest_copy.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def emit_experiment_artifact_reports(out_dir: str | Path) -> dict[str, Any]:
    """Emit per-experiment artifact matrix and normalized training-curve reports."""
    out = Path(out_dir).resolve()
    reports = out / "reports"
    training_dir = out / "training"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(out / "manifest.json")

    row = _artifact_row(out, manifest)
    curve_rows, curve_summary = _training_curve_rows(out, manifest)
    row["training_curve_contract_available"] = bool(curve_summary.get("contract_available"))
    row["training_curve_available"] = bool(curve_summary.get("available"))
    row["training_curve_source"] = curve_summary.get("source")
    row["training_curve_rows"] = curve_summary.get("row_count", 0)
    if curve_rows:
        row["final_loss"] = curve_summary.get("final_loss")
        row["final_accuracy"] = curve_summary.get("final_accuracy")
        row["best_accuracy"] = curve_summary.get("best_accuracy")

    matrix_payload = {
        "schema_version": 1,
        "artifact_kind": "experiment_artifact_matrix",
        "row_count": 1,
        "rows": [row],
        "claim_boundary": "The claim_level is derived from generated artifacts and reports. Real board-execution claims require board_runtime_report.json.",
    }
    _write_json(reports / "experiment_artifact_matrix.json", matrix_payload)
    _write_csv(reports / "experiment_artifact_matrix.csv", [row], _EXPERIMENT_FIELDS)
    _write_json(reports / "paper_experiment_row.json", row)
    _write_csv(reports / "paper_experiment_row.csv", [row], _EXPERIMENT_FIELDS)

    md_lines = [
        "# Experiment artifact matrix",
        "",
        f"- experiment_name: `{row['experiment_name']}`",
        f"- pipeline_mode: `{row['pipeline_mode']}`",
        f"- claim_level: `{row['claim_level']}`",
        f"- hls_ok: `{row['hls_ok']}`",
        f"- vivado_implemented: `{row['vivado_implemented']}`",
        f"- bitstream_generated: `{row['bitstream_generated']}`",
        f"- runtime_package_validated: `{row['runtime_package_validated']}`",
        f"- board_execution_claimed: `{row['board_execution_claimed']}`",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for field in _EXPERIMENT_FIELDS:
        md_lines.append(f"| {field} | `{_csv_value(row.get(field))}` |")
    md_lines.extend(["", "Board execution is not claimed unless a board runtime report exists and passes.", ""])
    (reports / "experiment_artifact_matrix.md").write_text("\n".join(md_lines), encoding="utf-8")

    training_paths: dict[str, str] = {}
    if curve_summary.get("available"):
        training_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(training_dir / "training_curve.csv", curve_rows, _TRAINING_CURVE_FIELDS)
        _write_json(training_dir / "training_curve.json", {"schema_version": 1, "rows": curve_rows, "summary": curve_summary})
        curve_md = [
            "# Training curve",
            "",
            f"- available: `{curve_summary['available']}`",
            f"- source: `{curve_summary['source']}`",
            f"- row_count: `{curve_summary['row_count']}`",
            f"- initial_loss: `{curve_summary['initial_loss']}`",
            f"- final_loss: `{curve_summary['final_loss']}`",
            f"- board_execution_claimed: `{curve_summary['board_execution_claimed']}`",
            "",
            "| step | source | loss | accuracy | gradient_cosine_vs_reference | weight_delta_cosine_vs_reference | status |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
        for item in curve_rows:
            curve_md.append(
                f"| {item.get('step')} | {item.get('source')} | {item.get('loss')} | {item.get('accuracy')} | {item.get('gradient_cosine_vs_reference')} | {item.get('weight_delta_cosine_vs_reference')} | {item.get('status')} |"
            )
        curve_md.append("")
        (training_dir / "training_curve.md").write_text("\n".join(curve_md), encoding="utf-8")

        report_payload = {
            "schema_version": 1,
            "artifact_kind": "training_curve_report",
            "available": True,
            "summary": curve_summary,
            "curve_csv": "training/training_curve.csv",
            "curve_json": "training/training_curve.json",
            "paper_training_curve_csv": "reports/paper_training_curve.csv",
            "claim_boundary": curve_summary["claim_boundary"],
        }
        _write_json(reports / "training_curve_report.json", report_payload)
        (reports / "training_curve_report.md").write_text("\n".join(curve_md), encoding="utf-8")
        _write_csv(reports / "paper_training_curve.csv", curve_rows, _TRAINING_CURVE_FIELDS)
        _write_json(reports / "paper_training_curve_summary.json", curve_summary)
        training_paths = {
            "training_curve_csv": "training/training_curve.csv",
            "training_curve_json": "training/training_curve.json",
            "training_curve_md": "training/training_curve.md",
            "training_curve_report_json": "reports/training_curve_report.json",
            "training_curve_report_md": "reports/training_curve_report.md",
            "paper_training_curve_csv": "reports/paper_training_curve.csv",
            "paper_training_curve_summary_json": "reports/paper_training_curve_summary.json",
        }
    else:
        contract_payload = {
            "schema_version": 1,
            "artifact_kind": "training_curve_contract",
            "available": False,
            "contract_available": bool(curve_summary.get("contract_available")),
            "source": curve_summary.get("source"),
            "expected_source_after_board_run": "kv260_board_runtime",
            "required_fields": _TRAINING_CURVE_FIELDS,
            "runtime_artifact_candidates": [
                "reports/board_training_curve.csv",
                "reports/training_curve_board.csv",
                "runtime_package/board_training_curve.csv",
                "runtime_package/training_curve_board.csv",
                "board_runtime/training_curve.csv",
                "reports/board_runtime_report.json:training_curve",
            ],
            "summary": curve_summary,
            "claim_boundary": curve_summary.get("claim_boundary"),
        }
        _write_json(reports / "training_curve_report.json", contract_payload)
        _write_json(reports / "training_curve_contract.json", contract_payload)
        contract_md = [
            "# Training curve contract",
            "",
            "- available: `False`",
            f"- contract_available: `{bool(curve_summary.get('contract_available'))}`",
            f"- source: `{curve_summary.get('source')}`",
            f"- reason: `{curve_summary.get('reason')}`",
            "- board_execution_claimed: `False`",
            "",
            "Real training-curve rows must be produced by board runtime, not by compile-time Python/HLS reference artifacts.",
            "",
            "## Required CSV fields",
        ]
        contract_md.extend(f"- `{field}`" for field in _TRAINING_CURVE_FIELDS)
        (reports / "training_curve_report.md").write_text("\n".join(contract_md) + "\n", encoding="utf-8")
        (reports / "training_curve_contract.md").write_text("\n".join(contract_md) + "\n", encoding="utf-8")
        training_paths = {
            "training_curve_report_json": "reports/training_curve_report.json",
            "training_curve_report_md": "reports/training_curve_report.md",
            "training_curve_contract_json": "reports/training_curve_contract.json",
            "training_curve_contract_md": "reports/training_curve_contract.md",
        }

    summary = {
        "status": "created",
        "claim_level": row["claim_level"],
        "runtime_package_validated": row["runtime_package_validated"],
        "training_curve_contract_available": row["training_curve_contract_available"],
        "training_curve_available": row["training_curve_available"],
        "training_curve_source": row["training_curve_source"],
        "training_curve_rows": row["training_curve_rows"],
        "board_execution_claimed": row["board_execution_claimed"],
        "board_execution_passed": row["board_execution_passed"],
        "paths": {
            "experiment_artifact_matrix_json": "reports/experiment_artifact_matrix.json",
            "experiment_artifact_matrix_csv": "reports/experiment_artifact_matrix.csv",
            "experiment_artifact_matrix_md": "reports/experiment_artifact_matrix.md",
            "paper_experiment_row_json": "reports/paper_experiment_row.json",
            "paper_experiment_row_csv": "reports/paper_experiment_row.csv",
            **training_paths,
        },
    }
    artifacts = {"summary": summary, "paths": summary["paths"]}
    _update_compile_manifest(out, artifacts)
    _update_package_manifest(out, artifacts)
    return artifacts
