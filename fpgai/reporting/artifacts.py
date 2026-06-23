"""Public report generation helpers for FPGAI experiment outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReportBuildResult:
    input_dir: Path
    out_dir: Path
    summary_md: Path
    results_table_csv: Path
    claim_traceability_md: Path
    result_count: int
    passed_count: int
    failed_count: int
    skipped_count: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc), "_path": str(path)}
    return data if isinstance(data, dict) else {"_read_error": "JSON root is not an object", "_path": str(path)}


def _iter_child_results(input_dir: Path) -> Iterable[tuple[str, Path, dict[str, Any]]]:
    for results_path in sorted(input_dir.glob("*/results.json")):
        name = results_path.parent.name
        yield name, results_path, _read_json(results_path)


def _status_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    passed = sum(1 for row in rows if str(row.get("status", "")).lower() == "passed")
    failed = sum(1 for row in rows if str(row.get("status", "")).lower() == "failed")
    skipped = sum(1 for row in rows if str(row.get("status", "")).lower() == "skipped")
    return passed, failed, skipped


def collect_result_rows(input_dir: str | Path) -> list[dict[str, Any]]:
    """Collect flattened result records from a paper experiment output directory."""
    root = Path(input_dir)
    rows: list[dict[str, Any]] = []

    for experiment_name, results_path, payload in _iter_child_results(root):
        records = payload.get("results")
        if not isinstance(records, list):
            rows.append(
                {
                    "experiment": experiment_name,
                    "results_path": str(results_path),
                    "status": "failed",
                    "error": payload.get("_read_error", "results.json has no results list"),
                }
            )
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            params = record.get("parameters") if isinstance(record.get("parameters"), dict) else {}
            metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            row = {
                "experiment": experiment_name,
                "results_path": str(results_path),
                "design_name": record.get("design_name", ""),
                "status": record.get("status", ""),
                "returncode": record.get("returncode", ""),
                "duration_sec": record.get("duration_sec", ""),
                "config_path": record.get("config_path", ""),
                "model_path": record.get("model_path", ""),
                "board": record.get("board", ""),
                "command": record.get("command", ""),
                "error": record.get("error", ""),
            }
            for key, value in sorted(params.items()):
                row[f"param.{key}"] = value
            for key, value in sorted(metrics.items()):
                row[f"metric.{key}"] = value
            rows.append(row)

    return rows


def _write_results_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = [
        "experiment",
        "design_name",
        "status",
        "returncode",
        "duration_sec",
        "config_path",
        "model_path",
        "board",
        "error",
        "results_path",
        "command",
    ]
    dynamic = sorted({key for row in rows for key in row if key not in base})
    fields = base + dynamic
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _load_status(input_dir: Path) -> dict[str, Any]:
    status_path = input_dir / "experiment_status.json"
    if status_path.exists():
        return _read_json(status_path)
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        return _read_json(manifest_path)
    return {}


def _write_summary(path: Path, input_dir: Path, rows: list[dict[str, Any]], status: dict[str, Any]) -> None:
    passed, failed, skipped = _status_counts(rows)
    child_items = status.get("items") if isinstance(status.get("items"), list) else []

    lines = [
        "# FPGAI Experiment Report",
        "",
        f"Input directory: `{input_dir}`",
        "",
        "## Summary",
        "",
        f"- Result records: {len(rows)}",
        f"- Passed records: {passed}",
        f"- Failed records: {failed}",
        f"- Skipped records: {skipped}",
    ]

    if status:
        lines.extend(
            [
                f"- Child sweeps passed: {status.get('passed_count', 'unknown')}",
                f"- Child sweeps failed: {status.get('failed_count', 'unknown')}",
                f"- Child sweeps skipped: {status.get('skipped_count', 'unknown')}",
            ]
        )

    if child_items:
        lines.extend(["", "## Child sweeps", ""])
        for item in child_items:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- `{item.get('name', '')}`: {item.get('status', '')} "
                f"({item.get('sweep_config', '')})"
            )

    lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `results_table.csv`",
            "- `claim_traceability.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_claim_traceability(path: Path, input_dir: Path, rows: list[dict[str, Any]], status: dict[str, Any]) -> None:
    child_items = status.get("items") if isinstance(status.get("items"), list) else []
    lines = [
        "# Claim Traceability",
        "",
        "This generated file links public experiment groups to the artifacts found in the input directory.",
        "It is intentionally conservative: unsupported or missing artifacts should be treated as not claimed.",
        "",
        "| Claim area | Implementation / command | Artifact | Status |",
        "|---|---|---|---|",
    ]

    if child_items:
        for item in child_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            command = str(item.get("sweep_config", ""))
            artifact = str(item.get("out_dir", ""))
            item_status = str(item.get("status", ""))
            lines.append(f"| {name} | `{command}` | `{artifact}` | {item_status} |")
    else:
        experiments = sorted({str(row.get("experiment", "")) for row in rows if row.get("experiment")})
        for name in experiments:
            status_set = {str(row.get("status", "")) for row in rows if str(row.get("experiment", "")) == name}
            item_status = "failed" if "failed" in status_set else "supported" if "passed" in status_set else "partial"
            lines.append(f"| {name} | collected result rows | `{input_dir / name}` | {item_status} |")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(input_dir: str | Path, out_dir: str | Path) -> ReportBuildResult:
    """Build a small public report from a paper experiment output directory."""
    input_path = Path(input_dir)
    output_path = Path(out_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    rows = collect_result_rows(input_path)
    status = _load_status(input_path)

    summary_md = output_path / "summary.md"
    results_table_csv = output_path / "results_table.csv"
    claim_traceability_md = output_path / "claim_traceability.md"

    _write_results_table(results_table_csv, rows)
    _write_summary(summary_md, input_path, rows, status)
    _write_claim_traceability(claim_traceability_md, input_path, rows, status)

    passed, failed, skipped = _status_counts(rows)
    return ReportBuildResult(
        input_dir=input_path,
        out_dir=output_path,
        summary_md=summary_md,
        results_table_csv=results_table_csv,
        claim_traceability_md=claim_traceability_md,
        result_count=len(rows),
        passed_count=passed,
        failed_count=failed,
        skipped_count=skipped,
    )
