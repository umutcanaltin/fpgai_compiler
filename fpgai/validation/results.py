"""Validation helpers for FPGAI experiment output directories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    path: str
    message: str


@dataclass(frozen=True)
class ResultsValidation:
    input_dir: Path
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    child_count: int = 0
    failed_child_count: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "warning")


def _read_json(path: Path, issues: list[ValidationIssue]) -> dict[str, Any] | None:
    if not path.exists():
        issues.append(ValidationIssue("error", str(path), "required file is missing"))
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(ValidationIssue("error", str(path), f"could not parse JSON: {exc}"))
        return None
    if not isinstance(data, dict):
        issues.append(ValidationIssue("error", str(path), "JSON root must be an object"))
        return None
    return data


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def validate_results(input_dir: str | Path) -> ResultsValidation:
    """Validate paper experiment output for false pass / missing child output."""
    root = Path(input_dir)
    issues: list[ValidationIssue] = []

    if not root.exists():
        issues.append(ValidationIssue("error", str(root), "input directory does not exist"))
        return ResultsValidation(root, False, issues)

    manifest = _read_json(root / "manifest.json", issues)
    status = _read_json(root / "experiment_status.json", issues)

    child_items = []
    if isinstance(status, dict) and isinstance(status.get("items"), list):
        child_items = [item for item in status["items"] if isinstance(item, dict)]
    elif isinstance(manifest, dict) and isinstance(manifest.get("items"), list):
        child_items = [item for item in manifest["items"] if isinstance(item, dict)]

    if not child_items:
        issues.append(ValidationIssue("error", str(root), "no child experiment items found"))

    failed_child_count = 0
    for item in child_items:
        name = str(item.get("name", "<unknown>"))
        child_status = str(item.get("status", "")).lower()
        out_dir = Path(str(item.get("out_dir", "")))
        if out_dir and not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir

        if child_status == "failed":
            failed_child_count += 1

        results_path = None
        child_summary = item.get("child_summary")
        if isinstance(child_summary, dict) and child_summary.get("results_path"):
            results_path = Path(str(child_summary["results_path"]))
        elif out_dir:
            results_path = out_dir / "results.json"

        if results_path is None or not results_path.exists():
            issues.append(ValidationIssue("error", name, "child results.json is missing"))
            continue

        child_payload = _read_json(results_path, issues)
        if not child_payload:
            continue

        child_failed_count = _as_int(child_payload.get("failed_count"))
        if child_status == "passed" and child_failed_count is not None and child_failed_count > 0:
            issues.append(
                ValidationIssue(
                    "error",
                    str(results_path),
                    f"false pass: child status passed but failed_count={child_failed_count}",
                )
            )

        records = child_payload.get("results")
        if not isinstance(records, list):
            issues.append(ValidationIssue("error", str(results_path), "child results list is missing"))
        elif not records:
            issues.append(ValidationIssue("warning", str(results_path), "child results list is empty"))

    if isinstance(status, dict):
        status_failed = _as_int(status.get("failed_count"))
        actual_failed = sum(1 for item in child_items if str(item.get("status", "")).lower() == "failed")
        if status_failed is not None and status_failed != actual_failed:
            issues.append(
                ValidationIssue(
                    "error",
                    str(root / "experiment_status.json"),
                    f"failed_count={status_failed} does not match failed child items={actual_failed}",
                )
            )

    passed = not any(issue.level == "error" for issue in issues)
    return ResultsValidation(root, passed, issues, len(child_items), failed_child_count)
