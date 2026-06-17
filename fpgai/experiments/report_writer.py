"""Markdown reporting for FPGAI experiment sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("\n", " ")


def write_summary_markdown(experiment_dir: str | Path, results_payload: Mapping[str, Any] | None = None) -> Path:
    experiment_dir = Path(experiment_dir)
    results_path = experiment_dir / "results.json"
    if results_payload is None:
        results_payload = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"results": []}
    records = list(results_payload.get("results", []))
    total = len(records)
    passed = sum(1 for r in records if r.get("status") == "passed")
    failed = sum(1 for r in records if r.get("status") == "failed")
    skipped = sum(1 for r in records if r.get("status") == "skipped")
    lines = [
        "# FPGAI Experiment Summary",
        "",
        f"Experiment directory: `{experiment_dir}`",
        "",
        "## Status",
        "",
        f"- Total design points: {total}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Skipped/resumed: {skipped}",
        "",
        "## Results",
        "",
        "| # | Design | Status | Board | Config | Duration (s) |",
        "|---:|---|---|---|---|---:|",
    ]
    for r in records:
        lines.append(
            "| {idx} | `{name}` | {status} | {board} | `{cfg}` | {dur} |".format(
                idx=_fmt(r.get("design_index")),
                name=_fmt(r.get("design_name")),
                status=_fmt(r.get("status")),
                board=_fmt(r.get("board")),
                cfg=_fmt(r.get("config_path")),
                dur=_fmt(r.get("duration_sec")),
            )
        )
    lines.extend([
        "",
        "## Reproducibility metadata",
        "",
        "Every row in `results.json` and `results.csv` includes commit hash, config path, model path, tool version, and board target when available.",
    ])
    out = experiment_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
