"""Experiment analysis utilities for FPGAI sweeps.

This module intentionally uses only the Python standard library so it can run
in lightweight CI environments. Plotting is isolated in plotting.py.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


METRIC_PREFIXES = ("raw_mape.", "cal_mape.", "calibration.")


@dataclass(frozen=True)
class ExperimentRecord:
    """Normalized view of a single sweep result record."""

    design_index: int
    design_name: str
    status: str
    board: Optional[str]
    config_path: Optional[str]
    model_path: Optional[str]
    command: Optional[str]
    duration_sec: Optional[float]
    commit_hash: Optional[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ExperimentAnalysis:
    """Normalized experiment payload and derived summaries."""

    experiment_dir: Path
    records: List[ExperimentRecord]
    summary: Dict[str, Any]


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _flatten_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in (metrics or {}).items():
        val = _as_float(value)
        if val is not None:
            out[str(key)] = val
    return out


def load_results_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {p}")
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"Missing list field 'results' in {p}")
    return payload


def load_experiment(experiment_dir: str | Path) -> ExperimentAnalysis:
    exp_dir = Path(experiment_dir)
    results_path = exp_dir / "results.json"
    payload = load_results_json(results_path)
    records: List[ExperimentRecord] = []
    for idx, item in enumerate(payload.get("results", [])):
        if not isinstance(item, dict):
            continue
        metrics = _flatten_metrics(item.get("metrics", {}))
        duration = _as_float(item.get("duration_sec"))
        rec = ExperimentRecord(
            design_index=int(item.get("design_index", idx)),
            design_name=str(item.get("design_name", f"design_{idx:03d}")),
            status=str(item.get("status", "unknown")),
            board=item.get("board"),
            config_path=item.get("config_path"),
            model_path=item.get("model_path"),
            command=item.get("command"),
            duration_sec=duration,
            commit_hash=item.get("commit_hash"),
            parameters=dict(item.get("parameters") or {}),
            metrics=metrics,
            metadata=dict(item.get("metadata") or {}),
        )
        records.append(rec)
    summary = summarize_records(records)
    return ExperimentAnalysis(experiment_dir=exp_dir, records=records, summary=summary)


def summarize_records(records: Sequence[ExperimentRecord]) -> Dict[str, Any]:
    statuses: Dict[str, int] = {}
    commit_hashes = set()
    durations: List[float] = []
    boards = set()
    metric_names = set()
    for rec in records:
        statuses[rec.status] = statuses.get(rec.status, 0) + 1
        if rec.commit_hash:
            commit_hashes.add(rec.commit_hash)
        if rec.duration_sec is not None:
            durations.append(rec.duration_sec)
        if rec.board:
            boards.add(rec.board)
        metric_names.update(rec.metrics.keys())
    return {
        "record_count": len(records),
        "statuses": dict(sorted(statuses.items())),
        "passed": statuses.get("passed", 0),
        "failed": statuses.get("failed", 0),
        "boards": sorted(boards),
        "commit_hashes": sorted(commit_hashes),
        "single_commit": len(commit_hashes) <= 1,
        "duration_sec_min": min(durations) if durations else None,
        "duration_sec_max": max(durations) if durations else None,
        "duration_sec_mean": sum(durations) / len(durations) if durations else None,
        "metric_names": sorted(metric_names),
    }


def records_to_rows(records: Sequence[ExperimentRecord]) -> List[Dict[str, Any]]:
    metric_names = sorted({m for rec in records for m in rec.metrics})
    param_names = sorted({p for rec in records for p in rec.parameters})
    rows: List[Dict[str, Any]] = []
    for rec in records:
        row: Dict[str, Any] = {
            "design_index": rec.design_index,
            "design_name": rec.design_name,
            "status": rec.status,
            "board": rec.board or "",
            "config_path": rec.config_path or "",
            "model_path": rec.model_path or "",
            "duration_sec": rec.duration_sec if rec.duration_sec is not None else "",
            "commit_hash": rec.commit_hash or "",
        }
        for p in param_names:
            row[f"param.{p}"] = rec.parameters.get(p, "")
        for m in metric_names:
            row[m] = rec.metrics.get(m, "")
        rows.append(row)
    return rows


def write_summary_csv(records: Sequence[ExperimentRecord], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = records_to_rows(records)
    fieldnames = list(rows[0].keys()) if rows else ["design_index", "design_name", "status"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def metric_average(records: Sequence[ExperimentRecord], metric_name: str, *, status: Optional[str] = None) -> Optional[float]:
    values: List[float] = []
    for rec in records:
        if status is not None and rec.status != status:
            continue
        val = rec.metrics.get(metric_name)
        if val is not None:
            values.append(val)
    if not values:
        return None
    return sum(values) / len(values)


def build_metrics_overview_markdown(analysis: ExperimentAnalysis) -> str:
    s = analysis.summary
    lines = [
        "# FPGAI Experiment Analysis",
        "",
        f"Experiment directory: `{analysis.experiment_dir}`",
        "",
        "## Run summary",
        "",
        f"- Total records: {s['record_count']}",
        f"- Passed: {s['passed']}",
        f"- Failed: {s['failed']}",
        f"- Boards: {', '.join(s['boards']) if s['boards'] else 'n/a'}",
        f"- Single commit: {'yes' if s['single_commit'] else 'no'}",
    ]
    if s["commit_hashes"]:
        lines.append(f"- Commit hashes: {', '.join(s['commit_hashes'])}")
    if s["duration_sec_mean"] is not None:
        lines.append(f"- Mean duration: {s['duration_sec_mean']:.2f} s")
    lines.extend(["", "## Metric averages", "", "| Metric | Average over passed designs |", "|---|---:|"])
    for metric in s["metric_names"]:
        avg = metric_average(analysis.records, metric, status="passed")
        if avg is not None:
            lines.append(f"| `{metric}` | {avg:.4g} |")
    lines.extend(["", "## Designs", "", "| # | Design | Status | Duration (s) | Config |", "|---:|---|---|---:|---|"])
    for rec in analysis.records:
        dur = "" if rec.duration_sec is None else f"{rec.duration_sec:.2f}"
        cfg = rec.config_path or ""
        lines.append(f"| {rec.design_index} | `{rec.design_name}` | {rec.status} | {dur} | `{cfg}` |")
    lines.append("")
    return "\n".join(lines)


def write_metrics_overview(analysis: ExperimentAnalysis, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_metrics_overview_markdown(analysis), encoding="utf-8")
    return out
