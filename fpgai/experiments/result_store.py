"""Persistent experiment result storage for FPGAI sweeps."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
import csv
import json
import os
import subprocess
from datetime import datetime, timezone


METADATA_FIELDS = [
    "commit_hash",
    "config_path",
    "model_path",
    "tool_version",
    "board",
]


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def get_git_commit(repo_root: str | Path | None = None) -> str:
    """Return current git commit hash, or 'unknown' outside git."""

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


class ResultStore:
    """Store sweep results in JSONL during execution and materialize JSON/CSV."""

    def __init__(self, experiment_dir: str | Path):
        self.experiment_dir = Path(experiment_dir)
        self.results_path = self.experiment_dir / "results.json"
        self.csv_path = self.experiment_dir / "results.csv"
        self.jsonl_path = self.experiment_dir / "results.jsonl"
        self.logs_dir = self.experiment_dir / "logs"
        self.configs_dir = self.experiment_dir / "configs"
        self.plots_dir = self.experiment_dir / "plots"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_records(self) -> List[Dict[str, Any]]:
        # JSONL is the append-only source of truth while a sweep is running.
        # results.json is a materialized view and may lag behind during append.
        if self.jsonl_path.exists():
            rows = []
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            return rows
        if self.results_path.exists():
            data = json.loads(self.results_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                return list(data["results"])
            if isinstance(data, list):
                return list(data)
        return []

    def completed_design_names(self) -> set[str]:
        return {str(r.get("design_name")) for r in self.load_records() if r.get("status") in {"passed", "failed", "skipped"}}

    def append_record(self, record: Mapping[str, Any]) -> None:
        row = dict(record)
        row.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
        self.materialize()

    def materialize(self) -> None:
        records = self.load_records()
        passed_count = sum(1 for row in records if row.get("status") == "passed")
        failed_count = sum(1 for row in records if row.get("status") == "failed")
        skipped_count = sum(1 for row in records if row.get("status") == "skipped")
        payload = {
            "schema_version": 1,
            "experiment_dir": str(self.experiment_dir),
            "result_count": len(records),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "results": records,
        }
        self.results_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        self._write_csv(records)

    def _write_csv(self, records: Iterable[Mapping[str, Any]]) -> None:
        records = list(records)
        base_fields = [
            "design_index",
            "design_name",
            "status",
            "returncode",
            "duration_sec",
            "commit_hash",
            "config_path",
            "model_path",
            "tool_version",
            "board",
            "command",
            "stdout_log",
            "stderr_log",
            "error",
        ]
        param_keys = sorted({k for r in records for k in (r.get("parameters") or {}).keys()})
        metric_keys = sorted({k for r in records for k in (r.get("metrics") or {}).keys()})
        fields = base_fields + [f"param.{k}" for k in param_keys] + [f"metric.{k}" for k in metric_keys]
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in records:
                row = {k: r.get(k) for k in base_fields}
                for k in param_keys:
                    row[f"param.{k}"] = (r.get("parameters") or {}).get(k)
                for k in metric_keys:
                    row[f"metric.{k}"] = (r.get("metrics") or {}).get(k)
                writer.writerow(row)
