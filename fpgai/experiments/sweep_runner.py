"""Run FPGAI experiment sweeps with resume and failure isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
import json
import os
import shlex
import subprocess
import time

import yaml
from dataclasses import replace

from .design_matrix import DesignPoint, expand_design_matrix, load_sweep_config, render_template
from .config_materializer import materialize_design_config
from .result_store import ResultStore, get_git_commit
from .report_writer import write_summary_markdown


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _extract_metrics_from_paths(paths: Iterable[str | Path]) -> Dict[str, Any]:
    """Best-effort metric extraction from known FPGAI JSON outputs."""

    metrics: Dict[str, Any] = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Common HLS calibration metrics
        summary = data.get("summary") if isinstance(data, dict) else None
        if isinstance(summary, dict):
            for prefix_key in ["raw_mean_absolute_percentage_error", "calibrated_mean_absolute_percentage_error"]:
                val = summary.get(prefix_key)
                if isinstance(val, dict):
                    short = "raw_mape" if prefix_key.startswith("raw") else "cal_mape"
                    for k, v in val.items():
                        metrics[f"{short}.{k}"] = v
            if "sample_count" in summary:
                metrics["calibration.sample_count"] = summary.get("sample_count")
        # Generic benchmark pass/fail metrics
        for key in ["accuracy", "latency_ms", "latency_cycles", "lut", "ff", "dsp", "bram", "passed"]:
            if isinstance(data, dict) and key in data:
                metrics[key] = data[key]
    return metrics


def _assign_design_artifact_out_dir(
    config_path: str | Path,
    *,
    experiment_dir: str | Path,
    design_name: str,
) -> Dict[str, Any]:
    """Make project.out_dir unique and inside the experiment directory.

    Earlier sweeps reused the base config's project.out_dir, for example
    build/fpgai_example_dense. That allowed later design points to overwrite
    earlier generated C/C++ artifacts and left no generated code under the
    experiment folder for precision design-effect checks.
    """

    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    project = data.get("project")
    if not isinstance(project, dict):
        project = {}
        data["project"] = project

    safe = _safe_name(design_name)
    artifact_dir = Path(experiment_dir) / "artifacts" / safe
    build_dir = artifact_dir / "build"
    previous = project.get("out_dir")

    project["out_dir"] = str(build_dir)
    project.setdefault("name", safe)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    return {
        "artifact_dir": str(artifact_dir),
        "project_out_dir": str(build_dir),
        "previous_project_out_dir": previous,
    }


class SweepRunner:
    def __init__(
        self,
        experiment_dir: str | Path,
        repo_root: str | Path | None = None,
        tool_version: str | None = None,
        dry_run: bool = False,
        resume: bool = True,
        timeout_sec: int | None = None,
        materialize_configs: Mapping[str, Any] | None = None,
        command_template: str | None = None,
    ) -> None:
        self.store = ResultStore(experiment_dir)
        self.repo_root = Path(repo_root or os.getcwd())
        self.tool_version = tool_version or os.environ.get("FPGAI_TOOL_VERSION", "unknown")
        self.dry_run = dry_run
        self.resume = resume
        self.timeout_sec = timeout_sec
        self.materialize_configs = dict(materialize_configs or {})
        self.command_template = command_template

    def run_points(self, points: Sequence[DesignPoint]) -> Dict[str, Any]:
        points = [self._materialize_point_config(point) for point in points]
        completed = self.store.completed_design_names() if self.resume else set()
        for point in points:
            if point.name in completed:
                self.store.append_record(self._record_for_skipped(point, "already_completed"))
                continue
            record = self._run_one(point)
            self.store.append_record(record)
        self.store.materialize()
        payload = json.loads(self.store.results_path.read_text(encoding="utf-8"))
        write_summary_markdown(self.store.experiment_dir, payload)
        return payload


    def _materialize_point_config(self, point: DesignPoint) -> DesignPoint:
        cfg = self.materialize_configs
        if not cfg or not cfg.get("enabled", False):
            return point
        base_config = point.config_path or point.parameters.get("config_path")
        if not base_config:
            return point
        base_path = Path(str(base_config))
        if not base_path.is_absolute():
            base_path = self.repo_root / base_path
        if not base_path.exists():
            # Keep original point; execution will fail normally if config is needed.
            return point
        config_dir_name = str(cfg.get("directory") or "configs")
        out_path = self.store.experiment_dir / config_dir_name / f"{_safe_name(point.name)}.yml"
        try:
            report = materialize_design_config(
                base_path,
                out_path,
                point.parameters,
                cfg,
                design_name=point.name,
                repo_root=self.repo_root,
            )
        except TypeError:
            # Backwards compatibility for older materializer implementations.
            report = materialize_design_config(base_path, out_path, point.parameters, cfg)

        artifact_info: Dict[str, Any] = {}
        if bool(cfg.get("preserve_artifacts", True)):
            artifact_info = _assign_design_artifact_out_dir(
                out_path,
                experiment_dir=self.store.experiment_dir,
                design_name=point.name,
            )
            if artifact_info:
                report.setdefault("applied", {})["project_out_dir"] = "project.out_dir"
                report.update(artifact_info)
                metadata_path = report.get("metadata_path")
                if metadata_path:
                    try:
                        Path(str(metadata_path)).write_text(
                            json.dumps(report, indent=2, sort_keys=True, default=str),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
        rel_out = out_path if out_path.is_absolute() else out_path
        try:
            rel_out = out_path.relative_to(self.repo_root)
        except Exception:
            pass
        new_params = dict(point.parameters)
        new_params["base_config_path"] = str(point.config_path)
        new_params["config_path"] = str(rel_out)
        metadata = dict(point.metadata)
        metadata["materialized_config"] = report
        command = point.command
        if self.command_template:
            command = render_template(self.command_template, new_params)
        elif command and point.config_path:
            command = command.replace(str(point.config_path), str(rel_out))
        return replace(
            point,
            parameters=new_params,
            command=command,
            config_path=str(rel_out),
            metadata=metadata,
        )

    def _record_base(self, point: DesignPoint) -> Dict[str, Any]:
        metadata = dict(point.metadata)
        materialized = metadata.get("materialized_config")
        if not isinstance(materialized, Mapping):
            materialized = {}
        record = {
            "schema_version": 1,
            "design_index": point.index,
            "design_name": point.name,
            "parameters": dict(point.parameters),
            "command": point.command,
            "config_path": point.config_path,
            "model_path": point.model_path,
            "board": point.board,
            "tool_version": self.tool_version,
            "commit_hash": get_git_commit(self.repo_root),
            "metadata": metadata,
        }
        if materialized.get("artifact_dir"):
            record["artifact_dir"] = materialized.get("artifact_dir")
        if materialized.get("project_out_dir"):
            record["project_out_dir"] = materialized.get("project_out_dir")
        return record

    def _record_for_skipped(self, point: DesignPoint, reason: str) -> Dict[str, Any]:
        r = self._record_base(point)
        r.update({"status": "skipped", "returncode": None, "duration_sec": 0.0, "error": reason})
        return r

    def _run_one(self, point: DesignPoint) -> Dict[str, Any]:
        base = self._record_base(point)
        log_base = self.store.logs_dir / _safe_name(point.name)
        stdout_path = log_base.with_suffix(".stdout.log")
        stderr_path = log_base.with_suffix(".stderr.log")
        base["stdout_log"] = str(stdout_path)
        base["stderr_log"] = str(stderr_path)
        if self.dry_run:
            stdout_path.write_text(f"DRY RUN: {point.command}\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            base.update({"status": "passed", "returncode": 0, "duration_sec": 0.0, "metrics": {"dry_run": True}})
            return base
        if not point.command:
            base.update({"status": "failed", "returncode": None, "duration_sec": 0.0, "error": "missing command"})
            return base
        start = time.time()
        try:
            proc = subprocess.run(
                point.command,
                cwd=str(self.repo_root),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_sec,
            )
            duration = time.time() - start
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")
            status = "passed" if proc.returncode == 0 else "failed"
            metric_paths = []
            # Look for metrics in explicit parameter paths first.
            for key in ["metrics_json", "calibration_json", "estimate_vs_hls_json"]:
                if key in point.parameters:
                    metric_paths.append(point.parameters[key])
            metrics = _extract_metrics_from_paths(metric_paths)
            base.update({"status": status, "returncode": proc.returncode, "duration_sec": duration, "metrics": metrics})
            if proc.returncode != 0:
                base["error"] = f"command failed with returncode {proc.returncode}"
            return base
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            base.update({"status": "failed", "returncode": None, "duration_sec": duration, "error": "timeout"})
            return base
        except Exception as exc:
            duration = time.time() - start
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            base.update({"status": "failed", "returncode": None, "duration_sec": duration, "error": str(exc)})
            return base


def run_sweep_config(
    config_path: str | Path,
    experiment_dir: str | Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    resume: bool = True,
    timeout_sec: int | None = None,
) -> Dict[str, Any]:
    config = load_sweep_config(config_path)
    points = expand_design_matrix(config, limit=limit)
    name = str(config.get("name") or Path(config_path).stem)
    out_dir = Path(experiment_dir or Path("experiments") / name)
    materialize_cfg = config.get("materialize_configs") or config.get("generated_configs") or {}
    command_template = config.get("command_template") or (config.get("defaults") or {}).get("command_template")
    runner = SweepRunner(
        out_dir,
        dry_run=dry_run,
        resume=resume,
        timeout_sec=timeout_sec,
        materialize_configs=materialize_cfg,
        command_template=str(command_template) if command_template else None,
    )
    return runner.run_points(points)
