from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


SweepRunner = Callable[..., int]


def default_sweep_for_experiment_name(name: str) -> str | None:
    """Map paper experiment names to public sweep configs."""
    mapping = {
        "precision_selection": "configs/sweeps/precision_selection.yml",
        "precision": "configs/sweeps/precision_selection.yml",
        "pipeline_policy": "configs/sweeps/pipeline_policy_strength.yml",
        "pipeline": "configs/sweeps/pipeline_policy_strength.yml",
        "parallel_envelope": "configs/sweeps/parallelism_feasible_envelope.yml",
        "parallelism": "configs/sweeps/parallelism_feasible_envelope.yml",
        "hardware_knobs": "configs/sweeps/hardware_knob_validation.yml",
        "memory_strategy": "configs/sweeps/memory_strategy.yml",
        "training_convergence": "configs/sweeps/training_multi_epoch_convergence.yml",
        "training": "configs/sweeps/training_multi_epoch_convergence.yml",
        "vivado_impl": "configs/sweeps/vivado_bridge.yml",
        "vivado_bridge": "configs/sweeps/vivado_bridge.yml",
    }
    return mapping.get(str(name))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - environment error
        raise RuntimeError(
            f"PyYAML is required to read experiment configs: {exc}"
        ) from exc

    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to parse experiment config {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Experiment config must contain a YAML mapping: {path}")

    return payload


def _load_child_sweep_summary(item_out: Path) -> dict[str, Any]:
    """Read child sweep status from its results.json when available.

    The public sweep runner can return successfully after writing a results file
    that contains failed design records. The paper-level coordinator must not
    report such a child as passed. This helper extracts stable counters from the
    child sweep output and returns an empty dict when no summary exists.
    """
    results_path = item_out / "results.json"
    if not results_path.exists():
        return {}

    try:
        payload = json.loads(results_path.read_text())
    except Exception as exc:
        return {
            "results_path": str(results_path),
            "results_read_error": str(exc),
        }

    if not isinstance(payload, dict):
        return {
            "results_path": str(results_path),
            "results_read_error": "results.json is not a JSON object",
        }

    summary: dict[str, Any] = {"results_path": str(results_path)}
    for key in (
        "failed_count",
        "passed_count",
        "dry_run_count",
        "skipped_count",
        "total_count",
        "design_points",
        "failed",
        "passed",
    ):
        if key in payload:
            summary[key] = payload.get(key)

    # Some result schemas store records under a list. Use that as a fallback.
    records = payload.get("records")
    if isinstance(records, list):
        summary.setdefault("total_count", len(records))
        failed_records = [
            record
            for record in records
            if isinstance(record, dict)
            and str(record.get("status", "")).lower() in {"failed", "error", "timeout"}
        ]
        summary.setdefault("failed_count", len(failed_records))
        summary.setdefault("passed_count", len(records) - len(failed_records))

    return summary


def _child_sweep_failed(returncode: int, child_summary: dict[str, Any]) -> tuple[bool, str | None]:
    """Classify a child sweep from return code plus results.json counters."""
    if returncode != 0:
        return True, f"child sweep returned non-zero exit code {returncode}"

    if child_summary.get("results_read_error"):
        return True, "child sweep results.json could not be read"

    failed_count = child_summary.get("failed_count")
    if isinstance(failed_count, bool):
        # bool is an int subclass; treat it separately to avoid confusing True as 1.
        failed_count = int(failed_count)

    if isinstance(failed_count, int) and failed_count > 0:
        return True, f"child sweep reported failed_count={failed_count}"

    failed_flag = child_summary.get("failed")
    if isinstance(failed_flag, bool) and failed_flag:
        return True, "child sweep reported failed=True"

    return False, None


def run_experiment_from_config(
    config_path: str,
    *,
    out_dir: str,
    run_sweep_callable: SweepRunner,
    max_design_points: int | None = None,
    timeout_sec: int | None = None,
    dry_run: bool = False,
    repo_root: str | None = None,
) -> int:
    """Run a paper experiment YAML through the public sweep runner."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] Experiment config not found: {cfg_path}")
        return 2

    try:
        payload = _load_yaml(cfg_path)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 2

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paper = payload.get("paper") or {}
    inputs = payload.get("inputs") or {}
    experiments = inputs.get("experiments") or {}

    if not isinstance(experiments, dict) or not experiments:
        print("[ERROR] Experiment config has no inputs.experiments entries")
        return 2

    manifest: dict[str, Any] = {
        "kind": "paper_experiment_run",
        "config": str(cfg_path),
        "out_dir": str(out_path),
        "paper": paper,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dry_run": bool(dry_run),
        "items": [],
    }

    failed = 0

    for name, source in experiments.items():
        source_str = str(source)

        if source_str.endswith((".yml", ".yaml")) and Path(source_str).exists():
            sweep_config = source_str
        else:
            sweep_config = default_sweep_for_experiment_name(str(name))

        item_out = out_path / str(name)

        item: dict[str, Any] = {
            "name": str(name),
            "source": source_str,
            "sweep_config": sweep_config,
            "out_dir": str(item_out),
            "status": "pending",
            "returncode": None,
        }

        if not sweep_config or not Path(sweep_config).exists():
            item["status"] = "skipped"
            item["reason"] = "no runnable public sweep config was found"
            manifest["items"].append(item)
            print(f"[SKIP] {name}: no runnable public sweep config found")
            continue

        print(f"[RUN] {name}: {sweep_config} -> {item_out}")

        if dry_run:
            item["status"] = "dry_run"
            item["returncode"] = 0
            manifest["items"].append(item)
            continue

        rc = run_sweep_callable(
            sweep_config,
            out_dir=str(item_out),
            max_design_points=max_design_points,
            timeout_sec=timeout_sec,
            dry_run=False,
            repo_root=repo_root,
        )

        item["returncode"] = int(rc)
        child_summary = _load_child_sweep_summary(item_out)
        if child_summary:
            item["child_summary"] = child_summary

        child_failed, reason = _child_sweep_failed(int(rc), child_summary)
        item["status"] = "failed" if child_failed else "passed"
        if reason:
            item["reason"] = reason

        if child_failed:
            failed += 1

        manifest["items"].append(item)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["failed_count"] = failed
    manifest["passed_count"] = sum(
        1 for item in manifest["items"] if item.get("status") == "passed"
    )
    manifest["skipped_count"] = sum(
        1 for item in manifest["items"] if item.get("status") == "skipped"
    )
    manifest["dry_run_count"] = sum(
        1 for item in manifest["items"] if item.get("status") == "dry_run"
    )

    manifest_path = out_path / "manifest.json"
    status_path = out_path / "experiment_status.json"

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    status_path.write_text(
        json.dumps(
            {
                "kind": "paper_experiment_status",
                "config": str(cfg_path),
                "out_dir": str(out_path),
                "failed_count": manifest["failed_count"],
                "passed_count": manifest["passed_count"],
                "skipped_count": manifest["skipped_count"],
                "dry_run_count": manifest["dry_run_count"],
                "items": manifest["items"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("============== FPGAI Experiment Run Summary ==============")
    print(f"Config        : {cfg_path}")
    print(f"Out dir       : {out_path}")
    print(f"Passed        : {manifest['passed_count']}")
    print(f"Failed        : {manifest['failed_count']}")
    print(f"Skipped       : {manifest['skipped_count']}")
    print(f"Dry run       : {manifest['dry_run_count']}")
    print(f"Manifest      : {manifest_path}")
    print("===========================================================")

    return 1 if failed else 0
