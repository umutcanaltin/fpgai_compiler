from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping


SweepRunner = Callable[..., int]


def default_sweep_for_experiment_name(name: str) -> str | None:
    """Map benchmark names to public sweep configs."""
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



def _write_single_compile_sweep(
    *,
    item_out: Path,
    name: str,
    config_path: str,
    required_artifacts: list[str] | None = None,
) -> Path:
    """Materialize one explicit compile config through the existing sweep runner.

    This is experiment orchestration only: it does not add a compiler design-space
    feature.  The generated sweep contains exactly one concrete design point and
    therefore preserves the user's explicit YAML architecture selection.
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - environment error
        raise RuntimeError(f"PyYAML is required to materialize compile experiment items: {exc}") from exc

    item_out.mkdir(parents=True, exist_ok=True)
    sweep_path = item_out / "single_compile.yml"
    payload = {
        "name": f"{name}_compile",
        "command_template": "PYTHONPATH=. python -B main.py compile --config {config_path}",
        "defaults": {"config_path": str(config_path)},
        "design_points": [{"name": str(name)}],
        "required_artifacts": list(required_artifacts or []),
        "metadata": {
            "generated_by": "fpgai experiment run",
            "experiment_kind": "explicit_compile",
            "design_space_sweep": False,
        },
    }
    sweep_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return sweep_path


def _collect_child_compile_outputs(items: list[dict[str, Any]]) -> list[Path]:
    """Collect unique compile output directories recorded by child result stores."""
    outputs: list[Path] = []
    seen: set[str] = set()
    for item in items:
        child = item.get("child_summary") or {}
        results_path = child.get("results_path")
        if not results_path:
            continue
        path = Path(str(results_path))
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict) or str(record.get("status")) != "passed":
                continue
            raw = record.get("out_dir") or record.get("project_out_dir")
            if not raw:
                continue
            out = Path(str(raw))
            key = str(out.resolve()) if out.exists() else str(out)
            if key in seen:
                continue
            seen.add(key)
            outputs.append(out)
    return outputs

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
    that contains failed design records. The benchmark-level coordinator must not
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
    """Run a benchmark YAML through the public sweep runner."""
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

    benchmark = payload.get("benchmark") or {}
    inputs = payload.get("inputs") or {}
    experiments = inputs.get("experiments") or {}

    if not isinstance(experiments, dict) or not experiments:
        print("[ERROR] Experiment config has no inputs.experiments entries")
        return 2

    manifest: dict[str, Any] = {
        "kind": "benchmark_run",
        "config": str(cfg_path),
        "out_dir": str(out_path),
        "benchmark": benchmark,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dry_run": bool(dry_run),
        "items": [],
    }

    failed = 0

    for name, source in experiments.items():
        item_out = out_path / str(name)
        item_kind = "sweep"
        required_artifacts: list[str] = []

        if isinstance(source, Mapping):
            item_kind = str(source.get("kind") or "compile").strip().lower()
            if item_kind != "compile":
                source_str = str(source.get("config") or source.get("sweep") or "")
                sweep_config = source_str if source_str else default_sweep_for_experiment_name(str(name))
            else:
                source_str = str(source.get("config") or "")
                required_artifacts = [str(x) for x in (source.get("required_artifacts") or [])]
                if source_str and Path(source_str).exists():
                    sweep_config = str(
                        _write_single_compile_sweep(
                            item_out=item_out,
                            name=str(name),
                            config_path=source_str,
                            required_artifacts=required_artifacts,
                        )
                    )
                else:
                    sweep_config = None
        else:
            source_str = str(source)
            if source_str.endswith((".yml", ".yaml")) and Path(source_str).exists():
                sweep_config = source_str
            else:
                sweep_config = default_sweep_for_experiment_name(str(name))

        item: dict[str, Any] = {
            "name": str(name),
            "kind": item_kind,
            "source": source_str,
            "sweep_config": sweep_config,
            "out_dir": str(item_out),
            "status": "pending",
            "returncode": None,
        }
        if required_artifacts:
            item["required_artifacts"] = required_artifacts

        if not sweep_config or not Path(sweep_config).exists():
            item["status"] = "failed" if item_kind == "compile" else "skipped"
            item["reason"] = (
                "explicit compile config was not found"
                if item_kind == "compile"
                else "no runnable public sweep config was found"
            )
            manifest["items"].append(item)
            if item_kind == "compile":
                failed += 1
                print(f"[FAIL] {name}: explicit compile config not found: {source_str}")
            else:
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

    compile_outputs = _collect_child_compile_outputs(manifest["items"])
    if compile_outputs and not dry_run:
        try:
            from fpgai.reporting.benchmark_results import write_master_results

            master_json = out_path / "master_results.json"
            master_csv = out_path / "master_results.csv"
            master_md = out_path / "master_results.md"
            schema_dir = out_path / "schema"
            master = write_master_results(
                compile_outputs,
                output_json=master_json,
                output_csv=master_csv,
                output_md=master_md,
                schema_json=schema_dir / "master_result_schema.json",
                schema_md=schema_dir / "master_result_schema.md",
            )
            manifest["master_results"] = {
                "status": master.get("status"),
                "summary": master.get("summary"),
                "json": str(master_json),
                "csv": str(master_csv),
                "markdown": str(master_md),
            }
        except Exception as exc:
            manifest["master_results"] = {"status": "failed", "error": str(exc)}
            failed += 1

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
                "kind": "benchmark_status",
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
