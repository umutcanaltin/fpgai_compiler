from __future__ import annotations

import json
from pathlib import Path

from fpgai.reporting.artifacts import build_report
from fpgai.validation.results import validate_results


def _write_smoke_tree(root: Path) -> None:
    child = root / "training_convergence"
    child.mkdir(parents=True)
    (child / "results.json").write_text(
        json.dumps(
            {
                "failed_count": 0,
                "passed_count": 1,
                "skipped_count": 0,
                "results": [
                    {
                        "design_name": "d0",
                        "status": "passed",
                        "returncode": 0,
                        "parameters": {"policy": "balanced"},
                        "metrics": {"latency_ms": 1.25},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status = {
        "kind": "paper_experiment_status",
        "failed_count": 0,
        "passed_count": 1,
        "skipped_count": 0,
        "dry_run_count": 0,
        "items": [
            {
                "name": "training_convergence",
                "status": "passed",
                "out_dir": str(child),
                "sweep_config": "configs/sweeps/training_multi_epoch_convergence.yml",
                "child_summary": {
                    "results_path": str(child / "results.json"),
                    "failed_count": 0,
                    "passed_count": 1,
                },
            }
        ],
    }
    (root / "experiment_status.json").write_text(json.dumps(status), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"items": status["items"]}), encoding="utf-8")


def test_build_report_writes_public_artifacts(tmp_path: Path):
    src = tmp_path / "paper"
    src.mkdir()
    _write_smoke_tree(src)

    out = tmp_path / "report"
    result = build_report(src, out)

    assert result.result_count == 1
    assert result.failed_count == 0
    assert (out / "summary.md").exists()
    assert (out / "results_table.csv").exists()
    assert (out / "claim_traceability.md").exists()


def test_validate_results_accepts_consistent_tree(tmp_path: Path):
    src = tmp_path / "paper"
    src.mkdir()
    _write_smoke_tree(src)

    result = validate_results(src)

    assert result.passed
    assert result.error_count == 0
    assert result.child_count == 1


def test_validate_results_rejects_false_pass(tmp_path: Path):
    src = tmp_path / "paper"
    src.mkdir()
    _write_smoke_tree(src)

    child_results = src / "training_convergence" / "results.json"
    payload = json.loads(child_results.read_text(encoding="utf-8"))
    payload["failed_count"] = 1
    payload["results"][0]["status"] = "failed"
    child_results.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_results(src)

    assert not result.passed
    assert result.error_count == 1
    assert "false pass" in result.issues[0].message
