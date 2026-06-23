import json

from fpgai.experiments.design_matrix import expand_design_matrix
from fpgai.experiments.sweep_runner import SweepRunner


def test_sweep_runner_dry_run_creates_outputs(tmp_path):
    cfg = {
        "name": "dry",
        "command_template": "echo {x}",
        "defaults": {"config_path": "configs/examples/default_compile.yml", "model_path": "model.onnx", "board": "kv260"},
        "parameters": {"x": [1, 2, 3]},
    }
    points = expand_design_matrix(cfg)
    runner = SweepRunner(tmp_path / "experiments" / "dry", repo_root=tmp_path, dry_run=True)
    payload = runner.run_points(points)
    assert len(payload["results"]) == 3
    assert all(r["status"] == "passed" for r in payload["results"])
    assert (tmp_path / "experiments" / "dry" / "summary.md").exists()
    assert (tmp_path / "experiments" / "dry" / "logs").exists()


def test_sweep_runner_failure_does_not_stop(tmp_path):
    cfg = {
        "name": "fail_continue",
        "defaults": {"board": "kv260"},
        "design_points": [
            {"name": "bad", "command": "python -c 'import sys; sys.exit(3)'"},
            {"name": "good", "command": "python -c 'print(42)'"},
        ],
    }
    points = expand_design_matrix(cfg)
    runner = SweepRunner(tmp_path / "exp", repo_root=tmp_path, dry_run=False)
    payload = runner.run_points(points)
    statuses = {r["design_name"]: r["status"] for r in payload["results"]}
    assert statuses["bad"] == "failed"
    assert statuses["good"] == "passed"
