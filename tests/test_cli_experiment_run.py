from pathlib import Path
import json
import sys
import pytest

from fpgai.cli import main as cli_main
from fpgai.experiments.paper_runner import run_experiment_from_config


def test_experiment_run_dry_run_writes_manifest(tmp_path, monkeypatch):
    config = Path("configs/experiments/arxiv_paper.yml")
    assert config.exists(), "configs/experiments/arxiv_paper.yml is missing"

    out = tmp_path / "paper_experiment"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fpgai",
            "experiment",
            "run",
            "--config",
            str(config),
            "--out",
            str(out),
            "--dry-run",
            "--max-design-points",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli_main()

    assert exc.value.code == 0

    manifest = out / "manifest.json"
    status = out / "experiment_status.json"

    assert manifest.exists()
    assert status.exists()

    data = json.loads(manifest.read_text())
    assert data["kind"] == "paper_experiment_run"
    assert data["dry_run"] is True
    assert "items" in data
    assert len(data["items"]) > 0
    assert all(item["status"] in {"dry_run", "skipped"} for item in data["items"])


def test_experiment_run_propagates_child_sweep_failed_count(tmp_path):
    sweep_cfg = tmp_path / "toy_sweep.yml"
    sweep_cfg.write_text("name: toy_sweep\n")

    experiment_cfg = tmp_path / "paper.yml"
    experiment_cfg.write_text(
        f"""
version: 1
paper:
  title: Toy
  stage: test
inputs:
  experiments:
    toy: {sweep_cfg}
""".strip()
        + "\n"
    )

    def fake_sweep_runner(config_path, *, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(
                {
                    "kind": "sweep_results",
                    "failed_count": 1,
                    "passed_count": 0,
                    "records": [{"status": "failed"}],
                }
            )
        )
        return 0

    out_dir = tmp_path / "paper_out"
    rc = run_experiment_from_config(
        str(experiment_cfg),
        out_dir=str(out_dir),
        run_sweep_callable=fake_sweep_runner,
    )

    assert rc == 1
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["failed_count"] == 1
    assert manifest["passed_count"] == 0
    assert manifest["items"][0]["status"] == "failed"
    assert manifest["items"][0]["reason"] == "child sweep reported failed_count=1"
