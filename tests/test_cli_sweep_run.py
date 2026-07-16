from __future__ import annotations

from pathlib import Path

import fpgai.cli as cli


def test_sweep_run_dispatches_to_package_runner(monkeypatch, tmp_path, capsys):
    calls = {}
    cfg = tmp_path / "sweep.yml"
    cfg.write_text("name: smoke\n", encoding="utf-8")

    def fake_run_sweep_config(
        config_path,
        *,
        experiment_dir=None,
        limit=None,
        dry_run=False,
        timeout_sec=None,
    ):
        calls["config_path"] = Path(config_path)
        calls["experiment_dir"] = Path(experiment_dir)
        calls["limit"] = limit
        calls["dry_run"] = dry_run
        calls["timeout_sec"] = timeout_sec
        return {
            "experiment_dir": str(experiment_dir),
            "result_count": 1,
            "failed_count": 0,
            "results": [{"status": "passed"}],
        }

    monkeypatch.setattr(cli, "run_sweep_config", fake_run_sweep_config)

    rc = cli.run_sweep_from_config(
        str(cfg),
        out_dir="experiments/smoke",
        max_design_points=1,
        timeout_sec=12,
        dry_run=True,
        repo_root=str(tmp_path),
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert calls["config_path"] == cfg
    assert calls["experiment_dir"] == tmp_path / "experiments" / "smoke"
    assert calls["limit"] == 1
    assert calls["dry_run"] is True
    assert calls["timeout_sec"] == 12
    assert "FPGAI Sweep Run" in out
    assert "[OK] Sweep completed successfully." in out


def test_parser_accepts_sweep_run_arguments():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "sweep",
            "run",
            "--config",
            "configs/sweeps/inference_precision.yml",
            "--out",
            "experiments/smoke",
            "--max-design-points",
            "1",
            "--timeout-sec",
            "1200",
            "--dry-run",
        ]
    )

    assert args.command == "sweep"
    assert args.sweep_command == "run"
    assert args.config == "configs/sweeps/inference_precision.yml"
    assert args.out == "experiments/smoke"
    assert args.max_design_points == 1
    assert args.timeout_sec == 1200
    assert args.dry_run is True


def test_documented_multi_epoch_sweep_command_includes_required_out_argument() -> None:
    readme = Path("configs/examples/README.md").read_text(encoding="utf-8")
    marker = "--config configs/sweeps/training_multi_epoch_convergence.yml"
    marker_index = readme.index(marker)
    block_start = readme.rfind("```bash", 0, marker_index)
    block_end = readme.index("```", marker_index)
    command_block = readme[block_start:block_end]

    assert readme.count(marker) == 1
    assert "python -m fpgai.cli sweep run" in command_block
    assert marker in command_block
    assert "--out paper_experiments/training_multi_epoch_convergence" in command_block



def test_sweep_run_reports_failed_design_details_and_returns_nonzero(monkeypatch, tmp_path, capsys):
    cfg = tmp_path / "sweep.yml"
    cfg.write_text("name: fail\n", encoding="utf-8")
    stderr_log = tmp_path / "bad.stderr.log"
    stderr_log.write_text("first line\nroot cause line\n", encoding="utf-8")

    def fake_run_sweep_config(*args, **kwargs):
        return {
            "experiment_dir": str(tmp_path / "exp"),
            "result_count": 1,
            "failed_count": 1,
            "results": [
                {
                    "design_name": "bad_point",
                    "status": "failed",
                    "returncode": 3,
                    "error": "command failed with returncode 3",
                    "config_path": "generated/bad.yml",
                    "stdout_log": str(tmp_path / "bad.stdout.log"),
                    "stderr_log": str(stderr_log),
                }
            ],
        }

    monkeypatch.setattr(cli, "run_sweep_config", fake_run_sweep_config)

    rc = cli.run_sweep_from_config(str(cfg), out_dir=str(tmp_path / "exp"), repo_root=str(tmp_path))
    out = capsys.readouterr().out

    assert rc == 1
    assert "Sweep failure details" in out
    assert "bad_point" in out
    assert "root cause line" in out
    assert "[ERROR] Sweep completed with 1 failed record(s)." in out
