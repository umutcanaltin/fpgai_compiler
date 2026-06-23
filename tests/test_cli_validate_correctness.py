from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import fpgai.cli as cli


@dataclass(frozen=True)
class DummyCorrectnessResult:
    config_path: Path
    pipeline_mode: str
    requested: bool
    executed: bool
    passed: bool
    reason: str | None = None
    build_dir: Path | None = None
    bench_dir: Path | None = None
    metrics_json: Path | None = None
    summary_txt: Path | None = None


def test_validate_correctness_dispatch_passes(monkeypatch, tmp_path: Path, capsys):
    config = tmp_path / "inference.yml"
    config.write_text("pipeline:\n  mode: inference\n", encoding="utf-8")

    calls = {"config": None}

    def fake_validate_correctness(config_path):
        calls["config"] = str(config_path)
        return DummyCorrectnessResult(
            config_path=Path(config_path),
            pipeline_mode="inference",
            requested=True,
            executed=True,
            passed=True,
            build_dir=tmp_path / "build",
            bench_dir=tmp_path / "build" / "bench",
            metrics_json=tmp_path / "build" / "bench" / "metrics.json",
            summary_txt=tmp_path / "build" / "bench" / "summary.txt",
        )

    monkeypatch.setattr(cli, "validate_correctness", fake_validate_correctness)
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "validate", "correctness", "--config", str(config)],
    )

    rc = cli.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert calls["config"] == str(config)
    assert "FPGAI Correctness Validation" in out
    assert "Pipeline mode  : inference" in out
    assert "Executed       : True" in out
    assert "Passed         : True" in out


def test_validate_correctness_dispatch_skipped_non_inference(monkeypatch, tmp_path: Path, capsys):
    config = tmp_path / "training.yml"
    config.write_text("pipeline:\n  mode: training_on_device\n", encoding="utf-8")

    def fake_validate_correctness(config_path):
        return DummyCorrectnessResult(
            config_path=Path(config_path),
            pipeline_mode="training_on_device",
            requested=True,
            executed=False,
            passed=False,
            reason=(
                "correctness validation currently supports "
                "pipeline.mode=inference only; got pipeline.mode=training_on_device"
            ),
        )

    monkeypatch.setattr(cli, "validate_correctness", fake_validate_correctness)
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "validate", "correctness", "--config", str(config)],
    )

    rc = cli.main()
    out = capsys.readouterr().out

    assert rc == 1
    assert "Pipeline mode  : training_on_device" in out
    assert "Executed       : False" in out
    assert "Passed         : False" in out
    assert "pipeline.mode=inference only" in out


def test_validate_correctness_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py", "validate", "correctness", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "correctness" in out
    assert "--config" in out
