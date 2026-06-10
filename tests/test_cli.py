from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import fpgai.cli as cli


def _config(
    *,
    mode: str,
    benchmark_enabled: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(mode=mode),
        raw={
            "benchmark": {
                "enabled": benchmark_enabled,
            }
        },
    )


def _compile_result(
    tmp_path: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        out_dir=tmp_path,
        summary=lambda: "compile summary",
    )


def _benchmark_result(
    tmp_path: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        passed=True,
        metrics_json=tmp_path / "metrics.json",
        summary_txt=tmp_path / "summary.txt",
        quant_metrics_json=None,
        quant_summary_txt=None,
        precision_sweep_results_json=None,
        precision_sweep_summary_txt=None,
    )


def test_compile_action_does_not_run_enabled_benchmark(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _config(
        mode="inference",
        benchmark_enabled=True,
    )
    calls = {
        "compile": 0,
        "benchmark": 0,
    }

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: cfg,
    )
    monkeypatch.setattr(
        cli,
        "print_summary",
        lambda _: None,
    )

    class FakeCompiler:
        def __init__(self, config) -> None:
            assert config is cfg

        def compile(self):
            calls["compile"] += 1
            return _compile_result(tmp_path)

    def fake_benchmark(
        *,
        config_path: str,
    ):
        assert config_path == "config.yml"
        calls["benchmark"] += 1
        return _benchmark_result(tmp_path)

    monkeypatch.setattr(
        cli,
        "Compiler",
        FakeCompiler,
    )
    monkeypatch.setattr(
        cli,
        "run_compile_correctness_benchmark",
        fake_benchmark,
    )

    result = cli.run_from_config(
        "config.yml",
        action="compile",
    )

    assert result == 0
    assert calls == {
        "compile": 1,
        "benchmark": 0,
    }


def test_benchmark_action_runs_when_yaml_flag_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _config(
        mode="inference",
        benchmark_enabled=False,
    )
    calls = {
        "compile": 0,
        "benchmark": 0,
    }

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: cfg,
    )
    monkeypatch.setattr(
        cli,
        "print_summary",
        lambda _: None,
    )

    class FakeCompiler:
        def __init__(self, config) -> None:
            assert config is cfg

        def compile(self):
            calls["compile"] += 1
            return _compile_result(tmp_path)

    def fake_benchmark(
        *,
        config_path: str,
    ):
        assert config_path == "config.yml"
        calls["benchmark"] += 1
        return _benchmark_result(tmp_path)

    monkeypatch.setattr(
        cli,
        "Compiler",
        FakeCompiler,
    )
    monkeypatch.setattr(
        cli,
        "run_compile_correctness_benchmark",
        fake_benchmark,
    )

    result = cli.run_from_config(
        "config.yml",
        action="benchmark",
    )

    assert result == 0
    assert calls == {
        "compile": 0,
        "benchmark": 1,
    }


def test_benchmark_action_rejects_training_mode(
    monkeypatch,
    capsys,
) -> None:
    cfg = _config(
        mode="training_on_device",
        benchmark_enabled=True,
    )

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: cfg,
    )
    monkeypatch.setattr(
        cli,
        "print_summary",
        lambda _: None,
    )

    result = cli.run_from_config(
        "config.yml",
        action="benchmark",
    )

    captured = capsys.readouterr()

    assert result == 2
    assert (
        "supports pipeline.mode=inference only"
        in captured.err
    )


def test_auto_action_preserves_legacy_config_behavior(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _config(
        mode="inference",
        benchmark_enabled=True,
    )
    calls = {
        "benchmark": 0,
    }

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: cfg,
    )
    monkeypatch.setattr(
        cli,
        "print_summary",
        lambda _: None,
    )

    def fake_benchmark(
        *,
        config_path: str,
    ):
        assert config_path == "config.yml"
        calls["benchmark"] += 1
        return _benchmark_result(tmp_path)

    monkeypatch.setattr(
        cli,
        "run_compile_correctness_benchmark",
        fake_benchmark,
    )

    result = cli.run_from_config(
        "config.yml",
        action="auto",
    )

    assert result == 0
    assert calls["benchmark"] == 1


def test_inspect_action_writes_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _config(
        mode="inference",
        benchmark_enabled=False,
    )
    output_path = tmp_path / "inspection.json"
    calls = {
        "inspect": 0,
    }

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: cfg,
    )

    class FakeReport:
        compilation_ready = True

        @staticmethod
        def summary() -> str:
            return "inspection summary"

        @staticmethod
        def write_json(path: str) -> Path:
            calls["inspect"] += 1

            output = Path(path)
            output.write_text(
                "{}\n",
                encoding="utf-8",
            )
            return output

    monkeypatch.setattr(
        cli,
        "inspect_config",
        lambda config: FakeReport(),
    )

    result = cli.inspect_from_config(
        "config.yml",
        json_output=str(output_path),
    )

    assert result == 0
    assert output_path.read_text(
        encoding="utf-8",
    ) == "{}\n"
    assert calls["inspect"] == 1


def test_inspect_returns_failure_for_unsupported_model(
    monkeypatch,
) -> None:
    cfg = _config(
        mode="inference",
        benchmark_enabled=False,
    )

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: cfg,
    )

    class FakeReport:
        compilation_ready = False

        @staticmethod
        def summary() -> str:
            return "unsupported model"

    monkeypatch.setattr(
        cli,
        "inspect_config",
        lambda config: FakeReport(),
    )

    result = cli.inspect_from_config(
        "config.yml",
    )

    assert result == 1