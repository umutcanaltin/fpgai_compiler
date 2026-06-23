from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from fpgai.benchmark.pipeline import run_compile_correctness_benchmark
from fpgai.engine.compiler import Compiler


SUPPORTED_MODES = {"inference", "training_on_device"}


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None

    return content if isinstance(content, dict) else None


def _pipeline_mode(config: dict[str, Any]) -> str:
    pipeline = config.get("pipeline", {})

    if not isinstance(pipeline, dict):
        return ""

    return str(pipeline.get("mode", "")).strip().lower()


def _vitis_config(config: dict[str, Any]) -> dict[str, Any]:
    backends = config.get("backends", {})

    if not isinstance(backends, dict):
        return {}

    hls = backends.get("hls", {})

    if not isinstance(hls, dict):
        return {}

    vitis = hls.get("vitis", {})

    return vitis if isinstance(vitis, dict) else {}


def _is_testable_config(config: dict[str, Any]) -> bool:
    mode = _pipeline_mode(config)

    if mode not in SUPPORTED_MODES:
        return False

    backends = config.get("backends", {})
    if not isinstance(backends, dict):
        return False

    hls = backends.get("hls", {})
    if not isinstance(hls, dict):
        return False

    vitis = _vitis_config(config)
    vitis_mode = str(vitis.get("mode", "")).strip().lower()

    return (
        hls.get("enabled") is True
        and vitis.get("enabled") is True
        and vitis_mode in {"csim", "all"}
    )


def _candidate_config_paths() -> list[Path]:
    root = Path.cwd()

    search_locations = [
        root,
        root / "configs",
        root / "tests",
        root / "tests" / "configs",
    ]

    candidates: set[Path] = set()

    for directory in search_locations:
        if not directory.is_dir():
            continue

        candidates.update(directory.glob("*.yml"))
        candidates.update(directory.glob("*.yaml"))

    return sorted(candidates)


def _find_test_config() -> tuple[Path, dict[str, Any]]:
    configured_path = os.getenv("FPGAI_TEST_CONFIG")

    if configured_path:
        path = Path(configured_path).expanduser().resolve()

        if not path.is_file():
            pytest.fail(f"FPGAI_TEST_CONFIG does not exist: {path}")

        config = _load_yaml(path)

        if config is None:
            pytest.fail(f"Cannot load FPGAI YAML configuration: {path}")

        if not _is_testable_config(config):
            pytest.fail(
                f"{path} is not configured for HLS C simulation. "
                "Enable backends.hls, backends.hls.vitis, and csim mode."
            )

        return path, config

    for path in _candidate_config_paths():
        config = _load_yaml(path)

        if config is not None and _is_testable_config(config):
            return path.resolve(), config

    pytest.skip(
        "No testable FPGAI YAML configuration was found. "
        "Set FPGAI_TEST_CONFIG or provide a YAML with HLS and Vitis csim enabled."
    )


def _test_inference(config_path: Path) -> None:
    result = run_compile_correctness_benchmark(
        config_path=config_path,
    )

    assert result.passed, (
        f"Inference benchmark failed using {config_path}. "
        f"See {result.summary_txt}"
    )

    assert result.hls_output_npy.is_file(), (
        f"HLS output was not produced: {result.hls_output_npy}"
    )


def _test_training(config_path: Path) -> None:
    compiler = Compiler.from_yaml(str(config_path))
    result = compiler.compile()

    assert result.hls_ran, "Vitis HLS did not run."
    assert result.hls_ok is True, (
        "Vitis HLS failed. "
        f"See {result.hls_stdout_log} and {result.hls_stderr_log}"
    )

    assert result.training_plan_json is not None
    assert result.training_plan_json.is_file(), (
        f"Training plan was not produced: {result.training_plan_json}"
    )

    assert result.training_summary_txt is not None
    assert result.training_summary_txt.is_file(), (
        f"Training summary was not produced: {result.training_summary_txt}"
    )


def test_compile_correctness() -> None:
    config_path, config = _find_test_config()
    mode = _pipeline_mode(config)

    print(f"Using FPGAI test configuration: {config_path}")
    print(f"Pipeline mode: {mode}")

    if mode == "inference":
        _test_inference(config_path)
    elif mode == "training_on_device":
        _test_training(config_path)
    else:
        pytest.fail(f"Unsupported pipeline mode: {mode}")