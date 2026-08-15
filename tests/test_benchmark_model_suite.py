from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from fpgai.experiments.model_suite import (
    MODEL_SPECS,
    BENCHMARK_MODEL_CONFIGS,
    make_benchmark_config,
)


def test_benchmark_model_suite_contains_compact_medium_and_large_workloads() -> None:
    model_names = {name for name, _model, _shape in MODEL_SPECS}
    assert "mlp_mnist" in model_names
    assert "cifar_small_cnn" in model_names
    assert "large_ddr_stress_cnn" in model_names
    assert "tiny_yolo_like" in model_names

    benchmark_names = set(BENCHMARK_MODEL_CONFIGS)
    assert "compact_onchip_mnist_mlp" in benchmark_names
    assert "compact_onchip_mnist_training" in benchmark_names
    assert "medium_ddr_cifar_cnn" in benchmark_names
    assert "medium_ddr_cifar_training" in benchmark_names
    assert "large_ddr_stress_cnn" in benchmark_names
    assert "large_ddr_yolo_like" in benchmark_names


def test_benchmark_model_configs_use_existing_compiler_schema() -> None:
    for name, spec in BENCHMARK_MODEL_CONFIGS.items():
        cfg = make_benchmark_config(name, spec)
        assert cfg["version"] == 1
        assert cfg["project"]["out_dir"] == f"build/benchmark/{name}"
        assert cfg["model"]["format"] == "onnx"
        assert cfg["model"]["path"].startswith("models/suite/")
        assert cfg["pipeline"]["mode"] in {"inference", "training_on_device"}
        assert cfg["targets"]["platform"]["board"] == "kv260"
        assert cfg["benchmark"]["model_class"] == name
        assert cfg["benchmark"]["memory_regime"] in {
            "onchip",
            "ddr_backed",
            "ddr_backed_tiled",
        }


def test_generated_static_benchmark_config_files_are_valid_yaml() -> None:
    # These files are included in the patch so users can compile the benchmark set
    # without first running the model-suite generator.
    root = Path("examples/benchmark/models")
    if not root.exists():
        pytest.skip("optional benchmark example pack is not present in this repository archive")
    expected = {
        "compact_onchip_mnist_mlp.yml",
        "compact_onchip_mnist_training.yml",
        "medium_ddr_cifar_cnn.yml",
        "medium_ddr_cifar_training.yml",
        "large_ddr_stress_cnn.yml",
        "large_ddr_yolo_like.yml",
    }
    for filename in expected:
        p = root / filename
        assert p.exists(), p
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert "benchmark" not in data
        assert data["project"]["name"].startswith("benchmark_")
        assert data["model"]["format"] == "onnx"
        assert data["pipeline"]["mode"] in {"inference", "training_on_device"}
