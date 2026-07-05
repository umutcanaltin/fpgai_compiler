from __future__ import annotations

from pathlib import Path

import yaml

from fpgai.experiments.model_suite import (
    MODEL_SPECS,
    PAPER_MODEL_CONFIGS,
    make_paper_config,
)


def test_paper_model_suite_contains_compact_medium_and_large_workloads() -> None:
    model_names = {name for name, _model, _shape in MODEL_SPECS}
    assert "mlp_mnist" in model_names
    assert "cifar_small_cnn" in model_names
    assert "large_ddr_stress_cnn" in model_names
    assert "tiny_yolo_like" in model_names

    paper_names = set(PAPER_MODEL_CONFIGS)
    assert "compact_onchip_mnist_mlp" in paper_names
    assert "compact_onchip_mnist_training" in paper_names
    assert "medium_ddr_cifar_cnn" in paper_names
    assert "medium_ddr_cifar_training" in paper_names
    assert "large_ddr_stress_cnn" in paper_names
    assert "large_ddr_yolo_like" in paper_names


def test_paper_model_configs_use_existing_compiler_schema() -> None:
    for name, spec in PAPER_MODEL_CONFIGS.items():
        cfg = make_paper_config(name, spec)
        assert cfg["version"] == 1
        assert cfg["project"]["out_dir"] == f"build/paper/{name}"
        assert cfg["model"]["format"] == "onnx"
        assert cfg["model"]["path"].startswith("models/suite/")
        assert cfg["pipeline"]["mode"] in {"inference", "training_on_device"}
        assert cfg["targets"]["platform"]["board"] == "kv260"
        assert cfg["paper"]["model_class"] == name
        assert cfg["paper"]["memory_regime"] in {
            "onchip",
            "ddr_backed",
            "ddr_backed_tiled",
        }


def test_generated_static_paper_config_files_are_valid_yaml() -> None:
    # These files are included in the patch so users can compile the paper set
    # without first running the model-suite generator.
    root = Path("examples/paper/models")
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
        assert "paper" not in data
        assert data["project"]["name"].startswith("paper_")
        assert data["model"]["format"] == "onnx"
        assert data["pipeline"]["mode"] in {"inference", "training_on_device"}
