from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fpgai.analysis.performance_estimator import estimate_performance
from fpgai.config.access import get_path
from fpgai.config.loader import ConfigError, load_config
from fpgai.engine.planner import make_compile_plan


def _base_config(model_path: Path) -> dict:
    return {
        "version": 1,
        "project": {"out_dir": "build/test", "clean": True},
        "pipeline": {
            "mode": "inference",
            "outputs": {"top_kernel_name": "test_top"},
        },
        "model": {"path": str(model_path)},
        "targets": {
            "platform": {
                "board": "kv260",
                "part": "xck26-sfvc784-2LV-c",
                "clocks": [{"name": "pl_clk0", "target_mhz": 173}],
            }
        },
        "operators": {"supported": ["Dense"]},
        "optimization": {"parallel_policy": "Balanced"},
        "data_movement": {
            "ps_pl": {"weights": {"mode": "embedded"}}
        },
        "backends": {
            "hls": {"enabled": True},
            "host_cpp": {"enabled": True},
        },
        "toolchain": {"vitis_hls": {"enabled": False}},
    }


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_get_path_traverses_mapping_and_list_indices() -> None:
    raw = {"targets": {"platform": {"clocks": [{"target_mhz": 173}]}}}
    assert get_path(raw, "targets.platform.clocks.0.target_mhz") == 173
    assert get_path(raw, "targets.platform.clocks.1.target_mhz", 200) == 200


def test_clock_reaches_planner_and_estimator(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    cfg = load_config(str(_write_config(tmp_path, _base_config(model_path))))

    plan = make_compile_plan(cfg, [])
    estimate = estimate_performance(
        resource_estimate={"layers": [], "totals": {}},
        raw_cfg=cfg.raw,
    )

    assert plan.clock_mhz == 173.0
    assert plan.notes["requested_clock_mhz"] == 173
    assert estimate["clock_mhz"] == 173.0


def test_unknown_policy_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    raw = _base_config(model_path)
    raw["optimization"]["parallel_policy"] = "Balnced"

    with pytest.raises(ConfigError, match="Unknown policy 'Balnced'"):
        load_config(str(_write_config(tmp_path, raw)))


def test_unknown_top_level_section_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    raw = _base_config(model_path)
    raw["outputs"] = {"top_kernel_name": "wrong_place"}

    with pytest.raises(
        ConfigError,
        match="Unknown top-level configuration section 'outputs'",
    ):
        load_config(str(_write_config(tmp_path, raw)))


def test_conflicting_policy_paths_are_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    raw = _base_config(model_path)
    raw["analysis"] = {
        "design_space": {"policy_name": "Latency-First"}
    }

    with pytest.raises(ConfigError, match="Conflicts with"):
        load_config(str(_write_config(tmp_path, raw)))



def test_memory_first_policy_is_accepted(tmp_path: Path) -> None:
    from fpgai.config.loader import load_config

    cfg = tmp_path / "memory_first.yml"
    cfg.write_text(
        """
version: 1
project:
  name: memory_first_test
  out_dir: build/memory_first_test
pipeline:
  mode: inference
model:
  format: onnx
  path: models/cnn_mnist.onnx
targets:
  platform:
    board: kv260
    part: xck26-sfvc784-2LV-c
    clocks:
      - name: pl_clk0
        target_mhz: 200
operators:
  supported:
    - Dense
    - Conv
    - MaxPool
    - Relu
    - Softmax
numerics:
  defaults:
    activation:
      type: ap_fixed
      total_bits: 16
      int_bits: 6
    weight:
      type: ap_fixed
      total_bits: 16
      int_bits: 6
    bias:
      type: ap_fixed
      total_bits: 24
      int_bits: 10
    accum:
      type: ap_fixed
      total_bits: 24
      int_bits: 10
optimization:
  parallel_policy: Memory-First
backends:
  hls:
    enabled: false
""",
        encoding="utf-8",
    )

    loaded = load_config(cfg)
    assert loaded.raw["optimization"]["parallel_policy"] == "Memory-First"



def test_build_fit_policy_block_over_limit_alias_is_accepted(tmp_path):
    import yaml

    from fpgai.config.loader import load_config

    src = Path("configs/examples/inference_compile.yml")
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    cfg.setdefault("project", {})
    cfg["project"]["out_dir"] = str(tmp_path / "out")
    cfg.setdefault("build", {})["fit_policy"] = "block_over_limit"

    path = tmp_path / "build_fit_policy_alias.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loaded = load_config(str(path))

    assert loaded.raw["build"]["fit_policy"] == "block_over_limit"

def test_invalid_fit_policy_is_rejected(tmp_path):
    import pytest
    import yaml

    from fpgai.config.loader import load_config

    src = Path("configs/examples/inference_compile.yml")
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    cfg.setdefault("project", {})
    cfg["project"]["out_dir"] = str(tmp_path / "out")

    cfg.setdefault("targets", {})
    cfg["targets"].setdefault("platform", {})
    cfg["targets"]["platform"]["fit_policy"] = "aggressive"

    path = tmp_path / "bad_fit_policy.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(Exception) as exc:
        load_config(str(path))

    msg = str(exc.value)
    assert "fit_policy" in msg
    assert "report_only" in msg
    assert "warn" in msg
    assert "enforce" in msg


def test_named_clock_without_target_mhz_is_allowed(tmp_path):
    import yaml

    from fpgai.config.loader import load_config

    src = Path("configs/examples/inference_compile.yml")
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg.setdefault("project", {})
    cfg["project"]["out_dir"] = str(tmp_path / "out")

    cfg.setdefault("targets", {})
    cfg["targets"].setdefault("platform", {})
    cfg["targets"]["platform"]["board"] = "pynq_z2"
    cfg["targets"]["platform"]["clocks"] = [{"name": "ap_clk"}]

    path = tmp_path / "named_clock_no_target.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loaded = load_config(str(path))
    assert loaded.raw["targets"]["platform"]["clocks"][0]["name"] == "ap_clk"
    assert "target_mhz" not in loaded.raw["targets"]["platform"]["clocks"][0]


def test_dataset_validation_section_is_accepted(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    dataset_path = tmp_path / "samples.npz"
    dataset_path.touch()
    raw = _base_config(model_path)
    raw["validation"] = {
        "task": "classification",
        "dataset": {
            "source": "npz",
            "path": str(dataset_path),
            "inputs_key": "inputs",
            "labels_key": "labels",
            "sample_selection": {"offset": 0, "count": 10},
        },
        "decision_thresholds": {
            "min_prediction_agreement": 0.95,
            "max_accuracy_drop_pct": 3.0,
        },
    }

    cfg = load_config(str(_write_config(tmp_path, raw)))

    assert cfg.raw["validation"]["task"] == "classification"
    assert cfg.raw["validation"]["dataset"]["source"] == "npz"
    assert cfg.raw["validation"]["dataset"]["sample_selection"]["count"] == 10


def test_dataset_validation_rejects_unknown_source(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    raw = _base_config(model_path)
    raw["validation"] = {
        "task": "classification",
        "dataset": {
            "source": "unknown_dataset_provider",
            "path": "unused",
        },
    }

    with pytest.raises(ConfigError, match="validation.dataset.source"):
        load_config(str(_write_config(tmp_path, raw)))


def test_torchvision_dataset_validation_section_is_accepted_without_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    raw = _base_config(model_path)
    raw["validation"] = {
        "task": "classification",
        "dataset": {
            "source": "torchvision",
            "name": "MNIST",
            "root": "datasets",
            "split": "test",
            "download": False,
            "sample_selection": {
                "mode": "balanced_per_class",
                "count": 100,
                "seed": 42,
                "per_class_count": 10,
            },
            "preprocessing": {
                "normalize": True,
                "flatten": True,
            },
        },
    }

    cfg = load_config(str(_write_config(tmp_path, raw)))
    dataset = cfg.raw["validation"]["dataset"]
    assert dataset["source"] == "torchvision"
    assert dataset["name"] == "MNIST"
    assert dataset["download"] is False
    assert dataset["sample_selection"]["mode"] == "balanced_per_class"


def test_torchvision_dataset_validation_rejects_invalid_name(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    raw = _base_config(model_path)
    raw["validation"] = {
        "dataset": {
            "source": "torchvision",
            "name": "CIFAR100",
        },
    }

    with pytest.raises(ConfigError, match="validation.dataset.name"):
        load_config(str(_write_config(tmp_path, raw)))
