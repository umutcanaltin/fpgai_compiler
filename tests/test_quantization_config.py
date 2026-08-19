from pathlib import Path

import pytest
import yaml

from fpgai.config.loader import ConfigError, load_config


def _base() -> dict:
    raw = yaml.safe_load(Path("configs/examples/quick_compile.yml").read_text(encoding="utf-8"))
    return raw


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "quant.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_ptq_config_is_accepted(tmp_path: Path):
    raw = _base()
    raw.setdefault("numerics", {})["quantization"] = {
        "mode": "ptq",
        "weights": {"bits": 8, "scheme": "symmetric", "granularity": "per_channel", "axis": 0},
        "activations": {"bits": 8, "scheme": "symmetric", "granularity": "per_tensor"},
        "calibration": {"method": "percentile", "percentile": 99.9, "samples": 128},
    }
    cfg = load_config(_write(tmp_path, raw))
    assert cfg.raw["numerics"]["quantization"]["mode"] == "ptq"


def test_ptq_requires_calibration(tmp_path: Path):
    raw = _base()
    raw.setdefault("numerics", {})["quantization"] = {"mode": "ptq"}
    with pytest.raises(ConfigError, match="PTQ mode requires"):
        load_config(_write(tmp_path, raw))


def test_qat_config_is_accepted(tmp_path: Path):
    raw = _base()
    raw.setdefault("numerics", {})["quantization"] = {
        "mode": "qat",
        "weights": {"bits": 8},
        "activations": {"bits": 8},
        "qat": {"fake_quant": True, "straight_through_estimator": True, "freeze_after_updates": 100},
    }
    cfg = load_config(_write(tmp_path, raw))
    assert cfg.raw["numerics"]["quantization"]["mode"] == "qat"


def test_ptq_calibration_dataset_fields_are_validated(tmp_path):
    import copy
    from fpgai.config.loader import ConfigError, load_config
    base = {
        "version": 1,
        "project": {"name": "quant_cfg", "out_dir": str(tmp_path / "out")},
        "pipeline": {"mode": "inference", "outputs": {"top_kernel_name": "deeplearn"}},
        "model": {"format": "onnx", "path": "models/example.onnx"},
        "targets": {"platform": {"board": "kv260", "part": "xck26-sfvc784-2LV-c"}},
        "operators": {"supported": ["Dense"]},
        "numerics": {
            "quantization": {
                "mode": "ptq",
                "weights": {"bits": 8},
                "activations": {"bits": 8},
                "calibration": {"method": "min_max", "dataset": 7, "array_key": ""},
            }
        },
    }
    path = tmp_path / "bad.yml"
    import yaml
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    try:
        load_config(str(path))
    except ConfigError as exc:
        text = str(exc)
        assert "calibration.dataset" in text
        assert "calibration.array_key" in text
    else:
        raise AssertionError("invalid calibration dataset fields should be rejected")
