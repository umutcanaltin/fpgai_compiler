from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml


def _load_training_config() -> dict:
    candidates = [
        Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix/configs/training_kv260_aggressive_fx8_3.yml"),
        Path("paper_experiments/full_pipeline_gate/sprint27h_full_rerun/configs_hls/training_kv260_aggressive_fx8_3.yml"),
        Path("configs/examples/training_compile_smoke.yml"),
    ]
    for p in candidates:
        if p.exists():
            data = yaml.safe_load(p.read_text())
            if isinstance(data, dict):
                return data
    pytest.skip("training config not available")


def _make_config(raw: dict, tmp_path: Path):
    cfg_path = tmp_path / "compile.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    from fpgai.config.loader import load_config

    return load_config(str(cfg_path))


def _compile_raw(raw: dict, tmp_path: Path):
    from fpgai.engine.compiler import Compiler

    cfg = _make_config(raw, tmp_path)
    return Compiler(cfg).compile()


def test_training_uram_weight_storage_is_rejected_until_real_hls_backend_exists(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_uram_rejected")

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "uram"
    raw.setdefault("memory", {})["weight_storage"] = "uram"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "uram"

    with pytest.raises(ValueError, match="Training runtime/external weight storage is not implemented"):
        _compile_raw(raw, tmp_path)


def test_training_ddr_weight_storage_is_rejected_until_real_hls_backend_exists(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_ddr_rejected")

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "ddr"
    raw.setdefault("memory", {})["weight_storage"] = "ddr"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "ddr"
    raw.setdefault("data_movement", {}).setdefault("weights", {}).setdefault("load", {})["interface"] = "ddr"

    with pytest.raises(ValueError, match="Training runtime/external weight storage is not implemented"):
        _compile_raw(raw, tmp_path)


def test_training_bram_weight_storage_still_compiles(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_bram_ok")

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "bram"
    raw.setdefault("memory", {})["weight_storage"] = "bram"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "bram"

    raw.get("data_movement", {}).pop("weights", None)
    if "ps_pl" in raw.get("data_movement", {}):
        raw["data_movement"]["ps_pl"].pop("weights", None)

    result = _compile_raw(raw, tmp_path)

    assert result is not None
    out_dir = Path(raw["project"]["out_dir"])
    assert (out_dir / "hls/src/deeplearn.cpp").exists()
