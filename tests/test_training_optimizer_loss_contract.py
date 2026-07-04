from __future__ import annotations

import copy
import json
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
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    pytest.skip("training config not available")


def _make_config(raw: dict, tmp_path: Path):
    cfg_path = tmp_path / "compile.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    from fpgai.config.loader import load_config

    return load_config(str(cfg_path))


def _compile_raw(raw: dict, tmp_path: Path):
    pytest.importorskip("onnx")
    from fpgai.engine.compiler import Compiler

    return Compiler(_make_config(raw, tmp_path)).compile()


def _cpp_only(raw: dict) -> None:
    raw.setdefault("build", {})["stages"] = {
        "cpp": True,
        "testbench": True,
        "hls_project": False,
        "hls_synthesis": False,
        "vivado_project": False,
        "vivado_implementation": False,
        "bitstream": False,
        "runtime_package": True,
        "reports": True,
    }
    raw.setdefault("toolchain", {}).setdefault("vitis_hls", {})["enabled"] = False


def test_training_sgd_optimizer_loss_contract_reports_default_no_state(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_sgd_contract")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {}).setdefault("optimizer", {})["type"] = "sgd"
    raw["training"].setdefault("loss", {})["type"] = "mse"
    raw["training"].setdefault("storage", {}).pop("optimizer_state", None)
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    opt = json.loads((out_dir / "reports/training_optimizer_state.json").read_text(encoding="utf-8"))
    loss = json.loads((out_dir / "reports/training_loss_contract.json").read_text(encoding="utf-8"))

    assert opt["optimizer"]["type"] == "sgd"
    assert opt["optimizer"]["hls_update_status"] == "implemented"
    assert opt["optimizer_state"]["required"] is False
    assert opt["optimizer_state"]["storage"] == "none"
    assert loss["loss"]["type"] == "mse"
    assert loss["loss"]["hls_status"] == "implemented"


def test_training_optimizer_state_bram_import_export_generates_port_storage_and_reports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_optimizer_state_bram")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {}).setdefault("optimizer", {})["type"] = "sgd"
    raw["training"].setdefault("storage", {})["optimizer_state"] = "bram"
    dm = raw.setdefault("data_movement", {}).setdefault("optimizer_state", {})
    dm["import"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"}
    dm["export"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"}
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* optimizer_state_mem" in source
    assert "m_axi port=optimizer_state_mem" in source
    assert "optimizer_state_tile[FPGAI_OPTIMIZER_STATE_TILE_SIZE]" in source
    assert "impl=bram" in source

    opt = json.loads((out_dir / "reports/training_optimizer_state.json").read_text(encoding="utf-8"))
    assert opt["optimizer_state"]["storage"] == "bram"
    assert opt["optimizer_state"]["import"]["resolved"] == "m_axi_import_full"
    assert opt["optimizer_state"]["export"]["resolved"] == "m_axi_export_full"
    assert opt["optimizer_state"]["generated_interface"] is True


def test_training_optimizer_state_uram_storage_generates_uram_binding(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_optimizer_state_uram")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {}).setdefault("optimizer", {})["type"] = "sgd"
    raw["training"].setdefault("storage", {})["optimizer_state"] = "uram"
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    source = (Path(result.out_dir) / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "optimizer_state_tile[FPGAI_OPTIMIZER_STATE_TILE_SIZE]" in source
    assert "impl=uram" in source



def test_training_optimizer_state_ddr_tiled_generates_m_axi_tile_and_reports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_optimizer_state_ddr_tiled")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {}).setdefault("optimizer", {})["type"] = "sgd"
    raw["training"].setdefault("storage", {})["optimizer_state"] = "ddr"
    dm = raw.setdefault("data_movement", {}).setdefault("optimizer_state", {})
    dm["import"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}
    dm["export"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* optimizer_state_mem" in source
    assert "m_axi port=optimizer_state_mem" in source
    assert "optimizer_state_tile[FPGAI_OPTIMIZER_STATE_TILE_SIZE]" in source
    assert "optimizer-state ddr backing" in source

    opt = json.loads((out_dir / "reports/training_optimizer_state.json").read_text(encoding="utf-8"))
    assert opt["optimizer_state"]["storage"] == "ddr"
    assert opt["optimizer_state"]["storage_supported"] is True
    assert opt["optimizer_state"]["import"]["resolved"] == "m_axi_import_tiled"
    assert opt["optimizer_state"]["export"]["resolved"] == "m_axi_export_tiled"
    assert opt["optimizer_state"]["generated_interface"] is True

def test_training_momentum_optimizer_generates_real_update_kernel_and_reports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_momentum_real")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    opt_cfg = raw.setdefault("training", {}).setdefault("optimizer", {})
    opt_cfg["type"] = "momentum"
    opt_cfg["learning_rate"] = 0.001
    opt_cfg["momentum"] = 0.9
    raw["training"].setdefault("storage", {})["optimizer_state"] = "bram"
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI Momentum optimizer update kernel" in source
    assert "FPGAI persistent momentum optimizer velocity state" in source
    assert "FPGAI_MOMENTUM_W_" in source
    assert "FPGAI_MOMENTUM_B_" in source
    assert "V = momentum * V - learning_rate * dParam" in source
    assert "fpgai::sgd_update_wgt_typed" not in source
    opt = json.loads((out_dir / "reports/training_optimizer_state.json").read_text(encoding="utf-8"))
    assert opt["optimizer"]["type"] == "momentum"
    assert opt["optimizer"]["hls_update_status"] == "implemented"
    assert opt["optimizer"]["numeric_validation_status"] == "implemented"
    assert opt["optimizer_state"]["required"] is True


def test_training_adam_optimizer_generates_real_update_kernel_and_reports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_adam_real")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    opt_cfg = raw.setdefault("training", {}).setdefault("optimizer", {})
    opt_cfg["type"] = "adam"
    opt_cfg["learning_rate"] = 0.001
    opt_cfg["beta1"] = 0.9
    opt_cfg["beta2"] = 0.999
    opt_cfg["epsilon"] = 1.0e-8
    raw["training"].setdefault("storage", {})["optimizer_state"] = "bram"
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI Adam optimizer update kernel" in source
    assert "FPGAI persistent Adam optimizer first/second moment state" in source
    assert "FPGAI_ADAM_M_W_" in source
    assert "FPGAI_ADAM_V_W_" in source
    assert "M = beta1 * M + (1-beta1) * dParam" in source
    assert "V = beta2 * V + (1-beta2) * dParam*dParam" in source
    assert "fpgai::sgd_update_wgt_typed" not in source
    opt = json.loads((out_dir / "reports/training_optimizer_state.json").read_text(encoding="utf-8"))
    assert opt["optimizer"]["type"] == "adam"
    assert opt["optimizer"]["hls_update_status"] == "implemented"
    assert opt["optimizer"]["numeric_validation_status"] == "implemented"
    assert opt["optimizer"]["beta1"] == 0.9
    assert opt["optimizer"]["beta2"] == 0.999
    assert opt["optimizer_state"]["required"] is True



def test_training_adam_optimizer_state_export_generates_capture_mode_and_reports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_adam_state_export")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    opt_cfg = raw.setdefault("training", {}).setdefault("optimizer", {})
    opt_cfg["type"] = "adam"
    opt_cfg["learning_rate"] = 0.001
    opt_cfg["beta1"] = 0.9
    opt_cfg["beta2"] = 0.999
    opt_cfg["epsilon"] = 1.0e-8
    raw["training"].setdefault("storage", {})["optimizer_state"] = "bram"
    dm = raw.setdefault("data_movement", {}).setdefault("optimizer_state", {})
    dm["export"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"}
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI optimizer-state export/capture mode" in source
    assert "FPGAI_MODE_EXPORT_OPTIMIZER_STATE = 9" in source
    assert "export_optimizer_state runtime command" in source
    assert "optimizer_state_mem" in source
    assert "FPGAI_ADAM_M_W_" in source
    assert "FPGAI_ADAM_V_W_" in source
    assert "fpgai_pack_optimizer_state_float32" in source

    opt = json.loads((out_dir / "reports/training_optimizer_state.json").read_text(encoding="utf-8"))
    assert opt["optimizer_state"]["export"]["resolved"] == "m_axi_export_full"
    assert opt["optimizer_state"]["export_capture_mode"] == 9
    assert opt["optimizer_state"]["export_capture_status"] == "generated_hls_mode"

    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    assert numeric["optimizer_state_validation"]["status"] == "generated_export_capture_supported"
    assert numeric["optimizer_state_validation"]["export_capture_mode"] == 9
    assert numeric["optimizer_state_validation"]["passed"] is False

def test_training_cross_entropy_generates_real_loss_kernel_and_reports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_cross_entropy_real")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {}).setdefault("loss", {})["type"] = "cross_entropy"
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI cross_entropy loss kernel" in source
    assert "softmax_denom" in source
    assert "probability - target_value" in source
    assert "logf((float)probability" in source

    loss = json.loads((out_dir / "reports/training_loss_contract.json").read_text(encoding="utf-8"))
    assert loss["loss"]["type"] == "cross_entropy"
    assert loss["loss"]["hls_status"] == "implemented"
    assert loss["loss"]["numeric_validation_status"] == "implemented"


def test_training_cross_entropy_reference_artifacts_and_loss_validation_report(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_cross_entropy_numeric")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {}).setdefault("loss", {})["type"] = "cross_entropy"
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    ref_dir = out_dir / "training_reference"

    assert (ref_dir / "logits_ref.bin").exists()
    assert (ref_dir / "softmax_ref.bin").exists()
    assert (ref_dir / "cross_entropy_loss_ref.json").exists()
    assert (ref_dir / "dlogits_ref.bin").exists()

    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    loss_validation = numeric["loss_validation"]
    assert loss_validation["requested"] is True
    assert loss_validation["loss_type"] == "cross_entropy"
    assert loss_validation["softmax_stable"] is True
    assert loss_validation["reference"]["logits_ref_exists"] is True
    assert loss_validation["reference"]["softmax_ref_exists"] is True
    assert loss_validation["reference"]["cross_entropy_loss_ref_exists"] is True
    assert loss_validation["reference"]["dlogits_ref_exists"] is True
    assert loss_validation["status"] == "artifact_missing"
    assert loss_validation["passed"] is False
    assert numeric["training"]["reference"]["loss_type"] == "cross_entropy"
    assert numeric["training"]["reference"]["softmax_ref_bin"].endswith("training_reference/softmax_ref.bin")


def test_training_mse_does_not_emit_cross_entropy_loss_kernel_or_validation(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_mse_no_cross_entropy")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {}).setdefault("loss", {})["type"] = "mse"
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))

    assert "FPGAI cross_entropy loss kernel" not in source
    assert "softmax_denom" not in source
    assert "cross_entropy_loss_ref.json" not in source
    assert numeric["loss_validation"]["status"] == "not_requested"
    assert numeric["loss_validation"]["loss_type"] == "mse"
