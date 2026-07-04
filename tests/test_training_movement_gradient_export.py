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


def test_training_m_axi_tiled_inputs_labels_outputs_generate_ports_and_reports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_tiled_io")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    dm = raw.setdefault("data_movement", {})
    dm.setdefault("inputs", {})["import"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}
    dm.setdefault("labels", {})["import"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}
    dm.setdefault("outputs", {})["export"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* input_mem" in source
    assert "ap_uint<32>* label_mem" in source
    assert "ap_uint<32>* output_mem" in source
    assert "m_axi port=input_mem" in source
    assert "m_axi port=label_mem" in source
    assert "m_axi port=output_mem" in source
    assert "input_tile[FPGAI_TRAIN_INPUT_TILE_SIZE]" in source
    assert "label_tile[FPGAI_TRAIN_LABEL_TILE_SIZE]" in source
    assert "output_tile[FPGAI_TRAIN_OUTPUT_TILE_SIZE]" in source

    report = json.loads((out_dir / "reports/training_io_movement.json").read_text(encoding="utf-8"))
    assert report["inputs"]["import"]["resolved"] == "m_axi_import_tiled"
    assert report["labels"]["import"]["resolved"] == "m_axi_import_tiled"
    assert report["outputs"]["export"]["resolved"] == "m_axi_export_tiled"

    # The training/Python comparison fixture path remains present.
    # The canonical location is the training_reference directory used by csim_train_tcl.
    ref_dir = out_dir / "training_reference"
    assert (ref_dir / "weights_before_ref.bin").exists()
    assert (ref_dir / "grads_ref.bin").exists()
    assert (ref_dir / "weights_after_ref.bin").exists()
    assert (ref_dir / "tiled_inputs_ref.bin").exists()
    assert (ref_dir / "tiled_labels_ref.bin").exists()
    assert (ref_dir / "tiled_outputs_ref.bin").exists()
    assert (ref_dir / "tiled_gradients_ref.bin").exists()
    assert (ref_dir / "tiled_weights_after_ref.bin").exists()

    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    tiled = numeric["training_tiled_io"]
    assert tiled["requested"] is True
    assert tiled["interface"] == "m_axi"
    assert tiled["input_tiled"] is True
    assert tiled["labels_tiled"] is True
    assert tiled["output_tiled"] is True
    assert tiled["compute_fused"] is True
    assert tiled["reference_available"] is True
    assert tiled["captures_available"] is False
    assert tiled["status"] == "artifact_missing"
    assert tiled["passed"] is False


def test_training_gradient_export_full_generates_m_axi_port_command_and_runtime_support(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_gradient_export")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("data_movement", {}).setdefault("gradients", {})["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "full",
    }
    raw.setdefault("runtime", {})["sequence"] = [
        {"run_training": {"steps": 1}},
        "export_gradients",
    ]

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* gradients_mem" in source
    assert "m_axi port=gradients_mem" in source
    assert "FPGAI_MODE_EXPORT_GRADIENTS" in source
    assert "if (mode == FPGAI_MODE_EXPORT_GRADIENTS)" in source

    report = json.loads((out_dir / "reports/gradient_export.json").read_text(encoding="utf-8"))
    assert report["resolved"] == "m_axi_export_full"
    assert report["supported"] is True

    manifest = json.loads((out_dir / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_sequence"]["supported_commands"]["export_gradients"] is True
    assert manifest["runtime_sequence"]["sequence"][-1]["command"] == "export_gradients"


def test_training_gradient_export_tiled_generates_tile_buffer_command_and_runtime_support(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_gradient_export_tiled")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("data_movement", {}).setdefault("gradients", {})["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "tiled",
    }
    raw.setdefault("runtime", {})["sequence"] = [
        {"run_training": {"steps": 1}},
        "export_gradients",
    ]

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* gradients_mem" in source
    assert "m_axi port=gradients_mem" in source
    assert "FPGAI_MODE_EXPORT_GRADIENTS" in source
    assert "gradient_export_tile[FPGAI_GRADIENT_EXPORT_TILE_SIZE]" in source
    assert "tile_base" in source
    assert "gradient_export tiled mode" in source

    report = json.loads((out_dir / "reports/gradient_export.json").read_text(encoding="utf-8"))
    assert report["resolved"] == "m_axi_export_tiled"
    assert report["supported"] is True

    manifest = json.loads((out_dir / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_sequence"]["supported_commands"]["export_gradients"] is True
    assert manifest["runtime_sequence"]["sequence"][-1]["command"] == "export_gradients"



def test_training_axi_stream_tiled_inputs_labels_outputs_generate_real_tile_readers_and_writers(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_axis_tiled_io")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    dm = raw.setdefault("data_movement", {})
    dm.setdefault("inputs", {})["import"] = {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}
    dm.setdefault("labels", {})["import"] = {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}
    dm.setdefault("outputs", {})["export"] = {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI training AXI-stream tiled input import" in source
    assert "FPGAI training AXI-stream tiled label import" in source
    assert "FPGAI training AXI-stream tiled output export" in source
    assert "FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE" in source
    assert "FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE" in source
    assert "FPGAI_TRAIN_AXIS_OUTPUT_TILE_SIZE" in source
    assert "axis_input_tile[FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE]" in source
    assert "axis_label_tile[FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE]" in source
    assert "emit_stream_tiled_block" in source
    assert "write_f32(out, axis_output_tile[lane], last)" in source

    report = json.loads((out_dir / "reports/stream_tiled_io.json").read_text(encoding="utf-8"))
    assert report["inputs"]["import"]["status"] == "generated_interface_supported"
    assert report["outputs"]["export"]["status"] == "generated_interface_supported"

    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    tiled = numeric["training_tiled_io"]
    assert tiled["requested"] is True
    assert tiled["interface"] == "axi_stream"
    assert tiled["axi_stream"]["tlast_required"] is True
    assert tiled["reference_available"] is True
    assert tiled["status"] == "artifact_missing"



def test_normal_training_omits_tiled_io_markers_and_reports_not_requested(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_no_tiled_io")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    # Force plain movement so stale defaults cannot request tiled training I/O.
    dm = raw.setdefault("data_movement", {})
    dm.setdefault("inputs", {})["import"] = {"interface": "none", "policy": "none"}
    dm.setdefault("labels", {})["import"] = {"interface": "none", "policy": "none"}
    dm.setdefault("outputs", {})["export"] = {"interface": "none", "policy": "none"}

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI training tiled input import" not in source
    assert "FPGAI training tiled label import" not in source
    assert "FPGAI training tiled output export" not in source
    assert "FPGAI training AXI-stream tiled input import" not in source
    assert "FPGAI training AXI-stream tiled label import" not in source
    assert "FPGAI training AXI-stream tiled output export" not in source

    ref_dir = out_dir / "training_reference"
    assert not (ref_dir / "tiled_inputs_ref.bin").exists()
    assert not (ref_dir / "tiled_labels_ref.bin").exists()
    assert not (ref_dir / "tiled_outputs_ref.bin").exists()

    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    assert numeric["training_tiled_io"]["requested"] is False
    assert numeric["training_tiled_io"]["status"] == "not_requested"
