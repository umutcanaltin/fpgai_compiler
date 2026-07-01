from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from fpgai.engine.compiler import (
    _resolve_stream_tiled_io_contract,
    _resolve_training_batch_accumulation_contract,
)


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


def _load_inference_config() -> dict:
    for p in [Path("configs/examples/inference_compile.yml"), Path("configs/examples/default_compile.yml")]:
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    pytest.skip("inference config not available")


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


def test_training_batch_accumulation_contract_resolves_batch_and_native_mode() -> None:
    contract = _resolve_training_batch_accumulation_contract(
        {
            "training": {
                "batch_size": 4,
                "gradient_accumulation": {"steps": 3, "mode": "native"},
            }
        }
    )
    assert contract["batch_size"] == 4
    assert contract["accumulation_steps"] == 3
    assert contract["mode"] == "native"
    assert contract["active"] is True
    assert contract["native_update_boundary"] is True
    assert contract["generated_hls_status"] == "implemented"


def test_training_batch_accumulation_invalid_mode_rejects() -> None:
    with pytest.raises(ValueError, match="training.gradient_accumulation.mode"):
        _resolve_training_batch_accumulation_contract(
            {"training": {"gradient_accumulation": {"steps": 2, "mode": "magic"}}}
        )


def test_stream_tiled_io_contract_distinguishes_inference_and_training() -> None:
    raw = {
        "data_movement": {
            "inputs": {"import": {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}},
            "outputs": {"export": {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}},
        }
    }
    inference = _resolve_stream_tiled_io_contract(raw, pipeline_mode="inference")
    training = _resolve_stream_tiled_io_contract(raw, pipeline_mode="training_on_device")
    assert inference["inputs"]["import"]["status"] == "generated_interface_supported"
    assert inference["outputs"]["export"]["status"] == "generated_interface_supported"
    assert training["inputs"]["import"]["status"] == "generated_interface_supported"


def test_training_batch_and_hardware_contract_reports_are_emitted(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_batch_contract")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {})["batch_size"] = 2
    raw["training"]["gradient_accumulation"] = {"steps": 2, "mode": "native"}
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    reports = Path(result.out_dir) / "reports"
    batch = json.loads((reports / "training_batch_accumulation.json").read_text(encoding="utf-8"))
    hardware = json.loads((reports / "hardware_knob_contract.json").read_text(encoding="utf-8"))

    assert batch["batch_size"] == 2
    assert batch["accumulation_steps"] == 2
    assert batch["mode"] == "native"
    assert "board_fit_status" in hardware
    assert "truth_boundary" in hardware


def test_inference_axi_stream_tiled_io_generates_tile_buffers_and_tlast(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "inference_stream_tiled")
    dm = raw.setdefault("data_movement", {})
    dm.setdefault("inputs", {})["import"] = {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}
    dm.setdefault("outputs", {})["export"] = {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI AXI-stream tiled input/output movement" in source
    assert "FPGAI_AXIS_INPUT_TILE_SIZE" in source
    assert "FPGAI_AXIS_OUTPUT_TILE_SIZE" in source
    assert "input_tile[FPGAI_AXIS_INPUT_TILE_SIZE]" in source
    assert "output_tile[FPGAI_AXIS_OUTPUT_TILE_SIZE]" in source
    assert "in_stream.read()" in source
    assert "out_stream.write(packet)" in source
    assert "packet.last = ((tile_base + lane + 1)" in source
    report = json.loads((out_dir / "reports/stream_tiled_io.json").read_text(encoding="utf-8"))
    assert report["inputs"]["import"]["status"] == "generated_interface_supported"
    assert report["outputs"]["export"]["status"] == "generated_interface_supported"


def test_native_batch_accumulation_generates_hls_modes_and_runtime_commands(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_native_accum_runtime")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {})["batch_size"] = 2
    raw["training"]["gradient_accumulation"] = {"steps": 2, "mode": "native"}
    raw.setdefault("runtime", {})["sequence"] = [
        "reset_accumulators",
        {"accumulate_gradients": {"steps": 2}},
        "apply_accumulated_gradients",
        "export_gradients",
    ]
    raw.setdefault("data_movement", {}).setdefault("gradients", {})["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "tiled",
    }
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI native gradient accumulation HLS modes" in source
    assert "FPGAI_MODE_ACCUMULATE_GRADIENTS = 3" in source
    assert "FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS = 4" in source
    assert "FPGAI_MODE_RESET_ACCUMULATORS = 5" in source
    assert "FPGAI accumulate_gradients runtime command" in source
    assert "FPGAI apply_accumulated_gradients runtime command" in source
    assert "FPGAI reset_accumulators runtime command" in source

    batch = json.loads((out_dir / "reports/training_batch_accumulation.json").read_text(encoding="utf-8"))
    assert batch["hls_modes"]["accumulate_gradients"] == 3
    assert batch["hls_modes"]["apply_accumulated_gradients"] == 4
    assert batch["hls_modes"]["reset_accumulators"] == 5
    assert "accumulate_gradients" in batch["runtime_commands"]

    runtime_sequence = json.loads((out_dir / "reports/runtime_sequence.json").read_text(encoding="utf-8"))
    assert runtime_sequence["supported_commands"]["reset_accumulators"] is True
    assert runtime_sequence["supported_commands"]["accumulate_gradients"] is True
    assert runtime_sequence["supported_commands"]["apply_accumulated_gradients"] is True

    api = (out_dir / "runtime_package/runtime_api.py").read_text(encoding="utf-8")
    assert "def reset_accumulators" in api
    assert "def accumulate_gradients" in api
    assert "def apply_accumulated_gradients" in api
