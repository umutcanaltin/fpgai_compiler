from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml


def _load_inference_config() -> dict:
    return yaml.safe_load(Path("configs/examples/inference_compile.yml").read_text(encoding="utf-8"))


def _load_training_config() -> dict:
    return yaml.safe_load(Path("configs/examples/training_compile_smoke.yml").read_text(encoding="utf-8"))


def _make_config(raw: dict, tmp_path: Path):
    from fpgai.config.loader import load_config

    cfg_path = tmp_path / "compile.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_config(str(cfg_path))


def _compile_raw(raw: dict, tmp_path: Path):
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


def test_vivado_bd_contract_runtime_api_and_claim_audit_for_cpp_only_inference(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "runtime_truth_inference")
    raw.setdefault("weights", {})["mode"] = "import_export"
    raw.setdefault("runtime", {})["sequence"] = ["import_weights", {"run_inference": {"repeat": 2}}, "export_weights"]
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    reports = out_dir / "reports"

    bd = json.loads((reports / "vivado_bd_contract.json").read_text(encoding="utf-8"))
    assert bd["format"] == "fpgai.vivado_bd_contract.v1"
    assert bd["status"] == "not_requested"
    assert "axi_lite_control" in bd["required_blocks"]
    assert bd["source_ports"]["source_present"] is True
    assert "truth_boundary" in bd

    api = out_dir / "runtime_package" / "runtime_api.py"
    assert api.exists()
    api_source = api.read_text(encoding="utf-8")
    assert "def import_weights" in api_source
    assert "def run_inference" in api_source
    assert "def export_weights" in api_source
    assert "def run_sequence" in api_source

    manifest = json.loads((out_dir / "runtime_package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_api"]["present"] is True
    assert "board-specific" in manifest["runtime_api"]["truth_boundary"]

    feature = json.loads((reports / "feature_contract.json").read_text(encoding="utf-8"))
    assert feature["source_generated"] is True
    assert feature["runtime_packaged"] is True
    assert feature["hls_synthesized"] is False
    assert feature["numeric_validated"] is False
    assert feature["paper_safe"] is False
    audit = (reports / "claim_audit.md").read_text(encoding="utf-8")
    assert "source_generation" in audit
    assert "fpga_execution" in audit


def test_vivado_bd_contract_for_m_axi_training_ports_and_runtime_api(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "runtime_truth_training")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("runtime", {})["sequence"] = [{"run_training": {"steps": 1}}]
    raw.setdefault("data_movement", {}).setdefault("gradients", {})["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "full",
    }
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    bd = json.loads((out_dir / "reports" / "vivado_bd_contract.json").read_text(encoding="utf-8"))
    assert bd["pipeline_mode"] == "training_on_device"
    assert "training_aux_buffers" in bd["required_blocks"]
    assert "axi_lite_control" in bd["required_blocks"]

    api_source = (out_dir / "runtime_package" / "runtime_api.py").read_text(encoding="utf-8")
    assert "def run_training" in api_source
    assert "def export_gradients" in api_source

    feature = json.loads((out_dir / "reports" / "feature_contract.json").read_text(encoding="utf-8"))
    assert feature["runtime_packaged"] is True
    assert any(item["feature"] == "vivado_bd_wiring" for item in feature["features"])
