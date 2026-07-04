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


def _vivado_project_only(raw: dict) -> None:
    raw.setdefault("build", {})["existing_hls_ip"] = True
    raw.setdefault("build", {})["stages"] = {
        "cpp": True,
        "testbench": True,
        "hls_project": False,
        "hls_synthesis": False,
        "vivado_project": True,
        "vivado_implementation": False,
        "bitstream": False,
        "runtime_package": True,
        "reports": True,
    }
    raw.setdefault("toolchain", {}).setdefault("vitis_hls", {})["enabled"] = False


def test_vivado_project_stage_emits_direct_tcl_and_validation_report(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "vivado_direct_inference")
    raw.setdefault("weights", {})["mode"] = "embedded"
    raw.setdefault("runtime", {})["sequence"] = ["run_inference"]
    _vivado_project_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    vivado_dir = out_dir / "vivado"

    assert (vivado_dir / "project.tcl").exists()
    assert (vivado_dir / "bd.tcl").exists()
    assert (vivado_dir / "run_vivado.tcl").exists()
    assert not (out_dir / "hls/run_hls.tcl").exists()

    bd_source = (vivado_dir / "bd.tcl").read_text(encoding="utf-8")
    assert "zynq_ultra_ps_e" in bd_source
    assert "axi_ctrl_interconnect" in bd_source
    assert "s_axi_control" in bd_source
    assert "axi_dma_0" not in bd_source
    assert "gradients_mem" not in bd_source
    assert "optimizer_state_mem" not in bd_source

    validation = json.loads((out_dir / "reports/vivado_bd_validation.json").read_text(encoding="utf-8"))
    assert validation["format"] == "fpgai.vivado_bd_validation.v1"
    assert validation["status"] == "passed"
    assert validation["vivado_project_requested"] is True
    assert validation["checks"]["ps_block"] is True
    assert validation["checks"]["axi_lite_control_path"] is True
    assert validation["checks"]["axi_dma_required"] is False
    assert validation["checks"]["axi_dma_present"] is False

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vivado_handoff_artifacts"]["vivado_project_tcl"].endswith("vivado/project.tcl")


def test_vivado_cpp_only_mode_omits_direct_tcl_but_writes_not_requested_validation(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "vivado_not_requested")
    raw.setdefault("runtime", {})["sequence"] = ["run_inference"]
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    assert not (out_dir / "vivado/project.tcl").exists()
    assert not (out_dir / "vivado/bd.tcl").exists()
    assert not (out_dir / "vivado/run_vivado.tcl").exists()
    validation = json.loads((out_dir / "reports/vivado_bd_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "not_requested"
    assert validation["vivado_project_requested"] is False


def _vivado_implementation_only(raw: dict) -> None:
    raw.setdefault("build", {})["existing_hls_ip"] = True
    raw.setdefault("build", {})["stages"] = {
        "cpp": True,
        "testbench": True,
        "hls_project": False,
        "hls_synthesis": False,
        "vivado_project": True,
        "vivado_implementation": True,
        "bitstream": False,
        "runtime_package": True,
        "reports": True,
    }
    raw.setdefault("toolchain", {}).setdefault("vitis_hls", {})["enabled"] = False
    raw.setdefault("toolchain", {}).setdefault("vivado", {})["executable"] = "__fpgai_missing_vivado__"


def _vivado_bitstream_requested(raw: dict) -> None:
    raw.setdefault("build", {})["existing_hls_ip"] = True
    raw.setdefault("build", {})["stages"] = {
        "cpp": True,
        "testbench": True,
        "hls_project": False,
        "hls_synthesis": False,
        "vivado_project": True,
        "vivado_implementation": True,
        "bitstream": True,
        "runtime_package": True,
        "reports": True,
    }
    raw.setdefault("toolchain", {}).setdefault("vitis_hls", {})["enabled"] = False
    raw.setdefault("toolchain", {}).setdefault("vivado", {})["executable"] = "__fpgai_missing_vivado__"


def test_vivado_implementation_stage_reports_missing_tool_without_fake_success(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "vivado_impl_truth")
    raw.setdefault("weights", {})["mode"] = "embedded"
    raw.setdefault("runtime", {})["sequence"] = ["run_inference"]
    _vivado_implementation_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)

    validation = json.loads((out_dir / "reports/vivado_validation_report.json").read_text(encoding="utf-8"))
    impl = json.loads((out_dir / "reports/vivado_implementation_report.json").read_text(encoding="utf-8"))
    bit = json.loads((out_dir / "reports/bitstream_report.json").read_text(encoding="utf-8"))

    assert validation["format"] == "fpgai.vivado_validation_report.v1"
    assert validation["status"] == "tool_missing"
    assert validation["claimed_success"] is False
    assert impl["format"] == "fpgai.vivado_implementation_report.v1"
    assert impl["requested"] is True
    assert impl["status"] == "tool_missing"
    assert impl["claimed_success"] is False
    assert bit["format"] == "fpgai.bitstream_report.v1"
    assert bit["requested"] is False
    assert bit["status"] == "not_requested"
    assert bit["claimed_success"] is False

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vivado_truth_artifacts"]["vivado_implementation_report_json"].endswith("vivado_implementation_report.json")


def test_bitstream_stage_reports_missing_artifact_without_fake_success(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "vivado_bitstream_truth")
    raw.setdefault("weights", {})["mode"] = "embedded"
    raw.setdefault("runtime", {})["sequence"] = ["run_inference"]
    _vivado_bitstream_requested(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    bit = json.loads((out_dir / "reports/bitstream_report.json").read_text(encoding="utf-8"))
    impl = json.loads((out_dir / "reports/vivado_implementation_report.json").read_text(encoding="utf-8"))

    assert impl["status"] == "tool_missing"
    assert impl["claimed_success"] is False
    assert bit["requested"] is True
    assert bit["status"] == "tool_missing"
    assert bit["claimed_success"] is False
    assert bit["artifact"] is None
    assert "implementation has not passed" in bit["reason"]


def test_vivado_impl_and_bitstream_blocked_when_board_fit_fails(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "vivado_board_fit_blocked")
    raw.setdefault("targets", {}).setdefault("platform", {})["board"] = "pynq_z2"
    raw.setdefault("targets", {}).setdefault("platform", {})["part"] = "xc7z020clg400-1"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "uram"
    raw.setdefault("weights", {})["mode"] = "embedded"
    raw.setdefault("runtime", {})["sequence"] = ["run_inference"]
    _vivado_bitstream_requested(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    board_fit = json.loads((out_dir / "reports/board_fit.json").read_text(encoding="utf-8"))
    impl = json.loads((out_dir / "reports/vivado_implementation_report.json").read_text(encoding="utf-8"))
    bit = json.loads((out_dir / "reports/bitstream_report.json").read_text(encoding="utf-8"))

    assert board_fit["status"] == "over_limit"
    assert board_fit["fit"]["resources"]["uram"]["status"] == "over_limit"
    assert board_fit["vivado_implementation_allowed"] is False
    assert board_fit["bitstream_allowed"] is False
    assert impl["status"] == "blocked_by_board_fit"
    assert impl["claimed_success"] is False
    assert impl["board_fit_status"] == "over_limit"
    assert bit["status"] == "blocked_by_board_fit"
    assert bit["claimed_success"] is False
    assert bit["board_fit_status"] == "over_limit"
