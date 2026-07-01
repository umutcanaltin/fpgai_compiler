from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from fpgai.engine.compiler import _resolve_build_stages, _resolve_runtime_sequence, _resolve_codegen_readability
from fpgai.runtime.package import emit_runtime_package


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


def test_resolve_build_stages_cpp_only_contract() -> None:
    stages = _resolve_build_stages(
        {
            "build": {
                "stages": {
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
            }
        }
    )

    assert stages["cpp"] is True
    assert stages["testbench"] is True
    assert stages["hls_project"] is False
    assert stages["hls_synthesis"] is False
    assert stages["vivado_project"] is False
    assert stages["bitstream"] is False
    assert stages["runtime_package"] is True
    assert stages["reports"] is True


@pytest.mark.parametrize(
    "stages, message",
    [
        ({"cpp": True, "hls_project": False, "hls_synthesis": True}, "hls_synthesis=true requires"),
        ({"cpp": True, "hls_project": True, "hls_synthesis": False, "vivado_project": True}, "vivado_project=true requires"),
        ({"cpp": True, "vivado_implementation": True, "vivado_project": False}, "vivado_implementation=true requires"),
        ({"cpp": True, "bitstream": True, "vivado_project": True, "vivado_implementation": False, "hls_synthesis": True, "hls_project": True}, "bitstream=true requires"),
        ({"cpp": False, "testbench": True}, "testbench=true requires"),
    ],
)
def test_resolve_build_stage_invalid_dependencies_reject(stages: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_build_stages({"build": {"stages": stages}})


def test_resolve_runtime_sequence_validates_commands_against_generated_modes() -> None:
    seq = _resolve_runtime_sequence(
        {"runtime": {"sequence": ["import_weights", {"run_inference": {"repeat": 3}}]}},
        pipeline_mode="inference",
        memory_semantics_mode="bram_import_full",
    )

    assert seq["explicit"] is True
    assert seq["sequence"][0]["command"] == "import_weights"
    assert seq["sequence"][1]["args"]["repeat"] == 3

    with pytest.raises(ValueError, match="not supported by generated artifacts"):
        _resolve_runtime_sequence(
            {"runtime": {"sequence": ["import_weights", "run_inference"]}},
            pipeline_mode="inference",
            memory_semantics_mode="bram_static",
        )


def test_resolve_codegen_readability_accepts_known_modes_and_rejects_typos() -> None:
    assert _resolve_codegen_readability({}) == "high"
    assert _resolve_codegen_readability({"codegen": {"readability": "compact"}}) == "compact"
    with pytest.raises(ValueError, match="codegen.readability"):
        _resolve_codegen_readability({"codegen": {"readability": "verbose-ish"}})


def test_runtime_package_records_selected_build_stages(tmp_path: Path) -> None:
    stages = {
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

    emit_runtime_package(tmp_path, build_stages=stages, weights_mode="bram_static")
    manifest = json.loads((tmp_path / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))

    assert manifest["build_stages"]["cpp"] is True
    assert manifest["build_stages"]["hls_project"] is False
    assert manifest["build_stages"]["bitstream"] is False


def test_compile_cpp_only_emits_sources_without_hls_run_script_and_records_manifest(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from fpgai.engine.compiler import Compiler

    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "cpp_only")
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
    raw.setdefault("toolchain", {}).setdefault("vitis_hls", {})["enabled"] = True

    result = Compiler(_make_config(raw, tmp_path)).compile()
    out_dir = Path(result.out_dir)

    assert (out_dir / "hls/src/deeplearn.cpp").exists()
    assert (out_dir / "hls/include/fpgai_params.h").exists()
    assert not (out_dir / "hls/run_hls.tcl").exists()
    assert result.hls_ran is False

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_stages"]["cpp"] is True
    assert manifest["build_stages"]["hls_project"] is False
    assert manifest["build_stages"]["hls_synthesis"] is False
    assert manifest["runtime_package"]["status"] == "created"
    assert manifest["runtime_sequence"]["sequence"][0]["command"] == "run_inference"
    assert manifest["resolved_config_artifacts"]["resolved_config_json"].endswith("reports/resolved_config.json")

    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI generated HLS top" in source
    assert "Weight semantics:" in source

    resolved_config = json.loads((out_dir / "reports/resolved_config.json").read_text(encoding="utf-8"))
    assert resolved_config["build_stages"]["hls_project"] is False
    assert resolved_config["runtime_sequence"]["sequence"][0]["command"] == "run_inference"
    assert (out_dir / "reports/config_contract.json").exists()
    assert (out_dir / "reports/runtime_sequence.json").exists()
    assert (out_dir / "reports/numeric_validation.json").exists()
    assert (out_dir / "reports/paper_verification.json").exists()
    assert (out_dir / "reports/paper_row.json").exists()
    assert (out_dir / "reports/model_compatibility.json").exists()
    assert (out_dir / "reports/layer_knob_contract.json").exists()

    numeric_validation = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    assert numeric_validation["status"] == "not_run"
    assert numeric_validation["paper_claim_allowed"]["numeric_correctness"] is False

    paper_row = json.loads((out_dir / "reports/paper_row.json").read_text(encoding="utf-8"))
    assert paper_row["source_generated"] is True
    assert paper_row["numeric_validated"] is False
    assert paper_row["paper_safe"] is False

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["numeric_validation_artifacts"]["numeric_validation_json"].endswith("reports/numeric_validation.json")
    assert manifest["paper_verification_artifacts"]["paper_row_json"].endswith("reports/paper_row.json")

    runtime_manifest = json.loads((out_dir / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    assert runtime_manifest["build_stages"]["cpp"] is True
    assert runtime_manifest["build_stages"]["hls_project"] is False
    assert runtime_manifest["runtime_sequence"]["sequence"][0]["command"] == "run_inference"
    assert (out_dir / "runtime_package/run_sequence.json").exists()
