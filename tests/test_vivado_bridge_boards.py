from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpgai.backends.vivado.boards import board_names, get_board
from fpgai.backends.vivado.vivado_bridge import generate_vivado_bridge_for_artifact


def _minimal_hls_artifact(tmp_path: Path) -> Path:
    build = tmp_path / "artifact" / "build"
    hls = build / "hls"
    hls.mkdir(parents=True)
    (hls / "run_hls.tcl").write_text("# hls script\n", encoding="utf-8")
    (build / "manifest.json").write_text(
        json.dumps({"top_name": "deeplearn"}),
        encoding="utf-8",
    )
    return tmp_path / "artifact"


def test_board_registry_contains_supported_vivado_bridge_boards() -> None:
    assert set(board_names()) >= {"pynq_z2", "kv260", "kr260"}

    pynq = get_board("pynq-z2")
    kv260 = get_board("kv260")
    kr260 = get_board("kr260")

    assert pynq.ps_type == "processing_system7"
    assert kv260.ps_type == "zynq_ultra_ps_e"
    assert kr260.ps_type == "zynq_ultra_ps_e"

    assert pynq.supports_bridge_generation is True
    assert kv260.supports_bridge_generation is True
    assert kr260.supports_bridge_generation is True
    assert kv260.supports_vivado_impl is True
    assert kr260.supports_vivado_impl is True


@pytest.mark.parametrize(
    ("board", "expected_ps", "forbidden_ps", "expected_part"),
    [
        ("pynq_z2", "processing_system7", "zynq_ultra_ps_e", "xc7z020clg400-1"),
        ("kv260", "zynq_ultra_ps_e", "processing_system7", "xck26-sfvc784-2LV-c"),
        ("kr260", "zynq_ultra_ps_e", "processing_system7", "xck26-sfvc784-2LV-c"),
    ],
)
def test_generate_vivado_bridge_uses_board_specific_ps(
    tmp_path: Path,
    board: str,
    expected_ps: str,
    forbidden_ps: str,
    expected_part: str,
) -> None:
    artifact_dir = _minimal_hls_artifact(tmp_path)

    result = generate_vivado_bridge_for_artifact(
        artifact_dir,
        board_name=board,
        run_impl_default=True,
    )

    bridge = Path(result["vivado_bridge_dir"])
    create_bd = (bridge / "scripts" / "create_bd.tcl").read_text(encoding="utf-8")
    run_vivado = (bridge / "scripts" / "run_vivado.tcl").read_text(encoding="utf-8")
    manifest = json.loads((bridge / "vivado_bridge_manifest.json").read_text(encoding="utf-8"))

    assert expected_ps in create_bd
    assert forbidden_ps not in create_bd
    assert f'set part "{expected_part}"' in run_vivado

    assert manifest["board"] == board
    assert manifest["part"] == expected_part
    assert manifest["ps_type"] == expected_ps
    assert manifest["supports_bridge_generation"] is True
    assert manifest["supports_vivado_synth"] is True
    assert manifest["supports_vivado_impl"] is True
    assert manifest["vivado_impl_requested"] is True
    assert manifest["bitstream_requested"] is True


def test_unknown_vivado_board_fails_cleanly(tmp_path: Path) -> None:
    artifact_dir = _minimal_hls_artifact(tmp_path)

    with pytest.raises(KeyError, match="Unknown Vivado board"):
        generate_vivado_bridge_for_artifact(artifact_dir, board_name="unknown_board")


def test_board_registry_exposes_resource_limits_for_supported_boards() -> None:
    pynq = get_board("pynq-z2")
    kv260 = get_board("kv260")
    kr260 = get_board("kr260")

    assert pynq.resource_limits()["lut"] == 53200
    assert pynq.resource_limits()["ff"] == 106400
    assert pynq.resource_limits()["bram_18k"] == 280
    assert pynq.resource_limits()["uram"] == 0
    assert pynq.resource_limits()["dsp"] == 220
    assert pynq.resource_limits()["ddr_bytes"] == 512 * 1024 * 1024
    assert pynq.resource_limits()["safe_clock_mhz"] == 100.0

    assert kv260.resource_limits()["lut"] == 117120
    assert kv260.resource_limits()["ff"] == 234240
    assert kv260.resource_limits()["bram_18k"] == 288
    assert kv260.resource_limits()["uram"] == 64
    assert kv260.resource_limits()["dsp"] == 1248
    assert kv260.resource_limits()["ddr_bytes"] == 4 * 1024 ** 3
    assert kv260.resource_limits()["safe_clock_mhz"] == 100.0

    assert kr260.resource_limits() == kv260.resource_limits()


def test_board_fit_classifier_marks_kv260_over_limit_dsp() -> None:
    from fpgai.reporting.hardware_feasibility import classify_board_fit

    fit = classify_board_fit(
        {"lut": 80890, "ff": 78034, "bram_18k": 50, "uram": 0, "dsp": 2333},
        board="kv260",
    )

    assert fit["status"] == "over_limit"
    assert fit["limiting_resource"] == "dsp"
    assert fit["resources"]["dsp"]["available"] == 1248
    assert fit["resources"]["dsp"]["status"] == "over_limit"
    assert fit["vivado_allowed"] is False


def test_board_fit_classifier_marks_kv260_near_limit() -> None:
    from fpgai.reporting.hardware_feasibility import classify_board_fit

    fit = classify_board_fit(
        {"lut": 50000, "ff": 50000, "bram_18k": 50, "uram": 0, "dsp": 1100},
        board="kv260",
    )

    assert fit["status"] == "near_limit"
    assert fit["limiting_resource"] == "dsp"
    assert fit["resources"]["dsp"]["status"] == "near_limit"
    assert fit["vivado_allowed"] is True


def test_board_fit_classifier_marks_small_kv260_design_fits() -> None:
    from fpgai.reporting.hardware_feasibility import classify_board_fit

    fit = classify_board_fit(
        {"lut": 9500, "ff": 4299, "bram_18k": 18, "uram": 0, "dsp": 16, "target_clock_mhz": 100},
        board="kv260",
    )

    assert fit["status"] == "fits"
    assert fit["resources"]["dsp"]["status"] == "fits"
    assert fit["resources"]["target_clock_mhz"]["status"] == "fits"
    assert fit["vivado_allowed"] is True


def test_board_fit_classifier_checks_uram_ddr_and_clock() -> None:
    from fpgai.reporting.hardware_feasibility import classify_board_fit

    uram_over = classify_board_fit({"uram": 65}, board="kv260")
    assert uram_over["status"] == "over_limit"
    assert uram_over["limiting_resource"] == "uram"

    ddr_over = classify_board_fit({"ddr_bytes": 5 * 1024 ** 3}, board="kv260")
    assert ddr_over["status"] == "over_limit"
    assert ddr_over["limiting_resource"] == "ddr_bytes"

    clock_warn = classify_board_fit({"target_clock_mhz": 200}, board="kv260")
    assert clock_warn["status"] == "near_limit"
    assert clock_warn["limiting_resource"] == "target_clock_mhz"
    assert clock_warn["vivado_allowed"] is True


def test_board_fit_extracts_prediction_totals_aliases(tmp_path) -> None:
    import json
    from pathlib import Path

    from fpgai.reporting.hardware_feasibility import emit_board_fit_report

    reports = tmp_path / "reports"
    art = emit_board_fit_report(
        reports,
        resource_data={
            "totals": {
                "predicted_lut": 2802,
                "predicted_ff": 3340,
                "predicted_dsp": 6,
                "predicted_bram18": 4,
            }
        },
        timing_data={"target_clock_mhz": 200},
        board="kv260",
        part="xck26-sfvc784-2LV-c",
        source="prediction",
    )

    payload = json.loads((reports / "board_fit.json").read_text())
    assert payload["normalized_resources"]["lut"] == 2802
    assert payload["normalized_resources"]["ff"] == 3340
    assert payload["normalized_resources"]["dsp"] == 6
    assert payload["normalized_resources"]["bram_18k"] == 4
    assert payload["fit"]["resources"]["dsp"]["status"] == "fits"
    assert payload["fit"]["resources"]["target_clock_mhz"]["status"] == "near_limit"
    assert payload["fit"]["status"] == "near_limit"
    assert art["status"] == "near_limit"
    assert art["vivado_allowed"] is True


def test_vivado_bridge_runner_honors_fit_policy_gate_in_source() -> None:
    source = Path("fpgai/backends/vivado/run_bridge.py").read_text(encoding="utf-8")

    assert "def _fit_policy_gate_from_manifest" in source
    assert "def _vivado_gate_block_reason" in source
    assert "fit_policy_gate_blocked" in source
    assert "fit_policy_gate_reason" in source
    assert "run_vivado_impl" in source
    assert "board_fit_status" in source
    assert "blocked_stages" in source
    assert "compile_project_dir = bridge.parent" in source
    assert "_vivado_gate_block_reason(compile_project_dir" in source
    assert "gate_block_reason = _vivado_gate_block_reason(project_dir" not in source
    assert "blocked_manifest" in source
    assert '"design": artifact.name' in source
    assert '"hls_ip_export_requested": bool(export_hls_ip or run_vivado_synth or run_vivado_impl)' in source
    assert '"vivado_reports_present": False' in source
    assert "bridge.mkdir(parents=True, exist_ok=True)" in source
    assert '"vivado_ran": False' in source
    assert '"bitstream_requested": bool(run_vivado_impl)' in source
