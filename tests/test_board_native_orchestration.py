"""No-vendor build orchestration tests: simulated vendors are not hardware evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from fpgai_research.toolchains.board import cli
from fpgai_research.toolchains.onboarding import VendorTool


def _args(tmp_path, *, stage="hls", accept=False):
    case = tmp_path / "original_case"
    case.mkdir()
    (case / "model.cpp").write_text("// immutable original source")
    # These tests intentionally stub the ONNX/Model IR verifier to exercise
    # vendor orchestration only. Keep its required manifest present so a
    # missing provenance file cannot mask all native orchestration assertions.
    (case / "manifest.json").write_text(json.dumps({
        "input_shape": [2, 3],
        "output_shape": [2, 2],
        "precision_bits": 16,
        "integer_bits": 6,
    }))
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"fake-original-onnx")
    return argparse.Namespace(
        onnx=str(onnx), case=str(case),
        output=str(tmp_path / "new_board"),
        board="pynq-z2", board_part="vendor.test:pynq-z2:part0:1.0",
        clock_ns=10.0, stage=stage,
        vitis="/does-not-run/vitis", vivado="/does-not-run/vivado",
        hls_timeout=7200, vivado_timeout=14400,
        execute_native=True, plan=False, accept_board_advisories=accept,
    )


def _mock_vendor(monkeypatch, drc_text="Violations found: 0\n",
                 slack=1.25, vendor_failed=False,
                 vivado_critical_lines="", vivado_stderr_critical_lines=""):
    monkeypatch.setattr(cli, "_verify_original_case", lambda case, onnx: {
        "input_shape": [2, 3], "output_shape": [2, 2],
        "precision_bits": 16, "integer_bits": 6,
    })
    monkeypatch.setattr(cli, "resolve_vendor_tool",
                        lambda name, root: VendorTool("/fake/" + name, None))
    monkeypatch.setattr(cli, "vendor_environment", lambda tools: None)
    calls = []

    def fake_run(executable, argv, cwd, environment, timeout, label):
        calls.append(label)
        (cwd / "logs").mkdir(exist_ok=True)
        (cwd / "logs" / "stderr.log").write_text(
            vivado_stderr_critical_lines if label == "vivado_board" else ""
        )
        if label == "vitis_board":
            assert argv == ["-f", "run_dma_hls.tcl"]
            assert (cwd / "fpgai_dma.cpp").is_file()
            assert (cwd / "fpgai_dma_tb.cpp").is_file()
            assert (cwd / "model.cpp").is_file()
            (cwd / "logs" / "stdout.log").write_text("FPGAI_BOARD_DMA_CSIM_PASS\n")
            ip = cwd / "fpgai_dma_hls_project" / "solution1" / "impl" / "ip"
            ip.mkdir(parents=True)
            (ip / "component.xml").write_text("<ip/>")
        else:
            assert label == "vivado_board"
            assert argv == ["-mode", "batch", "-source",
                            "run_board.tcl", "-notrace"]
            (cwd / "logs" / "stdout.log").write_text(
                vivado_critical_lines + "FPGAI_BOARD_ARTIFACTS_EXPORTED\n")
            artifacts = cwd / "board_artifacts"
            artifacts.mkdir()
            (artifacts / "timing_summary.rpt").write_text(
                "Design Timing Summary\nWNS (ns): " + str(slack) + "\n"
            )
            (artifacts / "drc.rpt").write_text(drc_text)
            (artifacts / "route_status.rpt").write_text(
                "Design Route Status\n"
                "# of routable nets: 5\n"
                "# of fully routed nets: 5\n"
                "# of nets with routing errors: 0\n"
            )
            (artifacts / "utilization.rpt").write_text(
                "| CLB LUTs | 100 |\n"
            )
            (artifacts / "fpgai_dma.bit").write_bytes(b"synthetic-testing-only")
            (artifacts / "fpgai_dma.hwh").write_bytes(b"<fake-hwh/>")
        return {"completed_successfully": label == "vitis_board" or not vendor_failed,
                "returncode": 1 if label == "vivado_board" and vendor_failed else 0}
    monkeypatch.setattr(cli, "_run_vendor", fake_run)
    return calls


def test_native_hls_stage_requires_qualified_new_ip_and_preserves_source(
        monkeypatch, tmp_path, capsys):
    args = _args(tmp_path)
    calls = _mock_vendor(monkeypatch)
    original = (tmp_path / "original_case" / "model.cpp").read_bytes()
    assert cli.run(args) == 0
    assert calls == ["vitis_board"]
    summary = json.loads((tmp_path / "new_board" / "board_build_summary.json").read_text())
    assert summary["state"] == "BOARD_IP_EXPORTED_NOT_BITSTREAM"
    assert summary["hardware_programmed"] is False
    assert (tmp_path / "original_case" / "model.cpp").read_bytes() == original
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.run(args)


def test_native_board_build_with_synthetic_clean_reports_is_not_device_inference(
        monkeypatch, tmp_path):
    args = _args(tmp_path, stage="bitstream")
    calls = _mock_vendor(monkeypatch)
    assert cli.run(args) == 0
    assert calls == ["vitis_board", "vivado_board"]
    result = json.loads((tmp_path / "new_board" / "board_build_summary.json").read_text())
    assert result["state"] == "BOARD_BITSTREAM_DRC_CLEAN_UNTESTED_ON_DEVICE"
    assert result["bitstream_generated"] is True
    assert result["on_device_inference_verified"] is False
    assert result["drc_clean"] is True
    assert result["routing_verified_complete"] is True
    assert len(result["board_report_sha256"]) == 4
    copied = tmp_path / "new_board" / "board_artifacts"
    assert (copied / "fpgai_pynq_runtime.py").is_file()
    assert (copied / "board_build_summary.json").is_file()
    assert (copied / "fpgai_dma.hwh").is_file()


def test_warning_only_board_build_requires_optin(monkeypatch, tmp_path):
    args = _args(tmp_path, stage="bitstream")
    advisory = (
        "Violations found: 2\n"
        "| Rule | Severity | Description | Violations |\n"
        "| DPIP-2 | Warning | Input pipelining | 2 |\n"
    )
    _mock_vendor(monkeypatch, drc_text=advisory)
    assert cli.run(args) == 3
    r = json.loads((tmp_path / "new_board" / "board_build_summary.json").read_text())
    assert r["state"] == "BOARD_NOT_QUALIFIED"
    assert r["warning_only_eligible"] is True
    assert r["strict_board_drc_clean_qualified"] is False


def test_warning_only_board_build_explicit_optin(monkeypatch, tmp_path):
    args = _args(tmp_path, stage="bitstream", accept=True)
    advisory = (
        "Violations found: 2\n"
        "| Rule | Severity | Description | Violations |\n"
        "| DPIP-2 | Warning | Input pipelining | 2 |\n"
    )
    _mock_vendor(monkeypatch, drc_text=advisory)
    assert cli.run(args) == 0
    r = json.loads((tmp_path / "new_board" / "board_build_summary.json").read_text())
    assert r["state"] == "BOARD_BITSTREAM_WITH_ADVISORIES_NOT_DEVICE_TESTED"
    assert r["strict_board_drc_clean_qualified"] is False
    assert r["on_device_inference_verified"] is False


@pytest.mark.parametrize("drc_text,slack,vendor_failed", [
    ("Violations found: 1\n| Rule | Severity | Description | Violations |\n"
     "| UCIO-1 | Warning | Unconstrained IO | 1 |\n", 1.0, False),
    ("Violations found: 1\n| Rule | Severity | Description | Violations |\n"
     "| DPIP-2 | Critical Warning | Input pipelining | 1 |\n", 1.0, False),
    ("Violations found: 2\n| Rule | Severity | Description | Violations |\n"
     "| DPIP-2 | Warning | Input pipelining | 1 |\n", 1.0, False),
    ("Violations found: 0\n", -0.4, False),
    ("Violations found: 0\n", 1.0, True),
])
def test_unknown_rule_critical_count_mismatch_timing_or_native_failure_rejected(
        monkeypatch, tmp_path, drc_text, slack, vendor_failed):
    args = _args(tmp_path, stage="bitstream", accept=True)
    _mock_vendor(monkeypatch, drc_text=drc_text, slack=slack,
                 vendor_failed=vendor_failed)
    assert cli.run(args) == 3
    report = json.loads((tmp_path / "new_board" / "board_build_summary.json").read_text())
    assert report["state"] == "BOARD_NOT_QUALIFIED"



def test_missing_original_manifest_rejected_before_vendor_calls(
        monkeypatch, tmp_path):
    args = _args(tmp_path)
    calls = _mock_vendor(monkeypatch)
    (tmp_path / "original_case" / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        cli.run(args)
    assert calls == []
    assert not (tmp_path / "new_board").exists()



def test_native_hls_export_works_without_vivado_board_preset(
        monkeypatch, tmp_path):
    """The mocked HLS export is part-qualified; no board_part is required."""
    args = _args(tmp_path, stage="hls")
    args.board_part = None
    calls = _mock_vendor(monkeypatch)
    assert cli.run(args) == 0
    assert calls == ["vitis_board"]
    report = json.loads(
        (tmp_path / "new_board" / "board_build_summary.json").read_text()
    )
    assert report["state"] == "BOARD_IP_EXPORTED_NOT_BITSTREAM"
    assert report["board_part"] is None
    assert report["board_preset_used_for_hls"] is False
    assert report["part"] == "xc7z020clg400-1"


@pytest.mark.parametrize("native_warning,stderr_warning", [
    ("CRITICAL WARNING: [PSU-1] Negative PS DDR clock-to-DQS delay\n", ""),
    ("CRITICAL WARNING: [BD 41-737] Reset polarity property read-only\n", ""),
    ("", "CRITICAL WARNING: [PSU-4] DDR skew violation\n"),
    ("CRITICAL WARNING: [UNEXPECTED-77] Unknown board warning\n", ""),
])
def test_native_critical_warnings_block_board_qualification_even_if_drc_clean(
        monkeypatch, tmp_path, native_warning, stderr_warning):
    """Real Vivado can generate a bitstream despite critical PS/BD warnings."""
    args = _args(tmp_path, stage="bitstream", accept=True)
    _mock_vendor(monkeypatch, vivado_critical_lines=native_warning,
                 vivado_stderr_critical_lines=stderr_warning)
    assert cli.run(args) == 3
    summary = json.loads(
        (tmp_path / "new_board" / "board_build_summary.json").read_text()
    )
    assert summary["state"] == "BOARD_NOT_QUALIFIED"
    assert summary["bitstream_generated"] is True
    assert summary["drc_clean"] is True
    assert summary["drc_critical_warnings"] == 0
    assert summary["vivado_log_diagnostics"]["critical_warning_count"] == 1
    assert summary["strict_board_drc_clean_qualified"] is False
    assert not (tmp_path / "new_board" / "board_artifacts" /
                "fpgai_pynq_runtime.py").exists()


def test_board_warning_audit_counts_independent_ps_and_bd_messages(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "stdout.log").write_text(
        "CRITICAL WARNING: [PSU-1] negative DDR lane 0\n"
        "CRITICAL WARNING: [PSU-2] negative DDR lane 1\n"
        "CRITICAL WARNING: [PSU-3] negative DDR lane 2\n"
        "CRITICAL WARNING: [PSU-4] negative DDR lane 3\n"
        "CRITICAL WARNING: [BD 41-737] invalid reset polarity parameter\n"
        "WARNING: [DRC RTSTAT-10] no routable loads\n"
        "INFO: [DRC REQP-181] writefirst\n"
    )
    (log_dir / "stderr.log").write_text("")
    result = cli._vivado_log_diagnostics(tmp_path)
    assert result["logs_inspected"] is True
    assert result["critical_warning_count"] == 5
    assert result["ps_ddr_critical_warning_count"] == 4
    assert result["bd_configuration_critical_warning_count"] == 1
    assert result["critical_warning_codes"]["BD 41-737"] == 1
    assert "RTSTAT-10" not in result["critical_warning_codes"]


def test_missing_vivado_log_blocks_qualification_even_with_fake_zero_drc(
        monkeypatch, tmp_path):
    args = _args(tmp_path, stage="bitstream")
    calls = _mock_vendor(monkeypatch)
    base_vendor = cli._run_vendor

    def missing_stderr(*positional, **keyword):
        receipt = base_vendor(*positional, **keyword)
        if positional[-1] == "vivado_board":
            (positional[2] / "logs" / "stderr.log").unlink()
        return receipt

    monkeypatch.setattr(cli, "_run_vendor", missing_stderr)
    assert cli.run(args) == 3
    summary = json.loads(
        (tmp_path / "new_board" / "board_build_summary.json").read_text()
    )
    assert summary["vivado_log_diagnostics"]["logs_inspected"] is False
    assert summary["state"] == "BOARD_NOT_QUALIFIED"
    assert calls == ["vitis_board", "vivado_board"]
