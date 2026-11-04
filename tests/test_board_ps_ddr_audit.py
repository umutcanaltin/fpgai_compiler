"""No-vendor, read-only PYNQ-Z2 PS/DDR board warning provenance tests."""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.toolchains.board.ps_audit import (
    REFERENCE, audit_board_ddr, main,
)


def _sha(data):
    return sha256(data).hexdigest()


def _fixture(tmp_path):
    repo = tmp_path / "trusted_copy_of_pynq_board_defs"
    preset = repo / "pynq-z2" / "A.0" / "preset.xml"
    preset.parent.mkdir(parents=True)
    values = (-0.051, -0.006, -0.009, -0.033)
    preset.write_text(
        '<ip_presets schema="1.0"><ip_preset><ip>'
        + "".join(
            f'<user_parameter name="CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_{i}" '
            f'value="{value:.3f}"/>'
            for i, value in enumerate(values)
        )
        + "</ip></ip_preset></ip_presets>"
    )
    case = tmp_path / "board_pynq_toy_bit_02"
    logs = case / "logs"
    logs.mkdir(parents=True)
    logs.joinpath("stdout.log").write_text(
        "".join(
            f"CRITICAL WARNING: [PSU-{i+1}] Parameter : "
            f"PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_{i} has negative "
            f"value {value:.3f} . PS DDR interfaces might fail.\n"
            for i, value in enumerate(values)
        )
    )
    logs.joinpath("stderr.log").write_text("")
    artifacts = case / "board_artifacts"
    artifacts.mkdir()
    hashes = {}
    for name in (
        "route_status.rpt", "timing_summary.rpt", "drc.rpt", "utilization.rpt",
    ):
        data = (name + "\n").encode()
        artifacts.joinpath(name).write_bytes(data)
        hashes[name] = _sha(data)
    bit = artifacts / "fpgai_dma.bit"
    bit.write_bytes(b"actual-test-fixture-only-bit")
    hwh = artifacts / "fpgai_dma.hwh"
    hwh.write_bytes(b"<hardware-test-fixture/>")
    receipt = {
        "schema": "fpgai-research/board-dma-build/v1",
        "board": "pynq-z2",
        "board_part": "tul.com.tw:pynq-z2:part0:1.0",
        "part": "xc7z020clg400-1",
        "stage": "bitstream",
        "state": "BOARD_NOT_QUALIFIED",
        "routing_verified_complete": True,
        "bitstream_generated": True,
        "vivado_exit_code": 0,
        "timing_slack_ns": 2.311,
        "board_report_sha256": hashes,
        "bitstream_sha256": _sha(bit.read_bytes()),
        "hwh_sha256": _sha(hwh.read_bytes()),
        "vivado_log_diagnostics": {
            "critical_warning_count": 4,
            "critical_warning_codes": {
                "PSU-1": 1, "PSU-2": 1, "PSU-3": 1, "PSU-4": 1,
            },
            "logs_inspected": True,
        },
    }
    (case / "board_build_summary.json").write_text(json.dumps(receipt))
    return case, repo


def test_pynq_ps_ddr_audit_correlates_original_warnings_without_qualification(tmp_path):
    case, repo = _fixture(tmp_path)
    prior = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    }
    report = audit_board_ddr(case, repo)
    assert report["status"] == "EVIDENCE_CORRELATED_BOARD_REMAINS_UNQUALIFIED"
    assert report["case_artifact_hashes_verified"] is True
    assert report["native_warnings_match_original_preset"] is True
    assert report["board_qualified_by_this_audit"] is False
    assert report["board_programmed_by_this_audit"] is False
    assert report["differing_reference_lanes"] == [0, 1]
    assert [lane["public_pynq_reference_ns"] for lane in report["lanes"]] == list(REFERENCE)
    assert [lane["negative_delay_warning_seen"] for lane in report["lanes"]] == [True] * 4
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    } == prior


def test_readonly_cli_outputs_metadata_and_does_not_write_case(tmp_path, capsys):
    case, repo = _fixture(tmp_path)
    assert main(["--case", str(case), "--board-repo", str(repo)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["vendor_exit_code"] == 0
    assert result["board_qualified_by_this_audit"] is False
    assert result["board_programmed_by_this_audit"] is False
    assert not (case / "ps_ddr_audit.json").exists()


@pytest.mark.parametrize("mutation,pattern", [
    ("bit", "Bitstream or hardware-handoff digest mismatch"),
    ("hwh", "Bitstream or hardware-handoff digest mismatch"),
    ("report", "Vivado report digest mismatch"),
    ("native_log", "reported DDR delay differs"),
    ("missing_lane", "complete four-lane"),
    ("wrong_board", "routed, unqualified PYNQ-Z2 bitstream"),
    ("warning_summary", "do not match stored case receipt"),
])
def test_mismatch_never_qualifies_altered_or_untrusted_evidence(
        tmp_path, mutation, pattern):
    case, repo = _fixture(tmp_path)
    if mutation == "bit":
        (case / "board_artifacts" / "fpgai_dma.bit").write_bytes(b"changed")
    elif mutation == "hwh":
        (case / "board_artifacts" / "fpgai_dma.hwh").write_bytes(b"changed")
    elif mutation == "report":
        (case / "board_artifacts" / "drc.rpt").write_text("changed")
    elif mutation == "native_log":
        log = case / "logs" / "stdout.log"
        log.write_text(log.read_text().replace("-0.051", "-0.050"))
    elif mutation == "missing_lane":
        preset = repo / "pynq-z2" / "A.0" / "preset.xml"
        preset.write_text(preset.read_text().replace(
            '<user_parameter name="CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_3" value="-0.033"/>',
            "",
        ))
    elif mutation == "wrong_board":
        p = case / "board_build_summary.json"
        data = json.loads(p.read_text())
        data["board"] = "kv260"
        p.write_text(json.dumps(data))
    elif mutation == "warning_summary":
        p = case / "board_build_summary.json"
        data = json.loads(p.read_text())
        data["vivado_log_diagnostics"]["critical_warning_codes"] = {"PSU-1": 1}
        p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=pattern):
        audit_board_ddr(case, repo)


def test_symlinked_vendor_board_profile_is_rejected(tmp_path):
    case, repo = _fixture(tmp_path)
    preset = repo / "pynq-z2" / "A.0" / "preset.xml"
    backup = tmp_path / "elsewhere.xml"
    backup.write_bytes(preset.read_bytes())
    preset.unlink()
    preset.symlink_to(backup)
    with pytest.raises(ValueError, match="symlinked"):
        audit_board_ddr(case, repo)
