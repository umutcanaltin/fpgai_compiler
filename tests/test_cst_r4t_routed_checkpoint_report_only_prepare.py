"""R4t post-hoc source-pinned DCP copy and inert Tcl checks.

All checkpoints and signed witnesses are SYNTHETIC; no vendor tools run
and no new physical E4 result is created.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4t_routed_checkpoint_report_only_prepare as sut


@pytest.fixture
def original(tmp_path, monkeypatch):
    roots = {
        "study": tmp_path / "original_r4p_native",
        "r4o_study": tmp_path / "original_r4o",
        "r4k_study": tmp_path / "original_r4k",
        "original_r4d": tmp_path / "original_r4d",
        "original_r4a": tmp_path / "original_r4a",
    }
    for folder in roots.values():
        folder.mkdir()
    arms = []
    for name, wns in (("full_storage", -2.170),
                      ("row_storage", -3.718)):
        file = (roots["study"] / name / "vivado" / "outputs" /
                "routed.dcp")
        file.parent.mkdir(parents=True)
        file.write_bytes(("SYNTHETIC-TEST-DCP-" + name).encode())
        arms.append({
            "arm": name,
            "original_signed_timing_summary_sha256": "a" * 64,
            "original_signed_drc_report_sha256": "b" * 64,
            "original_routed_wns_ns": wns,
            "timing_path_source_details": {
                "status": "UNKNOWN_NO_COMPLETE_PATH_DETAILS_IN_ORIGINAL_TIMING_SUMMARY",
            },
        })
    signed_r4s = {
        "schema": sut.r4s.SCHEMA,
        "status": "SIGNED_ORIGINAL_R4P_TIMING_DRC_TEXT_FORENSICS_NOT_E4",
        "original_source_revision": sut.SOURCE_REVISION,
        "source_attested_arms": 2,
        "physical_E4": None,
        "original_vendor_artifacts_modified": False,
        "no_new_vivado_or_vitis_calls": True,
        "arms": arms,
    }
    monkeypatch.setattr(sut.r4s, "inspect",
                        lambda **kw: signed_r4s)
    return roots, signed_r4s


def test_inspect_posthoc_checkpoint_attestation_not_original_vendor_hash(
    original,
):
    roots, _ = original
    result = sut.inspect(**roots)
    assert result["status"] == (
        "ORIGINAL_R4P_ROUTED_DCP_POSTHOC_PINNED_REPORT_ONLY_READY")
    assert result["original_report_sha_and_vendor_receipts_verified"] is True
    assert result["planned_new_report_only_vivado_calls"] == 2
    assert result["new_native_vivado_calls"] == 0
    assert result["strict_physical_E4"] is None
    assert result["board_qualified"] is False
    assert list(result["arms"]) == list(sut.ARMS)
    for key, arm in result["arms"].items():
        original_file = (roots["study"] / key / "vivado" /
                         "outputs" / "routed.dcp")
        assert arm["posthoc_original_checkpoint_sha256"] == (
            sha256(original_file.read_bytes()).hexdigest())
        assert arm["checkpoint_bytes"] == original_file.stat().st_size
        assert arm["original_dcp_sha_contemporaneous_in_R4p_vendor_receipt"] is False


def test_creates_isolated_checkpoint_copies_and_reports_only_scripts(
    original, tmp_path,
):
    roots, _ = original
    before = {
        str(path): sha256(path.read_bytes()).hexdigest()
        for root in roots.values() for path in root.rglob("*")
        if path.is_file()
    }
    out = tmp_path / "new_R4t_original_checkpoint_reports"
    result = sut.prepare(**roots, output=out)
    assert result["status"] == "R4T_ISOLATED_CHECKPOINT_REPORT_ONLY_SOURCE_PREPARED"
    for name in sut.ARMS:
        case = out / name
        checkpoint = case / "input" / "routed.dcp"
        original_dcp = (roots["study"] / name / "vivado" /
                        "outputs" / "routed.dcp")
        assert checkpoint.read_bytes() == original_dcp.read_bytes()
        text = (case / "report_only.tcl").read_text()
        assert "open_checkpoint $r4t_dcp" in text
        assert "report_timing -delay_type max -path_type full" in text
        assert "report_timing_summary -delay_type max" in text
        assert "report_drc -file" in text
        assert "max_delay_paths.rpt" in text
        assert "set r4t_case [file normalize [file dirname [info script]]]" in text
        for banned in (
                "synth_design", "opt_design", "place_design", "route_design",
                "write_checkpoint", "write_bitstream"):
            assert banned not in text
        assert not (case / "reports").exists()
    sha_manifest = json.loads((out / "artifact_sha256.json").read_text())
    assert sha_manifest["summary.json"] == sha256(
        (out / "summary.json").read_bytes()).hexdigest()
    assert len(sha_manifest) == 5
    for path, digest in sha_manifest.items():
        assert sha256((out / path).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="create-only"):
        sut.prepare(**roots, output=out)
    with pytest.raises(ValueError, match="overlaps originals"):
        sut.prepare(**roots, output=roots["study"] / "never_create")
    after = {
        str(path): sha256(path.read_bytes()).hexdigest()
        for root in roots.values() for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_missing_checkpoint_never_stages_fake_routed_design(original):
    roots, _ = original
    file = (roots["study"] / "row_storage" / "vivado" /
            "outputs" / "routed.dcp")
    file.unlink()
    with pytest.raises(ValueError, match="missing"):
        sut.inspect(**roots)


def test_symlinked_checkpoint_rejected_even_if_bytes_identical(original, tmp_path):
    roots, _ = original
    file = (roots["study"] / "row_storage" / "vivado" /
            "outputs" / "routed.dcp")
    elsewhere = tmp_path / "external_original.dcp"
    elsewhere.write_bytes(file.read_bytes())
    file.unlink()
    file.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlinked"):
        sut.inspect(**roots)


def test_zero_length_checkpoint_refused(original):
    roots, _ = original
    file = (roots["study"] / "full_storage" / "vivado" /
            "outputs" / "routed.dcp")
    file.write_bytes(b"")
    with pytest.raises(ValueError, match="size unsupported"):
        sut.inspect(**roots)


def test_bad_original_r4s_provenance_never_exposes_checkpoint(original):
    roots, signed = original
    signed["original_source_revision"] = "f" * 40
    with pytest.raises(ValueError, match="evidence unqualified"):
        sut.inspect(**roots)
    signed["original_source_revision"] = sut.SOURCE_REVISION
    signed["source_attested_arms"] = 1
    with pytest.raises(ValueError, match="evidence unqualified"):
        sut.inspect(**roots)


def test_only_original_two_arm_population_accepted(original):
    roots, signed = original
    signed["arms"][1]["arm"] = "full_storage"
    with pytest.raises(ValueError, match="evidence unqualified"):
        sut.inspect(**roots)
