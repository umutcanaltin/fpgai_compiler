"""R4p isolated OOC staging/native protection and strict read-only DRC audits.

All Vivado invocations here are mocks. Synthetic report text is NOT
measured original FPGA results, and no mock creates a physical E4 result.
"""
from __future__ import annotations

from hashlib import sha256
import json
import sys

import pytest

from fpgai_research.experiments import cst_r4p_matched_vivado_ooc_native as sut


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return digest(path)


@pytest.fixture
def source(tmp_path, monkeypatch):
    r4o = tmp_path / "R4o"
    r4k = tmp_path / "R4k"
    r4d = tmp_path / "R4d"
    r4a = tmp_path / "R4a"
    for root in (r4o, r4k, r4d, r4a):
        root.mkdir()
    v = ("module fpgai_model(input ap_clk, output reg done);\n"
         "always @(posedge ap_clk) done <= 1;\nendmodule\n")
    rom = "00ff\n"
    arms = {}
    for arm in sut.ARMS:
        stage = r4o / arm / "rtl"
        stage.mkdir(parents=True)
        vsha = write(stage / "fpgai_model.v", v)
        romsha = write(stage / "weight.dat", rom)
        script = sut.r4o._tcl(rtl_names=["fpgai_model.v"])
        write(r4o / arm / "run_vivado.tcl", script)
        arms[arm] = {
            "original_package_path": str(r4k / arm / "vitis_source"),
            "rtl_sha256": {"fpgai_model.v": vsha},
            "rom_sha256": {"weight.dat": romsha},
            "original_rtl_file_count": 1,
            "original_rom_file_count": 1,
            "HLS_estimated_BRAM18K": 13,
            "source_cpp_sha256": ("a" if arm == "full_storage" else "b") * 64,
        }
    meta = {
        "schema": sut.r4o.SCHEMA,
        "status": "ORIGINAL_R4L_RTL_ROM_SOURCE_PINNED_READY_FOR_OOC_STAGING",
        "original_R4k_source_revision": "f" * 40,
        "original_R4d_source_revision": "d" * 40,
        "original_R4a_source_revision": "a" * 40,
        "target_part": "xck26-sfvc784-2LV-c",
        "requested_clock_ns": 12.0,
        "source_clock_uncertainty_ns": 1.5,
        "top": "fpgai_model",
        "arms": arms,
        "both_R4l_E2_E3_qualified": True,
        "physical_E4": None,
        "OOC_board_qualified": False,
        "new_vivado_calls": 0,
    }
    write(r4o / "summary.json", json.dumps({
        **meta, "status": "SOURCE_ONLY_VIVADO_OOC_PREPARED_NOT_EXECUTED"}))
    hashlist = {
        str(p.relative_to(r4o)): digest(p)
        for p in r4o.rglob("*") if p.is_file()
    }
    write(r4o / "artifact_sha256.json", json.dumps(hashlist))
    monkeypatch.setattr(sut.r4o, "inspect", lambda **kw: meta)
    return {"r4o_study": r4o, "r4k_study": r4k,
            "original_r4d": r4d, "original_r4a": r4a}


@pytest.fixture
def prepared(source, tmp_path):
    stage = tmp_path / "new_R4p_physical"
    assert sut.prepare(**source, output=stage)[
        "status"] == "R4P_NATIVE_CASES_PREPARED_NOT_EXECUTED"
    return source, stage


def args(prepared):
    original, root = prepared
    return {"study": root, **original}


def test_prepare_preserves_original_and_matches_canonical_runner_layout(
    prepared,
):
    source, root = prepared
    assert sut.audit_prepared(**args(prepared))["source_gate"] == (
        "ALL_R4L_R4O_R4P_SOURCE_BYTES_SHA_ATTESTED")
    files = json.loads((root / "artifact_sha256.json").read_text())
    assert "summary.json" in files
    for arm in sut.ARMS:
        case = root / arm
        canonical = (case / "fpgai_hls_project" / "solution1"
                     / "syn" / "verilog")
        working = case / "vivado"
        for suffix in ("fpgai_model.v", "weight.dat"):
            a, b = canonical / suffix, working / "rtl" / suffix
            assert a.read_bytes() == b.read_bytes()
            assert digest(a) == files[str(a.relative_to(root))]
            assert digest(b) == files[str(b.relative_to(root))]
        assert digest(working / "weight.dat") == (
            files[str((working / "weight.dat").relative_to(root))])
        assert (working / "run_vivado.tcl").read_text() == (
            source["r4o_study"] / arm / "run_vivado.tcl").read_text()
        assert not (working / "vivado_invocation.json").exists()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.prepare(**source, output=root)
    assert sut.execute(**args(prepared))["status"] == "DRY_RUN_NO_VENDOR_CALLS"


def test_tamper_original_r4o_fails_before_any_prepare(source, tmp_path):
    p = source["r4o_study"] / "row_storage" / "rtl" / "weight.dat"
    p.write_text(p.read_text() + "changed\n")
    with pytest.raises(ValueError, match="source-only SHA inventory mismatch"):
        sut.prepare(**source, output=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_tamper_new_staged_rtl_or_rom_fails_preflight(prepared):
    source, root = prepared
    p = root / "full_storage" / "vivado" / "rtl" / "fpgai_model.v"
    p.write_text(p.read_text() + "// unauthorized\n")
    with pytest.raises(ValueError, match="SHA"):
        sut.execute(**args(prepared))
    assert not (root / "full_storage" / "vivado" / "logs").exists()


def test_previous_partial_attempt_in_either_case_prevents_launch(
    prepared, monkeypatch,
):
    source, root = prepared
    (root / "row_storage" / "vivado" / "logs").mkdir()
    monkeypatch.setattr(
        sut, "run_vivado",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("should not launch native Vivado")))
    with pytest.raises(FileExistsError, match="already attempted"):
        sut.execute(**args(prepared), execute_native=True,
                    confirm_ooc_native=True, executable=sys.executable)


def test_native_requires_explicit_confirmation_and_bounded_timeout(prepared):
    with pytest.raises(ValueError, match="explicit confirmation"):
        sut.execute(**args(prepared), execute_native=True)
    with pytest.raises(ValueError, match="timeout"):
        sut.execute(**args(prepared), execute_native=True,
                    confirm_ooc_native=True,
                    executable=sys.executable, timeout_seconds=0)


def test_native_executes_only_separate_cases_and_stops_on_first_failure(
    prepared, monkeypatch,
):
    _, root = prepared
    calls = []
    def fake_run(folder, **kw):
        calls.append(folder)
        assert folder == root / "full_storage" / "vivado"
        (folder / "logs").mkdir()
        return {"completed_successfully": False}
    monkeypatch.setattr(sut, "run_vivado", fake_run)
    result = sut.execute(**args(prepared), execute_native=True,
                         confirm_ooc_native=True,
                         executable=sys.executable)
    assert result["status"] == (
        "R4P_NATIVE_ATTEMPT_FAILED_PRESERVE_ALL_OUTPUTS")
    assert result["native_vendor_calls_executed"] == 1
    assert calls == [root / "full_storage" / "vivado"]
    assert not (root / "row_storage" / "vivado" / "logs").exists()
    with pytest.raises(FileExistsError, match="already attempted"):
        sut.execute(**args(prepared), execute_native=True,
                    confirm_ooc_native=True, executable=sys.executable)


def _write_mock_receipt(case, *, source, wns=0.25,
                        drc_violations=0, critical=False,
                        fully_routed=True):
    """Synthetic report only: not a real vendor invocation."""
    native = case / "vivado"
    reports = native / "reports"
    reports.mkdir(exist_ok=True)
    write(reports / "route_status.rpt", (
        "Design Route Status\n"
        "# of routable nets : 10\n"
        f"# of fully routed nets : {10 if fully_routed else 9}\n"
        "# of nets with routing errors : 0\n"
        f"# of unrouted nets : {0 if fully_routed else 1}\n"))
    write(reports / "timing_summary.rpt",
          f"Design Timing Summary\nWNS (ns): {wns:.3f}\n")
    write(reports / "utilization.rpt", (
        "| CLB LUTs | 100 |\n"
        "| CLB Registers | 150 |\n"
        "| DSPs | 2 |\n"
        "| Block RAM Tile | 2.0 |\n"
        "| URAM | 0 |\n"
    ))
    drc = (f"Violations found: {drc_violations}\n")
    if drc_violations:
        kind = "Error" if critical else "Warning"
        drc += f"| NSTD-1 | {kind} | example | {drc_violations} |\n"
    write(reports / "drc.rpt", drc)
    logs = native / "logs"
    write(logs / "vivado.stdout.log", "SYNTHETIC VIVADO EXAMPLE ONLY\\n")
    write(logs / "vivado.stderr.log", "")
    script_sha = digest(native / "run_vivado.tcl")
    receipt = {
        "schema": "fpgai-research/vendor-invocation/v1",
        "tool": "vivado",
        "executable": "/tools/mock/vivado",
        "argv": ["/tools/mock/vivado", "-mode", "batch", "-source",
                 "run_vivado.tcl", "-notrace"],
        "cwd": str(native.resolve()),
        "started_utc": "2026-11-06T12:00:00+00:00",
        "duration_seconds": 3.0,
        "input_hashes_before_execution": {"tcl": script_sha},
        "input_hashes_after_execution": {"tcl": script_sha},
        "stdout_sha256": digest(logs / "vivado.stdout.log"),
        "stderr_sha256": digest(logs / "vivado.stderr.log"),
        "completed_successfully": True,
        "returncode": 0,
        "timed_out": False,
        "inputs_stable_during_execution": True,
        "rom_inputs_stable_during_execution": True,
        "staged_rom_inputs_stable_during_execution": True,
        "rtl_input_sha256": source["rtl_sha256"],
        "rom_input_sha256": source["rom_sha256"],
        "script_sha256": digest(native / "run_vivado.tcl"),
        "staged_rom_input_integrity_before": {
            "matches_original_roms": True},
        "staged_rom_input_integrity_after": {
            "matches_original_roms": True},
        "allow_verified_readmem_post_absence": False,
        "fresh_report_sha256": {
            name: digest(reports / name)
            for name in sut._REPORTS},
    }
    write(native / "vivado_invocation.json", json.dumps(receipt))


def test_audit_without_native_receipts_preserves_all_physical_unknowns(
    prepared,
):
    record = sut.audit_native(**args(prepared))
    assert record["status"] == "UNKNOWN_INCOMPLETE_OR_FAILED_PHYSICAL_PAIR"
    assert record["original_native_invocation_receipts_present"] == 0
    assert record["both_strict_clean_DRC_OOC_E4"] is False
    assert record["descriptive_physical_resource_delta_row_minus_full"] == {
        k: None for k in sut._RESOURCE}
    assert record["new_vivado_calls_during_audit"] == 0
    assert record["physical_achieved_board_Fmax"] is None


def test_mock_clean_routed_two_arm_pair_requires_source_receipts_and_drc(
    prepared, monkeypatch,
):
    source, root = prepared
    orig = sut._original_r4o(**source)
    for arm in sut.ARMS:
        _write_mock_receipt(root / arm, source=orig["arms"][arm])
    report = sut.audit_native(**args(prepared))
    assert report["status"] == "STRICT_CLEAN_DRC_BOTH_OOC_ROUTED"
    assert report["original_native_invocation_receipts_present"] == 2
    assert report["both_source_attributed_routed_observations_valid"]
    assert report["both_strict_clean_DRC_OOC_E4"]
    assert report["strict_clean_DRC_resource_delta_row_minus_full"]["bram"] == 0
    assert report["strict_clean_DRC_WNS_delta_ns"] == 0
    assert report["board_qualified"] is False
    assert report["physical_achieved_board_Fmax"] is None


def test_mock_nonclean_drc_is_observation_not_strict_clean_e4(prepared):
    source, root = prepared
    orig = sut._original_r4o(**source)
    _write_mock_receipt(root / "full_storage",
                        source=orig["arms"]["full_storage"])
    _write_mock_receipt(root / "row_storage",
                        source=orig["arms"]["row_storage"],
                        drc_violations=1, critical=True)
    report = sut.audit_native(**args(prepared))
    assert report["status"] == "ROUTED_NONCLEAN_DRC_PAIR"
    assert report["both_source_attributed_routed_observations_valid"] is True
    assert report["both_strict_clean_DRC_OOC_E4"] is False
    assert report["strict_clean_DRC_resource_delta_row_minus_full"]["bram"] is None
    assert report["arms"][1]["drc_errors_or_critical_warnings_present"] is True
    assert report["arms"][1]["drc_warning_only_routed_observation"] is False


@pytest.mark.parametrize("bad", ["tampered_report", "incomplete_route",
                                  "negative_WNS", "missing_DRC"])
def test_mock_incomplete_physical_reports_never_qualify(
    prepared, bad,
):
    source, root = prepared
    orig = sut._original_r4o(**source)
    for arm in sut.ARMS:
        _write_mock_receipt(root / arm, source=orig["arms"][arm])
    folder = root / "row_storage" / "vivado" / "reports"
    if bad == "tampered_report":
        (folder / "timing_summary.rpt").write_text("modified\n")
    elif bad == "incomplete_route":
        _write_mock_receipt(
            root / "row_storage", source=orig["arms"]["row_storage"],
            fully_routed=False)
    elif bad == "negative_WNS":
        _write_mock_receipt(
            root / "row_storage", source=orig["arms"]["row_storage"],
            wns=-0.5)
    elif bad == "missing_DRC":
        (folder / "drc.rpt").unlink()
    report = sut.audit_native(**args(prepared))
    assert report["both_strict_clean_DRC_OOC_E4"] is False
    assert report["descriptive_physical_resource_delta_row_minus_full"]["bram"] is None
    assert report["physical_achieved_board_Fmax"] is None



@pytest.mark.parametrize("tamper", [
    "wrong_working_directory", "changed_argv", "wrong_tcl_input_sha",
    "modified_stdout", "modified_rom_consumed",
])
def test_forged_or_tampered_native_receipt_does_not_qualify_e4(
    prepared, tamper,
):
    source, root = prepared
    orig = sut._original_r4o(**source)
    for arm in sut.ARMS:
        _write_mock_receipt(root / arm, source=orig["arms"][arm])
    native = root / "row_storage" / "vivado"
    p = native / "vivado_invocation.json"
    receipt = json.loads(p.read_text())
    if tamper == "wrong_working_directory":
        receipt["cwd"] = "/tmp/other_case"
    elif tamper == "changed_argv":
        receipt["argv"] = ["/tools/mock/vivado", "-mode", "batch",
                           "-source", "untracked.tcl", "-notrace"]
    elif tamper == "wrong_tcl_input_sha":
        receipt["input_hashes_before_execution"]["tcl"] = "0" * 64
    elif tamper == "modified_stdout":
        (native / "logs" / "vivado.stdout.log").write_text("altered original\n")
    elif tamper == "modified_rom_consumed":
        (native / "rtl" / "weight.dat").write_text("altered input ROM\n")
    if tamper not in ("modified_stdout", "modified_rom_consumed"):
        p.write_text(json.dumps(receipt))
    if tamper == "modified_rom_consumed":
        with pytest.raises(ValueError, match="SHA"):
            sut.audit_native(**args(prepared))
    else:
        result = sut.audit_native(**args(prepared))
        assert result["both_strict_clean_DRC_OOC_E4"] is False
        assert result["descriptive_physical_resource_delta_row_minus_full"][
            "bram"] is None
        assert result["arms"][1]["status"] == (
            "REJECTED_VENDOR_OR_SOURCE_PROVENANCE")
