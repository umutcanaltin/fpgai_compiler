"""R4e: test original SHA, negative native gates and UNKNOWN HLS outcomes.

Tests only launch host g++ in temporary R4a/R4d sources; NEVER Vitis.
"""
from __future__ import annotations

from hashlib import sha256
import json
import shutil
import sys

import pytest

from fpgai_research.experiments import cst_r4a_row_retention_study as r4a
from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments import cst_r4e_numeric_format_native as sut


@pytest.fixture()
def prepared_original_studies(tmp_path, monkeypatch):
    if shutil.which("g++") is None:
        pytest.skip("Real g++ required for original E1 scientific source receipts")
    monkeypatch.setattr(r4a, "_git_revision", lambda: "c" * 40)
    original = tmp_path / "r4a_original"
    staged = tmp_path / "r4d_original"
    r4a.run(output=original, profile="paper")
    r4d.run(original_study=original, output=staged)
    return original, staged


def _digests(root):
    return {
        str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file()
    }


def test_original_source_audit_and_dry_native_plan_no_vitis(
    prepared_original_studies,
):
    original, staged = prepared_original_studies
    unchanged = _digests(staged)
    evidence = sut.audit_prepared(study=staged, original_study=original)
    assert evidence["format_pairs"] == 3
    assert evidence["source_attested_arms"] == 6
    assert evidence["source_and_testbench_attested"] is True
    assert evidence["original_R4a_source_revision"] == "c" * 40
    assert evidence["original_failed_E2_result_retained"] is True
    dry = sut.execute(
        study=staged, original_study=original, start_format=1, max_formats=1)
    assert dry["status"] == "DRY_RUN_NO_VENDOR_CALLS"
    assert dry["selected_formats"] == [[32, 6]]
    assert dry["planned_native_vendor_calls"] == 2
    assert dry["new_vendor_calls"] == 0
    summary = sut.audit_native(study=staged, original_study=original)
    assert summary["prespecified_matched_pairs"] == 3
    assert summary["original_native_invocation_files"] == 0
    assert summary["E2_qualified_arms"] == 0
    assert summary["E3_qualified_pairs"] == 0
    assert len(summary["observations"]) == 6
    assert summary["native_vendor_calls_during_read_only_audit"] == 0
    assert all(row["CSim_failed_outputs"] is None for row in summary["observations"])
    assert all(row["physical_E4"] is None for row in summary["observations"])
    assert all(row["HLS_estimated_resource_delta_row_minus_full"]["bram_18k"]
               is None for row in summary["pairs"])
    assert _digests(staged) == unchanged


def test_reject_bad_ranges_or_missing_native_confirmation(
    prepared_original_studies,
):
    original, staged = prepared_original_studies
    for start, length in ((-1, 1), (3, 1), (0, 0), (2, 2)):
        with pytest.raises(ValueError, match="selection"):
            sut.execute(
                study=staged, original_study=original,
                start_format=start, max_formats=length)
    with pytest.raises(ValueError, match="confirmation"):
        sut.execute(
            study=staged, original_study=original,
            execute_native=True, confirm_isolated_native=False,
            vitis_executable=sys.executable)
    with pytest.raises(ValueError, match="timeout"):
        sut.execute(
            study=staged, original_study=original,
            execute_native=True, confirm_isolated_native=True,
            vitis_executable=sys.executable, timeout_seconds=0)


def test_all_selected_arms_checked_before_vendor_launch(
    prepared_original_studies, monkeypatch,
):
    original, staged = prepared_original_studies
    # Pretend a historical arm began execution, while the matching
    # full_H arm is untouched. Tool launch must NOT occur.
    pkg = staged / "format_16_06" / "row_H" / "vitis_source"
    (pkg / "logs").mkdir()
    monkeypatch.setattr(sut, "run_vitis", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("Vitis launched")))
    with pytest.raises(FileExistsError, match="already attempted"):
        sut.execute(
            study=staged, original_study=original,
            execute_native=True, confirm_isolated_native=True,
            vitis_executable=sys.executable)
    # A different selected numerical format remains independently
    # eligible for a vendor-free dry run.
    dry = sut.execute(
        study=staged, original_study=original,
        start_format=2, max_formats=1)
    assert dry["selected_formats"] == [[32, 8]]


def test_tampered_source_is_rejected_before_attempt(
    prepared_original_studies, monkeypatch,
):
    original, staged = prepared_original_studies
    model = staged / "format_32_06" / "full_H" / "vitis_source" / "model.cpp"
    model.write_bytes(model.read_bytes() + b"\n// AFTER_SOURCE_FREEZE\n")
    monkeypatch.setattr(sut, "run_vitis", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("Vitis launched")))
    with pytest.raises(ValueError, match="inventory mismatch"):
        sut.execute(
            study=staged, original_study=original, start_format=1,
            execute_native=True, confirm_isolated_native=True,
            vitis_executable=sys.executable)


def test_tampered_original_r4a_baseline_rejected(
    prepared_original_studies,
):
    original, staged = prepared_original_studies
    model = original / "case_003" / "full_H" / "vitis_source" / "model.cpp"
    model.write_bytes(model.read_bytes() + b"\n// TAMPER\n")
    with pytest.raises(ValueError, match="SHA"):
        sut.audit_prepared(study=staged, original_study=original)


def test_separate_original_studies_required(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        sut.audit_prepared(
            study=tmp_path / "missing", original_study=tmp_path / "missing")
