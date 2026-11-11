"""New row-materialization native gate: source hash checks and safe dry runs.

Tests never launch Vitis or Vivado; they inspect fresh original GCC
outputs from a temp directory and deliberately tamper source receipts.
"""
from __future__ import annotations

import json
import shutil

import pytest

from fpgai_research.experiments import cst_r4a_row_retention_study as r4a
from fpgai_research.experiments import cst_r4b_row_retention_native as sut


@pytest.mark.skipif(shutil.which("g++") is None, reason="Host g++ missing")
def test_read_only_original_source_audit_and_zero_call_native_dry_run(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(r4a, "_git_revision", lambda: "d"*40)
    root = tmp_path / "r4a_original"
    r4a.run(output=root, profile="quick")
    evidence = sut.audit_prepared(study=root)
    assert evidence["original_complete_E1_pairs"] == 3
    assert evidence["original_HLS_source_packages"] == 6
    assert evidence["source_and_testbench_attested"] is True
    assert evidence["original_vendor_calls_during_audit"] == 0
    dry = sut.execute(study=root, start_case=0, max_cases=2)
    assert dry["status"] == "DRY_RUN_NO_VENDOR_CALLS"
    assert dry["planned_vendor_calls"] == 4
    assert dry["new_vendor_calls"] == 0
    summary = sut.audit_native(study=root)
    assert summary["planned_HLS_pairs"] == 3
    assert summary["qualified_HLS_pairs"] == 0
    assert summary["unqualified_HLS_pairs"] == 3
    assert summary["physical_E4"] is None
    assert all(
        p["HLS_estimated_resource_delta_intervention_minus_control"][
            "bram_18k"] is None for p in summary["pairs"]
    )
    with pytest.raises(ValueError, match="confirmation"):
        sut.execute(study=root, start_case=0, max_cases=1,
                    execute_native=True, confirm_isolated_native=False)
    with pytest.raises(ValueError, match="range"):
        sut.execute(study=root, start_case=3, max_cases=1)
    with pytest.raises(ValueError, match="range"):
        sut.execute(study=root, start_case=0, max_cases=0)
    source = root / "case_000" / "row_H" / "vitis_source" / "model.cpp"
    source.write_bytes(source.read_bytes() + b"\n// POSTRUN_TAMPER\n")
    with pytest.raises(ValueError, match="SHA"):
        sut.audit_prepared(study=root)


def test_unknown_and_invalid_root_refused(tmp_path):
    with pytest.raises(ValueError):
        sut.audit_prepared(study=tmp_path / "missing")
    with pytest.raises(ValueError):
        sut.execute(study=tmp_path / "missing")


def test_native_schema_and_responses_are_separate_from_physical_claims():
    assert sut.SCHEMA.endswith("/v1")
    assert sut.RESOURCES == ("bram_18k", "lut", "ff", "dsp", "uram")
    assert sut.ARMS == ("full_H", "row_H")
