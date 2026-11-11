"""R4c numerical triage is read-only and cannot promote failed E2 evidence.

Synthetic receipt tests validate interpretation ONLY. The integration test
generates actual host GCC R4a E1 evidence, NEVER invokes Vitis/Vivado.
"""
from __future__ import annotations

from hashlib import sha256
import json
import shutil

import pytest

from fpgai_research.experiments import cst_r4a_row_retention_study as r4a
from fpgai_research.experiments import cst_r4c_csim_numerical_triage as sut


def receipt(*, failures=28, maximum=0.055395125,
            rmse=0.0183012798956, cosine=0.999989333846, passed=False):
    return {
        "passed": passed,
        "marker": "FPGAI_CSIM_STATUS=failed" if not passed else "CSim completed",
        "source": "logs/vitis.stdout.log",
        "samples": 1,
        "outputs_per_sample": 128,
        "cosine": cosine,
        "rmse": rmse,
        "max_absolute_error": maximum,
        "failures": failures,
    }


def test_original_observed_metrics_mean_identical_aggregates_not_proof():
    r = sut.paired_numeric_diagnostic(
        receipt(), receipt(), tolerance=0.02, samples=1, outputs=128)
    assert r["status"] == "IDENTICAL_AGGREGATES_NOT_POINTWISE_IDENTICAL"
    assert r["identical_reported_aggregate_metrics"] is True
    assert r["both_E2_qualified"] is False
    assert r["arms"]["full_H"]["failed_outputs"] == 28
    assert r["arms"]["row_H"]["failure_fraction"] == 28 / 128
    assert r["arms"]["full_H"]["max_error_to_tolerance_ratio"] == pytest.approx(
        0.055395125 / 0.02)
    assert r["E3_resource_delta"] is None
    assert r["physical_E4_delta"] is None
    assert "cannot establish" in r["inference_limit"]


def test_different_aggregates_are_not_automatically_a_row_regression():
    r = sut.paired_numeric_diagnostic(
        receipt(), receipt(failures=27),
        tolerance=0.02, samples=1, outputs=128)
    assert r["status"] == "DIFFERENT_AGGREGATES_CAUSE_UNKNOWN"
    assert r["identical_reported_aggregate_metrics"] is False
    assert r["both_E2_qualified"] is False


def test_missing_unattested_or_wrong_coverage_is_unknown():
    for suspect in (None, {**receipt(), "samples": 2},
                    {**receipt(), "rmse": float("nan")},
                    {**receipt(), "failures": 129}):
        r = sut.paired_numeric_diagnostic(
            receipt(), suspect, tolerance=0.02, samples=1, outputs=128)
        assert r["status"] == "UNKNOWN_MISSING_OR_UNATTESTED_PAIR"
        assert r["both_E2_qualified"] is False
        assert r["identical_reported_aggregate_metrics"] is None
        assert r["arms"]["row_H"]["failed_outputs"] is None


def test_valid_positive_controls_do_not_count_as_failed_e2():
    r = sut.paired_numeric_diagnostic(
        receipt(failures=0, maximum=0.001, rmse=0.0004, passed=True),
        receipt(failures=0, maximum=0.001, rmse=0.0004, passed=True),
        tolerance=0.02, samples=1, outputs=128)
    assert r["both_E2_qualified"] is True
    for tolerance in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="contract"):
            sut.paired_numeric_diagnostic(
                receipt(), receipt(), tolerance=tolerance,
                samples=1, outputs=128)


@pytest.mark.skipif(shutil.which("g++") is None, reason="Host g++ missing")
def test_original_e1_integration_is_read_only_and_create_only(tmp_path, monkeypatch):
    monkeypatch.setattr(r4a, "_git_revision", lambda: "e" * 40)
    original = tmp_path / "original_r4a"
    r4a.run(output=original, profile="quick")
    original_hashes = {
        str(p.relative_to(original)): sha256(p.read_bytes()).hexdigest()
        for p in original.rglob("*") if p.is_file()
    }
    report = sut.run(study=original, case_index=0)
    assert report["original_source_revision"] == "e" * 40
    assert report["original_population_pairs"] == 3
    assert report["paired_numeric_diagnostic"]["status"] == (
        "UNKNOWN_MISSING_OR_UNATTESTED_PAIR")
    assert report["native_vendor_calls_during_triage"] == 0
    assert report["HLS_E3_qualified_from_this_diagnostic"] is False
    assert report["physical_E4_qualified_from_this_diagnostic"] is False
    assert set(report["original_csim_receipts"]) == {"full_H", "row_H"}
    dest = tmp_path / "r4c_new_diagnostic"
    sut.run(study=original, case_index=0, output=dest)
    assert json.loads((dest / "diagnostic.json").read_text())["schema"] == sut.SCHEMA
    hashes = json.loads((dest / "artifact_sha256.json").read_text())
    assert hashes["diagnostic.json"] == sha256(
        (dest / "diagnostic.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(study=original, case_index=0, output=dest)
    with pytest.raises(ValueError, match="must not write"):
        sut.run(study=original, case_index=0,
                output=original / "forbidden_diagnostic")
    with pytest.raises(ValueError, match="outside original"):
        sut.inspect(study=original, case_index=3)
    assert original_hashes == {
        str(p.relative_to(original)): sha256(p.read_bytes()).hexdigest()
        for p in original.rglob("*") if p.is_file()
    }
