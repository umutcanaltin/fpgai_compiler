"""R4f report: pure-source provenance comparisons, no Vitis/Vivado launches.

All native observations here are synthetic test fixtures. Original E2/E3
measurement claims must ONLY derive from real R4e and R4c audits.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4f_paired_hls_review as sut


def observation(*, arm, format=(32, 6), e2=True, e3=True,
                cycles=22192, clock=10.302,
                bram=13, dsp=23, ff=1798, lut=4990):
    return {
        "format": list(format), "arm": arm,
        "status": "QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL"
                  if e3 else "UNKNOWN_NO_ORIGINAL_VITIS",
        "E2_qualified": e2, "E3_qualified": e3,
        "CSim_numerical_receipt_attested": e2,
        "CSim_failed_outputs": 0 if e2 else None,
        "CSim_max_absolute_error": 3.7956994241e-6 if e2 else None,
        "CSim_rmse": 8.1167162467e-7 if e2 else None,
        "HLS_worst_cycles": cycles if e3 else None,
        "HLS_estimated_clock_ns": clock if e3 else None,
        "HLS_estimated_resources": {
            "bram_18k": bram if e3 else None,
            "dsp": dsp if e3 else None,
            "ff": ff if e3 else None,
            "lut": lut if e3 else None,
            "uram": 0 if e3 else None,
        },
    }


def test_observed_32_6_numbers_are_only_hls_estimates():
    # Exact numbers reproduce the user's observations as SYNTHETIC
    # input to the pure comparison test, not new vendor measurements.
    control = observation(arm="full_H")
    intervention = observation(
        arm="row_H", cycles=22602, clock=9.5,
        bram=13, dsp=19, ff=1647, lut=4755)
    r = sut.compare_pair(
        control, intervention, bits=32, integer=6,
        requested_ns=12.0, uncertainty_ns=1.5)
    assert r["status"] == "QUALIFIED_E3_HLS_ESTIMATES_ONLY"
    assert r["both_E2_qualified"] and r["both_E3_qualified"]
    assert r["estimated_resource_delta_row_minus_full"] == {
        "bram_18k": 0, "lut": -235, "ff": -151, "dsp": -4, "uram": 0}
    assert r["HLS_worst_cycles_delta_row_minus_full"] == 410
    assert r["HLS_requested_period_latency_ns_control"] == 266304
    assert r["HLS_requested_period_latency_ns_intervention"] == 271224
    assert r["HLS_requested_period_latency_ns_delta_row_minus_full"] == 4920
    assert r["HLS_estimated_clock_ns_delta_row_minus_full"] == pytest.approx(-0.802)
    assert r["HLS_effective_budget_margin_ns_control"] == pytest.approx(0.198)
    assert r["HLS_effective_budget_margin_ns_intervention"] == pytest.approx(1.0)
    assert r["physical_E4"] is None
    assert r["physical_resource_delta"] is None


def test_missing_native_evidence_is_unknown_not_zero():
    full = observation(arm="full_H", format=(32, 8), e2=False, e3=False)
    row = observation(arm="row_H", format=(32, 8), e2=False, e3=False)
    r = sut.compare_pair(
        full, row, bits=32, integer=8, requested_ns=12, uncertainty_ns=1.5)
    assert r["status"] == "UNKNOWN_OR_REJECTED_E3_PAIR"
    assert r["both_E3_qualified"] is False
    assert r["estimated_resource_delta_row_minus_full"]["bram_18k"] is None
    assert r["HLS_worst_cycles_delta_row_minus_full"] is None
    assert r["HLS_requested_period_latency_ns_delta_row_minus_full"] is None
    assert r["numerical"]["full_H"]["failed_outputs"] is None


def test_csim_rejection_cannot_be_promoted_to_E3():
    full = observation(arm="full_H", e2=False, e3=True)
    row = observation(arm="row_H", e2=True, e3=True)
    r = sut.compare_pair(full, row, bits=32, integer=6,
                         requested_ns=12, uncertainty_ns=1.5)
    assert r["both_E2_qualified"] is False
    assert r["both_E3_qualified"] is False
    assert r["estimated_resource_delta_row_minus_full"]["dsp"] is None


def test_invalid_pairing_and_timing_are_rejected():
    full = observation(arm="full_H")
    row = observation(arm="row_H", cycles=22602, clock=9.5)
    with pytest.raises(ValueError, match="paired"):
        sut.compare_pair(full, row, bits=16, integer=6,
                         requested_ns=12, uncertainty_ns=1.5)
    with pytest.raises(ValueError, match="paired"):
        sut.compare_pair(full, row, bits=32, integer=6,
                         requested_ns=12, uncertainty_ns=12)
    with pytest.raises(ValueError, match="budget"):
        sut.compare_pair(observation(arm="full_H", clock=10.51),
                         row, bits=32, integer=6,
                         requested_ns=12, uncertainty_ns=1.5)
    with pytest.raises(ValueError, match="resources"):
        sut.compare_pair(observation(arm="full_H", bram=None),
                         row, bits=32, integer=6,
                         requested_ns=12, uncertainty_ns=1.5)


def test_read_only_independent_audits_and_create_only_output(
    monkeypatch, tmp_path,
):
    original = tmp_path / "original_r4a"
    new = tmp_path / "original_r4d"
    original.mkdir()
    new.mkdir()
    native = {
        "original_R4a_source_revision": "a" * 40,
        "source_revision": "b" * 40,
        "original_native_invocation_files": 2,
        "observations": [
            observation(arm="full_H", format=(16, 6), e2=False, e3=False),
            observation(arm="row_H", format=(16, 6), e2=False, e3=False),
            observation(arm="full_H"),
            observation(arm="row_H", cycles=22602, clock=9.5,
                        bram=13, dsp=19, ff=1647, lut=4755),
            observation(arm="full_H", format=(32, 8), e2=False, e3=False),
            observation(arm="row_H", format=(32, 8), e2=False, e3=False),
        ],
    }
    old = {
        "original_source_revision": "a" * 40,
        "inspected_case": "case_003",
        "paired_numeric_diagnostic": {
            "identical_reported_aggregate_metrics": True,
            "arms": {
                arm: {
                    "E2_passed": False,
                    "failed_outputs": 28,
                    "max_absolute_error": 0.055395125,
                } for arm in ("full_H", "row_H")
            },
        },
    }
    calls = []
    monkeypatch.setattr(sut.r4e, "audit_native",
                        lambda **kwargs: calls.append("r4e") or native)
    monkeypatch.setattr(sut.r4c, "inspect",
                        lambda **kwargs: calls.append("r4c") or old)
    r = sut.run(study=new, original_study=original)
    assert r["qualified_E3_format_pairs"] == 1
    assert r["prespecified_format_pairs"] == 3
    assert r["original_R4e_invocation_files"] == 2
    assert r["original_R4a_case_003"]["CSim_failed_outputs_full_H"] == 28
    assert r["new_native_Vitis_calls_during_report"] == 0
    assert calls == ["r4e", "r4c"]
    path = tmp_path / "new_r4f_report"
    r = sut.run(study=new, original_study=original, output=path)
    data = (path / "report.json").read_bytes()
    inventory = json.loads((path / "artifact_sha256.json").read_text())
    assert inventory["report.json"] == sha256(data).hexdigest()
    assert json.loads(data)["qualified_E3_format_pairs"] == 1
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(study=new, original_study=original, output=path)
    with pytest.raises(ValueError, match="outside both"):
        sut.run(study=new, original_study=original,
                output=new / "not_allowed")
    with pytest.raises(ValueError, match="outside both"):
        sut.run(study=new, original_study=original,
                output=original / "not_allowed")


def test_source_revision_disagreement_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sut.r4e, "audit_native",
        lambda **_: {"original_R4a_source_revision": "x" * 40})
    monkeypatch.setattr(
        sut.r4c, "inspect",
        lambda **_: {"original_source_revision": "y" * 40})
    with pytest.raises(ValueError, match="revision"):
        sut.inspect(study=tmp_path / "new", original_study=tmp_path / "old")
