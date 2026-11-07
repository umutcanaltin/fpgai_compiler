"""R4r tests use mocked R4q report values; NO Vivado, NO new physical data.

The numerical input matches the author's observed R4q stdout, but every
fixture is SYNTHETIC and does not validate the native vendor execution.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4r_original_routed_tradeoff_ledger as sut


def _arm(name, *, bram, dsp, lut, ff, wns, drc_dpop4):
    rules = {
        "DPIP-2": {"severity": "Warning", "violations": 29},
        "DPOP-3": {"severity": "Warning", "violations": 20},
        "DPOP-4": {"severity": "Warning", "violations": drc_dpop4},
        "RTSTAT-10": {"severity": "Warning", "violations": 1},
    }
    total = sum(rule["violations"] for rule in rules.values())
    return {
        "arm": name,
        "original_R4p_verdict": "UNKNOWN_INCOMPLETE_PHYSICAL_RESOURCE_OR_TIMING",
        "original_strict_clean_DRC_E4": False,
        "original_source_and_vendor_receipt_attested": True,
        "original_report_sha_verified": True,
        "source_attested_route_complete": True,
        "source_attested_routed_wns_ns": wns,
        "source_attested_timing_met": False,
        "source_attested_utilization": {
            "bram": bram, "dsp": dsp, "lut": lut, "ff": ff, "uram": 0,
        },
        "source_attested_drc_summary": {
            "available": True,
            "summary_verified": True,
            "violations": total,
            "critical_warnings": 0,
            "unconstrained_io": False,
            "board_deployment_qualified": False,
            "clean": False,
            "rules": rules,
        },
        "blocked_conditions": ["NEGATIVE_ROUTED_WNS", "DRC_NONCLEAN"],
        "unrecognized_or_missing_required_resources": [],
        "fresh_report_sha256": {k: "a"*64 for k in sut.r4q.REPORTS},
        "physical_E4": None,
        "board_qualified": False,
    }


@pytest.fixture
def signed_original_r4q():
    return {
        "schema": sut.r4q.SCHEMA,
        "status": "SIGNED_ORIGINAL_R4P_POSTRUN_DIAGNOSTICS_NOT_E4",
        "source_revision": sut.SOURCE_REVISION,
        "original_R4p_pair_verdict": "UNKNOWN_INCOMPLETE_OR_FAILED_PHYSICAL_PAIR",
        "original_physical_E4": None,
        "both_original_native_vendor_invocations_attested": True,
        "report_files_modified": False,
        "new_Vivado_calls": 0,
        "new_Vitis_calls": 0,
        "board_qualified": False,
        "physical_E4": None,
        "diagnostic_arms": [
            _arm("full_storage", bram=7, dsp=24, lut=1282, ff=883,
                 wns=-2.17, drc_dpop4=24),
            _arm("row_storage", bram=4, dsp=21, lut=1710, ff=1022,
                 wns=-3.718, drc_dpop4=21),
        ],
    }


def test_signed_original_routed_tradeoff_precise_and_does_not_promote_e4(
    signed_original_r4q,
):
    report = sut.summarize(signed_original_r4q)
    assert report["status"] == (
        "SIGNED_ORIGINAL_ROUTED_OOC_TRADEOFF_TIMING_AND_DRC_NOT_QUALIFIED")
    assert report["source_attested_original_routed_pair"] is True
    assert report["physical_resource_delta_row_minus_full"] == {
        "bram": -3, "dsp": -3, "lut": 428, "ff": 139, "uram": 0}
    assert report["physical_resource_percent_delta_row_minus_full"] == {
        "bram": -42.857, "dsp": -12.5, "lut": 33.385,
        "ff": 15.742, "uram": None}
    assert report["routed_WNS_delta_row_minus_full_ns"] == -1.548
    assert report["drc_total_delta_row_minus_full"] == -3
    assert report["drc_rule_deltas_row_minus_full"] == {
        "DPIP-2": 0, "DPOP-3": 0, "DPOP-4": -3, "RTSTAT-10": 0}
    assert report["original_arms"][0]["drc_total_violations"] == 74
    assert report["original_arms"][1]["drc_total_violations"] == 71
    assert report["physical_E4"] is None
    assert report["strict_clean_DRC_E4"] is False
    assert report["board_qualified"] is False
    assert report["native_vendor_calls_during_analysis"] == 0


def test_missing_or_invalid_original_resource_never_becomes_zero_delta(
    signed_original_r4q,
):
    report = deepcopy(signed_original_r4q)
    report["diagnostic_arms"][1]["source_attested_utilization"]["bram"] = None
    result = sut.summarize(report)
    assert result["status"].startswith("UNKNOWN")
    assert result["physical_resource_delta_row_minus_full"] is None
    assert result["physical_E4"] is None
    report["diagnostic_arms"][1]["source_attested_utilization"]["bram"] = 4
    report["diagnostic_arms"][1]["unrecognized_or_missing_required_resources"] = ["bram"]
    assert sut.summarize(report)["source_attested_original_routed_pair"] is False


@pytest.mark.parametrize("mutation", [
    "wrong_source", "not_attested", "missing_arm", "source_modified",
    "wrong_sha", "missing_route", "timing_contradiction", "bad_drc_sum",
    "missing_native_receipt",
])
def test_incomplete_or_forged_original_evidence_fails_closed(
    signed_original_r4q, mutation,
):
    data = deepcopy(signed_original_r4q)
    row = data["diagnostic_arms"][1]
    if mutation == "wrong_source":
        data["source_revision"] = "f" * 40
    elif mutation == "not_attested":
        row["original_report_sha_verified"] = False
    elif mutation == "missing_arm":
        data["diagnostic_arms"].pop()
    elif mutation == "source_modified":
        data["report_files_modified"] = True
    elif mutation == "wrong_sha":
        row["fresh_report_sha256"]["drc.rpt"] = "bad"
    elif mutation == "missing_route":
        row["source_attested_route_complete"] = False
    elif mutation == "timing_contradiction":
        row["source_attested_timing_met"] = True
    elif mutation == "bad_drc_sum":
        row["source_attested_drc_summary"]["violations"] += 1
    elif mutation == "missing_native_receipt":
        data["both_original_native_vendor_invocations_attested"] = False
    if mutation in ("wrong_source", "missing_arm", "source_modified",
                    "missing_native_receipt"):
        with pytest.raises(ValueError, match="R4r"):
            sut.summarize(data)
    else:
        result = sut.summarize(data)
        assert result["source_attested_original_routed_pair"] is False
        assert result["physical_resource_delta_row_minus_full"] is None


def test_original_negative_wns_and_drc_warning_are_preserved(
    signed_original_r4q,
):
    result = sut.summarize(signed_original_r4q)
    full, row = result["original_arms"]
    assert full["blocked_conditions"] == ["DRC_NONCLEAN", "NEGATIVE_ROUTED_WNS"]
    assert row["blocked_conditions"] == ["DRC_NONCLEAN", "NEGATIVE_ROUTED_WNS"]
    assert full["routed_timing_met"] is False
    assert row["routed_timing_met"] is False
    assert full["drc_critical_warnings"] == 0
    assert row["drc_critical_warnings"] == 0


def test_create_only_separate_new_ledger_and_originals_not_touched(
    signed_original_r4q, monkeypatch, tmp_path,
):
    original = [tmp_path / k for k in ("r4p", "r4o", "r4k", "r4d", "r4a")]
    for p in original:
        p.mkdir()
        (p / "original.txt").write_text(p.name)
    before = {str(p): sha256((p / "original.txt").read_bytes()).hexdigest()
              for p in original}
    monkeypatch.setattr(sut.r4q, "inspect",
                        lambda **kwargs: signed_original_r4q)
    args = {
        "study": original[0], "r4o_study": original[1],
        "r4k_study": original[2], "original_r4d": original[3],
        "original_r4a": original[4]}
    dest = tmp_path / "new_r4r_descriptive_ledger"
    result = sut.run(**args, output=dest)
    assert result["source_attested_original_routed_pair"] is True
    body = (dest / "report.json").read_bytes()
    manifest = json.loads((dest / "artifact_sha256.json").read_text())
    assert manifest["report.json"] == sha256(body).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(**args, output=dest)
    with pytest.raises(ValueError, match="outside immutable"):
        sut.run(**args, output=original[0] / "not_permitted")
    assert before == {
        str(p): sha256((p / "original.txt").read_bytes()).hexdigest()
        for p in original}
