"""R4m synthetic no-vendor tests of original source-controlled HLS ledger.

The examples use the author's observed R4e/R4l HLS values, but the tests
do NOT create new native evidence or assert independent physical FPGA
observations. Read-only audit integration is tested with mocked studies.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4m_three_point_source_contrast as sut

ROW_SHA = "aa0a7a2f684f3c0e6e64f6b4e3084c8f0dc93ae152318bd5339d9646b4894e9e"
ROW_REPORT_SHA = "fe929aaa9db08990e5f7556d750d7d1eb7ef081c94b855424cccea38eb2644eb"


def _arm(name, *, resource, cycles, clock, row=False, original=False):
    result = {
        "arm": name,
        "format": [32, 6],
        "E2_qualified": True,
        "E3_qualified": True,
        "HLS_estimated_resources": {
            "bram_18k": 13, "uram": 0,
            "dsp": resource[0], "lut": resource[1], "ff": resource[2],
        },
        "HLS_worst_cycles": cycles,
        "HLS_estimated_clock_ns": clock,
        "HLS_CSynth_report_sha256": (
            ROW_REPORT_SHA if row else "a" * 64),
    }
    if original:
        result["original_source_sha256"] = (
            ROW_SHA if row else "b" * 64)
    else:
        result["source_sha256"] = (
            ROW_SHA if row else "c" * 64)
    return result


def quartet():
    a = _arm("full_H", resource=(23, 4990, 1798),
             cycles=22192, clock=10.302, original=True)
    d = _arm("row_H", resource=(19, 4755, 1647),
             cycles=22602, clock=9.5, original=True, row=True)
    b = _arm("full_storage", resource=(21, 5095, 2120),
             cycles=22954, clock=9.857)
    c = _arm("row_storage", resource=(19, 4755, 1647),
             cycles=22602, clock=9.5, row=True)
    return a, d, b, c


def calc(four=None):
    a, d, b, c = four or quartet()
    return sut.compare_three(
        original_full=a, original_row=d, new_full=b, new_row=c,
        original_source_revision="b" * 40,
        new_source_revision="c" * 40)


def test_original_qualified_three_point_path_closes_with_bram_tie():
    r = calc()
    assert r["status"] == "SOURCE_PINNED_THREE_POINT_HLS_DESCRIPTIVE_LEDGER"
    assert r["E2_E3_qualified_all_four_observations"] is True
    assert r["exact_original_row_reproduction_at_E3"] is True
    assert r["common_row_source_sha256"] == ROW_SHA
    assert r["common_row_HLS_CSynth_xml_sha256"] == ROW_REPORT_SHA
    assert r["additive_path_identity_verified"] is True
    contrast = r["descriptive_source_contrasts"]
    schedule = contrast["schedule_change_with_full_H_storage_B_minus_A"]
    storage = contrast["storage_change_with_row_source_schedule_C_minus_B"]
    combined = contrast["combined_original_change_D_minus_A"]
    assert schedule == {
        "bram_18k": 0, "dsp": -2, "lut": 105, "ff": 322, "uram": 0,
        "worst_cycles": 762, "estimated_clock_period_ns": -0.445,
    }
    assert storage == {
        "bram_18k": 0, "dsp": -2, "lut": -340, "ff": -473, "uram": 0,
        "worst_cycles": -352, "estimated_clock_period_ns": -0.357,
    }
    assert combined == {
        "bram_18k": 0, "dsp": -4, "lut": -235, "ff": -151, "uram": 0,
        "worst_cycles": 410, "estimated_clock_period_ns": -0.802,
    }
    assert r["physical_BRAM_delta"] is None
    assert r["placed_routed_clock_delta"] is None
    assert r["vendor_calls_during_analysis"] == 0


def test_missing_or_failed_e2_retains_unknown_not_zero_improvement():
    four = list(quartet())
    four[2]["E2_qualified"] = False
    r = calc(four)
    assert r["E2_E3_qualified_all_four_observations"] is False
    assert r["exact_original_row_reproduction_at_E3"] is False
    assert r["descriptive_source_contrasts"][
        "storage_change_with_row_source_schedule_C_minus_B"] is None
    assert r["points"]["new_full_row_schedule"] is None
    assert r["physical_BRAM_delta"] is None


def test_row_source_hash_mismatch_is_fatal():
    four = list(quartet())
    four[3]["source_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="exactly equal"):
        calc(four)


def test_row_report_mismatch_does_not_impute_additive_path():
    four = list(quartet())
    four[3]["HLS_CSynth_report_sha256"] = "f" * 64
    r = calc(four)
    assert r["exact_original_row_reproduction_at_E3"] is False
    assert r["status"] == "UNKNOWN_UNQUALIFIED_OR_NONREPRODUCED_HLS_LEDGER"
    assert r["descriptive_source_contrasts"][
        "combined_original_change_D_minus_A"] is None


def test_even_matching_source_but_changed_qualified_qor_does_not_close_path():
    four = list(quartet())
    four[3]["HLS_estimated_resources"]["lut"] += 17
    r = calc(four)
    assert r["exact_original_row_reproduction_at_E3"] is False
    assert r["additive_path_identity_verified"] is None


def test_unqualified_or_invalid_numeric_report_has_no_deltas():
    four = list(quartet())
    four[2]["HLS_worst_cycles"] = None
    r = calc(four)
    assert r["status"].startswith("UNKNOWN")
    four = list(quartet())
    four[2]["HLS_estimated_resources"]["bram_18k"] = -1
    assert calc(four)["points"]["new_full_row_schedule"] is None


def _mock_native(monkeypatch):
    orig_full, orig_row, new_full, new_row = quartet()
    old = {
        "source_attested_arms": 6,
        "prespecified_matched_pairs": 3,
        "source_revision": "b" * 40,
        "original_R4a_source_revision": "a" * 40,
        "E3_qualified_pairs": 1,
        "observations": [
            {"format": [16, 6], "arm": "full_H"},
            {"format": [16, 6], "arm": "row_H"},
            orig_full, orig_row,
            {"format": [32, 8], "arm": "full_H"},
            {"format": [32, 8], "arm": "row_H"},
        ],
    }
    new = {
        "E1_matched_pair_accepted": True,
        "paired_HLS_source_arms": 2,
        "original_R4d_source_revision": "b" * 40,
        "original_R4a_source_revision": "a" * 40,
        "source_revision": "c" * 40,
        "both_E3_qualified": True,
        "package_hashes": {"row_storage": {"model.cpp": ROW_SHA}},
        "observations": [new_full, new_row],
    }
    monkeypatch.setattr(sut.r4e, "audit_native", lambda **kwargs: old)
    monkeypatch.setattr(sut.r4l, "audit_native", lambda **kwargs: new)
    return old, new


def test_read_only_audit_full_three_format_denominator(monkeypatch, tmp_path):
    _mock_native(monkeypatch)
    result = sut.inspect(
        r4k_study=tmp_path/"new", original_r4d=tmp_path/"old",
        original_r4a=tmp_path/"frozen")
    assert result["population"] == {
        "original_R4d_predeclared_numeric_format_pairs": 3,
        "original_R4d_E3_qualified_pairs": 1,
        "R4l_matched_32_6_E3_qualified_pair": True,
    }
    assert result["vendor_calls_during_analysis"] == 0
    assert result["status"] == "SOURCE_PINNED_THREE_POINT_HLS_DESCRIPTIVE_LEDGER"


def test_source_lineage_and_r4l_missing_arms_fail_closed(monkeypatch, tmp_path):
    old, new = _mock_native(monkeypatch)
    new["original_R4d_source_revision"] = "bad"
    with pytest.raises(ValueError, match="populations"):
        sut.inspect(r4k_study=tmp_path/"new",
                    original_r4d=tmp_path/"old",
                    original_r4a=tmp_path/"oldest")
    new["original_R4d_source_revision"] = "b" * 40
    new["observations"] = new["observations"][:1]
    with pytest.raises(ValueError, match="incomplete"):
        sut.inspect(r4k_study=tmp_path/"new",
                    original_r4d=tmp_path/"old",
                    original_r4a=tmp_path/"oldest")


def test_create_only_sha_inventory_no_original_files_written(
    monkeypatch, tmp_path,
):
    _mock_native(monkeypatch)
    roots = [tmp_path / name for name in ("new", "old", "frozen")]
    for root in roots:
        root.mkdir()
        (root / "original.txt").write_text(root.name)
    before = {
        str(root): sha256((root/"original.txt").read_bytes()).hexdigest()
        for root in roots
    }
    out = tmp_path/"new_R4m"
    report = sut.run(
        r4k_study=roots[0], original_r4d=roots[1], original_r4a=roots[2],
        output=out)
    assert report["additive_path_identity_verified"] is True
    contents = (out/"report.json").read_bytes()
    manifest = json.loads((out/"artifact_sha256.json").read_text())
    assert manifest["report.json"] == sha256(contents).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(r4k_study=roots[0], original_r4d=roots[1],
                original_r4a=roots[2], output=out)
    with pytest.raises(ValueError, match="outside original"):
        sut.run(r4k_study=roots[0], original_r4d=roots[1],
                original_r4a=roots[2], output=roots[0]/"forbidden")
    assert before == {
        str(root): sha256((root/"original.txt").read_bytes()).hexdigest()
        for root in roots
    }
