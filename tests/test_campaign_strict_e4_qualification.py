"""Strict read-only campaign E4 classification regression.

Synthetic source-attested report objects are not real Vivado results. Tests
exercise only campaign gating; no experiments or vendor tools are executed.
"""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments import reproduction_audit as sut


def _complete_campaign(tmp_path, monkeypatch):
    root = tmp_path / "synthetic_campaign"
    root.mkdir()
    rows = [{"case": f"case_{i:04d}", "status": "READY",
             "dimension": axis, "cpp_sha256": "a" * 64}
            for i, axis in enumerate(("baseline", *sut.AXES))]
    (root / "campaign.json").write_text(json.dumps({
        "schema": "fpgai-research/paper-campaign/v1",
        "cases": rows,
    }))
    reports = {}
    for row in rows:
        reports[row["case"]] = {
            "source_sha256": row["cpp_sha256"],
            "evidence": "E4-Vivado-implementation",
            "csim_passed": True,
            "vivado_implementation_mode": "out_of_context",
            "vivado_metrics": {
                "routed_nets_complete": True,
                "timing": {"wns_ns": 0.25, "timing_met": True},
                "drc": {
                    "available": True,
                    "summary_verified": True,
                    "violations": 0,
                    "critical_warnings": 0,
                    "unconstrained_io": False,
                    "clean": True,
                },
                "utilization": {
                    "lut": 100, "ff": 200, "dsp": 2,
                    "bram": 1, "uram": 0,
                },
            },
            "warnings": [],
        }
    monkeypatch.setattr(
        sut, "summarize_case",
        lambda path: reports[path.rsplit("/", 1)[-1]],
    )
    return root, reports


def test_complete_strict_e4_campaign_requires_all_axes_and_clean_drc(
        tmp_path, monkeypatch):
    root, _reports = _complete_campaign(tmp_path, monkeypatch)
    result = sut.audit_campaign(root, "E4")
    assert result["ready_cases"] == 8
    assert result["strict_e4_qualified_cases"] == 8
    assert result["meeting_required_evidence"] == 8
    assert result["campaign_complete_for_required_evidence"] is True
    assert result["paper_physical_results_complete"] is True
    assert all(row["strict_e4_qualified"] for row in result["cases"])


@pytest.mark.parametrize("mutation", [
    "drc_noncritical_violation",
    "drc_missing_summary",
    "drc_not_clean",
    "unrouted",
    "no_fixed_point_csim",
    "timing_failure",
    "wrong_mode",
    "missing_resource",
    "only_e3",
])
def test_attested_e4_report_is_not_automatically_strict_qualified(
        tmp_path, monkeypatch, mutation):
    root, reports = _complete_campaign(tmp_path, monkeypatch)
    record = reports["case_0000"]
    physical = record["vivado_metrics"]
    drc = physical["drc"]
    if mutation == "drc_noncritical_violation":
        drc["violations"] = 2
        drc["clean"] = False
        # The old campaign logic and critical-only qualifiers would pass
        # this warning-only E4 report (WNS, CSim, zero critical DRC).
        assert drc["critical_warnings"] == 0
        assert physical["timing"]["timing_met"] is True
    elif mutation == "drc_missing_summary":
        drc["summary_verified"] = False
    elif mutation == "drc_not_clean":
        drc["clean"] = False
    elif mutation == "unrouted":
        physical["routed_nets_complete"] = False
    elif mutation == "no_fixed_point_csim":
        record["csim_passed"] = False
    elif mutation == "timing_failure":
        physical["timing"]["wns_ns"] = -0.5
        physical["timing"]["timing_met"] = False
    elif mutation == "wrong_mode":
        record["vivado_implementation_mode"] = "standalone_pin_placed"
    elif mutation == "missing_resource":
        del physical["utilization"]["bram"]
    elif mutation == "only_e3":
        record["evidence"] = "E3-HLS-synthesis-report"

    report = sut.audit_campaign(root, "E4")
    assert report["strict_e4_qualified_cases"] == 7
    assert report["meeting_required_evidence"] == 7
    assert report["campaign_complete_for_required_evidence"] is False
    assert report["paper_physical_results_complete"] is False
    assert report["cases"][0]["strict_e4_qualified"] is False
    assert report["cases"][0]["e4_qualification_blockers"]


def test_lower_evidence_request_never_claims_complete_physical_results(
        tmp_path, monkeypatch):
    root, reports = _complete_campaign(tmp_path, monkeypatch)
    reports["case_0000"]["vivado_metrics"]["drc"]["violations"] = 1
    reports["case_0000"]["vivado_metrics"]["drc"]["clean"] = False
    e1 = sut.audit_campaign(root, "E1")
    assert e1["campaign_complete_for_required_evidence"] is True
    assert e1["meeting_required_evidence"] == 8
    assert e1["paper_physical_results_complete"] is False
    assert e1["strict_e4_qualified_cases"] == 7


def test_original_campaign_source_identity_cannot_be_skipped(
        tmp_path, monkeypatch):
    root, reports = _complete_campaign(tmp_path, monkeypatch)
    reports["case_0000"]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source hash mismatch"):
        sut.audit_campaign(root, "E4")


def test_canonical_shared_e4_qualifier_rejects_noncritical_drc_violations(
        tmp_path, monkeypatch):
    from fpgai_research.experiments.suite_native_batch import _qualify
    _root, reports = _complete_campaign(tmp_path, monkeypatch)
    row = reports["case_0000"]
    assert _qualify(row) == []
    drc = row["vivado_metrics"]["drc"]
    drc["violations"] = 1
    drc["clean"] = False
    assert drc["critical_warnings"] == 0
    assert any("Clean DRC" in failure for failure in _qualify(row))


def test_canonical_e4_qualifier_does_not_infer_drc_clean_from_zero_count(
        tmp_path, monkeypatch):
    from fpgai_research.experiments.suite_native_batch import _qualify
    _root, reports = _complete_campaign(tmp_path, monkeypatch)
    row = reports["case_0000"]
    del row["vivado_metrics"]["drc"]["clean"]
    assert any("Clean DRC" in failure for failure in _qualify(row))
