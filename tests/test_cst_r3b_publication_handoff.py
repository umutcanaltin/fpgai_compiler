"""R3b safety tests: no vendor tools or original study modification."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fpgai_research.experiments import cst_r3b_publication_handoff as r3b


def _source_rows():
    rows = []
    for i in range(6):
        qualified = i == 0
        rows.append({
            "case": f"case_{i:03d}",
            "network": ("cnn_pool_dense", "cnn_pool_dense",
                        "cnn_strided_dense", "cnn_strided_dense",
                        "cnn_two_conv_softmax", "cnn_two_conv_softmax")[i],
            "original_vitis_invocation_present": qualified,
            "native_CSim_E2_accepted": qualified,
            "native_CSynth_E3_qualified": qualified,
            "qualification_status": ("QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL"
                                     if qualified else "UNKNOWN_NO_ORIGINAL_VITIS"),
            "hls_worst_cycles": 100 if qualified else None,
            "hls_estimated_clock_ns": 8.0 if qualified else None,
            "HLS_estimated_resources": (
                {"bram_18k": 1, "dsp": 2, "ff": 3, "lut": 4, "uram": 0}
                if qualified else {k: None for k in r3b.r3a.METRICS}),
        })
    return rows


def _reports():
    rows = _source_rows()
    native = {
        "source_revision": "a" * 40,
        "original_source_cases": 6,
        "original_GCC_source_inputs_qualified": 12,
        "original_vitis_invocation_receipts": 1,
        "native_CSim_E2_qualified_cases": 1,
        "native_CSynth_E3_qualified_cases": 1,
        "cases": rows,
    }
    pairs = {
        "source_revision": "a" * 40,
        "prespecified_original_neural_pairs": 3,
        "E3_both_qualified_pairs": 0,
        "E3_unknown_or_rejected_pairs": 3,
        "pairs": [
            {"control": f"case_{i:03d}",
             "independent_Phi": f"case_{i+1:03d}", "both_E3": False}
            for i in (0, 2, 4)
        ],
    }
    return native, pairs


def test_original_pilot_snapshot_is_not_overclaimed(monkeypatch, tmp_path):
    native, pairs = _reports()
    monkeypatch.setattr(r3b.r3a, "audit", lambda **kw: deepcopy(native))
    monkeypatch.setattr(r3b.r3a, "contrasts", lambda **kw: deepcopy(pairs))
    report = r3b.snapshot(study=tmp_path)
    assert report["original_CSim_E2_qualified"] == 1
    assert report["original_CSynth_E3_qualified"] == 1
    assert report["matched_E3_qualified_pairs"] == 0
    assert report["physical_E4"] is None
    assert report["decision"] == "MAJOR_REVISION_NOT_SUBMISSION_READY"
    assert [x["state"] for x in report["cases"]] == (
        ["QUALIFIED_HLS_E3_ESTIMATES_ONLY"] + ["UNATTEMPTED_UNKNOWN"] * 5)
    assert r3b.next_case(report) == 1
    md = r3b.markdown_report(report)
    assert "Original Vitis CSynth E3 | 1 | 6" in md
    assert "UNKNOWN" in md
    assert "not routed measurements" in md


def test_partial_attempt_blocks_native_continuation(monkeypatch, tmp_path):
    native, pairs = _reports()
    (tmp_path / "case_001" / "logs").mkdir(parents=True)
    monkeypatch.setattr(r3b.r3a, "audit", lambda **kw: native)
    monkeypatch.setattr(r3b.r3a, "contrasts", lambda **kw: pairs)
    report = r3b.snapshot(study=tmp_path)
    assert report["cases"][1]["state"] == "BLOCKED_PARTIAL_OR_UNATTRIBUTED_ATTEMPT"
    with pytest.raises(RuntimeError, match="refusing continuation"):
        r3b.next_case(report)


def test_native_qualification_counts_are_fail_closed(monkeypatch, tmp_path):
    native, pairs = _reports()
    native["native_CSynth_E3_qualified_cases"] = 2
    monkeypatch.setattr(r3b.r3a, "audit", lambda **kw: native)
    monkeypatch.setattr(r3b.r3a, "contrasts", lambda **kw: pairs)
    with pytest.raises(ValueError, match="count inconsistent"):
        r3b.snapshot(study=tmp_path)


def test_native_e3_cannot_exist_without_original_receipt(monkeypatch, tmp_path):
    native, pairs = _reports()
    native["cases"][0]["original_vitis_invocation_present"] = False
    native["original_vitis_invocation_receipts"] = 0
    monkeypatch.setattr(r3b.r3a, "audit", lambda **kw: native)
    monkeypatch.setattr(r3b.r3a, "contrasts", lambda **kw: pairs)
    with pytest.raises(ValueError, match="E3 without"):
        r3b.snapshot(study=tmp_path)


def test_existing_report_folder_is_never_overwritten(tmp_path):
    report = {"source_revision": "x", "cases": [],
              "remaining_submission_blockers": []}
    folder = tmp_path / "frozen"
    folder.mkdir()
    with pytest.raises(FileExistsError, match="overwrit"):
        r3b._create_only_report(report, folder)
    assert not (folder / "read_only_evidence.json").exists()


def test_native_opt_in_and_first_failure_stop(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="BOTH"):
        r3b.continue_native(
            study=tmp_path, max_new_cases=5, execute_native=True,
            confirm_isolated_native=False, vitis_executable="vitis_hls",
            timeout_seconds=7200)

    old = {"cases": [
        {"case": f"case_{i:03d}",
         "state": ("QUALIFIED_HLS_E3_ESTIMATES_ONLY" if i == 0
                   else "UNATTEMPTED_UNKNOWN"),
         "E3_original_CSynth": i == 0}
        for i in range(6)]}
    new = deepcopy(old)
    new["cases"][1] = {
        "case": "case_001",
        "state": "ATTEMPTED_BUT_NOT_E3_QUALIFIED",
        "E3_original_CSynth": False}
    calls = []
    monkeypatch.setattr(r3b, "snapshot", lambda **kw: old if not calls else new)
    monkeypatch.setattr(r3b.r3a, "execute", lambda **kw: calls.append(kw))
    with pytest.raises(RuntimeError, match="STOP and inspect"):
        r3b.continue_native(
            study=tmp_path, max_new_cases=5, execute_native=True,
            confirm_isolated_native=True, vitis_executable="vitis_hls",
            timeout_seconds=7200)
    assert len(calls) == 1
    assert calls[0]["start_case"] == 1
    assert calls[0]["max_cases"] == 1
    assert calls[0]["execute_native"] is True


def test_read_only_report_create_once(monkeypatch, tmp_path):
    native, pairs = _reports()
    monkeypatch.setattr(r3b.r3a, "audit", lambda **kw: native)
    monkeypatch.setattr(r3b.r3a, "contrasts", lambda **kw: pairs)
    report = r3b.snapshot(study=tmp_path)
    folder = tmp_path / "new_report"
    r3b._create_only_report(report, folder)
    loaded = json.loads((folder / "read_only_evidence.json").read_text())
    assert loaded["physical_E4"] is None
    assert loaded["original_Vitis_invocation_receipts"] == 1
    with pytest.raises(FileExistsError):
        r3b._create_only_report(report, folder)



def _attested_clock_failure(case="case_002"):
    return {
        "case": case,
        "state": "ATTEMPTED_BUT_NOT_E3_QUALIFIED",
        "E2_original_CSim": True,
        "E3_original_CSynth": False,
        "qualification_status": "FAILED_HLS_EFFECTIVE_CLOCK_BUDGET",
        "original_vitis_invocation_present": True,
        "source_attested_evidence": "E3-HLS-synthesis-report",
        "attempt_artifacts": list(r3b.ARTIFACTS),
        "attested_HLS_timing_diagnostic": {
            "estimated_clock_ns": 9.3,
            "effective_budget_ns": 9.0,
            "margin_ns": -0.3,
            "feasible": False,
            "original_HLS_report_sha256": "a" * 64,
        },
    }


def _mixed_checkpoint():
    cases = [
        {"case": f"case_{i:03d}", "state": "UNATTEMPTED_UNKNOWN",
         "E3_original_CSynth": False} for i in range(6)]
    for i in (0, 1):
        cases[i] = {
            "case": f"case_{i:03d}",
            "state": "QUALIFIED_HLS_E3_ESTIMATES_ONLY",
            "E3_original_CSynth": True}
    cases[2] = _attested_clock_failure()
    return {"cases": cases}


def test_attested_clock_failure_is_preserved_not_promoted():
    report = _mixed_checkpoint()
    with pytest.raises(RuntimeError, match="refusing continuation"):
        r3b.next_case(report)
    assert r3b._is_attested_clock_failure(report["cases"][2])
    assert r3b.next_case(report, allow_attested_timing_failure=True) == 3
    assert report["cases"][2]["E3_original_CSynth"] is False


@pytest.mark.parametrize("field,bad", [
    ("E2_original_CSim", False),
    ("original_vitis_invocation_present", False),
    ("source_attested_evidence", "E1-HLS-source"),
    ("qualification_status", "UNKNOWN_INCOMPLETE_HLS_METRICS"),
    ("attempt_artifacts", ["vitis_invocation.json"]),
    ("attested_HLS_timing_diagnostic", None),
])
def test_unattributed_or_partial_failure_cannot_be_skipped(field, bad):
    report = _mixed_checkpoint()
    report["cases"][2][field] = bad
    assert not r3b._is_attested_clock_failure(report["cases"][2])
    with pytest.raises(RuntimeError, match="refusing continuation"):
        r3b.next_case(report, allow_attested_timing_failure=True)


@pytest.mark.parametrize("field,bad", [
    ("feasible", True),
    ("margin_ns", 0.0),
    ("estimated_clock_ns", 8.5),
    ("effective_budget_ns", None),
    ("original_HLS_report_sha256", "invalid"),
])
def test_inconsistent_clock_diagnostics_cannot_be_skipped(field, bad):
    report = _mixed_checkpoint()
    report["cases"][2]["attested_HLS_timing_diagnostic"][field] = bad
    with pytest.raises(RuntimeError, match="refusing continuation"):
        r3b.next_case(report, allow_attested_timing_failure=True)


def test_newly_attempted_clock_failure_may_be_skipped_but_never_retried(monkeypatch, tmp_path):
    original = _mixed_checkpoint()
    calls = []

    def status(**kw):
        report = deepcopy(original)
        for i in calls:
            if i == 3:
                report["cases"][i] = _attested_clock_failure("case_003")
            else:
                report["cases"][i] = {
                    "case": f"case_{i:03d}",
                    "state": "QUALIFIED_HLS_E3_ESTIMATES_ONLY",
                    "E3_original_CSynth": True}
        return report

    monkeypatch.setattr(r3b, "snapshot", status)
    monkeypatch.setattr(
        r3b.r3a, "execute",
        lambda **kw: calls.append(kw["start_case"]))
    result = r3b.continue_native(
        study=tmp_path, max_new_cases=3,
        execute_native=True, confirm_isolated_native=True,
        vitis_executable="vitis_hls", timeout_seconds=7200,
        allow_attested_timing_failure=True)
    assert calls == [3, 4, 5]
    assert result["new_original_Vitis_cases_in_this_command"] == [
        "case_003", "case_004", "case_005"]
    assert result["original_attested_timing_failures_retained"] == [
        "case_002", "case_003"]


def test_existing_failed_native_case_cannot_be_skipped_without_optin(monkeypatch, tmp_path):
    original = _mixed_checkpoint()
    monkeypatch.setattr(r3b, "snapshot", lambda **kw: original)
    monkeypatch.setattr(
        r3b.r3a, "execute",
        lambda **kw: pytest.fail("must not call native vendor"))
    with pytest.raises(RuntimeError, match="refusing continuation"):
        r3b.continue_native(
            study=tmp_path, max_new_cases=3,
            execute_native=True, confirm_isolated_native=True,
            vitis_executable="vitis_hls", timeout_seconds=7200)


def test_observed_clock_rejection_is_not_an_hls_resource_result(monkeypatch, tmp_path):
    native, pairs = _reports()
    failed = native["cases"][2]
    failed.update({
        "original_vitis_invocation_present": True,
        "native_CSim_E2_accepted": True,
        "qualification_status": "FAILED_HLS_EFFECTIVE_CLOCK_BUDGET",
        "source_attested_evidence": "E3-HLS-synthesis-report",
        "attested_HLS_timing_diagnostic":
            _attested_clock_failure()["attested_HLS_timing_diagnostic"],
    })
    (tmp_path / "case_002").mkdir()
    for name in r3b.ARTIFACTS:
        p = tmp_path / "case_002" / name
        if name == "vitis_invocation.json":
            p.write_text("{}")
        else:
            p.mkdir()
    native["original_vitis_invocation_receipts"] = 2
    native["native_CSim_E2_qualified_cases"] = 2
    monkeypatch.setattr(r3b.r3a, "audit", lambda **kw: native)
    monkeypatch.setattr(r3b.r3a, "contrasts", lambda **kw: pairs)
    report = r3b.snapshot(study=tmp_path)
    assert report["original_attested_timing_failures"] == 1
    assert report["original_native_attempted_not_E3"] == 1
    assert report["cases"][2]["HLS_estimated_resources"] is None
    assert report["cases"][2]["attested_HLS_timing_diagnostic"]["margin_ns"] < 0
    assert r3b.next_case(report, allow_attested_timing_failure=True) == 1
