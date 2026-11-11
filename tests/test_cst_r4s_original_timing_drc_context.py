"""R4s lexical original Vivado timing/DRC tests. Pure synthetic text.

No vendor tools. Fake timing path strings are not original real device
critical paths; they test that the parser NEVER invents them.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4s_original_timing_drc_context as sut


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return sha256(path.read_bytes()).hexdigest()


def _arm(name, *, wns, bram, dsp, lut, ff, dpop4, sha):
    rules = {
        "DPIP-2": {"severity": "Warning", "violations": 29},
        "DPOP-3": {"severity": "Warning", "violations": 20},
        "DPOP-4": {"severity": "Warning", "violations": dpop4},
        "RTSTAT-10": {"severity": "Warning", "violations": 1},
    }
    total = sum(item["violations"] for item in rules.values())
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
            "bram": bram, "dsp": dsp, "lut": lut, "ff": ff, "uram": 0},
        "source_attested_drc_summary": {
            "available": True, "summary_verified": True,
            "violations": total, "critical_warnings": 0,
            "board_deployment_qualified": False, "clean": False,
            "unconstrained_io": False, "rules": rules,
        },
        "blocked_conditions": ["NEGATIVE_ROUTED_WNS", "DRC_NONCLEAN"],
        "unrecognized_or_missing_required_resources": [],
        "fresh_report_sha256": sha,
        "physical_E4": None,
        "board_qualified": False,
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    original = {
        "study": tmp_path / "original_R4p",
        "r4o_study": tmp_path / "original_R4o",
        "r4k_study": tmp_path / "original_R4k",
        "original_r4d": tmp_path / "original_R4d",
        "original_r4a": tmp_path / "original_R4a",
    }
    for root in original.values():
        root.mkdir()
    arms = []
    for arm, wns, bram, dsp, lut, ff, dpop4 in (
        ("full_storage", -2.170, 7, 24, 1282, 883, 24),
        ("row_storage", -3.718, 4, 21, 1710, 1022, 21),
    ):
        directory = original["study"] / arm / "vivado" / "reports"
        detail = (
            "Design Timing Summary\n"
            f"WNS (ns): {wns}\n"
            "Startpoint: cpu/ctrl/q\n"
            "Endpoint: cpu/ctrl/d\n"
            "Path Group: ap_clk\n"
            "Path Type: max\n"
            f"Slack (VIOLATED) : {wns}ns\n"
        )
        drc = (
            f"Violations found: {74 if arm == 'full_storage' else 71}\n"
            "| DPIP-2 | Warning | synthetic example label | 29 |\n"
            "| DPOP-3 | Warning | synthetic example label | 20 |\n"
            f"| DPOP-4 | Warning | synthetic example label | {dpop4} |\n"
            "| RTSTAT-10 | Warning | synthetic example label | 1 |\n"
            "[DRC DPIP-2] synthetic message details only\n"
            "[DRC DPOP-4] synthetic port example\n"
        )
        contents = {
            "route_status.rpt": "SYNTHETIC FULLY ROUTED\n",
            "timing_summary.rpt": detail,
            "utilization.rpt": "SYNTHETIC UTILIZATION\n",
            "drc.rpt": drc,
        }
        sha = {name: _write(directory / name, body)
               for name, body in contents.items()}
        arms.append(_arm(arm, wns=wns, bram=bram, dsp=dsp, lut=lut,
                         ff=ff, dpop4=dpop4, sha=sha))
    r4q_report = {
        "schema": sut.r4q.SCHEMA,
        "status": "SIGNED_ORIGINAL_R4P_POSTRUN_DIAGNOSTICS_NOT_E4",
        "source_revision": sut.r4r.SOURCE_REVISION,
        "original_R4p_pair_verdict": "UNKNOWN_INCOMPLETE_OR_FAILED_PHYSICAL_PAIR",
        "original_physical_E4": None,
        "both_original_native_vendor_invocations_attested": True,
        "report_files_modified": False,
        "new_Vivado_calls": 0, "new_Vitis_calls": 0,
        "board_qualified": False,
        "physical_E4": None,
        "diagnostic_arms": arms,
    }
    monkeypatch.setattr(sut.r4q, "inspect",
                        lambda **kwargs: r4q_report)
    return original, r4q_report


def test_explicit_source_path_data_bounded_and_not_physical_causality(source):
    roots, _ = source
    result = sut.inspect(**roots)
    assert result["schema"] == sut.SCHEMA
    assert result["status"] == "SIGNED_ORIGINAL_R4P_TIMING_DRC_TEXT_FORENSICS_NOT_E4"
    assert result["source_attested_arms"] == 2
    assert result["source_to_source_WNS_delta_ns"] == -1.548
    assert result["physical_E4"] is None
    assert result["no_new_vivado_or_vitis_calls"] is True
    assert result["original_vendor_artifacts_modified"] is False
    for arm in result["arms"]:
        data = arm["timing_path_source_details"]
        assert data["status"] == (
            "EXPLICIT_ORIGINAL_PATH_DETAILS_PRESENT_NOT_CAUSAL_FORENSICS")
        assert data["sampled_original_path_count"] == 1
        path = data["sample_original_paths"][0]
        assert path["original_source"] == "cpu/ctrl/q"
        assert path["original_destination"] == "cpu/ctrl/d"
        assert path["original_path_group"] == "ap_clk"
        assert path["complete_path_detail_in_signed_text"] is True
        assert path["path_slack_ns"] == arm["original_routed_wns_ns"]
        assert data["critical_path_root_cause"] is None
        assert arm["routed_timing_failure_cause"] is None
        context = arm["drc_rule_source_context"]
        assert len(context["DPOP-4"]["sample_original_report_lines"]) == 2
        assert context["DPOP-4"]["sample_original_report_lines"][0][
            "original_report_text_excerpt"].startswith("| DPOP-4")
        assert arm["physical_E4"] is None


def test_no_detailed_path_from_wns_only_is_unknown(source):
    roots, report = source
    for arm in sut.ARMS:
        file = roots["study"] / arm / "vivado" / "reports" / "timing_summary.rpt"
        original = report["diagnostic_arms"][0 if arm == "full_storage" else 1]
        orig_wns = original["source_attested_routed_wns_ns"]
        original["fresh_report_sha256"]["timing_summary.rpt"] = _write(
            file, f"Design Timing Summary\nWNS (ns): {orig_wns}\n")
    result = sut.inspect(**roots)
    for arm in result["arms"]:
        assert arm["timing_path_source_details"]["status"] == (
            "UNKNOWN_NO_COMPLETE_PATH_DETAILS_IN_ORIGINAL_TIMING_SUMMARY")
        assert arm["timing_path_source_details"]["sample_original_paths"] == []
        assert arm["original_routed_wns_ns"] < 0
        assert arm["routed_timing_failure_cause"] is None


def test_path_missing_slack_or_endpoint_must_not_invent_source_path():
    text = "Startpoint: a/q\nEndpoint: a/d\nPath Group: clk\n"
    assert sut.extract_paths(text)["sampled_original_path_count"] == 0
    text = "Startpoint: a/q\nSlack (VIOLATED) : -1.0ns\n"
    assert sut.extract_paths(text)["sampled_original_path_count"] == 0


def test_source_path_parser_bounds_five_and_text_length():
    pattern = (
        "Startpoint: " + "x"*400 + "\n"
        "Endpoint: z/d\n"
        "Slack (VIOLATED) : -1.25ns\n"
    )
    result = sut.extract_paths(pattern * 8)
    assert result["sampled_original_path_count"] == sut.MAX_PATHS
    assert len(result["sample_original_paths"][0]["original_source"]) <= sut.MAX_TEXT


def test_drc_exact_labeled_examples_not_counted_as_new_violations():
    text = "\n".join(
        f"[DRC DPIP-2] example context {i}" for i in range(10))
    output = sut.extract_drc_context(
        text, {"DPIP-2": 29, "DPOP-3": 20})
    assert len(output["DPIP-2"]["sample_original_report_lines"]) == 3
    assert output["DPIP-2"]["signed_summary_violation_count"] == 29
    assert output["DPOP-3"]["sample_original_report_lines"] == []
    assert output["DPOP-3"]["status"] == "UNKNOWN_ORIGINAL_RULE_CONTEXT_ABSENT"


def test_tampered_original_report_sha_rejected_before_context(source):
    roots, _ = source
    p = roots["study"] / "full_storage" / "vivado" / "reports" / "drc.rpt"
    p.write_text(p.read_text() + "tampered after original vendor run\n")
    with pytest.raises(ValueError, match="SHA"):
        sut.inspect(**roots)


def test_wrong_original_source_revision_fails_closed(source):
    roots, result = source
    result["source_revision"] = "f"*40
    with pytest.raises(ValueError, match="source/physical provenance"):
        sut.inspect(**roots)


def test_create_only_output_and_original_bytes_unchanged(source, tmp_path):
    roots, _ = source
    before = {
        str(p): sha256(p.read_bytes()).hexdigest()
        for root in roots.values()
        for p in root.rglob("*") if p.is_file()
    }
    output = tmp_path / "new_r4s_source_only_report"
    result = sut.run(**roots, output=output)
    assert result["original_vendor_artifacts_modified"] is False
    h = json.loads((output / "artifact_sha256.json").read_text())
    assert h["report.json"] == sha256((output / "report.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(**roots, output=output)
    with pytest.raises(ValueError, match="outside original native"):
        sut.run(**roots, output=roots["study"] / "do_not_write")
    after = {
        str(p): sha256(p.read_bytes()).hexdigest()
        for root in roots.values()
        for p in root.rglob("*") if p.is_file()
    }
    assert before == after
