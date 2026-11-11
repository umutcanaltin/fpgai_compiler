"""R4q forensic report tests, all synthetic; NEVER launch vendor tools.

Mocked originals are fixtures only, not actual Vivado E4 evidence.
The original R4p result remains UNKNOWN under negative WNS or bad QoR.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4q_original_r4p_physical_diagnostics as sut


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    roots = {
        "study": tmp_path / "original_R4p",
        "r4o_study": tmp_path / "original_R4o",
        "r4k_study": tmp_path / "original_R4k",
        "original_r4d": tmp_path / "original_R4d",
        "original_r4a": tmp_path / "original_R4a",
    }
    for folder in roots.values():
        folder.mkdir()
    arms = []
    for name in sut.r4p.ARMS:
        dir = roots["study"] / name / "vivado" / "reports"
        reports = {
            "route_status.rpt": (
                "Design Route Status\n"
                "# of routable nets : 10\n"
                "# of fully routed nets : 10\n"
                "# of nets with routing errors : 0\n"
                "# of unrouted nets : 0\n"),
            "timing_summary.rpt": "Design Timing Summary\nWNS (ns): 0.2\n",
            "utilization.rpt": (
                "| CLB LUTs | 100 |\n"
                "| CLB Registers | 150 |\n"
                "| DSPs | 2 |\n"
                "| Block RAM Tile | 2.0 |\n"
                "| URAM | 0 |\n"),
            "drc.rpt": "Violations found: 0\n",
        }
        hashes = {filename: write(dir / filename, content)
                  for filename, content in reports.items()}
        arms.append({
            "arm": name,
            "status": "UNKNOWN_INCOMPLETE_PHYSICAL_RESOURCE_OR_TIMING",
            "strict_clean_DRC_E4": False,
            "original_RTL_ROM_stable": True,
            "vendor_completed": True,
            "fresh_original_report_sha256": hashes,
        })
    original_audit = {
        "schema": sut.r4p.SCHEMA,
        "status": "UNKNOWN_INCOMPLETE_OR_FAILED_PHYSICAL_PAIR",
        "source_revision": "7" * 40,
        "one_time_original_attempt_lock_verified": True,
        "source_Vivado_inputs_sha_attested": True,
        "original_native_invocation_receipts_present": 2,
        "arms": arms,
        "physical_E4": None,
    }
    monkeypatch.setattr(sut.r4p, "audit_native",
                        lambda **kwargs: original_audit)
    return roots, original_audit


def test_readonly_original_signed_reports_expose_true_source_rows(evidence):
    kwargs, original = evidence
    result = sut.inspect(**kwargs)
    assert result["schema"] == sut.SCHEMA
    assert result["new_Vivado_calls"] == 0
    assert result["original_R4p_pair_verdict"] == (
        "UNKNOWN_INCOMPLETE_OR_FAILED_PHYSICAL_PAIR")
    assert result["original_physical_E4"] is None
    for arm in result["diagnostic_arms"]:
        assert arm["original_report_sha_verified"] is True
        assert arm["source_attested_route_complete"] is True
        assert arm["source_attested_routed_wns_ns"] == pytest.approx(0.2)
        assert arm["source_attested_timing_met"] is True
        assert arm["source_attested_utilization"]["bram"] == 2
        assert arm["unrecognized_or_missing_required_resources"] == []
        assert arm["blocked_conditions"] == []
        assert arm["original_strict_clean_DRC_E4"] is False
        assert arm["physical_E4"] is None
    assert result["physical_E4"] is None


def test_real_report_negative_wns_is_diagnostic_not_new_e4(evidence):
    kwargs, audit = evidence
    p = kwargs["study"] / "row_storage" / "vivado" / "reports" / "timing_summary.rpt"
    audit["arms"][1]["fresh_original_report_sha256"]["timing_summary.rpt"] = (
        write(p, "Design Timing Summary\nWNS (ns): -1.234\n"))
    result = sut.inspect(**kwargs)
    row = result["diagnostic_arms"][1]
    assert row["source_attested_routed_wns_ns"] == pytest.approx(-1.234)
    assert row["source_attested_timing_met"] is False
    assert "NEGATIVE_ROUTED_WNS" in row["blocked_conditions"]
    assert row["source_attested_utilization"]["lut"] == 100
    assert row["physical_E4"] is None
    assert result["physical_E4"] is None


def test_unrecognized_bram_resource_retains_raw_named_row_and_unknown(
    evidence,
):
    kwargs, audit = evidence
    p = kwargs["study"] / "row_storage" / "vivado" / "reports" / "utilization.rpt"
    audit["arms"][1]["fresh_original_report_sha256"]["utilization.rpt"] = write(
        p, "| CLB LUTs | 100 |\n"
           "| CLB Registers | 150 |\n"
           "| DSPs | 2 |\n"
           "| Block RAM Tile (invented-format) | 3.0 |\n"
           "| URAM | 0 |\n")
    result = sut.inspect(**kwargs)
    arm = result["diagnostic_arms"][1]
    assert arm["unrecognized_or_missing_required_resources"] == ["bram"]
    assert arm["source_attested_utilization"]["bram"] is None
    assert "MISSING_OR_INVALID_PHYSICAL_UTILIZATION" in arm["blocked_conditions"]
    assert any("Block RAM" in x["source_label"]
               for x in arm["sample_original_utilization_rows"])
    assert arm["physical_E4"] is None


def test_original_report_tampering_does_not_create_new_qualified_numbers(evidence):
    kwargs, audit = evidence
    p = kwargs["study"] / "full_storage" / "vivado" / "reports" / "drc.rpt"
    p.write_text("Violations found: 999\n")
    with pytest.raises(ValueError, match="SHA drift"):
        sut.inspect(**kwargs)


def test_not_attested_original_receipt_never_reads_stale_report(evidence):
    kwargs, audit = evidence
    audit["arms"][0]["original_RTL_ROM_stable"] = False
    result = sut.inspect(**kwargs)
    diag = result["diagnostic_arms"][0]
    assert diag["original_report_sha_verified"] is False
    assert diag["source_attested_routed_wns_ns"] is None
    assert diag["blocked_conditions"] == [
        "ORIGINAL_VENDOR_OR_SOURCE_PROVENANCE_NOT_ATTESTED"]
    assert diag["physical_E4"] is None


def test_missing_original_population_or_lock_rejected(evidence):
    kwargs, audit = evidence
    audit["one_time_original_attempt_lock_verified"] = False
    with pytest.raises(ValueError, match="population"):
        sut.inspect(**kwargs)
    audit["one_time_original_attempt_lock_verified"] = True
    audit["arms"] = audit["arms"][:1]
    with pytest.raises(ValueError, match="population"):
        sut.inspect(**kwargs)


def test_creates_only_new_output_and_never_updates_originals(evidence, tmp_path):
    kwargs, _ = evidence
    before = {
        str(p): sha256(p.read_bytes()).hexdigest()
        for folder in kwargs.values() for p in folder.rglob("*") if p.is_file()
    }
    output = tmp_path / "r4q_new_diagnostics"
    result = sut.run(**kwargs, output=output)
    assert result["report_files_modified"] is False
    assert result["new_Vivado_calls"] == 0
    h = json.loads((output / "artifact_sha256.json").read_text())
    assert h["report.json"] == sha256(
        (output / "report.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(**kwargs, output=output)
    with pytest.raises(ValueError, match="outside original"):
        sut.run(**kwargs, output=kwargs["study"] / "forbidden")
    after = {
        str(p): sha256(p.read_bytes()).hexdigest()
        for folder in kwargs.values() for p in folder.rglob("*") if p.is_file()
    }
    assert before == after
