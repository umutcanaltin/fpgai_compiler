"""Native HLS source-only dry-run, hostile evidence and explicit opt-in guard."""
import json
import pytest

from fpgai_research.experiments.cst_nchw_native_execute import (
    audit_native,execute,
)
from fpgai_research.experiments.cst_nchw_native_pairs import prepare


def test_no_vitis_runs_during_default_and_read_only_native_audit(tmp_path,monkeypatch):
    root=tmp_path/"original"
    prepared=prepare(output=root)
    calls=[]
    def forbidden(*a,**k):
        calls.append((a,k))
        raise AssertionError("Native Vitis must NEVER run in CPU tests")
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_native_execute.run_vitis",
        forbidden)
    plan=execute(root,max_pairs=2)
    assert plan["status"]=="DRY_RUN_NO_VENDOR_CALLS"
    assert len(plan["planned_source_cases"])==4
    assert plan["original_vitis_calls"]==0
    result=audit_native(root)
    assert result["cases"]==prepared["case_count"]==12
    assert result["matched_pairs"]==6
    assert result["native_invocation_receipts_present"]==0
    assert result["original_CSim_CSynth_attested_cases"]==0
    assert result["original_matched_report_pairs"]==0
    assert result["physical_qualified_cases"]==0
    assert all(r["reported_hls_worst_cycles"] is None for r in result["cases_audited"])
    assert all(r["reported_hls_bram_18k"] is None for r in result["cases_audited"])
    assert all(not p["both_original_CSim_CSynth_attested"]
               for p in result["pairs_audited"])
    assert calls==[]
    with pytest.raises(ValueError,match="confirm-isolated-native"):
        execute(root,execute_native=True,max_pairs=1)
    assert calls==[]


def test_unattributed_csynth_xml_cannot_be_promoted(tmp_path):
    root=tmp_path/"original"
    prepare(output=root)
    xml=root/"case_000"/"fpgai_hls_project"/"solution1"/"syn"/"report"
    xml.mkdir(parents=True)
    (xml/"fpgai_model_csynth.xml").write_text(
        "<Profile><AreaEstimates><Resources><BRAM_18K>0</BRAM_18K>"
        "</Resources></AreaEstimates><PerformanceEstimates>"
        "<SummaryOfOverallLatency><Best-caseLatency>1</Best-caseLatency>"
        "<Worst-caseLatency>1</Worst-caseLatency></SummaryOfOverallLatency>"
        "</PerformanceEstimates></Profile>")
    result=audit_native(root)
    first=result["cases_audited"][0]
    assert first["attested_evidence_level"]=="E1-HLS-source"
    assert first["reported_hls_bram_18k"] is None
    assert first["reported_hls_worst_cycles"] is None
    assert not first["observed_numerical_and_hls_report"]
    assert result["original_matched_report_pairs"]==0


def test_strict_pairs_and_stale_preflight_fail_closed(tmp_path):
    root=tmp_path/"original"
    prepare(output=root)
    for kw in ({"max_pairs":0},{"max_pairs":7},{"start_pair":6},
               {"start_pair":5,"max_pairs":2}):
        with pytest.raises(ValueError,match="matched native"):
            execute(root,**kw)
    (root/"case_000"/"model.cpp").write_text("malicious source")
    with pytest.raises(ValueError,match="Tampered"):
        execute(root)
    with pytest.raises(ValueError,match="Tampered"):
        audit_native(root)
