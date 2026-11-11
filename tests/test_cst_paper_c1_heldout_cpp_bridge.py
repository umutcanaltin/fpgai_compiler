"""C1 finite E0 -> real generated E1 GCC source refinement bridge tests.

Synthetic original E0 receipts in this module are fixtures, NOT the
author's prior 486-event experiment. The real compiler/GCC test runs in
a temporary create-only directory, not in frozen original evidence.
"""
from __future__ import annotations

from hashlib import sha256
import json
import shutil

import pytest

from fpgai_research.experiments import cst_paper_c1_heldout_cpp_bridge as sut
from fpgai_research.experiments.cst_paper_c1_seeded_metamorphic import (
    _architecture, configurations,
)

def _synthetic_original_e0(tmp_path, monkeypatch):
    root = tmp_path / "original_synthetic_c1_e0"
    root.mkdir()
    rows = []
    for case in configurations():
        _, _, events, _ = _architecture(case)
        rows.append({
            "seed":case[0], "readers":case[1],
            "shape":list(case[2:6]), "mapping":case[6],
            "mutations":[
                {"mutation":"canonical",
                 "event_trace_sha256":sut._source_trace_sha(events)}
            ]+[{"mutation":"selected_invalid"} for _ in range(8)],
        })
    assert len(rows) == 54
    report = {
        "schema": sut.ORIGINAL_C1_SCHEMA,
        "experiment_source_revision": "a"*40,
        "profiles": 54, "traces": 486,
        "value_oracle_disagreements":0,"issue_oracle_disagreements":0,
        "source_reports":rows,
    }
    path = root/"report.json"
    path.write_text(json.dumps(report, indent=2,sort_keys=True)+"\n")
    digest = sha256(path.read_bytes()).hexdigest()
    (root/"artifact_sha256.json").write_text(json.dumps({
        "report.json":digest},indent=2,sort_keys=True)+"\n")
    monkeypatch.setattr(sut,"ORIGINAL_C1_REPORT_SHA256",digest)
    monkeypatch.setattr(sut,"ORIGINAL_C1_SOURCE_REVISION","a"*40)
    monkeypatch.setattr(sut,"_git_revision",lambda:"b"*40)
    return root, digest


def test_pinned_c1_original_54_profile_read_only_population(tmp_path,monkeypatch):
    original, digest = _synthetic_original_e0(tmp_path,monkeypatch)
    cohort = sut.load_original_c1(original)
    assert len(cohort)==54
    assert len(set(cohort.values()))>=2
    assert digest == sut.ORIGINAL_C1_REPORT_SHA256


def test_original_c1_receipt_tampering_rejected(tmp_path,monkeypatch):
    original,_=_synthetic_original_e0(tmp_path,monkeypatch)
    report=original/"report.json"
    report.write_text(report.read_text()+" ")
    with pytest.raises(ValueError,match="SHA differs"):
        sut.load_original_c1(original)


def test_cross_revision_mismatched_e0_canonical_events_rejected(tmp_path,monkeypatch):
    original,_=_synthetic_original_e0(tmp_path,monkeypatch)
    cases=configurations()
    # Ensure a validator failure is not rationalized as an acceptable
    # alternative source trace.
    with pytest.raises(ValueError,match="canonical event changed"):
        sut._assert_source_parent(cases[0],"f"*64)


@pytest.mark.skipif(shutil.which("g++") is None,
                    reason="Requires actual GNU g++ to validate emitted C++")
def test_real_gcc_scalar_and_mixed_phi_paired_source_e1_smoke(tmp_path,monkeypatch):
    original,_=_synthetic_original_e0(tmp_path,monkeypatch)
    output=tmp_path/"fresh_gcc_bridge"
    result=sut.run(original_c1_study=original,output=output,quick=True)
    assert result["profiles"]==2
    assert result["gcc_compilations"]==4
    assert result["gcc_executions"]==4
    assert result["no_vendor_calls"] is True
    assert result["physical_compute_latency"] is None
    verified=sut.audit(output)
    assert verified["profiles"]==2
    assert verified["source_gcc_compilations"]==4
    assert verified["source_artifact_count"]>=7
    with pytest.raises(FileExistsError,match="create-only"):
        sut.run(original_c1_study=original,output=output,quick=True)


@pytest.mark.skipif(shutil.which("g++") is None,
                    reason="Requires actual GNU g++ to validate emitted C++")
def test_e1_source_or_address_evidence_tampering_detected(tmp_path,monkeypatch):
    original,_=_synthetic_original_e0(tmp_path,monkeypatch)
    out=tmp_path/"fresh_e1"
    sut.run(original_c1_study=original,output=out,quick=True)
    path=out/"case_000/disjoint.cpp"
    path.write_text(path.read_text()+"// mutated without original evidence approval\n")
    with pytest.raises(ValueError,match="original generated source or receipts changed"):
        sut.audit(out)


def test_e1_original_failure_preserved_with_source_receipt(tmp_path,monkeypatch):
    original,_=_synthetic_original_e0(tmp_path,monkeypatch)
    monkeypatch.setattr(sut,"_evaluate_one",lambda *a,**kw: (
        _ for _ in ()).throw(AssertionError("synthetic E1 counterexample")))
    # No actual GCC calls: the exception is injected before emission.
    monkeypatch.setattr(sut.shutil,"which",lambda _: "/usr/bin/g++")
    dest=tmp_path/"failed_e1"
    with pytest.raises(AssertionError,match="synthetic E1 counterexample"):
        sut.run(original_c1_study=original,output=dest,quick=True)
    failed=json.loads((dest/"FAILED.json").read_text())
    assert failed["completed_profiles_before_failure"]==0
    assert failed["status"]=="PRESERVE_ORIGINAL_SOURCE_COUNTEREXAMPLE"
    assert failed["native_Vitis_calls"]==failed["native_Vivado_calls"]==0
    with pytest.raises(FileExistsError):
        sut.run(original_c1_study=original,output=dest,quick=True)
