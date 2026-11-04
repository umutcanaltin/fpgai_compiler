"""Publication gates are conservative; synthetic data cannot prove novelty."""
import copy
import json

import pytest

from fpgai_research.experiments import cst_paper_gate as gate


def studies(tmp_path):
    native=tmp_path/"native"
    physical=tmp_path/"physical"
    native.mkdir()
    physical.mkdir()
    (native/"native_source_manifest.json").write_text('{"source_revision":"original"}\n')
    (physical/"physical_study_manifest.json").write_text('{"source_revision":"original"}\n')
    ns=[]
    ps=[]
    for i in range(12):
        source=f"sha_{i:03d}"
        shared=i%2==1
        graph=("branch_concat","pool_fork","broadcast_fork")[i//4]
        mapping="independent_phi" if (i//2)%2 else "scalar"
        original={"case":f"case_{i:03d}","graph":graph,"mapping":mapping,
                  "source_storage":"real_shared" if shared else "real_disjoint",
                  "source_cpp_sha256":source}
        ns.append({**original,"observed_numerical_and_hls_report":True})
        ps.append({**original,"routed_qualified":True,
                   "routed_observation_valid":True})
    hn={"cases":12,"matched_pairs":6,"cases_audited":ns,
        "source_revision":"8814bd0005e8a852348b49f19131b2410117216b",
        "pairs_audited":[{"both_original_CSim_CSynth_attested":True}
                         for i in range(6)]}
    ph={"prespecified_cases":12,"cases":ps,
        "original_hls_source_revision":hn["source_revision"]}
    return native,physical,hn,ph


def fake_audits(monkeypatch,hn,ph):
    monkeypatch.setattr(gate,"audit_native",lambda *_,
                        require_current_source=False:copy.deepcopy(hn))
    monkeypatch.setattr(gate,"audit_routed_study",lambda *_:copy.deepcopy(ph))


def test_complete_hardware_cannot_claim_peer_novelty_or_automatic_release(
    monkeypatch,tmp_path
):
    n,p,hn,ph=studies(tmp_path)
    fake_audits(monkeypatch,hn,ph)
    result=gate.evaluate(native_study=n,physical_study=p)
    assert result["strict_clean_drc_e4_pairs"]==6
    assert result["strict_physical_primary_gate"] is True
    assert result["paper_submission_ready"] is False
    assert result["mapping_conditioned_certificate_firstness_established"] is False
    assert all(x=="NOT_ESTABLISHED" for x in
               result["independently_executed_exact_peer_artifact_parity"].values())
    assert result["original_vendor_calls_this_command"]==0


def test_separate_warning_only_physical_tier_remains_not_strict_e4(
    monkeypatch,tmp_path
):
    n,p,hn,ph=studies(tmp_path)
    ph["cases"][0]["routed_qualified"]=False
    ph["cases"][0]["routed_observation_valid"]=True
    fake_audits(monkeypatch,hn,ph)
    result=gate.evaluate(native_study=n,physical_study=p)
    assert result["strict_clean_drc_e4_pairs"]==5
    assert result["descriptive_routed_observation_pairs"]==6
    assert result["strict_physical_primary_gate"] is False
    assert result["exploratory_observation_complete"] is True
    assert result["paper_submission_ready"] is False


@pytest.mark.parametrize("mutation",[
    "reordered_case","different_hls_revision","different_graph",
    "different_source_cpp","missing_case","incomplete_pair",
])
def test_missing_or_mismatched_primary_vendor_evidence_rejected(
    monkeypatch,tmp_path,mutation
):
    n,p,hn,ph=studies(tmp_path)
    if mutation=="reordered_case":ph["cases"][0],ph["cases"][1]=(
        ph["cases"][1],ph["cases"][0])
    elif mutation=="different_hls_revision":
        ph["original_hls_source_revision"]="0"*40
    elif mutation=="different_graph":
        ph["cases"][1]["graph"]="mismatched"
    elif mutation=="different_source_cpp":
        ph["cases"][0]["source_cpp_sha256"]="forged"
    elif mutation=="missing_case":ph["cases"].pop()
    elif mutation=="incomplete_pair":hn["pairs_audited"].pop()
    fake_audits(monkeypatch,hn,ph)
    with pytest.raises(ValueError):
        gate.evaluate(native_study=n,physical_study=p)


def test_create_only_audited_paper_readiness_report(
    monkeypatch,tmp_path
):
    n,p,hn,ph=studies(tmp_path)
    fake_audits(monkeypatch,hn,ph)
    output=tmp_path/"readiness"
    result=gate.create_report(native_study=n,physical_study=p,output=output)
    assert result["paper_submission_ready"] is False
    assert json.loads((output/"readiness.json").read_text())==result
    with pytest.raises(FileExistsError,match="create-only"):
        gate.create_report(native_study=n,physical_study=p,output=output)
