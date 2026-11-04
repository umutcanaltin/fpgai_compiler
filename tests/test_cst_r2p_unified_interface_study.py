"""R2p source-pinned cross-domain policy facade original GCC and receipt tests."""
from __future__ import annotations

import json
import shutil

import pytest

from fpgai_research.experiments import cst_r2p_unified_interface_study as sut


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("Original GCC required")
    destination=tmp_path_factory.mktemp("cst_r2p")/"original"
    summary=sut.run(output=destination)
    return destination,summary


def _copy(study,tmp_path):
    result=tmp_path/"copy"
    shutil.copytree(study[0],result)
    return result


def test_original_source_both_domains_and_conservative_6_action_denominator(study):
    root,result=study
    assert result["original_graph_domains"]==list(sut.DOMAINS)
    assert result["original_graph_count"]==2
    assert result["original_action_count"]==6
    assert result["status_denominators"]=={
        "E1_GCC_QUALIFIED":3, "REJECTED":2, "UNKNOWN_E0":1,
    }
    assert result["original_gcc_e1_attempts"]==3
    assert result["original_gcc_e1_qualified"]==3
    assert result["original_vitis_calls"]==0
    assert result["original_vivado_calls"]==0
    assert result["rl_training_calls"]==0
    for domain in sut.DOMAINS:
        p=root/(domain+".json")
        assert p.is_file()
        receipt=json.loads(p.read_text())
        assert receipt["source_revision"]==result["original_source_revision"]
        assert receipt["e0_mask"]["candidate_count"]==3
        assert receipt["e1_mask"]["candidate_count"]==3
        assert receipt["policy_observation"]["domain"]==domain
        assert "full_source_state_sha256" not in str(
            receipt["policy_observation"])
    audit=sut.audit(root)
    assert audit["observed_original_actions"]==6
    assert audit["outcome_counts"]==result["status_denominators"]
    assert audit["gcc_e1_attempts"]==3
    assert audit["original_vendor_calls_during_audit"]==0


def test_original_output_create_only_and_audit_never_mutates(study,tmp_path):
    root,summary=study
    with pytest.raises(FileExistsError,match="create-only"):
        sut.run(output=root)
    before={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
    assert sut.audit(root)["observed_original_actions"]==6
    after={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
    assert before==after
    with pytest.raises(ValueError,match="missing or symlinked"):
        sut.audit(tmp_path/"absent")


def test_tamper_original_input_digest_is_caught(study,tmp_path):
    root=_copy(study,tmp_path)
    p=root/"sequential.json"
    d=json.loads(p.read_text())
    d["original_fixed_input_sha256"]="c"*64
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="original E0/E1 or graph identity invalid"):
        sut.audit(root)


def test_forged_e1_status_from_unknown_is_caught(study,tmp_path):
    root=_copy(study,tmp_path)
    p=root/"sequential.json"
    d=json.loads(p.read_text())
    case=d["e1_mask"]["rows"][2]
    assert case["status"]=="UNKNOWN_E0"
    case["status"]="E1_GCC_QUALIFIED"
    case["e1_gcc_qualified"]=True
    d["e1_mask"]["statuses"][2]="E1_GCC_QUALIFIED"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="falsely claimed GCC E1"):
        sut.audit(root)


def test_inconsistent_e0_and_e1_compiler_lineage_caught(study,tmp_path):
    root=_copy(study,tmp_path)
    p=root/"multi_reader_fanout.json"
    d=json.loads(p.read_text())
    d["e1_mask"]["rows"][0]["cpp_sha256"]="f"*64
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="action treatment or source CPP"):
        sut.audit(root)


def test_wrong_summary_denominator_and_missing_domain_are_rejected(study,tmp_path):
    root=_copy(study,tmp_path)
    p=root/"summary.json"
    d=json.loads(p.read_text())
    d["original_action_count"]=9
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="population forged"):
        sut.audit(root)
    p=root/"sequential.json"
    p.unlink()
    with pytest.raises(ValueError,match="missing or symlinked"):
        sut.audit(root)


def test_symlinked_source_receipt_is_rejected(study,tmp_path):
    root=_copy(study,tmp_path)
    p=root/"sequential.json"
    p.unlink()
    p.symlink_to(root/"multi_reader_fanout.json")
    with pytest.raises(ValueError,match="missing or symlinked"):
        sut.audit(root)


def test_original_study_must_have_clean_source_and_finite_legacy_cohort(
        monkeypatch,tmp_path):
    monkeypatch.setattr(sut,"_source_sha",lambda: (_ for _ in ()).throw(
        RuntimeError("dirty source revision")))
    dest=tmp_path/"not_created"
    with pytest.raises(RuntimeError,match="dirty source revision"):
        sut.run(output=dest)
    assert not dest.exists()



def test_tampered_original_e1_gcc_input_digest_cannot_be_accepted(study,tmp_path):
    root=_copy(study,tmp_path)
    p=root/"sequential.json"
    d=json.loads(p.read_text())
    case=d["e1_mask"]["rows"][0]
    assert case["status"]=="E1_GCC_QUALIFIED"
    case["gcc_receipt"]["original_input_sha256"]="a"*64
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="original GCC receipt input SHA mismatch"):
        sut.audit(root)


def test_tampered_original_fanout_gcc_source_sha_cannot_be_accepted(study,tmp_path):
    root=_copy(study,tmp_path)
    p=root/"multi_reader_fanout.json"
    d=json.loads(p.read_text())
    case=d["e1_mask"]["rows"][0]
    assert case["status"]=="E1_GCC_QUALIFIED"
    case["gcc_receipt"]["source_cpp_sha256"]="b"*64
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="fanout original GCC source/trace receipt missing"):
        sut.audit(root)
