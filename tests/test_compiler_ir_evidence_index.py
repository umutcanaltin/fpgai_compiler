"""Source-verified combined CST scientific evidence index, never native QoR fiction."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _sut():
    path=Path(__file__).resolve().parents[1]/"experiments/compiler_ir/export_evidence.py"
    spec=importlib.util.spec_from_file_location("compiler_ir_evidence",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    sut=_sut()
    root=tmp_path/"series"
    root.mkdir()
    revision="a"*40
    conf=root/"source_plan.json"
    conf.write_text('{"checked":"configuration"}\n')
    study=root/"array"
    study.mkdir()
    file=study/"summary.json"
    file.write_text(json.dumps({
        "schema":"fpgai-research/compositional-ir-cpu-campaign/v1",
        "source_revision":revision,
        "cases":2,"host_checked_cases":1,"host_numeric_passes":1,
    })+"\n")
    meta={
        "schema":"fpgai-research/compiler-ir-experiment-series/v1",
        "source_revision":revision,
        "configuration_sha256":sut.digest(conf),
        "steps":{
            "array":{
                "status":"PASSED","evidence":"E0/E1",
                "artifacts":[{"path":"summary.json","sha256":sut.digest(file)}],
            },
            "trained_mlp_preflight":{
                "status":"BLOCKED","evidence":"E0",
                "reason":"Private original trained assets absent",
            },
        },
    }
    (root/"series_manifest.json").write_text(json.dumps(meta)+"\n")
    return sut,root,file


def test_evidence_index_distinguishes_original_source_results_from_missing(tmp_path):
    sut,root,_file=_fixture(tmp_path)
    result=sut.export(root,tmp_path/"paper")
    assert result["stage_counts"]=={"PASSED":1,"FAILED":0,"BLOCKED":1}
    assert result["source_revision"]=="a"*40
    assert not result["published_prior_art_novelty_established"]
    assert not result["original_vitis_hardware_quality_claim_established"]
    assert result["stages"][0]["original_counters"]["host_numeric_passes"]==1
    assert result["stages"][1]["status"]=="BLOCKED"
    assert (tmp_path/"paper"/"paper_evidence.json").is_file()
    assert "trained_mlp_preflight" in (
        tmp_path/"paper"/"paper_evidence.md").read_text()
    with pytest.raises(FileExistsError,match="Create-only"):
        sut.export(root,tmp_path/"paper")


def test_tampered_raw_evidence_is_rejected_even_if_summary_remains_readable(tmp_path):
    sut,root,file=_fixture(tmp_path)
    file.write_text(file.read_text().replace('"host_numeric_passes": 1',
                                            '"host_numeric_passes": 999'))
    with pytest.raises(ValueError,match="changed"):
        sut.collect(root)


def test_cross_revision_statistics_cannot_be_combined(tmp_path):
    sut,root,file=_fixture(tmp_path)
    raw=json.loads(file.read_text())
    raw["source_revision"]="b"*40
    file.write_text(json.dumps(raw)+"\n")
    manifest=root/"series_manifest.json"
    data=json.loads(manifest.read_text())
    data["steps"]["array"]["artifacts"][0]["sha256"]=sut.digest(file)
    manifest.write_text(json.dumps(data)+"\n")
    with pytest.raises(ValueError,match="across revisions"):
        sut.collect(root)


def test_source_plan_checksum_mismatch_is_rejected(tmp_path):
    sut,root,_file=_fixture(tmp_path)
    (root/"source_plan.json").write_text('{"evil":"changed"}\n')
    with pytest.raises(ValueError,match="Source plan"):
        sut.collect(root)


def test_three_cst_contribution_protocol_status_is_never_paper_novelty(tmp_path):
    sut,root,_=_fixture(tmp_path)
    # The fixture contains only array E1; C1/C2/C3 must remain incomplete.
    report=sut.collect(root)
    assert not report["all_three_contribution_source_protocols_complete"]
    groups=report["three_research_contribution_source_protocols"]
    assert set(groups)=={
        "C1_compositional_CST_IR",
        "C2_bounded_certificate_and_independent_oracle",
        "C3_policy_independent_RL_compatible_interface",
    }
    assert all(not row["novelty_established"] for row in groups.values())
    assert report["actual_RL_training_benefit_established"] is False


def test_phi_counterfactual_must_match_exact_original_gcc_csv_bytes(tmp_path):
    sut,root,_=_fixture(tmp_path)
    original=root/"two_phi_readers_e1"
    original.mkdir()
    cases=original/"cases.csv"
    cases.write_text("case,value\n000,1\n")
    (root/"two_phi_counterfactual_e1").mkdir()
    after=root/"two_phi_counterfactual_e1"/"summary.json"
    import hashlib
    original_digest=hashlib.sha256(cases.read_bytes()).hexdigest()
    after.write_text(json.dumps({
        "schema":"fpgai-research/cst-two-reader-phi-causal-effects/v1",
        "input_csv_sha256":original_digest,
        "original_cases":24,"controlled_phi_contrasts":18,
        "novelty_relative_to_prior_art_established":False,
    })+"\n")
    meta_path=root/"series_manifest.json"
    meta=json.loads(meta_path.read_text())
    meta["steps"]["two_phi_readers_e1"]={
        "status":"PASSED","evidence":"E1-original-GCC",
        "artifacts":[{"path":"cases.csv","sha256":sut.digest(cases)}],
    }
    meta["steps"]["two_phi_counterfactual_e1"]={
        "status":"PASSED","evidence":"E1-secondary",
        "artifacts":[{"path":"summary.json","sha256":sut.digest(after)}],
    }
    meta_path.write_text(json.dumps(meta)+"\n")
    accepted=sut.collect(root)
    assert accepted["all_three_contribution_source_protocols_complete"] is False
    assert next(row for row in accepted["stages"]
                if row["stage"]=="two_phi_counterfactual_e1")[
                    "original_counters"]["controlled_phi_contrasts"]==18
    # Change the referenced case bytes and update their OWN case hash so
    # generic artifact integrity still passes; the cross-stage
    # differential source binding must catch the inconsistency.
    cases.write_text("case,value\n000,9\n")
    meta=json.loads(meta_path.read_text())
    meta["steps"]["two_phi_readers_e1"]["artifacts"][0]["sha256"]=sut.digest(cases)
    meta_path.write_text(json.dumps(meta)+"\n")
    with pytest.raises(ValueError,match="causal CSV"):
        sut.collect(root)
