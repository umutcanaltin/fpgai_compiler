"""R3k C2 matched exact source-limit interventions on twelve held-out models."""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments import cst_r3k_c2_target_limit_intervention as sut


@pytest.fixture(scope="module")
def complete_pair_report():
    return sut.run()


@pytest.mark.parametrize("case",sut.CASES)
def test_true_causal_target_limit_contrast_same_source_and_actions(
    case,complete_pair_report
):
    row=complete_pair_report["rows"][sut.CASES.index(case)]
    assert row["source_model"]==list(case)
    assert row["candidate_population_per_limit"]==8
    assert row["target_intervention_flip_indices"]==[0,1]
    assert row["target_intervention_flip_count"]==2
    assert row["hls_or_routed_resource_feasibility_tested"] is False
    assert row["trained_agents"]==0
    assert len(row["identical_candidate_actions"])==8
    strict,relaxed=row["results"]
    assert strict["source_parallel_limit"]==1
    assert relaxed["source_parallel_limit"]==4
    assert strict["statuses"]==list(sut.EXPECTED_STRICT)
    assert relaxed["statuses"]==list(sut.EXPECTED_RELAXED)
    assert strict["e0_certified"]==1
    assert relaxed["e0_certified"]==3
    assert strict["rejected"]==7
    assert relaxed["rejected"]==5
    assert strict["e0_unknown"]==relaxed["e0_unknown"]==0
    assert strict["source_mask_compiler_transitions"]==8
    assert relaxed["source_mask_compiler_transitions"]==8
    assert strict["blind_replay_compiler_transitions"]==8
    assert relaxed["blind_replay_compiler_transitions"]==8
    assert strict["original_source_state_sha256"]!=relaxed[
        "original_source_state_sha256"]
    assert strict["graph_public_features_sha256"]!=relaxed[
        "graph_public_features_sha256"]
    for item in (strict,relaxed):
        assert len(item["source_cpp_sha256"])==8
        assert item["original_vendor_calls"]==0


def test_complete_12_model_192_action_matched_population(
    complete_pair_report
):
    r=complete_pair_report
    assert r["schema"]==sut.SCHEMA
    assert r["matched_static_source_model_pairs"]==12
    assert r["source_parallel_limits"]==[1,4]
    assert r["actions_per_limit"]==8
    assert r["total_compiler_masked_candidates"]==192
    assert r["total_independently_replayed_candidates"]==192
    assert r["target_limit_status_flips"]==24
    assert r["certified"]==[12,36]
    assert r["rejected"]==[84,60]
    assert r["unknown_e0"]==[0,0]
    assert r["gcc_calls"]==0
    assert r["vitis_calls"]==0
    assert r["trained_agents"]==0
    assert len(set(sut.CASES))==12


def test_bad_registered_case_and_modified_status_oracle_rejected(
    monkeypatch
):
    with pytest.raises(ValueError,match="outside predeclared"):
        sut.evaluate((9,9,9,9,9))
    monkeypatch.setattr(sut,"EXPECTED_RELAXED",sut.EXPECTED_STRICT)
    with pytest.raises(AssertionError,match="source constraints"):
        sut.evaluate(sut.CASES[0])


def test_source_receipt_read_only_and_create_only(tmp_path,complete_pair_report):
    root=tmp_path/"original"
    sut.create_only(complete_pair_report,output=root,source_revision="a"*40)
    report=sut.audit(root)
    assert report["model_pairs"]==12
    assert report["candidate_population"]==192
    assert report["causally_changed_source_action_statuses"]==24
    assert len(report["receipt_sha256"])==64
    assert report["new_compiler_vendor_calls"]==0
    with pytest.raises(FileExistsError,match="create-only"):
        sut.create_only(complete_pair_report,output=root,source_revision="a"*40)


@pytest.mark.parametrize("tamper",[
    lambda r:r.update(target_limit_status_flips=23),
    lambda r:r.update(total_compiler_masked_candidates=191),
    lambda r:r["rows"][0]["results"][0]["statuses"].reverse(),
    lambda r:r["rows"][1]["results"][1].update(source_parallel_limit=9),
    lambda r:r.update(trained_agents=1),
    lambda r:r.update(source_revision="invalid"),
])
def test_bad_source_receipt_mutation_fails_closed(
    tmp_path,complete_pair_report,tamper
):
    root=tmp_path/"orig"
    sut.create_only(complete_pair_report,output=root,source_revision="b"*40)
    file=root/"r3k_matched_target_limit_source_masks.json"
    raw=json.loads(file.read_text())
    tamper(raw)
    file.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="R3k"):
        sut.audit(root)
