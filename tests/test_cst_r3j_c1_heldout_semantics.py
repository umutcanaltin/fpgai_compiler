"""R3j: held-out C1 source semantics with varied shapes and Phi mappings.

The predeclared 24 source profiles have 216 canonical/adverse source traces,
including genuine safe-but-Phi-order-invalid controls.
No general arbitrary-graph/physical correctness claim is permitted.
"""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments import cst_r3j_c1_heldout_semantics as sut


@pytest.fixture(scope="module")
def complete_report():
    return sut.run()


@pytest.mark.parametrize("config", sut.CASES)
def test_all_heldout_profiles_have_both_positive_and_negative_controls(
    config, complete_report
):
    position=sut.CASES.index(config)
    row=complete_report["rows"][position]
    assert row["case"]==list(config)
    assert row["readers"]==config[0]
    assert row["source_only"] is True
    assert row["variant_population"]==1+len(sut.MUTANTS)==9
    assert row["semantic_oracle_disagreements"]==0
    assert row["valid_source_trace"]==1
    assert row["safe_but_phi_issue_nonconformant"]==1
    assert row["expected_semantic_violations"]==7
    assert row["classification_counts"]=={
        "safe_and_issue":1,
        "safe_wrong_issue":1,
        "unsafe":7,
    }
    assert [v["name"] for v in row["outcomes"]]==[
        "canonical",*sut.MUTANTS]
    assert all(len(v["source_trace_sha256"])==64 for v in row["outcomes"])
    assert all(len(v["source_offsets_sha256"])==64 for v in row["outcomes"])
    assert (row["outcomes"][0]["source_offsets_sha256"] !=
            row["outcomes"][7]["source_offsets_sha256"])
    assert (row["outcomes"][0]["source_trace_sha256"] ==
            row["outcomes"][7]["source_trace_sha256"])


def test_complete_prespecified_shape_phi_factorial_has_no_hidden_exclusions(
    complete_report
):
    report=complete_report
    assert len(sut.CASES)==24
    assert len(set(sut.CASES))==24
    assert len(sut.SHAPES)==4
    assert len(sut.READER_COUNTS)==3
    assert len(sut.POLICIES)==2
    assert report["schema"]==sut.SCHEMA
    assert report["prespecified_source_cases"]==24
    assert report["variants_per_case"]==9
    assert report["total_trace_variants"]==216
    assert report["matched_independent_oracle"]==216
    assert report["independent_oracle_disagreements"]==0
    assert report["safe_wrong_phi_issue_control_count"]==24
    assert report["unsafe_source_mutants_count"]==168
    assert report["original_gcc_calls"]==0
    assert report["original_vitis_calls"]==0
    assert report["trained_agents"]==0


def test_mixed_phi_intervention_changes_actual_source_issue_owner():
    scalar=sut._source((2,1,2,2,2,"scalar"))
    mixed=sut._source((2,1,2,2,2,"mixed_phi"))
    scalar_H=[e for e in scalar[2] if e.kind=="READ" and e.tensor==scalar[1].h]
    mixed_H=[e for e in mixed[2] if e.kind=="READ" and e.tensor==mixed[1].h]
    assert len(scalar_H)==len(mixed_H)
    assert any(a.phi_owner!=b.phi_owner or a.output_element!=b.output_element
               for a,b in zip(scalar_H,mixed_H))
    assert sut.evaluate((2,1,2,2,2,"mixed_phi"))[
        "semantic_oracle_disagreements"]==0


def test_rejects_nonregistered_graph_and_unsound_mutant_handling(monkeypatch):
    with pytest.raises(ValueError,match="undeclared"):
        sut._source((4,9,9,9,9,"invented"))
    original=sut.reference_event_oracle
    def dishonest(arch,events,offsets):
        s,i=original(arch,events,offsets)
        return True,i
    monkeypatch.setattr(sut,"reference_event_oracle",dishonest)
    with pytest.raises(AssertionError,match="independent oracle/compiler mismatch"):
        sut.evaluate(sut.CASES[0])


def test_saved_original_receipt_create_only_and_readonly_audit(
    tmp_path,complete_report
):
    path=tmp_path/"original"
    sut.create_only(complete_report,output=path,source_revision="a"*40)
    result=sut.audit(path)
    assert result["source_cases"]==24
    assert result["tested_trace_variants"]==216
    assert result["independent_oracle_disagreements"]==0
    assert result["original_compiler_or_vendor_calls_during_audit"]==0
    assert len(result["read_only_receipt_sha256"])==64
    with pytest.raises(FileExistsError,match="create-only"):
        sut.create_only(complete_report,output=path,source_revision="a"*40)


@pytest.mark.parametrize("tamper",[
    lambda a:a.update(total_trace_variants=215),
    lambda a:a["rows"][1]["outcomes"][0].update(safe=False),
    lambda a:a["rows"][3].update(semantic_oracle_disagreements=1),
    lambda a:a["rows"][4]["outcomes"][2].update(safe=True),
    lambda a:a.update(trained_agents=1),
    lambda a:a.update(source_revision="not-a-source-sha"),
])
def test_readonly_audit_rejects_modified_population_receipts(
    tmp_path,complete_report,tamper
):
    root=tmp_path/"study"
    sut.create_only(complete_report,output=root,source_revision="b"*40)
    file=root/"r3j_heldout_source_semantics.json"
    data=json.loads(file.read_text())
    tamper(data)
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="R3j"):
        sut.audit(root)
