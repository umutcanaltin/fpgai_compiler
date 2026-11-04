"""R3g C1: independent source event semantics on all adjacent swap mutations.

This is finite trace differential checking, not a general proof or physical
hardware verification. It never trains RL policies or calls native vendor tools.
"""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments import cst_r3g_c1_independent_event_semantics as sut


@pytest.fixture(scope="module")
def complete_reference_report():
    return sut.run()


@pytest.mark.parametrize("case", sut.CASES)
def test_complete_finite_trace_differential_oracle(case):
    observed = sut.evaluate(case)
    assert observed["case"] == list(case)
    assert observed["source_event_count"] > 0
    assert observed["adjacent_swaps"] == observed["source_event_count"]-1
    assert observed["variant_count"] == observed["adjacent_swaps"]+6
    assert observed["false_acceptances"] == 0
    assert observed["false_rejections"] == 0
    assert observed["value_safety_and_issue_both_true"] >= 1
    assert observed["value_safety_and_issue_both_false"] >= 1
    assert sum(observed[k] for k in (
        "value_safety_and_issue_both_true",
        "value_safe_but_issue_nonconformant",
        "value_unsafe_but_issue_conformant",
        "value_safety_and_issue_both_false",
    )) == observed["variant_count"]


def test_source_event_reference_separates_value_safety_and_phi_issue():
    arch, original, offsets = sut._case(sut.CASES[0])
    assert sut.reference_event_oracle(arch,original,offsets) == (True,True)
    reads = [i for i,e in enumerate(original)
             if e.kind == "READ" and e.tensor == "h"]
    a,b=reads[0],reads[1]
    changed=list(original)
    changed[a],changed[b]=changed[b],changed[a]
    assert sut.reference_event_oracle(arch,tuple(changed),offsets) == (
        True,False)


def test_reference_oracle_rejects_missing_read_and_invalid_offset():
    arch, original, offsets=sut._case(sut.CASES[1])
    assert sut.reference_event_oracle(arch,original[:-1],offsets)[0] is False
    malformed={**offsets, "new_tensor":0}
    assert sut.reference_event_oracle(arch,original,malformed)==(False,False)


def test_restricted_case_registry_refuses_unknown_graph():
    with pytest.raises(ValueError,match="unregistered"):
        sut.evaluate((9,1,1,1,1,1))


def test_create_only_and_read_only_saved_semantic_receipt(tmp_path,complete_reference_report):
    report=complete_reference_report
    assert report["schema"]==sut.SCHEMA
    assert report["original_source_cases"]==5
    assert report["tested_event_trace_variants"]>5
    assert report["oracle_disagreements"]==0
    assert report["original_gcc_calls"]==0
    assert report["native_vitis_calls"]==0
    assert report["rl_training_calls"]==0
    folder=tmp_path/"evidence"
    sut.create_only(report,output=folder,source_revision="a"*40)
    audit=sut.audit(folder)
    assert audit["tested_event_trace_variants"]==report["tested_event_trace_variants"]
    assert audit["independent_oracle_disagreements"]==0
    assert len(audit["source_receipt_sha256"])==64
    with pytest.raises(FileExistsError):
        sut.create_only(report,output=folder,source_revision="a"*40)


@pytest.mark.parametrize("mutation", [
    lambda a: a.update(oracle_disagreements=1),
    lambda a: a["rows"][0].update(variant_count=0),
    lambda a: a["rows"][1].update(false_rejections=2),
    lambda a: a.update(tested_event_trace_variants=0),
    lambda a: a["rows"][3].update(adjacent_swaps=1000),
    lambda a: a.update(source_revision="invalid"),
])
def test_saved_oracle_denominator_mutation_rejected(tmp_path,mutation,complete_reference_report):
    folder=tmp_path/"r3g"
    report=complete_reference_report
    sut.create_only(report,output=folder,source_revision="b"*40)
    file=folder/"r3g_independent_event_oracle.json"
    changed=json.loads(file.read_text())
    mutation(changed)
    file.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="R3g:"):
        sut.audit(folder)
