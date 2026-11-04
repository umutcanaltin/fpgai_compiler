"""R3e C1 bounded independent oracle and adverse value-event mutation tests."""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments import cst_r3e_c1_fanout_counterexamples as study


@pytest.mark.parametrize("case", study.CASES)
def test_original_finite_oracle_and_all_rejected_mutation_classes(case):
    row = study.one_case(case)
    assert row["readers"] == case[0]
    assert row["shape"] == list(case[1:5])
    assert row["reader0_pes"] == case[5]
    assert row["valid_typed_event_stream"] is True
    assert row["valid_phi_issue"] is True
    assert row["source_unit_slots"] == row["independently_exhaustive_minimum_unit_slots"]
    assert row["source_unit_slots"] >= 1
    assert row["mutant_denominator"] == 5
    assert set(row["rejected_mutant_reasons"]) == set(study.MUTANTS)
    assert row["rejected_mutant_reasons"]["early_consumer_read"] == (
        "FANOUT_READ_BEFORE_READY_OR_STALE")
    assert row["rejected_mutant_reasons"]["premature_producer_commit"] == (
        "PREMATURE_H_PRODUCER_COMMIT")
    assert row["rejected_mutant_reasons"]["wrong_phi_owner"] == (
        "WRONG_FANOUT_PHI_COORDINATE_OR_OWNER")
    assert row["rejected_mutant_reasons"]["wrong_phi_issue_order"] == (
        "FANOUT_PHI_ISSUE_OR_OWNER_ORDER_MISMATCH")
    assert row["rejected_mutant_reasons"]["overwrite_live_value"] in (
        "OVERWRITE_LIVE_VALUE", "READ_STALE_OR_UNREADY_VALUE")
    assert len(row["source_cpp_sha256"]) == 64
    assert len(row["original_source_event_sha256"]) == 64


def test_full_prespecified_c1_mutation_denominator_not_cherrypicked():
    result = study.run()
    assert result["schema"] == study.SCHEMA
    assert result["cases_prespecified"] == len(study.CASES) == 5
    assert result["mutants_per_case"] == len(study.MUTANTS) == 5
    assert result["rejected_mutants"] == 25
    assert result["oracle_agreement_cases"] == 5
    assert len(result["rows"]) == 5
    assert result["new_gcc_calls"] == 0
    assert result["new_vitis_calls"] == 0
    assert result["new_vivado_calls"] == 0
    assert result["trained_rl_agents"] == 0
    assert "finite" in result["formal_scope"]


def test_broken_reference_oracle_must_not_be_counted_as_evidence(monkeypatch):
    monkeypatch.setattr(study, "exhaustive_minimum_slots",
                        lambda *_a, **_kw: 999)
    with pytest.raises(AssertionError, match="independent finite coloring oracle"):
        study.one_case(study.CASES[0])


def test_corrupted_alias_not_silently_reclassified(monkeypatch):
    original = study.verify_allocation
    def false_accepted(events, addresses, *, unit_slots):
        try:
            return original(events, addresses, unit_slots=unit_slots)
        except ValueError:
            return {"status": "EXACT_SOURCE_SLOT_REPLAY_VERIFIED"}
    monkeypatch.setattr(study, "verify_allocation", false_accepted)
    with pytest.raises(AssertionError, match="overlapping live scratch alias"):
        study.one_case(study.CASES[0])


def test_create_only_refuses_original_overwrite_and_bad_revision(tmp_path):
    result = {
        "schema": study.SCHEMA,
        "cases_prespecified": 5,
        "mutants_per_case": 5,
        "rejected_mutants": 25,
    }
    output = tmp_path / "original"
    with pytest.raises(ValueError, match="revision"):
        study.create_only(output, result, git_revision="wrong")
    assert not output.exists()
    study.create_only(output, result, git_revision="a" * 40)
    data = json.loads((output / "r3e_bounded_c1_results.json").read_text())
    assert data["source_revision"] == "a" * 40
    assert data["rejected_mutants"] == 25
    with pytest.raises(FileExistsError):
        study.create_only(output, result, git_revision="a" * 40)


def test_unknown_population_cannot_be_smuggled_into_prespecified_oracle():
    with pytest.raises(ValueError, match="prespecified"):
        study.one_case((5,1,1,2,1,1))



def test_read_only_original_oracle_audit_rejects_mutated_receipt(tmp_path):
    full = study.run()
    target = tmp_path / "evidence"
    study.create_only(target, full, git_revision="c" * 40)
    original = study.audit(target)
    assert original["original_case_count"] == len(study.CASES)
    assert original["rejected_mutants"] == 25
    assert original["independent_minimum_slot_oracle_agreements"] == 5
    assert original["new_compiler_gcc_vitis_vivado_calls"] == 0
    receipt = target / "r3e_bounded_c1_results.json"
    corrupted = json.loads(receipt.read_text())
    corrupted["rows"][0]["rejected_mutant_reasons"]["wrong_phi_owner"] = (
        "DIFFERENT_OR_UNEXPLAINED_REJECTION")
    receipt.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="saved original counterexample"):
        study.audit(target)
