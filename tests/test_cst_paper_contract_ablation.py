"""Ablation isolates Phi source issue-order checks from value safety.

This is a bounded internal source comparison, not an external baseline,
a numeric output-equivalence benchmark or a trained search experiment.
"""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments import cst_paper_contract_ablation as sut


def fixture_source_profiles():
    shapes = ((1, 2, 2, 2), (2, 1, 2, 3),
              (1, 2, 3, 2), (2, 2, 3, 3))
    rows = []
    for readers in (2, 3, 4):
        for shape in shapes:
            for policy in ("scalar", "mixed_phi"):
                outcomes = []
                for label in sut.LABELS:
                    outcomes.append({
                        "name": label,
                        "safe": label in ("canonical", "safe_wrong_issue_order"),
                        "phi_issue_conformant": (
                            True if label == "canonical"
                            else False if label == "safe_wrong_issue_order"
                            else None),
                        "source_trace_sha256": "a" * 64,
                        "source_offsets_sha256": "b" * 64,
                    })
                rows.append({
                    "case": [readers, *shape, policy],
                    "outcomes": outcomes,
                })
    return {
        "schema": sut.INPUT_SCHEMA,
        "prespecified_source_cases": 24,
        "total_trace_variants": 216,
        "variants_per_case": 9,
        "matched_independent_oracle": 216,
        "independent_oracle_disagreements": 0,
        "original_vitis_calls": 0,
        "original_vivado_calls": 0,
        "trained_agents": 0,
        "rows": rows,
    }


def test_identical_event_population_differentiates_contract_predicates():
    result = sut.summarize(fixture_source_profiles())
    assert result["compared_event_traces"] == 216
    assert result["source_graph_mapping_profiles"] == 24
    assert result["declared_valid_traces"] == 24
    assert result["selected_invalid_traces"] == 192
    assert result["value_safety_only_accepts"] == 48
    assert result["full_contract_accepts"] == 24
    assert result["value_only_accepts_invalid_issue_controls"] == 24
    assert result["full_contract_accepts_invalid_issue_controls"] == 0
    assert result["external_compiler_baseline"] == "NOT_RUN"
    assert result["hardware_BRAM"] is None
    assert all(r["population"] == 24 for r in result["by_mutation_class"])


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(original_vitis_calls=1),
    lambda data: data.update(total_trace_variants=215),
    lambda data: data["rows"][0]["outcomes"][0].update(safe=False),
    lambda data: data["rows"][0]["outcomes"][1].update(
        phi_issue_conformant=True),
    lambda data: data["rows"][0]["outcomes"][2].update(safe=True),
    lambda data: data["rows"][1].update(case=data["rows"][0]["case"]),
    lambda data: data["rows"][2]["outcomes"][1].update(
        source_trace_sha256="not a hash"),
])
def test_bad_population_or_dishonest_verdict_fails_closed(mutation):
    data = fixture_source_profiles()
    mutation(data)
    with pytest.raises(ValueError):
        sut.summarize(data)


def test_create_only_receipts_and_no_original_rewriting(tmp_path):
    original = fixture_source_profiles()
    before = json.dumps(original, sort_keys=True)
    report = sut.summarize(original)
    dest = tmp_path / "new"
    sut.export_create_only(report, output=dest,
                           source_revision="f" * 40)
    assert json.loads((dest / "summary.json").read_text())[
        "experiment_source_revision"] == "f" * 40
    assert (dest / "observations.csv").is_file()
    assert len(json.loads((dest / "derived_sha256.json").read_text())) == 2
    assert before == json.dumps(original, sort_keys=True)
    with pytest.raises(FileExistsError):
        sut.export_create_only(report, output=dest,
                               source_revision="f" * 40)
    with pytest.raises(ValueError):
        sut.export_create_only(report, output=tmp_path / "bad",
                               source_revision="unknown")
