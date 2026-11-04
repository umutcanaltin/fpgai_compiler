"""Adversarial accounting controls from the independent scientific review.

No GitHub Actions, no vendor tools, no RL training, no frozen experiments.
"""
from __future__ import annotations

import json

import pytest

from fpgai_research.model_ir.branch_matmul_contract import BranchEvent
from fpgai_research.model_ir.element_scratch_allocation import (
    allocate_minimum_slots, extract_intervals, verify_allocation,
)
from fpgai_research.model_ir.source_allocation_baselines import (
    reader_inplace_intervals, source_baseline_ladder, tensor_blocks,
    whole_tensor_first_fit, whole_tensor_peak_lower_bound, _peak,
)


def _synthetic():
    # H stays live across the reader's FIRST MAC. Committed A starts
    # only after the final MAC. In-place A starts at the first MAC.
    return (
        BranchEvent("COMMIT", "H", 0),
        BranchEvent("COMMIT", "H", 1),
        BranchEvent("READ", "H", 0, "reader0", 0),
        BranchEvent("READ", "H", 1, "reader0", 0),
        BranchEvent("READ", "H", 0, "reader0", 0),
        BranchEvent("COMMIT", "A", 0),
        BranchEvent("READ", "A", 0, "relu", 0),
    )


def test_counterfactual_first_mac_changes_liveness_not_source_event_order():
    events = _synthetic()
    original = extract_intervals(events)
    inplace = reader_inplace_intervals(
        events, original, producer_tensor="H",
        reader_outputs={"reader0": "A"})
    assert _peak(original) == 2
    assert _peak(inplace) == 3
    assert [iv.commit_index for iv in original if iv.tensor == "A"] == [5]
    assert [iv.commit_index for iv in inplace if iv.tensor == "A"] == [2]
    assert allocate_minimum_slots(events).unit_slots == 2


def test_first_mac_at_last_h_read_does_not_reuse_same_event_slot_without_proof():
    events = (
        BranchEvent("COMMIT", "H", 0),
        BranchEvent("READ", "H", 0, "reader0", 0),
        BranchEvent("COMMIT", "A", 0),
        BranchEvent("READ", "A", 0, "relu", 0),
    )
    intervals = extract_intervals(events)
    hypothetical = reader_inplace_intervals(
        events, intervals, producer_tensor="H",
        reader_outputs={"reader0": "A"})
    assert _peak(intervals) == 1
    # A becomes live *during* event 1; absent a proven micro-operation
    # order, H's last READ and A's first MAC cannot share one slot.
    assert _peak(hypothetical) == 2


def test_tensor_firstfit_is_feasible_and_lb_is_not_called_a_real_placement():
    events = _synthetic()
    blocks = tensor_blocks(extract_intervals(events))
    slots, bases = whole_tensor_first_fit(blocks)
    lower = whole_tensor_peak_lower_bound(blocks)
    assert lower <= slots <= 3
    # Separate tensor base is conservative and source-replay accepted.
    assignment = {
        (block.tensor, i): bases[block.tensor] + i
        for block in blocks for i in range(block.size)
    }
    assert verify_allocation(
        events, assignment, unit_slots=slots
    )["status"] == "EXACT_SOURCE_SLOT_REPLAY_VERIFIED"


def test_full_memory_ladder_distinguishes_real_and_hypothetical_counts():
    result = source_baseline_ladder(
        _synthetic(), producer_tensor="H",
        reader_outputs={"reader0": "A"},
        accumulator_elements=4, counter_elements=4,
    )
    assert result["element_exact_committed_value_slots"] == 2
    assert result["hypothetical_inplace_reader_value_slots"] == 3
    assert result["actual_r2h_all_declared_mutable_elements_element_exact"] == 10
    assert result["hypothetical_fused_inplace_plus_counters_elements"] == 7
    assert result["physical_BRAM_measured"] is False
    assert result["native_vitis_calls"] == 0
    assert result["native_vivado_calls"] == 0


def test_reader_output_completeness_and_malformed_lifetimes_fail_closed():
    events = _synthetic()
    intervals = extract_intervals(events)
    with pytest.raises(ValueError, match="all reader accumulations"):
        reader_inplace_intervals(
            events, intervals, producer_tensor="H",
            reader_outputs={"reader1": "A"})
    with pytest.raises(ValueError, match="distinct"):
        reader_inplace_intervals(
            events, intervals, producer_tensor="H",
            reader_outputs={"reader0": "A", "reader1": "A"})
    with pytest.raises(ValueError, match="nonnegative"):
        source_baseline_ladder(
            events, producer_tensor="H", reader_outputs={"reader0": "A"},
            accumulator_elements=-1, counter_elements=0,
        )


@pytest.fixture(scope="module")
def original_catalog():
    from fpgai_research.experiments.cst_r2q_accounting_baselines import (
        calculate_catalog,
    )
    return calculate_catalog()


def test_original_all_36_fanout_mapping_cases_have_feasible_baseline_ladder(
        original_catalog):
    assert len(original_catalog) == 36
    assert len(set(row["case"] for row in original_catalog)) == 36
    for row in original_catalog:
        d = row["disjoint_no_reuse_slots"]
        t = row["whole_tensor_first_fit_feasible_slots"]
        lb = row["whole_tensor_concurrency_lower_bound_slots"]
        e = row["element_exact_committed_value_slots"]
        i = row["hypothetical_inplace_reader_value_slots"]
        assert e <= lb <= t <= d
        assert e <= i
        assert row["value_interval_count"] == d
        assert row["fixed_order_optimal_equals_peak"] is True
        assert row["actual_r2h_all_declared_mutable_elements_element_exact"] == (
            e + 2 * row["fanout_readers"] * row["shape"][0] * row["shape"][3])
        assert len(row["original_r2h_disjoint_cpp_sha256"]) == 64
        assert len(row["original_r2h_optimized_cpp_sha256"]) == 64
        assert row["gcc_executed_by_this_analysis"] is False
        assert row["physical_BRAM_measured"] is False


def test_reviewers_original_small_counterexample_is_explicit_not_ignored(
        original_catalog):
    cells = [r for r in original_catalog
             if r["fanout_readers"] == 2 and r["shape"] == [2, 2, 3, 4]]
    scalar = next(r for r in cells if r["policy"] == "scalar")
    all_pes2 = next(r for r in cells if r["policy"] == "all_pes2")
    # Reviewer-reported under the two accounting conventions:
    assert scalar["element_exact_committed_value_slots"] == 17
    assert all_pes2["element_exact_committed_value_slots"] == 16
    assert scalar["hypothetical_inplace_reader_value_slots"] == 19
    assert all_pes2["hypothetical_inplace_reader_value_slots"] == 19


def test_source_summary_keeps_phi_and_baseline_denominators_separate(
        original_catalog):
    from fpgai_research.experiments.cst_r2q_accounting_baselines import (
        summarize, SCHEMA,
    )
    result = summarize(original_catalog, "a" * 40)
    assert result["schema"] == SCHEMA
    assert result["original_catalog_cases"] == 36
    assert result["phi_intervention_count"] == 27
    assert sum(result["phi_effects"][prefix + "_" + suffix]
               for suffix in ("reduced", "tied", "increased")
               for prefix in ("committed",)) == 27
    assert sum(result["phi_effects"][prefix + "_" + suffix]
               for suffix in ("reduced", "tied", "increased")
               for prefix in ("inplace",)) == 27
    assert result["new_vitis_calls"] == 0
    assert result["new_vivado_calls"] == 0


def test_original_source_study_create_only_with_read_only_recompute(
        original_catalog, tmp_path):
    from fpgai_research.experiments import cst_r2q_accounting_baselines as study
    target = tmp_path / "r2q"
    result = study.run(output=target)
    assert result["original_catalog_cases"] == 36
    before = {f.name: f.read_bytes() for f in target.iterdir()}
    read_only = study.audit(target)
    assert read_only["audited_cases"] == 36
    assert read_only["fresh_original_gcc_calls"] == 0
    assert before == {f.name: f.read_bytes() for f in target.iterdir()}
    with pytest.raises(FileExistsError, match="create-only"):
        study.run(output=target)


def test_baseline_results_tampering_is_detected(tmp_path):
    from fpgai_research.experiments import cst_r2q_accounting_baselines as study
    target = tmp_path / "r2q"
    study.run(output=target)
    path = target / "cases.json"
    cases = json.loads(path.read_text())
    cases[0]["whole_tensor_first_fit_feasible_slots"] += 1
    path.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="recomputation"):
        study.audit(target)
