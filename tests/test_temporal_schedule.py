"""Falsifiable finite schedule semantics with real positive/negative controls.

These tests intentionally compare the same unmodified operator-local Phi
against distinct proposed cross-operator schedules. A per-operator refinement
checker does not validate the independent producer/consumer event order.
"""
from dataclasses import replace

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.row_chain import resolve_row_chain
from fpgai_research.model_ir.temporal_schedule import (
    ComputeEvent, blocked_schedule, check_schedule,
)
from fpgai_research.model_ir.semantics import verify_refinement


def model_ir(rows=5):
    f = make_chain(rows, (3, 5, 4, 2))
    return resolve_row_chain(resolve(f, ArchitectureRequest()))


@pytest.mark.parametrize("rows,block", [
    (2, 1), (3, 1), (3, 2), (5, 2), (5, 3), (5, 5),
])
def test_valid_blocked_schedules_have_bounded_independent_certificates(rows, block):
    ir = model_ir(rows)
    schedule = blocked_schedule(rows, len(ir.chain), block)
    evidence = check_schedule(ir, schedule, row_slots=block)
    assert evidence["status"] == "VERIFIED"
    assert evidence["checked_element_events"] == (
        2*rows*sum(n for _, n in ir.scratch_by_tensor))
    assert evidence["source_scratch_elements"] == block * sum(
        n for _, n in ir.scratch_by_tensor)
    assert evidence["fully_consumed"]
    assert evidence["peak_live_elements"] <= evidence["source_scratch_elements"]


def test_insufficient_capacity_produces_actual_aliasing_counterexample():
    ir = model_ir(5)
    schedule = blocked_schedule(5, len(ir.chain), 2)
    witness = check_schedule(ir, schedule, row_slots=1)
    assert witness["status"] == "REJECTED"
    assert witness["counterexample"]["reason"] == "OVERWRITE_UNCONSUMED_VALUE"
    assert witness["counterexample"]["tensor"] == "h0"
    assert witness["counterexample"]["event_index"] == 1


def test_bad_event_order_missing_duplicate_and_wrong_stage_are_rejected():
    ir = model_ir(3)
    normal = blocked_schedule(3, len(ir.chain), 2)
    cases = [
        ((ComputeEvent(0, 1),) + normal, "READ_BEFORE_PRODUCE_OR_STALE_ROW"),
        ((normal[0],) + normal, "DUPLICATE_STAGE_EXECUTION"),
        (normal[:-1], "MISSING_STAGE_EXECUTION"),
        ((ComputeEvent(-1, 0),) + normal, "INVALID_STAGE_OR_ROW"),
    ]
    for schedule, reason in cases:
        answer = check_schedule(ir, schedule, row_slots=2)
        assert answer["status"] == "REJECTED"
        assert answer["counterexample"]["reason"] == reason
        assert type(answer["counterexample"]["event_index"]) is int


def test_bound_exceeded_not_a_false_certificate():
    ir = model_ir(5)
    schedule = blocked_schedule(5, len(ir.chain), 2)
    found = check_schedule(ir, schedule, row_slots=2, max_element_events=1)
    assert found["status"] == "BOUND_EXCEEDED"
    with pytest.raises(ValueError, match="row_slots"):
        check_schedule(ir, schedule, row_slots=0)


def test_operator_level_refinement_remains_unchanged_for_invalid_schedule():
    ir = model_ir(3)
    valid = blocked_schedule(3, len(ir.chain), 2)
    invalid = (ComputeEvent(0, 1),) + valid
    # This is a scoped ablation: the existing per-op refinement does not
    # even receive the candidate cross-operator event ordering.
    model_only = verify_refinement(ir.composed.lowered)
    assert model_only["stage_invariants_verified"]
    assert check_schedule(ir, valid, row_slots=2)["status"] == "VERIFIED"
    assert check_schedule(ir, invalid, row_slots=2)["status"] == "REJECTED"
