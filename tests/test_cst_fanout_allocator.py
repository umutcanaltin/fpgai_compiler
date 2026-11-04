"""L5c: independently enumerate all finite E0 tensor-base assignments.

This proves optimality ONLY inside the sequential consumer-ACK interval
model; no real C++ buffer allocation or hardware port assumptions.
"""
from itertools import product

import pytest

from fpgai_research.experiments.cst_fanout_campaign import branch_graph
from fpgai_research.model_ir.fanout_allocator import exact_fanout_slot_plan
from fpgai_research.model_ir.fanout_lifetime import (
    canonical_fanout_events, verify_fanout_events,
)


def brute_force_minimum(model, tensors, *, max_slots=8):
    """Independent checker: enumerate offsets then run existing E0 event replay.

    Intentionally does not use allocator's derived lifetime intervals or
    collision predicate. Small domains only.
    """
    tensors = tuple(sorted(tensors))
    sizes = {t: model.tensor(t).shape[0] * model.tensor(t).shape[1]
             for t in tensors}
    for capacity in range(max(sizes.values()), max_slots + 1):
        options = [range(capacity - sizes[t] + 1) for t in tensors]
        for offsets in product(*options):
            mapping = dict(zip(tensors, offsets))
            events = canonical_fanout_events(model, mapping)
            if verify_fanout_events(model, mapping, events)["status"] == "VERIFIED":
                return capacity
    return None


@pytest.mark.parametrize("names", [
    ("h", "g"),
    ("h", "a"),
    ("a", "b"),
    ("h", "a", "b", "g"),
])
def test_exact_minimum_matches_independent_full_offset_enumeration(names):
    model = branch_graph()
    planned = exact_fanout_slot_plan(model, names, max_slots=8)
    reference = brute_force_minimum(model, names)
    assert planned["status"] == "OPTIMAL_E0"
    assert planned["physical_slots"] == reference
    assert planned["optimal_within_model"] is True
    assert planned["original_vitis_calls"] == 0
    assert planned["event_count"] > 0
    assert planned["verification_sha256"] == planned["logical_event_sha256"]
    assert planned["physical_slots"] <= planned["dense_disjoint_slots"]
    assert "No hardware BRAM" in " ".join(planned["limits"])


def test_two_reader_scratch_is_reused_only_after_last_reader():
    model = branch_graph()
    planned = exact_fanout_slot_plan(model, ("h", "g"), max_slots=4)
    assert planned["physical_slots"] == 2
    assert planned["offsets"]["h"] == planned["offsets"]["g"]
    assert verify_fanout_events(
        model, planned["offsets"],
        canonical_fanout_events(model, planned["offsets"])
    )["status"] == "VERIFIED"


def test_no_solution_and_search_budget_must_never_be_reported_optimal():
    graph = branch_graph()
    tiny = exact_fanout_slot_plan(graph, ("h", "a"), max_slots=2)
    assert tiny["status"] == "INFEASIBLE_WITHIN_CAPACITY"
    assert tiny["physical_slots"] is None
    limited = exact_fanout_slot_plan(
        graph, ("h", "a", "b", "g"), max_slots=8, max_candidates=1)
    assert limited["status"] == "UNKNOWN_BUDGET"
    assert limited["physical_slots"] is None
    assert not limited["optimal_within_model"]
    event_bound = exact_fanout_slot_plan(graph, ("h", "a"), max_events=1)
    assert event_bound["status"] == "UNKNOWN_EVENT_BOUND"
    assert event_bound["physical_slots"] is None


def test_invalid_inputs_reject_instead_of_silent_repair():
    graph = branch_graph()
    for tensors in ((), ("h", "h"), ("x",), ("out",), ("missing",)):
        with pytest.raises((ValueError, KeyError)):
            exact_fanout_slot_plan(graph, tensors)
    with pytest.raises(ValueError):
        exact_fanout_slot_plan(graph, ("h",), max_slots=True)
    with pytest.raises(ValueError):
        exact_fanout_slot_plan(graph, ("h",), max_candidates=0)
    with pytest.raises(ValueError):
        exact_fanout_slot_plan(graph, ("h",), max_events=0)
    with pytest.raises(ValueError):
        exact_fanout_slot_plan(graph, ("h", "a", "b", "g", "x", "out", "extra"))
