"""Exact bounded scratch allocation for the E0 fan-out consumer-ACK model.

Finds minimum contiguous tensor-base address space by exhaustive finite
enumeration. Element lifetimes come from canonical COMMIT / last CONSUME
events or an explicitly supplied, independently checked source event order,
*not* the HLS/RTL clock schedule. Returned plans are independently
replayed through fanout_lifetime.verify_fanout_events.

This is an E0 address/alias lower-bound experiment. It does not model
simultaneous port access, dataflow concurrency, banking, cycle timing,
per-Phi reductions, or actual C++ storage allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from fpgai_research.model_ir.contracts import FunctionalModel
from fpgai_research.model_ir.fanout_lifetime import (
    LifetimeEvent, canonical_fanout_events, fanout_contract, verify_fanout_events,
)

SCHEMA = "fpgai-research/cst-fanout-exact-slot-allocation-e0/v1"


@dataclass(frozen=True)
class _Interval:
    tensor: str
    element: int
    begin: int
    end: int


def _lifetime_intervals(model: FunctionalModel, tensors: tuple[str, ...],
                        *, max_events: int,
                        event_sequence: tuple[LifetimeEvent, ...] | None = None):
    # Use disjoint addresses for initial admission. All tensor names are
    # independently verified by the production E0 fan-out contract.
    from math import prod
    disjoint = {}
    sizes = {}
    for name in tensors:
        sizes[name] = prod(model.tensor(name).shape)
    offset = 0
    for name in tensors:
        disjoint[name] = offset
        offset += sizes[name]
    fanout_contract(model, disjoint)
    events = (canonical_fanout_events(model, disjoint)
              if event_sequence is None else event_sequence)
    if len(events) > max_events:
        return None, sizes, len(events), "UNKNOWN_EVENT_BOUND"
    checked = verify_fanout_events(model, disjoint, events, max_events=max_events)
    if checked["status"] != "VERIFIED":
        raise ValueError("Fan-out event schedule is not valid: " +
                             str(checked.get("reason")))
    started = {}
    read = {}
    for index, event in enumerate(events):
        key = (event.tensor, event.element)
        if event.kind == "COMMIT":
            if key in started:
                raise AssertionError("Duplicate producer in canonical schedule")
            started[key] = index
        elif event.kind == "CONSUME":
            if key not in started:
                raise AssertionError("Read precedes production in canonical schedule")
            read[key] = index
        else:
            raise AssertionError("Unknown canonical E0 event")
    if set(started) != set(read):
        raise AssertionError("Unreleased canonical tensor values")
    intervals = tuple(_Interval(name, element, started[(name, element)],
                                read[(name, element)])
                      for name in tensors for element in range(sizes[name]))
    return intervals, sizes, len(events), checked["event_sha256"]


def exact_fanout_slot_plan(
    model: FunctionalModel,
    tensor_names: tuple[str, ...],
    *,
    max_slots: int = 64,
    max_candidates: int = 100_000,
    max_events: int = 20_000,
    event_sequence: tuple[LifetimeEvent, ...] | None = None,
) -> dict:
    """Exhaustively find the first minimum-width contiguous offset layout.

    `max_candidates` is an actual explored-assignment-node budget. Reaching
    it without completing the search returns UNKNOWN_BUDGET, NEVER 'optimal'.
    `max_slots` is an explicit physical address search cap, not a device
    capacity or a guaranteed mapping to on-chip BRAMs. When supplied,
    `event_sequence` must be a full typed schedule accepted by the existing
    E0 checker against nonoverlapping source addresses.
    """
    model.validate()
    if (type(tensor_names) is not tuple or not tensor_names or
            any(type(name) is not str or not name for name in tensor_names) or
            len(set(tensor_names)) != len(tensor_names)):
        raise ValueError("Distinct nonempty tensor identities required")
    if len(tensor_names) > 6:
        raise ValueError("At most six tensors supported in bounded E0 allocation")
    if any(type(v) is not int or v < 1 for v in
           (max_slots, max_candidates, max_events)):
        raise ValueError("Positive integer search and event bounds required")
    if event_sequence is not None and (type(event_sequence) is not tuple or
            any(not isinstance(x, LifetimeEvent) for x in event_sequence)):
        raise TypeError("An explicit schedule must be immutable typed LifetimeEvents")
    tensors = tuple(sorted(tensor_names))
    intervals, sizes, count, event_digest = _lifetime_intervals(
        model, tensors, max_events=max_events, event_sequence=event_sequence)
    total = sum(sizes.values())
    base = {
        "schema": SCHEMA,
        "status": "UNKNOWN",
        "tensors": list(tensors),
        "element_count": total,
        "event_count": count,
        "max_slots": max_slots,
        "max_candidates": max_candidates,
        "max_events": max_events,
        "logical_event_sha256": event_digest if intervals is not None else None,
        "event_schedule": ("canonical-stage-ACK" if event_sequence is None
                           else "explicit-source-order-ACK"),
        "original_vitis_calls": 0,
        "evidence": "E0-abstract-consumer-ACK-interval-allocation-only",
        "limits": [
            "No generated branching C++ source or executed port accesses",
            "No Phi-conditioned reductions, clock-order or bank-port model",
            "Optimal ONLY over finite contiguous-base E0 address assignments",
            "No hardware BRAM, timing, throughput or routed resource evidence",
        ],
    }
    if intervals is None:
        return {**base, "status": "UNKNOWN_EVENT_BOUND",
                "examined_candidates": 0, "physical_slots": None,
                "offsets": None, "optimal_within_model": False}

    by_tensor = {name: tuple(x for x in intervals if x.tensor == name)
                 for name in tensors}
    examined = 0
    exhausted = False

    def conflict(name, offset, other, other_offset):
        # One physical address may be reused only when *all* readers have
        # acknowledged the old value. Same-step tie remains conflicting.
        for left in by_tensor[name]:
            address = offset + left.element
            for right in by_tensor[other]:
                if (address == other_offset + right.element and
                        left.begin <= right.end and right.begin <= left.end):
                    return True
        return False

    def search(capacity: int):
        nonlocal examined, exhausted
        assigned = {}
        def visit(index):
            nonlocal examined, exhausted
            if index == len(tensors):
                return dict(assigned)
            name = tensors[index]
            for offset in range(capacity - sizes[name] + 1):
                if examined >= max_candidates:
                    exhausted = True
                    return None
                examined += 1
                if any(conflict(name, offset, other, assigned[other])
                       for other in assigned):
                    continue
                assigned[name] = offset
                found = visit(index + 1)
                if found is not None:
                    return found
                del assigned[name]
                if exhausted:
                    return None
            return None
        return visit(0)

    # Capacity lower bound is max tensor extent; larger counts are checked
    # in ascending order, making first complete feasible answer optimal for
    # this precise E0 allocation model.
    for slots in range(max(sizes.values()), max_slots + 1):
        solution = search(slots)
        if solution is not None:
            chosen_events = (canonical_fanout_events(model, solution)
                             if event_sequence is None else event_sequence)
            proof = verify_fanout_events(
                model, solution, chosen_events, max_events=max_events)
            if proof["status"] != "VERIFIED":
                raise AssertionError("Chosen allocation failed independent E0 replay")
            return {
                **base, "status": "OPTIMAL_E0",
                "offsets": solution, "physical_slots": slots,
                "dense_disjoint_slots": total,
                "optimal_within_model": True,
                "examined_candidates": examined,
                "verification_sha256": proof["event_sha256"],
                "allocation_sha256": sha256(json.dumps(
                    {"tensors": tensors, "slots": slots,
                     "offsets": solution, "events": event_digest},
                    sort_keys=True).encode()).hexdigest(),
            }
        if exhausted:
            return {**base, "status": "UNKNOWN_BUDGET",
                    "offsets": None, "physical_slots": None,
                    "optimal_within_model": False,
                    "examined_candidates": examined}
    return {
        **base, "status": "INFEASIBLE_WITHIN_CAPACITY",
        "offsets": None, "physical_slots": None,
        "optimal_within_model": False,
        "examined_candidates": examined,
        "searched_capacity_lower_bound": max(sizes.values()),
    }
