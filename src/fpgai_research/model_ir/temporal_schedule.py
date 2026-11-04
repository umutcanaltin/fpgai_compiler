"""Finite counterexample-producing checker for proposed tensor-lifetime schedules.

The input is an explicit sequence of (row, operator-stage) compute events,
*not* an implicitly known-good emitter traversal. Each event must preserve
logical read-before-write, full intermediate-row lifetime, distinct
stage/row completion, and no overwrite of an unconsumed aliased address.

For a batch of B rows, every internal edge has B independent row slots; the
next B-row tile can reuse them only after every corresponding consumer has
completed. This checker is E0 schedule semantics only, not executable C++
equivalence, RTL or physical timing.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable

from fpgai_research.model_ir.row_chain import RowChainIR


@dataclass(frozen=True)
class ComputeEvent:
    row: int
    stage: int


def blocked_schedule(row_count: int, stage_count: int,
                     row_block: int) -> tuple[ComputeEvent, ...]:
    """Return a prespecified wavefront-blocked (not invented-cycle) schedule."""
    if any(type(x) is not int for x in (row_count, stage_count, row_block)):
        raise TypeError("Schedule dimensions must be integers")
    if row_count < 1 or stage_count < 2 or not 1 <= row_block <= row_count:
        raise ValueError("Unsupported row-block schedule dimensions")
    return tuple(
        ComputeEvent(row, stage)
        for start in range(0, row_count, row_block)
        for stage in range(stage_count)
        for row in range(start, min(start + row_block, row_count))
    )


def check_schedule(ir: RowChainIR, events: Iterable[ComputeEvent], *,
                   row_slots: int = 1, max_element_events: int = 200_000) -> dict:
    """Check an *independently proposed* schedule; return a counterexample.

    Each stage's output becomes visible after completing its reduction, and
    source stage-level input and output scratch are conservatively overlapping.
    An invalid schedule produces a deterministic first failure with event and
    address, not a false pass or a silently repaired schedule.
    """
    if not isinstance(ir, RowChainIR):
        raise TypeError("Expected typed RowChainIR")
    if type(row_slots) is not int or not 1 <= row_slots <= ir.row_count:
        raise ValueError("row_slots must be 1..row_count")
    if type(max_element_events) is not int or max_element_events < 1:
        raise ValueError("Positive element-event budget required")
    edges = ir.composed.boundaries
    count = len(ir.chain)
    if len(edges) != count-1:
        raise ValueError("Missing tensor boundaries in composed Model IR")
    for stage, (edge, pair) in enumerate(zip(edges, ir.scratch_by_tensor)):
        name, width = pair
        if (edge.tensor != name or edge.producer != ir.chain[stage]
                or edge.consumers != (ir.chain[stage+1],)
                or edge.shape != (ir.row_count, width)
                or width < 1 or edge.storage_layout != "row_buffer"
                or edge.exchange_policy != "producer_row_complete_then_consumer_row"):
            raise ValueError("Unsupported or inconsistent edge semantics")
    schedule = tuple(events)
    if any(not isinstance(e, ComputeEvent) for e in schedule):
        raise TypeError("All schedule events must be ComputeEvent")
    required = 2*ir.row_count*sum(width for _, width in ir.scratch_by_tensor)
    base = {
        "schema": "fpgai-research/independent-tensor-schedule-check/v1",
        "proposed_event_count": len(schedule),
        "required_stage_events": ir.row_count*count,
        "required_element_events": required,
        "row_slots": row_slots,
        "source_scratch_elements": row_slots*sum(width for _, width in ir.scratch_by_tensor),
        "native_vitis_calls": 0,
        "evidence": "E0-finite-candidate-schedule-semantics",
    }
    def invalid(reason, index, event=None, *, tensor=None, address=None,
                current_live_row=None):
        return {
            **base, "status": "REJECTED",
            "counterexample": {
                "reason": reason, "event_index": index,
                "row": event.row if event is not None else None,
                "stage": event.stage if event is not None else None,
                "tensor": tensor, "physical_address": address,
                "current_live_row": current_live_row,
            },
            "checked_element_events": operations,
        }
    if required > max_element_events:
        return {**base, "status": "BOUND_EXCEEDED",
                "max_element_events": max_element_events,
                "counterexample": None}
    done: set[tuple[int, int]] = set()
    live: dict[tuple[str, int], int] = {}
    operations = 0
    peak = 0
    trace = []
    for idx, e in enumerate(schedule):
        row, stage = e.row, e.stage
        if (type(row) is not int or type(stage) is not int
                or not 0 <= row < ir.row_count or not 0 <= stage < count):
            return invalid("INVALID_STAGE_OR_ROW", idx, e)
        if (row, stage) in done:
            return invalid("DUPLICATE_STAGE_EXECUTION", idx, e)
        # Before writing any output, *all* input row elements must be live.
        # A stage may overlap input and output scratch, but neither may be
        # silently aliased across graph edges.
        if stage:
            src_edge = edges[stage-1]
            for column in range(src_edge.shape[1]):
                slot_address = (row % row_slots)*src_edge.shape[1] + column
                occupied = live.get((src_edge.tensor, slot_address))
                if occupied != row:
                    return invalid("READ_BEFORE_PRODUCE_OR_STALE_ROW", idx, e,
                                   tensor=src_edge.tensor, address=slot_address,
                                   current_live_row=occupied)
        if stage < count-1:
            out_edge = edges[stage]
            for column in range(out_edge.shape[1]):
                address = (row % row_slots)*out_edge.shape[1] + column
                occupied = live.get((out_edge.tensor, address))
                if occupied is not None:
                    return invalid("OVERWRITE_UNCONSUMED_VALUE", idx, e,
                                   tensor=out_edge.tensor, address=address,
                                   current_live_row=occupied)
            # Stage-level output materialization, before releasing inputs:
            for column in range(out_edge.shape[1]):
                address = (row % row_slots)*out_edge.shape[1] + column
                live[(out_edge.tensor, address)] = row
                operations += 1
                trace.append(("produce", out_edge.tensor, row, stage, address))
            peak = max(peak, len(live))
        if stage:
            src_edge = edges[stage-1]
            for column in range(src_edge.shape[1]):
                address = (row % row_slots)*src_edge.shape[1] + column
                del live[(src_edge.tensor, address)]
                operations += 1
                trace.append(("consume", src_edge.tensor, row, stage, address))
        done.add((row, stage))
    if len(done) != ir.row_count*count:
        missing = next((ComputeEvent(row, stage)
                        for row in range(ir.row_count)
                        for stage in range(count)
                        if (row, stage) not in done), None)
        return invalid("MISSING_STAGE_EXECUTION", len(schedule), missing)
    if live:
        (tensor, address), occupied = next(iter(live.items()))
        return invalid("LIVE_VALUES_REMAIN", len(schedule),
                       tensor=tensor, address=address,
                       current_live_row=occupied)
    if operations != required:
        raise AssertionError("Internal element event count inconsistent")
    return {
        **base, "status": "VERIFIED",
        "checked_element_events": operations,
        "peak_live_elements": peak,
        "fully_consumed": True,
        "counterexample": None,
        "schedule_sha256": sha256(json.dumps(
            [(e.row, e.stage) for e in schedule],
            separators=(",", ":")).encode()).hexdigest(),
        "semantic_trace_sha256": sha256(json.dumps(
            trace, separators=(",", ":")).encode()).hexdigest(),
    }
