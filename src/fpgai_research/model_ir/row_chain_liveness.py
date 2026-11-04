"""Independent bounded E0 verification of CST row-chain storage lifetimes.

Models actual value liveness per tensor-element address across the explicit
sequential per-row operator schedule. No logical value may overwrite a live
physical address; each producer value must be consumed exactly once by the
next stage before its address can be reused for the following logical row.

This verifies a *finite row-chain scheduling contract*, not arbitrary
generated C++, bit-exact fixed point, achieved II, or RTL memory allocation.
"""
from __future__ import annotations

from hashlib import sha256
import json

from fpgai_research.model_ir.row_chain import RowChainIR


def verify_row_chain_liveness(ir: RowChainIR, *, max_events: int = 100_000) -> dict:
    if not isinstance(ir, RowChainIR):
        raise TypeError("Expected typed RowChainIR")
    if type(max_events) is not int or max_events <= 0:
        raise ValueError("max_events requires positive integer")
    boundaries = ir.composed.boundaries
    n = len(ir.chain)
    if n < 2 or len(boundaries) != n-1 or len(ir.scratch_by_tensor) != n-1:
        raise ValueError("Row-chain IR has an incomplete edge lifetime contract")
    capacity = 0
    for idx, (edge, scratch) in enumerate(zip(boundaries, ir.scratch_by_tensor)):
        name, width = scratch
        if (edge.tensor != name or edge.producer != ir.chain[idx]
                or edge.consumers != (ir.chain[idx+1],)
                or edge.storage_layout != "row_buffer"
                or edge.exchange_policy != "producer_row_complete_then_consumer_row"
                or edge.physical_elements != width
                or edge.shape != (ir.row_count, width)
                or type(width) is not int or width < 1):
            raise ValueError("Invalid physical scratch or producer/consumer contract")
        capacity += width
    if capacity != ir.physical_scratch_elements:
        raise ValueError("CST scratch capacity does not match declared allocation")
    if capacity*ir.row_count != ir.full_intermediate_elements:
        raise ValueError("Logical and physical intermediate extents disagree")
    event_count = 2*ir.row_count*capacity
    if event_count > max_events:
        return {
            "schema": "fpgai-research/cst-row-chain-liveness/v1",
            "status": "BOUND_EXCEEDED",
            "required_element_events": event_count,
            "max_events": max_events,
            "vitis_executed": False,
        }
    # Each independent tensor scratch has its own address space; consecutive
    # logical rows alias addresses only once the previous row is consumed.
    live = {}
    events = []
    peak = 0
    for row in range(ir.row_count):
        for stage in range(n):
            # Conservative stage-level overlap: the next scratch can begin
            # receiving outputs while the input scratch is still required by
            # later output-channel reductions. Never release the input before
            # accounting for the new outputs. Distinct intermediate tensors
            # have distinct scratch arrays in the actual generated source.
            if stage < n-1:
                edge = boundaries[stage]
                for address in range(edge.physical_elements):
                    key = (edge.tensor, address)
                    if key in live:
                        raise AssertionError(
                            f"ROW_CHAIN_OVERWRITE_LIVE: {edge.tensor}[{address}] "
                            f"old_row={live[key]}, new_row={row}")
                    live[key] = row
                    events.append(("produce", edge.tensor, row, address, stage))
                peak = max(peak, len(live))
            if stage:
                edge = boundaries[stage-1]
                for address in range(edge.physical_elements):
                    key = (edge.tensor, address)
                    if live.get(key) != row:
                        raise AssertionError(
                            f"ROW_CHAIN_READ_BEFORE_WRITE: {edge.tensor}[{address}] row={row}")
                    del live[key]
                    events.append(("consume", edge.tensor, row, address, stage))
        if live:
            raise AssertionError("ROW_CHAIN_LIVE_VALUES_REMAIN_AFTER_ROW")
    if len(events) != event_count:
        raise AssertionError("Missing E0 tensor-element producer/consumer event")
    return {
        "schema": "fpgai-research/cst-row-chain-liveness/v1",
        "status": "EXHAUSTIVE_BOUNDED_LIVENESS_VERIFIED",
        "source_row_count": ir.row_count,
        "intermediate_tensor_count": n-1,
        "physical_scratch_elements": capacity,
        "logical_intermediate_elements": ir.full_intermediate_elements,
        "producer_consumer_element_events": event_count,
        "peak_simultaneously_live_scratch_values": peak,
        "every_write_consumed_before_address_reuse": True,
        "all_live_values_released_each_row": True,
        "event_sha256": sha256(json.dumps(events, separators=(",", ":")).encode()).hexdigest(),
        "evidence": "E0-bounded-independent-temporal-memory-lifetime-only",
        "host_executed": False,
        "vitis_executed": False,
        "hardware_memory_resource_proven": False,
    }
