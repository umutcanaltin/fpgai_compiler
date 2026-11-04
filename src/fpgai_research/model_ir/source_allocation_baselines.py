"""Adversarial R2q source-memory baselines; no compiler or vendor execution.

All results are source-event accounting, not on-chip FPGA BRAM, hardware
ports, HLS cycles, achieved Fmax or proof of scientific novelty. The same
fixed, validated R2f COMMIT/READ event order is used in every arm.

Crucial distinction:
  * committed-value: source scratch becomes live at COMMIT.
  * in-place-reader-output: a hypothetical single-pool reader accumulator
    becomes live at its FIRST reduction READ of H, before the COMMIT.

The second model is NOT emitted by the existing R2h C++ code: R2h has
separate full-sized reader accumulators as well as the scratch pool.
We report it as an adversarial alternative rather than pretending it
is an implemented or area-qualified design.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from fpgai_research.model_ir.element_scratch_allocation import (
    ValueInterval, allocate_minimum_slots, extract_intervals,
    verify_allocation,
)


@dataclass(frozen=True)
class WholeTensorBlock:
    tensor: str
    size: int
    first_commit: int
    last_read: int


def _peak(intervals: tuple[ValueInterval, ...]) -> int:
    """Independent maximum overlap, inclusive and source-event ordered."""
    timeline = []
    for item in intervals:
        if item.commit_index > item.last_read_index or item.commit_index < 0:
            raise ValueError("Ill-ordered value interval")
        timeline.append((item.commit_index, 1))
        timeline.append((item.last_read_index, -1))
    live = maximum = 0
    # In the hypothetical in-place model, first-MAC accumulation begins
    # during an H READ event. Treat a new output as live BEFORE any value
    # ending at that same ordinal is released. Without a micro-operation
    # schedule/bank-port proof, same-event overwrite is not certified.
    # Original committed-only intervals have distinct event boundaries,
    # so this stricter tie rule changes only the alternative accounting.
    for _, change in sorted(timeline, key=lambda event: (event[0], -event[1])):
        live += change
        if live < 0:
            raise ValueError("Invalid lifetime timeline")
        maximum = max(maximum, live)
    if live:
        raise ValueError("Lifetime accounting ends with live values")
    return maximum


def tensor_blocks(intervals: tuple[ValueInterval, ...]) -> tuple[WholeTensorBlock, ...]:
    """Conservatively hold an ENTIRE tensor from first COMMIT to last READ."""
    grouped: dict[str, list[ValueInterval]] = {}
    for iv in intervals:
        grouped.setdefault(iv.tensor, []).append(iv)
    blocks = []
    for tensor, group in grouped.items():
        elements = sorted(x.element for x in group)
        if elements != list(range(len(elements))):
            raise ValueError("Missing, duplicate, or non-contiguous tensor elements")
        blocks.append(WholeTensorBlock(
            tensor=tensor, size=len(group),
            first_commit=min(x.commit_index for x in group),
            last_read=max(x.last_read_index for x in group),
        ))
    return tuple(sorted(blocks, key=lambda b: (b.first_commit, b.tensor)))


def whole_tensor_peak_lower_bound(blocks: tuple[WholeTensorBlock, ...]) -> int:
    """Weighted concurrency bound; may NOT be achievable with contiguous bases."""
    timeline = []
    for block in blocks:
        timeline.append((block.first_commit, block.size))
        timeline.append((block.last_read, -block.size))
    live = upper = 0
    for _, change in sorted(timeline):
        live += change
        if live < 0:
            raise ValueError("Negative tensor block liveness")
        upper = max(live, upper)
    if live:
        raise ValueError("Tensor block liveness not closed")
    return upper


def whole_tensor_first_fit(
    blocks: tuple[WholeTensorBlock, ...],
) -> tuple[int, dict[str, int]]:
    """Feasible greedy contiguous-base tensor reuse, checked on actual events."""
    offsets: dict[str, int] = {}
    assigned: list[tuple[WholeTensorBlock, int]] = []
    capacity = 0
    for block in blocks:
        base = 0
        # Scan existing allocated base endpoints, not each possible byte.
        while True:
            conflicting = [
                (other, address) for other, address in assigned
                if block.first_commit <= other.last_read
                   and other.first_commit <= block.last_read
                   and base < address + other.size
                   and address < base + block.size
            ]
            if not conflicting:
                break
            base = min(address + other.size for other, address in conflicting
                       if address + other.size > base)
        offsets[block.tensor] = base
        assigned.append((block, base))
        capacity = max(capacity, base + block.size)
    return capacity, offsets


def reader_inplace_intervals(
    events: tuple,
    committed: tuple[ValueInterval, ...],
    *,
    producer_tensor: str,
    reader_outputs: dict[str, str],
) -> tuple[ValueInterval, ...]:
    """Hypothetical reader outputs live from first multiply-accumulate.

    For READ of the produced H, the known reader's 'output_element'
    identifies the accumulator that receives that contribution. This is
    NOT a new generated C++ program; no claim of hardware feasibility.
    """
    if type(reader_outputs) is not dict or not reader_outputs:
        raise ValueError("Explicit reader->output mapping required")
    if len(set(reader_outputs.values())) != len(reader_outputs):
        raise ValueError("Reader output identities must be distinct")
    first: dict[tuple[str, int], int] = {}
    for order, event in enumerate(events):
        if (event.kind == "READ" and event.tensor == producer_tensor
                and event.consumer in reader_outputs):
            if type(event.output_element) is not int or event.output_element < 0:
                raise ValueError("Missing typed reader output element for early lifetime")
            key = (reader_outputs[event.consumer], event.output_element)
            first[key] = min(order, first.get(key, order))
    outputs = set(reader_outputs.values())
    affected = {(iv.tensor, iv.element) for iv in committed
                if iv.tensor in outputs}
    if affected != set(first):
        raise ValueError("In-place accounting requires all reader accumulations")
    modified = []
    for iv in committed:
        begin = first.get((iv.tensor, iv.element), iv.commit_index)
        if begin > iv.commit_index:
            raise ValueError("Output accumulation begins after commit")
        modified.append(replace(iv, commit_index=begin))
    return tuple(modified)


def source_baseline_ladder(
    events: tuple,
    *,
    producer_tensor: str,
    reader_outputs: dict[str, str],
    accumulator_elements: int,
    counter_elements: int,
) -> dict:
    """Return comparison with explicitly separated source memory objectives.

    Note: optimal unit-slot result is the KNOWN classical fixed-order
    interval coloring minimum. Tensor peak is a LOWER BOUND, not an
    automatically feasible tensor-granular layout.
    """
    if (type(accumulator_elements) is not int
            or type(counter_elements) is not int
            or accumulator_elements < 0 or counter_elements < 0):
        raise ValueError("Explicit nonnegative source-only mutable-array counts")
    actual = allocate_minimum_slots(events)
    committed = extract_intervals(events)
    if actual.unit_slots != _peak(committed):
        raise AssertionError("Source peak disagrees with original interval allocator")
    blocks = tensor_blocks(committed)
    disjoint = sum(b.size for b in blocks)
    lower_bound = whole_tensor_peak_lower_bound(blocks)
    greedy, bases = whole_tensor_first_fit(blocks)
    addresses = {
        (block.tensor, idx): bases[block.tensor] + idx
        for block in blocks for idx in range(block.size)
    }
    if verify_allocation(events, addresses, unit_slots=greedy)[
            "status"] != "EXACT_SOURCE_SLOT_REPLAY_VERIFIED":
        raise AssertionError("Tensor first-fit not safe on actual source events")
    if not actual.unit_slots <= lower_bound <= greedy <= disjoint:
        raise AssertionError("Inconsistent tensor-vs-element source capacity bounds")
    in_place = reader_inplace_intervals(
        events, committed, producer_tensor=producer_tensor,
        reader_outputs=reader_outputs)
    hypothetical_inplace_peak = _peak(in_place)
    if hypothetical_inplace_peak < actual.unit_slots:
        raise AssertionError("Earlier accumulator lifetime reduced source peak")
    # This number is actual *declarations* in the current R2h source,
    # not a hardware utilisation estimate. The fused counterfactual is
    # hypothetical, not emitted in R2h.
    gap = disjoint - actual.unit_slots
    return {
        "disjoint_no_reuse_slots": disjoint,
        "whole_tensor_concurrency_lower_bound_slots": lower_bound,
        "whole_tensor_first_fit_feasible_slots": greedy,
        "element_exact_committed_value_slots": actual.unit_slots,
        "hypothetical_inplace_reader_value_slots": hypothetical_inplace_peak,
        "fixed_order_optimal_equals_peak": (
            actual.unit_slots == actual.peak_simultaneous_live_values),
        "actual_r2h_accumulator_array_elements": accumulator_elements,
        "actual_r2h_counter_array_elements": counter_elements,
        "actual_r2h_all_declared_mutable_elements_disjoint": (
            disjoint + accumulator_elements + counter_elements),
        "actual_r2h_all_declared_mutable_elements_element_exact": (
            actual.unit_slots + accumulator_elements + counter_elements),
        "hypothetical_fused_inplace_plus_counters_elements": (
            hypothetical_inplace_peak + counter_elements),
        "reduction_fraction_captured_by_feasible_tensor_first_fit": (
            (disjoint - greedy) / gap if gap else None),
        "tensor_lower_bound_fraction_of_disjoint_to_element_gap": (
            (disjoint - lower_bound) / gap if gap else None),
        "value_interval_count": len(committed),
        "whole_tensor_count": len(blocks),
        "source_event_sha256": actual.schedule_sha256,
        "physical_BRAM_measured": False,
        "gcc_executed_by_this_analysis": False,
        "native_vitis_calls": 0,
        "native_vivado_calls": 0,
        "limits": [
            "Both tensor-granular baselines conservatively hold all elements from first COMMIT to last READ",
            "Tensor concurrency peak is a mathematical lower bound, NOT an attained feasible placement",
            "Exact element optimum is classical interval coloring for one fixed total order",
            "Hypothetical in-place model moves reader-output lifetimes to first MAC, but is NOT emitted C++",
            "Actual R2h includes separate accumulator and counter arrays; physical BRAM and port costs UNKNOWN",
        ],
    }
