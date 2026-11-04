"""R2t: exact contiguous whole-tensor source allocation for fixed lifetimes.

Each tensor is one indivisible contiguous block of its scalar elements,
held from the FIRST element COMMIT through the LAST element READ. This is
a conservative tensor baseline, *not* the element-exact or in-place
first-MAC objective. The block allocation is independent of any tensor
element slot coloring and can be checked against actual source events.

Exact search enumerates all global spatial orderings of the tensor blocks.
For any one spatial ordering, each block can be left-justified to the
largest end of its spatially preceding *temporally overlapping* blocks.
Every feasible allocation has at least one such global spatial order, so
exhaustive enumeration with a finite work budget is exact when completed.

The budget is fail-closed (UNKNOWN_BUDGET, not a proven optimum). No
ILP or nonfree numerical solver is required, but this exponential
reference is NOT a production allocator or a hardware BRAM model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from fpgai_research.model_ir.source_allocation_baselines import (
    WholeTensorBlock, whole_tensor_first_fit,
    whole_tensor_peak_lower_bound, tensor_blocks,
)
from fpgai_research.model_ir.element_scratch_allocation import (
    extract_intervals, verify_allocation,
)

SCHEMA = "fpgai-research/cst-R2t-optimal-contiguous-tensor-E0/v1"


@dataclass(frozen=True)
class TensorExactResult:
    status: str
    tensor_count: int
    lower_bound: int
    greedy_feasible_slots: int
    exact_feasible_slots: int | None
    bases: tuple[tuple[str, int], ...] | None
    search_nodes: int
    source_event_sha256: str | None
    schema: str = SCHEMA

    def as_dict(self) -> dict:
        return asdict(self)


def _validate(blocks: tuple[WholeTensorBlock, ...]) -> None:
    if type(blocks) is not tuple or not blocks or len(blocks) > 9:
        raise ValueError("R2t exact reference requires 1..9 typed tensor blocks")
    names: set[str] = set()
    for block in blocks:
        if (not isinstance(block, WholeTensorBlock)
                or not isinstance(block.tensor, str)
                or not block.tensor or block.tensor in names
                or type(block.size) is not int or block.size < 1
                or type(block.first_commit) is not int
                or type(block.last_read) is not int
                or block.first_commit < 0
                or block.last_read < block.first_commit):
            raise ValueError("R2t invalid tensor block/source lifetime")
        names.add(block.tensor)


def exact_tensor_bases(
    blocks: tuple[WholeTensorBlock, ...], *, max_search_nodes: int = 250_000
) -> TensorExactResult:
    _validate(blocks)
    if type(max_search_nodes) is not int or max_search_nodes < 1:
        raise ValueError("R2t exact solver requires finite positive node budget")
    lower = whole_tensor_peak_lower_bound(blocks)
    greedy, greedy_bases = whole_tensor_first_fit(blocks)
    if not lower <= greedy:
        raise AssertionError("R2t lower bound exceeds feasible first-fit")
    # Order choices are spatial order, not source event order. Each
    # complete permutation has a minimal left-justified placement.
    items = tuple(sorted(blocks, key=lambda b: (b.tensor, b.first_commit)))
    n = len(items)
    conflicts = [
        sum(
            (1 << j)
            for j, other in enumerate(items)
            if j != i and b.first_commit <= other.last_read
            and other.first_commit <= b.last_read
        )
        for i, b in enumerate(items)
    ]
    best = greedy
    best_bases = dict(greedy_bases)
    nodes = 0
    exhausted = False
    placement = [0] * n
    # A known certificate equal to the lower bound proves optimality
    # even if no search is necessary.
    if best == lower:
        return TensorExactResult(
            "OPTIMAL_PROVED", n, lower, greedy, best,
            tuple(sorted(best_bases.items())), 0, None,
        )

    def visit(mask: int, spatial_order: tuple[int, ...], height: int) -> None:
        nonlocal nodes, best, best_bases, exhausted
        if best == lower or exhausted:
            return
        if nodes >= max_search_nodes:
            exhausted = True
            return
        nodes += 1
        if mask == (1 << n) - 1:
            if height < best:
                best = height
                best_bases = {item.tensor: placement[i]
                              for i, item in enumerate(items)}
            return
        # Stable, performance-neutral heuristic chooses large blocks first.
        for i in sorted(
            (idx for idx in range(n) if not mask & (1 << idx)),
            key=lambda idx: (-items[idx].size, items[idx].tensor),
        ):
            base = max(
                (placement[j] + items[j].size for j in spatial_order
                 if conflicts[i] & (1 << j)), default=0,
            )
            cap = max(height, base + items[i].size)
            if cap >= best:
                continue
            placement[i] = base
            visit(mask | (1 << i), spatial_order + (i,), cap)
            if best == lower or exhausted:
                return

    visit(0, (), 0)
    if exhausted:
        return TensorExactResult(
            "UNKNOWN_BUDGET", n, lower, greedy, None, None, nodes, None,
        )
    return TensorExactResult(
        "OPTIMAL_PROVED", n, lower, greedy, best,
        tuple(sorted(best_bases.items())), nodes, None,
    )


def exact_tensor_source_baseline(
    events: tuple, *, max_search_nodes: int = 250_000
) -> TensorExactResult:
    intervals = extract_intervals(events)
    blocks = tensor_blocks(intervals)
    result = exact_tensor_bases(blocks, max_search_nodes=max_search_nodes)
    if result.status != "OPTIMAL_PROVED":
        return result
    bases = dict(result.bases or ())
    addresses = {
        (block.tensor, i): bases[block.tensor] + i
        for block in blocks for i in range(block.size)
    }
    verified = verify_allocation(
        events, addresses, unit_slots=result.exact_feasible_slots
    )
    if verified.get("status") != "EXACT_SOURCE_SLOT_REPLAY_VERIFIED":
        raise AssertionError("R2t exact whole-tensor bases fail actual source replay")
    # Source provenance, *not* a vendor flow identity.
    source_digest = sha256(json.dumps([
        {"kind": event.kind, "tensor": event.tensor,
         "element": event.element,
         "consumer": getattr(event, "consumer", None),
         "output_element": getattr(event, "output_element", None)}
        for event in events
    ], sort_keys=True, separators=(",",":")).encode()).hexdigest()
    return TensorExactResult(
        result.status, result.tensor_count, result.lower_bound,
        result.greedy_feasible_slots, result.exact_feasible_slots,
        result.bases, result.search_nodes, source_digest,
    )
