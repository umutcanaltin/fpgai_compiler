"""Independent *constructive* finite reference oracle for CST schedule legality.

Unlike check_schedule, this oracle does NOT replay a proposed sequence, keep
a tensor-value dictionary, or inspect the first failing event. Instead, it
constructs the complete language of valid atomic row-stage executions by DFS
over completed-stage sets. Candidate legality is set membership in that
language. The model is deliberately restricted and finite (small graphs).

The oracle shares the *stated atomic-stage specification*, not implementation
logic. Agreement tests are bounded cross-implementation checks, not a
machine-checked proof of full C++ or RTL correctness.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial

from fpgai_research.model_ir.temporal_schedule import ComputeEvent


def reference_valid_sequences(rows: int, stages: int, slots: int, *,
                              max_permutations: int = 50_000) -> frozenset[tuple[tuple[int, int], ...]]:
    """Enumerate all schedules allowed by logical deps and finite row slots.

    Each (row,stage) fires once. It may fire only after that row's previous
    stage. Producing an intermediate result requires an unused physical row
    slot for that tensor edge, where occupancy can be *derived solely from*
    completed predecessor and successor stages. No imperative memory map,
    counterexample code or actual implementation checker is reused.
    """
    if any(type(x) is not int for x in (rows, stages, slots, max_permutations)):
        raise TypeError("Reference domain must use integer dimensions")
    if rows < 1 or stages < 2 or not 1 <= slots <= rows:
        raise ValueError("Invalid finite reference domain")
    if max_permutations < 1 or factorial(rows * stages) > max_permutations:
        raise ValueError("Exact reference domain exceeds permutation budget")
    domain = tuple((r, s) for r in range(rows) for s in range(stages))
    whole = frozenset(domain)

    @lru_cache(maxsize=None)
    def continuations(done: frozenset[tuple[int, int]]):
        if done == whole:
            return ((),)
        sequences = []
        for r, s in domain:
            if (r, s) in done or (s > 0 and (r, s-1) not in done):
                continue
            if s < stages-1:
                # A scratch slot is in use by another row iff it produced an
                # edge value but has not yet executed its consuming stage.
                occupied = any(
                    other != r and other % slots == r % slots
                    and (other, s) in done and (other, s+1) not in done
                    for other in range(rows)
                )
                if occupied:
                    continue
            for suffix in continuations(done | {(r, s)}):
                sequences.append(((r, s),) + suffix)
        return tuple(sequences)

    return frozenset(continuations(frozenset()))


def reference_accepts(events: tuple[ComputeEvent, ...],
                      language: frozenset[tuple[tuple[int, int], ...]]) -> bool:
    if any(not isinstance(e, ComputeEvent) for e in events):
        raise TypeError("Reference requires typed stage events")
    return tuple((e.row, e.stage) for e in events) in language
