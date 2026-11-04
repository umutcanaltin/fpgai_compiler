"""Exhaustive differential study on every permutation of small stage graphs.

The actual legality checker and constructive oracle use structurally
different algorithms. No intentionally selected mutants are required to
establish agreement across the complete bounded permutation universe.
"""
from itertools import permutations

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.row_chain import resolve_row_chain
from fpgai_research.model_ir.temporal_schedule import ComputeEvent, check_schedule
from fpgai_research.validation.schedule_reference_oracle import (
    reference_accepts, reference_valid_sequences,
)


@pytest.mark.parametrize("rows,widths,slots", [
    (2, (3, 5, 4, 2), 1),
    (2, (3, 5, 4, 2), 2),
    (3, (3, 5, 2), 1),
    (3, (3, 5, 2), 2),
    (3, (3, 5, 2), 3),
])
def test_every_small_schedule_permutation_agrees_with_independent_oracle(
        rows, widths, slots):
    arch = resolve(make_chain(rows, widths), ArchitectureRequest())
    ir = resolve_row_chain(arch)
    stages = len(ir.chain)
    valid_language = reference_valid_sequences(rows, stages, slots)
    assert len(valid_language) > 0
    agreements = 0
    accepted = 0
    for proposal in permutations(
            tuple(ComputeEvent(r, s) for r in range(rows) for s in range(stages))):
        expected = reference_accepts(proposal, valid_language)
        got = check_schedule(ir, proposal, row_slots=slots)
        assert (got["status"] == "VERIFIED") == expected, (
            rows, widths, slots, proposal, got)
        if expected:
            accepted += 1
        else:
            assert got["status"] == "REJECTED"
            assert got["counterexample"] is not None
        agreements += 1
    assert accepted == len(valid_language)
    assert agreements == 720


def test_independent_oracle_rejects_incomplete_and_duplicate_sequence():
    language = reference_valid_sequences(2, 3, 2)
    candidate = next(iter(language))
    events = tuple(ComputeEvent(*x) for x in candidate)
    assert reference_accepts(events, language)
    assert not reference_accepts(events[:-1], language)
    assert not reference_accepts(events + events[:1], language)


def test_oracle_fails_closed_for_large_exhaustive_domains():
    with pytest.raises(ValueError, match="permutation budget"):
        reference_valid_sequences(3, 3, 2)
    with pytest.raises(ValueError, match="permutation budget"):
        reference_valid_sequences(2, 4, 2, max_permutations=200)
