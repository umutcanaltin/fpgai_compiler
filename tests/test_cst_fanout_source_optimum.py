"""L5c-E1: exact source-order alias allocation against independent oracle.

All optimality claims refer to finite abstract consumer-ACK schedules matching
the actually generated C++ event order; no real FPGA banking or timing.
"""
from itertools import product
import shutil

import numpy as np
import pytest

from fpgai_research.codegen.hls.fanout_diamond import (
    emit_fanout_diamond_hls, source_order_events,
)
from fpgai_research.experiments.cst_fanout_source_campaign import sample_diamond
from fpgai_research.model_ir.fanout_allocator import exact_fanout_slot_plan
from fpgai_research.model_ir.fanout_lifetime import verify_fanout_events
from fpgai_research.validation.cpp_fanout_trace import trace_fanout_cpp
from fpgai_research.validation.host import validate_host


def independent_source_order_minimum(model, event_schedule, *, upper=9):
    """Replays every physical offset tuple, not the allocator interval test."""
    tensors=("a","b","g","h")
    extent=model.tensor("h").shape[0]*model.tensor("h").shape[1]
    for capacity in range(extent,upper+1):
        for offsets in product(range(capacity-extent+1),repeat=4):
            mapping=dict(zip(tensors,offsets))
            if verify_fanout_events(model,mapping,event_schedule)["status"]=="VERIFIED":
                return capacity
    return None


def test_source_order_minimum_matches_separate_full_offset_oracle():
    model=sample_diamond(2,2)
    events=source_order_events(model)
    optimum=exact_fanout_slot_plan(
        model,("h","a","b","g"),event_sequence=events,max_slots=12)
    assert optimum["status"]=="OPTIMAL_E0"
    assert optimum["event_schedule"]=="explicit-source-order-ACK"
    assert optimum["physical_slots"]==independent_source_order_minimum(
        model,events,upper=12)
    assert optimum["physical_slots"]==8
    assert optimum["dense_disjoint_slots"]==16
    assert optimum["verification_sha256"]==optimum["logical_event_sha256"]


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
@pytest.mark.parametrize("m,n",[(2,2),(3,3),(5,2)])
def test_source_order_optimal_alias_is_executed_and_numerically_equal(m,n):
    model=sample_diamond(m,n)
    events=source_order_events(model)
    optimum=exact_fanout_slot_plan(
        model,("a","b","g","h"),event_sequence=events,
        max_slots=4*m*n,max_candidates=500_000)
    assert optimum["status"]=="OPTIMAL_E0"
    emitted=emit_fanout_diamond_hls(model,offsets=optimum["offsets"])
    assert emitted.manifest["scratch_elements"]==optimum["physical_slots"]
    assert emitted.manifest["source_e0_event_sha256"]==optimum["logical_event_sha256"]
    report=trace_fanout_cpp(model,emitted)
    assert report["status"]=="SOURCE_FANOUT_LIFETIME_VERIFIED"
    x=np.asarray([((-1)**i)*(i+1)/4 for i in range(m*2)],
                 dtype=np.float32).reshape(m,2)
    assert validate_host(model,emitted,x)["accepted"]


def test_source_order_verification_rejects_wrong_schedule_and_unknown():
    model=sample_diamond(2,2)
    sequence=source_order_events(model)
    with pytest.raises(TypeError,match="typed"):
        exact_fanout_slot_plan(model,("h","a","b","g"),event_sequence=[*sequence])
    with pytest.raises(ValueError,match="event schedule"):
        exact_fanout_slot_plan(
            model,("h","a","b","g"),event_sequence=sequence[1:])
    limited=exact_fanout_slot_plan(model,("h","a","b","g"),
             event_sequence=sequence,max_candidates=1,max_slots=16)
    assert limited["status"]=="UNKNOWN_BUDGET"
    assert limited["physical_slots"] is None
