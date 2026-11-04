"""R2c finite per-value source scratch: independent interval oracle and replay."""
from __future__ import annotations

from dataclasses import replace

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.interleaved_branch_schedule import interleaved_branch_events
from fpgai_research.model_ir.element_scratch_allocation import (
    allocate_minimum_slots, exhaustive_minimum_slots, extract_intervals,
    verify_allocation,
)
from fpgai_research.validation.r2c_independent_slot_oracle import (
    oracle_minimum_unit_slots,
)


def _events(*,m=1,k=1,n=1,q=1,r0=1,r1=1):
    model=sample_branch(m,k,n,q)
    base=ArchitectureRequest()
    overrides={}
    if r0!=1:
        overrides["reader0"]=replace(base,pes=r0)
    if r1!=1:
        overrides["reader1"]=replace(base,pes=r1)
    return interleaved_branch_events(resolve(model,base,overrides)).events


@pytest.mark.parametrize("dimensions",[
    (1,1,1,1),(1,2,2,1),(1,2,2,2),(1,2,3,2),
])
def test_exact_allocator_matches_independent_exhaustive_oracle(dimensions):
    events=_events(m=dimensions[0],k=dimensions[1],
                   n=dimensions[2],q=dimensions[3])
    result=allocate_minimum_slots(events)
    assert result.unit_slots==result.peak_simultaneous_live_values
    assert result.unit_slots==exhaustive_minimum_slots(events,max_values=12)
    # Truly independent liveness extraction PLUS exhaustive oracle:
    separate=oracle_minimum_unit_slots(events,max_values=11)
    assert separate["minimum_unit_slots"]==result.unit_slots
    assert separate["status"]=="EXACT_ORACLE_VERIFIED"
    assert len(result.addresses)==dimensions[0]*(dimensions[2]+3*dimensions[3])
    assert verify_allocation(events,result.slot_map(),
                             unit_slots=result.unit_slots)["status"]==(
                                 "EXACT_SOURCE_SLOT_REPLAY_VERIFIED")
    assert allocate_minimum_slots(events)==result


def test_early_commit_to_live_h_slot_is_refused_with_address_and_event():
    events=_events()
    good=allocate_minimum_slots(events)
    changed=good.slot_map()
    changed[("a",0)]=changed[("h",0)]
    with pytest.raises(ValueError,match="OVERWRITE_LIVE_VALUE.*slot="):
        verify_allocation(events,changed,unit_slots=good.unit_slots)


def test_reject_undelivered_final_read_and_wrong_address_set():
    events=_events()
    good=allocate_minimum_slots(events)
    with pytest.raises(ValueError,match="VALUE_WITHOUT_FINAL_READ"):
        allocate_minimum_slots(events[:-1])
    with pytest.raises(ValueError,match="MISSING_OR_EXTRA_VALUE_ADDRESS"):
        verify_allocation(events,{("h",0):0},unit_slots=good.unit_slots)
    with pytest.raises(ValueError,match="Invalid finite scratch"):
        verify_allocation(events,{key:-1 for key in good.slot_map()},unit_slots=3)


def test_two_phi_reader_mapping_changes_do_not_auto_claim_capacity_savings():
    versions=[]
    for reader1_pes in (1,2):
        events=_events(m=2,k=2,n=3,q=4,r1=reader1_pes)
        allocation=allocate_minimum_slots(events)
        assert allocation.unit_slots==allocation.peak_simultaneous_live_values
        assert verify_allocation(events,allocation.slot_map(),
                                 unit_slots=allocation.unit_slots)["status"]==(
                                     "EXACT_SOURCE_SLOT_REPLAY_VERIFIED")
        versions.append(allocation)
    # No assertion that Phi causes capacity savings. The experiment must
    # report original real case numbers and retain equality/regressions.
    assert versions[0].schedule_sha256!=versions[1].schedule_sha256
    assert all(row.unit_slots>0 for row in versions)


def test_exhaustive_oracle_aborts_on_large_finite_cases():
    events=_events(m=2,k=2,n=3,q=4)
    with pytest.raises(ValueError,match="bounded tiny"):
        exhaustive_minimum_slots(events,max_values=8)


def test_fully_independent_oracle_rejects_invalid_commit_and_missing_last_read():
    events=_events()
    with pytest.raises(ValueError,match="INDEPENDENT_ORACLE_INCOMPLETE_VALUE_LIFETIME"):
        oracle_minimum_unit_slots(events[:-1])
    extra=(events[0],)+events
    with pytest.raises(ValueError,match="INDEPENDENT_ORACLE_DUPLICATE_COMMIT"):
        oracle_minimum_unit_slots(extra)
    with pytest.raises(ValueError,match="tiny domain"):
        oracle_minimum_unit_slots(_events(m=2,k=2,n=3,q=4),max_values=8)
