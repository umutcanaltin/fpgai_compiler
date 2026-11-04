"""Mapping-conditioned bounded value obligations, not just Phi hashes."""
from dataclasses import replace

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.mapped_event_schedule import (
    canonical_value_events, verify_value_events,
)

def setup():
    graph=make_chain(3,(3,5,4,2))
    arch=resolve(graph,ArchitectureRequest())
    events=canonical_value_events(arch,row_block=2)
    return graph,arch,events

def test_canonical_logical_event_schedule_has_verified_reductions_and_last_use():
    _,arch,events=setup()
    result=verify_value_events(arch,events,row_block=2)
    assert result["status"]=="VERIFIED"
    assert result["events_checked"]==len(events)
    assert result["values_checked"]==3*(5+4)
    assert result["original_vitis_calls"]==0

def test_missing_contribution_causes_specific_premature_commit():
    _,arch,events=setup()
    first=next(i for i,e in enumerate(events) if e.kind=="CONTRIBUTE")
    bad=events[:first]+events[first+1:]
    result=verify_value_events(arch,bad,row_block=2)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="PREMATURE_REDUCTION_COMMIT"
    assert result["counterexample"]["completed"]

def test_changed_phi_rejects_stale_declared_owner_even_when_graph_same():
    graph,arch,events=setup()
    request=ArchitectureRequest()
    changed=resolve(graph,request,{"mm0":replace(request,pes=2,simd=2)})
    proof=verify_value_events(changed,events,row_block=2)
    assert proof["status"]=="REJECTED"
    assert proof["reason"] in ("PHI_PRODUCER_OWNER_MISMATCH",
                               "PHI_CONSUMER_OWNER_MISMATCH")
    new_events=canonical_value_events(changed,row_block=2)
    assert verify_value_events(changed,new_events,row_block=2)["status"]=="VERIFIED"

def test_early_read_is_rejected_and_cannot_consume_unproduced_value():
    _,arch,events=setup()
    first=next(e for e in events if e.kind=="READ")
    bad=(first,)+tuple(e for e in events if e is not first)
    result=verify_value_events(arch,bad,row_block=2)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="READ_BEFORE_READY"

def test_truncated_events_are_not_accepted():
    _,arch,events=setup()
    result=verify_value_events(arch,events[:-1],row_block=2)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="INCOMPLETE_VALUE_LIFETIMES"
