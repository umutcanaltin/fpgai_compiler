"""Branching graph last reader release controls. E0 only; no HLS claim."""
from fpgai_research.model_ir.contracts import FunctionalModel,Operator,Tensor
from fpgai_research.model_ir.fanout_lifetime import (
    LifetimeEvent,canonical_fanout_events,verify_fanout_events,
)

def graph():
    return FunctionalModel(
        tensors=(
            Tensor("x",(1,2)),Tensor("w",(2,2),data=(1.,0.,0.,1.)),
            Tensor("h",(1,2)),Tensor("a",(1,2)),
            Tensor("b",(1,2)),Tensor("g",(1,2)),Tensor("out",(1,2))),
        operators=(
            Operator("producer","MatMul",("x","w"),("h",)),
            Operator("reader0","Relu",("h",),("a",)),
            Operator("reader1","Relu",("h",),("b",)),
            Operator("merge","Add",("a","b"),("g",)),
            Operator("final","Relu",("g",),("out",))),
        inputs=("x",),outputs=("out",),
    )

def test_two_consumer_contract_releases_only_after_both_readers():
    m=graph(); mapping={"h":0,"g":0}
    events=canonical_fanout_events(m,mapping)
    res=verify_fanout_events(m,mapping,events)
    assert res["status"]=="VERIFIED"
    assert res["logical_values"]==4
    assert res["multi_reader_values"]==2
    assert res["native_vitis_calls"]==0

def test_premature_slot_reuse_after_first_reader_is_rejected():
    m=graph(); mapping={"h":0,"g":0}
    events=canonical_fanout_events(m,mapping)
    marker=next(i for i,e in enumerate(events) if e.kind=="CONSUME" and e.consumer=="reader0")
    bad=events[:marker+1]+(LifetimeEvent("COMMIT","g",0),)+events[marker+1:]
    result=verify_fanout_events(m,mapping,bad)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="OVERWRITE_BEFORE_LAST_READER"
    assert result["counterexample"]["previous_value"]==["h",0]

def test_incomplete_reader_release_and_missing_producer_are_detected():
    m=graph(); mapping={"h":0,"g":0}
    events=canonical_fanout_events(m,mapping)
    e=next(e for e in events if e.consumer=="reader1")
    truncated=tuple(x for x in events if x is not e)
    result=verify_fanout_events(m,mapping,truncated)
    assert result["status"]=="REJECTED"
    assert result["reason"] in ("OVERWRITE_BEFORE_LAST_READER",
                                "INCOMPLETE_READER_RELEASE")
    premature=(LifetimeEvent("CONSUME","h",0,"reader0"),)+events
    assert verify_fanout_events(m,mapping,premature)["reason"]=="READ_BEFORE_READY_OR_STALE"

def test_distinct_slots_are_allowed_without_early_alias():
    m=graph(); mapping={"h":0,"g":2}
    result=verify_fanout_events(m,mapping,canonical_fanout_events(m,mapping))
    assert result["status"]=="VERIFIED"
