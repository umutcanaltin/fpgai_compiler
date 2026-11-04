"""Bounded IR-level fan-out lifetime and physical alias verifier (E0 ONLY).

Supports multiple downstream consumers of a static intermediate tensor and
checks that its storage slot is not overwritten until ALL declared readers
acknowledge it. Consumer acknowledgments are logical stage-level events,
not actual per-iteration reads or generated-C++ evidence. The source backend
does not yet lower arbitrary fan-out; no native Vitis claim is made.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
from hashlib import sha256
import json
from math import prod

from fpgai_research.model_ir.contracts import FunctionalModel

@dataclass(frozen=True)
class LifetimeEvent:
    kind: str                  # COMMIT or CONSUME
    tensor: str
    element: int               # flattened logical element index
    consumer: str = ""         # operation identity if CONSUME

def fanout_contract(model: FunctionalModel, offsets: dict[str,int]) -> dict:
    model.validate()
    if type(offsets) is not dict or not offsets:
        raise ValueError("Nonempty tensor-to-physical-base mapping required")
    writers={t:op.name for op in model.operators for t in op.outputs}
    readers={name:tuple(op.name for op in model.operators if name in op.inputs)
             for name in offsets}
    for tensor,base in offsets.items():
        if tensor not in writers or tensor in model.outputs:
            raise ValueError("Only produced intermediate tensors admit scratch contracts")
        if type(base) is not int or base<0 or not readers[tensor]:
            raise ValueError("Positive physical address domain and consumers required")
        if len(set(readers[tensor]))!=len(readers[tensor]):
            raise ValueError("Repeated same-op consumer ports not yet modeled")
    return {
        "schema":"fpgai-research/fanout-lifetime-e0/v1",
        "tensors":{name:{
            "producer":writers[name],
            "consumers":list(readers[name]),
            "elements":prod(model.tensor(name).shape),
            "physical_base":base,
        } for name,base in sorted(offsets.items())},
        "scope":"logical consumer acknowledgments; no source read trace",
    }

def canonical_fanout_events(model: FunctionalModel, offsets: dict[str,int]):
    contract=fanout_contract(model,offsets)
    events=[]
    for op in model.operators:
        for name in op.inputs:
            if name in offsets:
                for i in range(contract["tensors"][name]["elements"]):
                    events.append(LifetimeEvent("CONSUME",name,i,op.name))
        for name in op.outputs:
            if name in offsets:
                for i in range(contract["tensors"][name]["elements"]):
                    events.append(LifetimeEvent("COMMIT",name,i))
    return tuple(events)

def verify_fanout_events(model: FunctionalModel, offsets: dict[str,int],
                         events: tuple[LifetimeEvent,...], *,
                         max_events:int=20000) -> dict:
    contract=fanout_contract(model,offsets)
    if type(events) is not tuple or any(not isinstance(e,LifetimeEvent) for e in events):
        raise TypeError("Typed immutable lifetime events required")
    if type(max_events) is not int or max_events<1 or len(events)>max_events:
        raise ValueError("Finite fan-out event bound required")
    live={}
    committed=set()
    consumed={}
    def fail(reason,index,event,**detail):
        return {"status":"REJECTED","reason":reason,"event_index":index,
                "counterexample":{"event":asdict(event),**detail},
                "evidence":"E0-abstract-consumer-acknowledgment-only",
                "native_vitis_calls":0}
    for i,event in enumerate(events):
        spec=contract["tensors"].get(event.tensor)
        if spec is None or type(event.element) is not int or not 0<=event.element<spec["elements"]:
            return fail("INVALID_TENSOR_ELEMENT",i,event)
        key=(event.tensor,event.element)
        addr=spec["physical_base"]+event.element
        if event.kind=="COMMIT":
            if event.consumer:
                return fail("INVALID_PRODUCER_EVENT",i,event)
            if key in committed:
                return fail("DUPLICATE_COMMIT",i,event)
            if addr in live:
                return fail("OVERWRITE_BEFORE_LAST_READER",i,event,
                            address=addr,previous_value=list(live[addr]))
            committed.add(key)
            live[addr]=key
        elif event.kind=="CONSUME":
            if event.consumer not in spec["consumers"]:
                return fail("UNKNOWN_CONSUMER",i,event)
            if key not in committed or live.get(addr)!=key:
                return fail("READ_BEFORE_READY_OR_STALE",i,event,address=addr)
            completed=consumed.setdefault(key,set())
            if event.consumer in completed:
                return fail("DUPLICATE_CONSUMER_ACK",i,event)
            completed.add(event.consumer)
            if len(completed)==len(spec["consumers"]):
                del live[addr]
        else:
            return fail("UNKNOWN_EVENT_KIND",i,event)
    expected=sum(x["elements"] for x in contract["tensors"].values())
    if len(committed)!=expected or live:
        return {"status":"REJECTED","reason":"INCOMPLETE_READER_RELEASE",
                "counterexample":{"committed":len(committed),"expected":expected,
                                  "live_addresses":len(live)},
                "evidence":"E0-abstract-consumer-acknowledgment-only",
                "native_vitis_calls":0}
    return {"schema":"fpgai-research/fanout-lifetime-check/v1",
            "status":"VERIFIED","logical_values":expected,
            "events_checked":len(events),
            "multi_reader_values":sum(spec["elements"] for spec in contract["tensors"].values()
                                      if len(spec["consumers"])>1),
            "event_sha256":sha256(json.dumps([asdict(e) for e in events],
                             sort_keys=True).encode()).hexdigest(),
            "evidence":"E0-stage-consumer-last-use-only",
            "native_vitis_calls":0,
            "limits":["Not a Phi-conditioned or source-level fan-out proof",
                      "No C++ lowering, HLS clock, routing or physical BRAM measurement"]}
