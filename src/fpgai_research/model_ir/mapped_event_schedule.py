"""Finite Phi-conditioned *logical* value-event verifier (E0 only).

This checks explicit mapped contribution, commit and consumer-read events for
the supported sequential CST row chain. A candidate event includes the
spatial Phi owner it claims. Changing Phi without changing its event owners
therefore yields a concrete mapping mismatch rather than only a hash drift.

No claim about cycle-level availability, port conflicts, LLVM/HLS scheduling,
RTL or support for fan-out. Actual generated source has a separate E1 tracer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.mapped_value_contract import resolve_mapped_values
from fpgai_research.model_ir.temporal_schedule import blocked_schedule

@dataclass(frozen=True)
class ValueEvent:
    kind: str                  # CONTRIBUTE / COMMIT / READ
    tensor: str
    row: int
    column: int
    index: int = -1             # reduction k or consumer output n
    phi_owner: tuple = ()      # expected (replica, PE, SIMD lane, reuse)
    
def _owner(point, index):
    p=coordinate(point,index)
    return (p["replica"], p["pe"], p["lane"], tuple(p["reuse_step"]))

def canonical_value_events(arch: ArchitecturalModel, *, row_block: int = 1):
    """Construct one legal row-atomic schedule; all event owners derive from Phi."""
    mapped=resolve_mapped_values(arch,row_block=row_block)
    ops=arch.functional.operators
    points=arch.points
    by_tensor={v.tensor:v for v in mapped.row_chain.composed.boundaries}
    events=[]
    for step in blocked_schedule(mapped.row_chain.row_count,len(ops),row_block):
        row,stage=step.row,step.stage
        if stage:
            edge=by_tensor[ops[stage-1].outputs[0]]
            width=edge.shape[1]
            out_width=arch.functional.tensor(ops[stage].outputs[0]).shape[1]
            for outcol in range(out_width):
                for col in range(width):
                    pos=(row,outcol,col) if ops[stage].kind in ("MatMul","Gemm") else (row,col)
                    if ops[stage].kind not in ("MatMul","Gemm") and outcol!=col:
                        continue
                    events.append(ValueEvent("READ",edge.tensor,row,col,outcol,
                                             _owner(points[stage],pos)))
        if stage<len(ops)-1:
            edge=by_tensor[ops[stage].outputs[0]]
            for col in range(edge.shape[1]):
                count=points[stage].iteration_shape[2] if ops[stage].kind in ("MatMul","Gemm") else 1
                for k in range(count):
                    pos=(row,col,k) if ops[stage].kind in ("MatMul","Gemm") else (row,col)
                    events.append(ValueEvent("CONTRIBUTE",edge.tensor,row,col,k,
                                             _owner(points[stage],pos)))
                events.append(ValueEvent("COMMIT",edge.tensor,row,col))
    return tuple(events)

def verify_value_events(arch: ArchitecturalModel, events: tuple[ValueEvent,...], *,
                        row_block: int=1, max_events: int=100_000) -> dict:
    if type(events) is not tuple or any(not isinstance(e,ValueEvent) for e in events):
        raise TypeError("Expected typed finite event sequence")
    if type(max_events) is not int or max_events<1:
        raise ValueError("Positive event budget required")
    if len(events)>max_events:
        raise ValueError("Explicit value events exceed bound")
    mapped=resolve_mapped_values(arch,row_block=row_block)
    f=arch.functional
    ops=f.operators
    points=arch.points
    value_map={(v.tensor,v.row,v.column):v for v in mapped.values}
    boundary_stage={b.tensor:i for i,b in enumerate(mapped.row_chain.composed.boundaries)}
    contributions={}
    ready=set()
    reads={}
    occupancy={}
    def failure(reason,idx,event,**extra):
        return {"schema":"fpgai-research/mapped-logical-events/v1","status":"REJECTED",
                "reason":reason,"event_index":idx,"event":asdict(event),
                "counterexample":extra,"original_vitis_calls":0}
    for idx,event in enumerate(events):
        key=(event.tensor,event.row,event.column)
        val=value_map.get(key)
        if val is None:
            return failure("UNKNOWN_TENSOR_ELEMENT",idx,event)
        stage=boundary_stage[event.tensor]
        if event.kind=="CONTRIBUTE":
            count=val.producer_required_contributions
            if not 0<=event.index<count:
                return failure("INVALID_REDUCTION",idx,event,required=count)
            pos=(event.row,event.column,event.index) if ops[stage].kind in ("MatMul","Gemm") else (event.row,event.column)
            owner=_owner(points[stage],pos)
            if event.phi_owner!=owner:
                return failure("PHI_PRODUCER_OWNER_MISMATCH",idx,event,expected=owner)
            if key in ready or event.index in contributions.get(key,set()):
                return failure("DUPLICATE_OR_LATE_CONTRIBUTION",idx,event)
            contributions.setdefault(key,set()).add(event.index)
        elif event.kind=="COMMIT":
            if event.index!=-1 or event.phi_owner:
                return failure("MALFORMED_COMMIT",idx,event)
            if key in ready:
                return failure("DUPLICATE_COMMIT",idx,event)
            if len(contributions.get(key,set()))!=val.producer_required_contributions:
                return failure("PREMATURE_REDUCTION_COMMIT",idx,event,
                               completed=sorted(contributions.get(key,set())),
                               required=val.producer_required_contributions)
            slot=(stage,val.source_slot_address)
            if slot in occupancy:
                return failure("OVERWRITE_LIVE_TENSOR_VALUE",idx,event,slot=slot,
                               previous=occupancy[slot])
            occupancy[slot]=key
            ready.add(key)
        elif event.kind=="READ":
            if key not in ready:
                return failure("READ_BEFORE_READY",idx,event)
            count=val.consumer_required_reads
            if not 0<=event.index<count:
                return failure("INVALID_CONSUMER_INDEX",idx,event,required=count)
            downstream=stage+1
            pos=((event.row,event.index,event.column)
                 if ops[downstream].kind in ("MatMul","Gemm")
                 else (event.row,event.column))
            if event.phi_owner!=_owner(points[downstream],pos):
                return failure("PHI_CONSUMER_OWNER_MISMATCH",idx,event)
            used=reads.setdefault(key,set())
            if event.index in used:
                return failure("DUPLICATE_CONSUMER_READ",idx,event)
            used.add(event.index)
            if len(used)==count:
                slot=(stage,val.source_slot_address)
                if occupancy.get(slot)!=key:
                    return failure("INVALID_LAST_USE_SLOT",idx,event)
                del occupancy[slot]
        else:
            return failure("INVALID_EVENT_KIND",idx,event)
    if len(ready)!=len(value_map) or any(len(reads.get(key,set()))!=v.consumer_required_reads
                                         for key,v in value_map.items()) or occupancy:
        return {"schema":"fpgai-research/mapped-logical-events/v1",
                "status":"REJECTED","reason":"INCOMPLETE_VALUE_LIFETIMES",
                "event_index":len(events),"counterexample":{
                    "committed":len(ready),"required":len(value_map),
                    "remaining_live":len(occupancy)},"original_vitis_calls":0}
    return {"schema":"fpgai-research/mapped-logical-events/v1",
            "status":"VERIFIED","events_checked":len(events),
            "values_checked":len(value_map),"mapping_projection_sha256":mapped.projection_sha256,
            "event_sha256":sha256(json.dumps([asdict(e) for e in events],
                     sort_keys=True,default=list).encode()).hexdigest(),
            "evidence":"E0-Phi-mapped-logical-value-order-only",
            "original_vitis_calls":0,
            "limits":["Not an RTL/cycle-level timing proof",
                      "Sequential single-reader rows only",
                      "Not a source trace or bank port feasibility proof"]}
