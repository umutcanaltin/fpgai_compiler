"""Bounded NCHW Conv2D fan-out CST read/last-use certificate (E0).

Graph: Conv -> (Relu, Conv) -> Add -> Relu, all produced NCHW
tensors. Group=1, dilation=(1,1), constant weights, static strides and
explicit padding. Phi coordinates are per-contribution owner obligations.
This is a LOGICAL source-value schedule and does not claim HLS source-loop
order, RTL scheduling, achieved ports or BRAM mapping.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
from math import prod

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint,functional_fingerprint,verify_refinement,
)

SCHEMA="fpgai-research/cst-conv-diamond-logical-E0/v1"


@dataclass(frozen=True)
class ConvValueEvent:
    kind: str
    tensor: str
    element: int
    consumer: str=""
    output_element: int=-1
    input_port: int=-1
    reduction_k: int=-1
    phi_owner: tuple[int,int,int,int]|None=None


def _hash(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def admissible_conv_diamond(arch:ArchitecturalModel):
    if not isinstance(arch,ArchitecturalModel):
        raise TypeError("Resolved ArchitecturalModel required")
    model=arch.functional
    model.validate()
    ops=model.operators
    if (len(ops)!=5 or len(model.inputs)!=1 or len(model.outputs)!=1 or
            tuple(op.kind for op in ops)!=("Conv","Relu","Conv","Add","Relu")):
        raise ValueError("Only static Conv -> (Relu,Conv) -> Add -> Relu admitted")
    p,r0,r1,join,last=ops
    h,a,b,g,out=(op.outputs[0] for op in ops)
    if (len(p.inputs)!=2 or p.inputs[0]!=model.inputs[0] or
            len(r0.inputs)!=1 or r0.inputs[0]!=h or
            len(r1.inputs)!=2 or r1.inputs[0]!=h or
            join.inputs!=(a,b) or last.inputs!=(g,) or
            model.outputs!=(out,)):
        raise ValueError("Incorrect Conv multi-reader graph wiring")
    if any(len(model.tensor(n).shape)!=4 for n in (model.inputs[0],h,a,b,g,out,
                                                   p.inputs[1],r1.inputs[1])):
        raise ValueError("All Conv branch tensors must be NCHW/4D")
    if (any(model.tensor(op.inputs[1]).data is None for op in (p,r1)) or
            any(model.tensor(n).data is not None
                for n in (model.inputs[0],h,a,b,g,out))):
        raise ValueError("Only Conv RHS weights may be constant")
    if any(model.tensor(n).shape!=model.tensor(h).shape
           for n in (a,b,g,out)):
        raise ValueError("Both Conv readers must preserve output shape")
    for op in (p,r1):
        if (int(op.attr("group",1))!=1 or
                tuple(op.attr("dilations",(1,1)))!=(1,1) or
                op.attr("auto_pad","NOTSET")!="NOTSET"):
            raise ValueError("Grouped/dilated/automatic padding not admitted")
        if set(dict(op.attributes))-{"strides","pads","group","dilations","kernel_shape"}:
            raise ValueError("Unknown Conv attribute")
    if any(op.attributes for op in (r0,join,last)):
        raise ValueError("Unlowered elementwise attributes")
    verify_refinement(lower(arch))
    if len({(point.effective.precision_bits,point.effective.integer_bits)
            for point in arch.points})!=1:
        raise ValueError("Mixed numerical formats require explicit casts")
    for point in (arch.points[1],arch.points[3],arch.points[4]):
        req=point.effective
        if ((req.parallelism,req.pes,req.simd,req.serialization)!=(1,1,1,1)
                or req.tile or req.pipeline_ii is not None):
            raise ValueError("Unlowered elementwise Phi mapping")
    return h,a,b,g,out


def _flat(shape,index):
    value=0
    for coord,length in zip(index,shape):
        value=value*length+coord
    return value


def _unflat(shape,index):
    result=[]
    for length in reversed(shape):
        result.append(index%length)
        index//=length
    return tuple(reversed(result))


def _read_sources(model,op,out_flat):
    """Canonical math multiset for an output element in logical reduction order."""
    shape=model.tensor(op.outputs[0]).shape
    pos=_unflat(shape,out_flat)
    if op.kind=="Conv":
        x=model.tensor(op.inputs[0]).shape
        w=model.tensor(op.inputs[1]).shape
        batch,oc,oy,ox=pos
        strides=tuple(op.attr("strides",(1,1)))
        pads=tuple(op.attr("pads",(0,0,0,0)))
        for ic in range(x[1]):
            for ky in range(w[2]):
                for kx in range(w[3]):
                    iy=oy*strides[0]+ky-pads[0]
                    ix=ox*strides[1]+kx-pads[1]
                    if 0<=iy<x[2] and 0<=ix<x[3]:
                        red=(ic*w[2]+ky)*w[3]+kx
                        yield (0,op.inputs[0],
                               _flat(x,(batch,ic,iy,ix)),
                               red,(batch,oc,oy,ox,ic,ky,kx))
    else:
        for port,name in enumerate(op.inputs):
            yield (port,name,out_flat,-1,None)


def conv_logical_events(arch:ArchitecturalModel,*,max_events:int=100_000):
    h,a,b,g,out=admissible_conv_diamond(arch)
    model=arch.functional
    if type(max_events) is not int or max_events<1:
        raise ValueError("Positive finite Conv E0 event bound required")
    produced={op.outputs[0] for op in model.operators}
    events=[]
    for ordinal,op in enumerate(model.operators):
        size=prod(model.tensor(op.outputs[0]).shape)
        for output_index in range(size):
            for port,name,source_index,reduction,coords in _read_sources(model,op,output_index):
                if name not in produced:
                    continue
                owner=None
                if coords is not None:
                    phi=coordinate(arch.points[ordinal],coords)
                    owner=(phi["replica"],phi["pe"],phi["lane"],phi["reuse_step"][4])
                events.append(ConvValueEvent("READ",name,source_index,
                                             op.name,output_index,port,reduction,owner))
            events.append(ConvValueEvent("COMMIT",op.outputs[0],output_index))
            if len(events)>max_events:
                raise ValueError("Conv source event budget exceeded")
    for index in range(prod(model.tensor(out).shape)):
        events.append(ConvValueEvent("READ",out,index,"EXTERNAL",index,0))
    if len(events)>max_events:
        raise ValueError("Conv sink event budget exceeded")
    return tuple(events)


def disjoint_conv_offsets(arch):
    admissible_conv_diamond(arch)
    offsets={}
    used=0
    for op in arch.functional.operators:
        name=op.outputs[0]
        offsets[name]=used
        used+=prod(arch.functional.tensor(name).shape)
    return offsets


def verify_conv_logical(arch:ArchitecturalModel,offsets:dict[str,int],
                        events:tuple[ConvValueEvent,...],*,max_events:int=100_000):
    admissible_conv_diamond(arch)
    model=arch.functional
    names={op.outputs[0] for op in model.operators}
    if (type(offsets) is not dict or set(offsets)!=names or
            any(type(v) is not int or v<0 for v in offsets.values())):
        raise ValueError("Exact nonnegative Conv scratch-base map required")
    if (type(events) is not tuple or len(events)>max_events or
            any(not isinstance(x,ConvValueEvent) for x in events)):
        raise ValueError("Bounded typed Conv event stream required")
    expected=Counter()
    for ordinal,op in enumerate(model.operators):
        for oi in range(prod(model.tensor(op.outputs[0]).shape)):
            for port,name,idx,red,coords in _read_sources(model,op,oi):
                if name in names:
                    owner=None
                    if coords is not None:
                        phi=coordinate(arch.points[ordinal],coords)
                        owner=(phi["replica"],phi["pe"],phi["lane"],phi["reuse_step"][4])
                    expected[(name,idx,op.name,oi,port,red,owner)]+=1
    for oi in range(prod(model.tensor(model.outputs[0]).shape)):
        expected[(model.outputs[0],oi,"EXTERNAL",oi,0,-1,None)]+=1
    sizes={name:prod(model.tensor(name).shape) for name in names}
    ready=set()
    live={}
    seen=Counter()
    releases={}
    def fail(reason,index,event,**other):
        return {"schema":SCHEMA,"status":"REJECTED","reason":reason,
                "event_index":index,"counterexample":{"event":asdict(event),**other},
                "original_vitis_calls":0}
    for index,event in enumerate(events):
        if (event.tensor not in names or type(event.element) is not int
                or not 0<=event.element<sizes[event.tensor]):
            return fail("UNKNOWN_SOURCE_ELEMENT",index,event)
        identity=(event.tensor,event.element)
        address=offsets[event.tensor]+event.element
        if event.kind=="COMMIT":
            if (event.consumer or event.output_element!=-1 or event.input_port!=-1
                    or event.reduction_k!=-1 or event.phi_owner is not None):
                return fail("INVALID_COMMIT_FIELDS",index,event)
            if identity in ready:
                return fail("DUPLICATE_COMMIT",index,event)
            if address in live:
                return fail("OVERWRITE_BEFORE_ALL_READS",index,event,
                            address=address,live_value=list(live[address]))
            ready.add(identity)
            live[address]=identity
        elif event.kind=="READ":
            token=(event.tensor,event.element,event.consumer,event.output_element,
                   event.input_port,event.reduction_k,event.phi_owner)
            if token not in expected:
                return fail("INVALID_CONV_READ_OR_PHI",index,event)
            if identity not in ready or live.get(address)!=identity:
                return fail("READ_BEFORE_COMMIT_OR_STALE",index,event,address=address)
            seen[token]+=1
            if seen[token]>expected[token]:
                return fail("DUPLICATE_REDUCTION_READ",index,event)
            if sum(seen[x] for x in expected if x[:2]==identity)==sum(
                    count for key,count in expected.items() if key[:2]==identity):
                releases[identity]=index
                del live[address]
        else:
            return fail("UNKNOWN_EVENT_KIND",index,event)
    if seen!=expected or len(ready)!=sum(sizes.values()) or live:
        return {"schema":SCHEMA,"status":"REJECTED",
                "reason":"INCOMPLETE_CONV_REDUCTION_OR_LAST_USE",
                "counterexample":{"actual_reads":sum(seen.values()),
                                  "needed_reads":sum(expected.values()),
                                  "live":len(live)},
                "original_vitis_calls":0}
    return {"schema":SCHEMA,"status":"VERIFIED_E0",
            "functional_sha256":functional_fingerprint(model),
            "architecture_sha256":architecture_fingerprint(arch),
            "source_event_sha256":_hash([asdict(e) for e in events]),
            "logical_commits":len(ready),"logical_reads":sum(expected.values()),
            "conv_branch_intermediate_reads":sum(
                v for k,v in expected.items() if k[0]==model.operators[0].outputs[0]),
            "last_use_sha256":_hash(sorted((a,b,i) for (a,b),i in releases.items())),
            "source_scratch_slots":max(offsets[n]+sizes[n] for n in names),
            "original_vitis_calls":0,
            "evidence":"E0-bounded-Conv-NCHW-per-contribution-read-and-last-use",
            "limitations":[
                "Mathematical producer-major output order, NOT emitted HLS source loop order",
                "No source trace comparison, hardware cycle schedule, bank ports or RTL proof",
                "Groups/dilation/automatic padding and general DAG remain unsupported",
            ]}


def conv_contract(arch,offsets,*,max_events:int=100_000):
    events=conv_logical_events(arch,max_events=max_events)
    result=verify_conv_logical(arch,offsets,events,max_events=max_events)
    if result["status"]!="VERIFIED_E0":
        raise ValueError("UNSAFE_CONV_BOUNDARY "+result["reason"])
    return result
