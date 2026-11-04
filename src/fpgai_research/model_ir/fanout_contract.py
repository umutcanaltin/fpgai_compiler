"""CST R2f: bounded compositional fan-out MatMul readiness/source-event certificates.

A separate, strictly scoped E0 research contract for 2..4 independent
MatMul readers of one MatMul producer, followed by an Add chain and Relu.

Unlike the historical fixed five-operator R2b/R2e contract:
* all producer X*W reduction contributions are explicit events;
* each reader independently maps its (row, column, reduction) to Phi;
* Add-chain edges each have typed operand READ and output COMMIT;
* source offset collisions and final per-element use are checked;
* issue-order conformance is a SEPARATE property from event safety.

Neither certificate is a hardware cycle/port/BRAM proof or a theorem
about arbitrary DAGs or generated executable C++.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod

from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import (
    verify_refinement, architecture_fingerprint, functional_fingerprint,
)

SCHEMA="fpgai-research/cst-R2f-fanout-2-to-4-source-events-E0/v1"


@dataclass(frozen=True)
class FanoutEvent:
    kind: str                       # CONTRIBUTE, COMMIT, READ
    tensor: str
    element: int
    operation: str = ""             # producer or consumer op identity
    output_element: int = -1        # for READ of an operand
    reduction: int = -1             # for producer CONTRIBUTE
    source_element: int = -1        # input X flat index
    weight_element: int = -1        # W0 flat index
    phi_owner: tuple[int,int,int,int] | None = None


@dataclass(frozen=True)
class FanoutContract:
    m: int
    k: int
    n: int
    q: int
    input: str
    producer: str
    h: str
    readers: tuple[str,...]
    reader_values: tuple[str,...]
    add_ops: tuple[str,...]
    add_values: tuple[str,...]
    final: str
    weights: tuple[str,...]

    @property
    def intermediates(self)->tuple[str,...]:
        return (self.h,)+self.reader_values+self.add_values


def admit_fanout(arch: ArchitecturalModel)->FanoutContract:
    if not isinstance(arch,ArchitecturalModel):
        raise TypeError("Resolved ArchitecturalModel required")
    model=arch.functional
    model.validate()
    ops=model.operators
    if len(ops)<5 or len(ops)>9:
        raise ValueError("Fanout compiler admits only 2..4 readers plus Add chain and Relu")
    number=(len(ops)-1)//2
    if (number not in (2,3,4) or len(ops)!=2*number+1
            or tuple(op.kind for op in ops)!=
               ("MatMul",)*(number+1)+("Add",)*(number-1)+("Relu",)
            or any(op.attributes for op in ops)):
        raise ValueError("Exact MatMul fanout 2..4, Add chain, Relu topology required")
    expected_names=tuple(op.name for op in ops)
    if (tuple(point.operator for point in arch.points)!=expected_names
            or any(point.kind!=op.kind for point,op in zip(arch.points,ops))):
        raise ValueError("Architecture points must preserve exact operator order/kind")
    verify_refinement(lower(arch))
    producer=ops[0]
    readers=ops[1:1+number]
    adds=ops[1+number:-1]
    final=ops[-1]
    inp=model.inputs[0] if len(model.inputs)==1 else None
    if (inp is None or len(producer.inputs)!=2 or
            producer.inputs[0]!=inp or len(producer.outputs)!=1 or
            len(model.outputs)!=1):
        raise ValueError("One input and single producer MatMul output required")
    h=producer.outputs[0]
    wx=producer.inputs[1]
    weights=(wx,)+tuple(r.inputs[1] for r in readers if len(r.inputs)==2)
    if len(weights)!=number+1 or len(set(weights))!=number+1:
        raise ValueError("Distinct constant MatMul weights required")
    if (any(model.tensor(w).data is None for w in weights)
            or any(model.tensor(t).data is not None for t in
                   (inp,h,*[op.outputs[0] for op in ops[1:]]))):
        raise ValueError("Weights constant; input/intermediate/output nonconstant")
    m,k=model.tensor(inp).shape
    wn,n=model.tensor(wx).shape
    if wn!=k or model.tensor(h).shape!=(m,n):
        raise ValueError("Producer H MatMul geometry mismatch")
    q=None
    for r in readers:
        if len(r.inputs)!=2 or r.inputs[0]!=h:
            raise ValueError("Every reader must consume the exact H producer")
        shape=model.tensor(r.inputs[1]).shape
        if len(shape)!=2 or shape[0]!=n or model.tensor(r.outputs[0]).shape!=(m,shape[1]):
            raise ValueError("Reader MatMul dimensions mismatch")
        if q is None:
            q=shape[1]
        elif q!=shape[1]:
            raise ValueError("Only equal-shaped reader outputs supported by Add chain")
    values=tuple(r.outputs[0] for r in readers)
    current=values[0]
    add_values=[]
    for i,add in enumerate(adds):
        if add.inputs!=(current,values[i+1]) or model.tensor(add.outputs[0]).shape!=(m,q):
            raise ValueError("Add chain must join each reader in fixed topological order")
        current=add.outputs[0]
        add_values.append(current)
    if final.inputs!=(current,) or model.outputs!=final.outputs:
        raise ValueError("Relu must consume final Add result and be output")
    if model.tensor(model.outputs[0]).shape!=(m,q):
        raise ValueError("Relu output dimensions mismatch")
    if len(model.tensors)!=1+(number+1)+1+number+(number-1)+1:
        raise ValueError("Unexpected unused or extra tensors in fixed fanout contract")
    if any(not all(c.isalnum() or c=="_" for c in op.name) for op in ops):
        raise ValueError("Invalid operator identity")
    for point in arch.points:
        e=point.effective
        if e.tile or e.pipeline_ii is not None or e.serialization!=1 or e.parallelism!=1:
            raise ValueError("R2f source issue supports no tiling/pipeline/serialization/replicas")
        if point.operator in (producer.name,*[op.name for op in adds],final.name):
            if (e.pes,e.simd)!=(1,1):
                raise ValueError("Producer/Add/Relu scalar-only in R2f")
        elif not 1<=e.pes<=q or not 1<=e.simd<=n:
            raise ValueError("Unsupported R2f reader PE/SIMD mapping")
    if any(type(d) is not int or not 1<=d<=16 for d in (m,k,n,q)):
        raise ValueError("R2f static dimensions outside bounded pilot")
    return FanoutContract(
        m,k,n,q,inp,producer.name,h,tuple(r.name for r in readers),
        values,tuple(op.name for op in adds),tuple(add_values),
        final.name,weights,
    )


def disjoint_fanout_offsets(contract: FanoutContract)->dict[str,int]:
    offsets={}
    cursor=0
    for tensor in contract.intermediates:
        offsets[tensor]=cursor
        cursor+=contract.m*(contract.n if tensor==contract.h else contract.q)
    return offsets


def _issue_coordinate(ordinal:int,*,m:int,n:int,q:int,width:int):
    """Runtime-style inverse rank decoder, independent of the forward oracle."""
    row=ordinal//(n*q)
    remaining=ordinal%(n*q)
    for start in range(0,q,width):
        active=min(width,q-start)
        if remaining<active*n:
            kk=remaining//active
            col=start+remaining%active
            return row,col,kk
        remaining-=active*n
    raise AssertionError("Complete finite source issue decoder exhausted")


def generate_fanout_events(arch:ArchitecturalModel, *,
                            max_events:int=100_000)->tuple[FanoutEvent,...]:
    spec=admit_fanout(arch)
    if type(max_events) is not int or max_events<1:
        raise ValueError("Positive original source-event budget required")
    m,k,n,q=spec.m,spec.k,spec.n,spec.q
    events=[]
    for row in range(m):
        for col in range(n):
            h_id=row*n+col
            for kk in range(k):
                events.append(FanoutEvent("CONTRIBUTE",spec.h,h_id,
                    spec.producer,reduction=kk,
                    source_element=row*k+kk,weight_element=kk*n+col))
            events.append(FanoutEvent("COMMIT",spec.h,h_id))
    counts=[{} for _ in spec.readers]
    for ordinal in range(m*n*q):
        for stage,reader in enumerate(spec.readers):
            point=arch.points[stage+1]
            width=point.effective.pes
            lanes=point.effective.simd
            row,col,kk=_issue_coordinate(ordinal,m=m,n=n,q=q,width=width)
            out=row*q+col
            events.append(FanoutEvent("READ",spec.h,row*n+kk,
                reader,output_element=out,phi_owner=(
                    0,col%width,kk%lanes,0)))
            counter=counts[stage]
            counter[out]=counter.get(out,0)+1
            if counter[out]==n:
                events.append(FanoutEvent("COMMIT",spec.reader_values[stage],out))
    # A topologically ordered Add-chain, with each edge READ named by
    # its actual consumer operation (supports shared reader outputs).
    current=spec.reader_values[0]
    for i,(name,result) in enumerate(zip(spec.add_ops,spec.add_values)):
        next_value=spec.reader_values[i+1]
        for out in range(m*q):
            events.append(FanoutEvent("READ",current,out,name,output_element=out))
            events.append(FanoutEvent("READ",next_value,out,name,output_element=out))
            events.append(FanoutEvent("COMMIT",result,out))
        current=result
    for out in range(m*q):
        events.append(FanoutEvent("READ",current,out,spec.final,output_element=out))
    if len(events)>max_events:
        raise ValueError("R2f source trace exceeds predeclared event bound")
    return tuple(events)


def _expected_issue_reads(arch: ArchitecturalModel,spec:FanoutContract)->tuple:
    """Independent forward closed-form rank: not the decoder or generator."""
    result=[]
    m,n,q=spec.m,spec.n,spec.q
    for reader_idx,reader in enumerate(spec.readers):
        e=arch.points[reader_idx+1].effective
        for row in range(m):
            for col in range(q):
                group=(col//e.pes)*e.pes
                active=min(e.pes,q-group)
                for kk in range(n):
                    rank=row*n*q+group*n+kk*active+(col-group)
                    result.append((rank,reader_idx,reader,row*n+kk,
                                   row*q+col,(0,col%e.pes,kk%e.simd,0)))
    return tuple((name,element,out,owner)
                 for _,_,name,element,out,owner in sorted(result))


def verify_fanout_issue(arch:ArchitecturalModel,
                         events:tuple[FanoutEvent,...])->dict:
    spec=admit_fanout(arch)
    expected=_expected_issue_reads(arch,spec)
    observed=tuple((e.operation,e.element,e.output_element,e.phi_owner)
                   for e in events if e.kind=="READ" and e.tensor==spec.h)
    if observed!=expected:
        first=next((i for i,(a,b) in enumerate(zip(observed,expected)) if a!=b),
                   min(len(observed),len(expected)))
        return {"schema":SCHEMA,"status":"REJECTED",
                "reason":"FANOUT_PHI_ISSUE_OR_OWNER_ORDER_MISMATCH",
                "first_H_read_index":first,
                "actual":observed[first] if first<len(observed) else None,
                "expected":expected[first] if first<len(expected) else None,
                "original_vendor_calls":0}
    return {"schema":SCHEMA,"status":"VERIFIED_E0",
            "H_read_population":len(observed),
            "independent_forward_rank_rule":True,
            "original_vendor_calls":0}


def verify_fanout_events(arch:ArchitecturalModel,
                          offsets:dict[str,int],
                          events:tuple[FanoutEvent,...],
                          *,max_events:int=100_000)->dict:
    """Independent value-readiness/reduction/live-storage check.

    Does NOT enforce the source policy's unique READ issue order.
    A different safe stream may be valid but not Phi-issue conforming.
    """
    spec=admit_fanout(arch)
    if (type(events) is not tuple
            or any(not isinstance(e,FanoutEvent) for e in events)
            or type(max_events) is not int or not 1<=max_events
            or len(events)>max_events):
        raise ValueError("Finite typed fanout event tuple required")
    sizes={tensor:spec.m*(spec.n if tensor==spec.h else spec.q)
           for tensor in spec.intermediates}
    if (type(offsets) is not dict or set(offsets)!=set(sizes)
            or any(type(v) is not int or v<0 for v in offsets.values())):
        raise ValueError("One exact nonnegative source offset per intermediate required")
    required={}
    for row in range(spec.m):
        for kk in range(spec.n):
            required[(spec.h,row*spec.n+kk)]={
                (reader,row*spec.q+col) for reader in spec.readers
                for col in range(spec.q)}
    merge_inputs={}
    current=spec.reader_values[0]
    for i,add_name in enumerate(spec.add_ops):
        merge_inputs[add_name]=(current,spec.reader_values[i+1])
        current=spec.add_values[i]
    for j,tensor in enumerate(spec.reader_values):
        consumer=spec.add_ops[0] if j<2 else spec.add_ops[j-1]
        for index in range(spec.m*spec.q):
            required[(tensor,index)]={(consumer,index)}
    for i,tensor in enumerate(spec.add_values):
        consumer=spec.add_ops[i+1] if i+1<len(spec.add_ops) else spec.final
        for index in range(spec.m*spec.q):
            required[(tensor,index)]={(consumer,index)}
    pending={key:set() for key in required}
    committed=set()
    live={}
    completed=set()
    producer_terms={}
    last_use={}
    def fail(reason,where,event,**detail):
        return {"schema":SCHEMA,"status":"REJECTED",
                "reason":reason,"event_index":where,
                "counterexample":{"event":asdict(event),**detail},
                "original_vendor_calls":0}
    for index,event in enumerate(events):
        key=(event.tensor,event.element)
        if (event.tensor not in sizes or type(event.element) is not int
                or not 0<=event.element<sizes[event.tensor]):
            return fail("INVALID_FANOUT_ELEMENT",index,event)
        address=offsets[event.tensor]+event.element
        if event.kind=="CONTRIBUTE":
            if (event.tensor!=spec.h or event.operation!=spec.producer
                    or event.output_element!=-1 or event.phi_owner is not None
                    or type(event.reduction) is not int
                    or not 0<=event.reduction<spec.k):
                return fail("INVALID_PRODUCER_CONTRIBUTE",index,event)
            row,col=divmod(event.element,spec.n)
            if (event.source_element!=row*spec.k+event.reduction
                    or event.weight_element!=event.reduction*spec.n+col):
                return fail("WRONG_PRODUCER_OPERAND_INDEX",index,event)
            if key in committed:
                return fail("PRODUCER_CONTRIBUTE_AFTER_COMMIT",index,event)
            seen=producer_terms.setdefault(key,set())
            if event.reduction in seen:
                return fail("DUPLICATE_PRODUCER_CONTRIBUTE",index,event)
            seen.add(event.reduction)
        elif event.kind=="COMMIT":
            if (event.operation or event.output_element!=-1
                    or event.reduction!=-1 or event.source_element!=-1
                    or event.weight_element!=-1 or event.phi_owner is not None):
                return fail("INVALID_FANOUT_COMMIT",index,event)
            if key in committed:
                return fail("DUPLICATE_FANOUT_COMMIT",index,event)
            if event.tensor==spec.h:
                missing=set(range(spec.k))-producer_terms.get(key,set())
                if missing:
                    return fail("PREMATURE_H_PRODUCER_COMMIT",index,event,
                                missing_terms=sorted(missing))
            elif event.tensor in spec.reader_values:
                reader=spec.readers[spec.reader_values.index(event.tensor)]
                row=event.element//spec.q
                missing=[kk for kk in range(spec.n)
                         if (reader,event.element) not in
                         pending[(spec.h,row*spec.n+kk)]]
                if missing:
                    return fail("PREMATURE_FANOUT_READER_COMMIT",index,event,
                                missing_reductions=missing)
            else:
                add_index=spec.add_values.index(event.tensor)
                add=spec.add_ops[add_index]
                inputs=merge_inputs[add]
                missing=[tensor for tensor in inputs
                         if (add,event.element) not in pending[(tensor,event.element)]]
                if missing:
                    return fail("PREMATURE_FANOUT_ADD_COMMIT",index,event,
                                missing_inputs=missing)
            if address in live:
                return fail("FANOUT_OVERWRITE_LIVE_VALUE",index,event,
                            live=live[address],address=address)
            live[address]=key
            committed.add(key)
        elif event.kind=="READ":
            if (event.reduction!=-1 or event.source_element!=-1
                    or event.weight_element!=-1):
                return fail("INVALID_READ_CONTRIBUTION_METADATA",index,event)
            if (event.operation,event.output_element) not in required[key]:
                return fail("UNKNOWN_FANOUT_CONSUMER_READ",index,event)
            if event.tensor==spec.h:
                if (type(event.phi_owner) is not tuple
                        or len(event.phi_owner)!=4
                        or any(type(v) is not int for v in event.phi_owner)):
                    return fail("INVALID_FANOUT_PHI_OWNER",index,event)
                reader_index=spec.readers.index(event.operation)
                e=arch.points[reader_index+1].effective
                row,kk=divmod(event.element,spec.n)
                output_row,col=divmod(event.output_element,spec.q)
                if row!=output_row or event.phi_owner!=(
                        0,col%e.pes,kk%e.simd,0):
                    return fail("WRONG_FANOUT_PHI_COORDINATE_OR_OWNER",index,event)
            elif event.phi_owner is not None:
                return fail("SPURIOUS_FANOUT_OWNER",index,event)
            if key not in committed or live.get(address)!=key:
                return fail("FANOUT_READ_BEFORE_READY_OR_STALE",index,event,
                            address=address)
            token=(event.operation,event.output_element)
            if token in pending[key]:
                return fail("DUPLICATE_FANOUT_READ",index,event)
            pending[key].add(token)
            if pending[key]==required[key]:
                del live[address]
                completed.add(key)
                last_use[key]=index
        else:
            return fail("UNKNOWN_FANOUT_EVENT_KIND",index,event)
    if (committed!=set(required) or completed!=set(required)
            or live or any(pending[key]!=need for key,need in required.items())):
        return {"schema":SCHEMA,"status":"REJECTED",
                "reason":"INCOMPLETE_FANOUT_DATAFLOW_OR_LAST_USE",
                "counterexample":{
                    "committed":len(committed),
                    "required_values":len(required),
                    "complete_last_reads":len(completed),
                    "live_addresses":len(live)},
                "original_vendor_calls":0}
    return {"schema":SCHEMA,"status":"VERIFIED_E0",
            "functional_sha256":functional_fingerprint(arch.functional),
            "architecture_sha256":architecture_fingerprint(arch),
            "fanout_readers":len(spec.readers),
            "producer_contributions":sum(len(a) for a in producer_terms.values()),
            "actual_H_reads":sum(len(pending[(spec.h,index)])
                                  for index in range(spec.m*spec.n)),
            "all_intermediate_value_count":len(required),
            "source_event_sha256":sha256(json.dumps(
                [asdict(e) for e in events],sort_keys=True,
                separators=(",",":"),allow_nan=False).encode()).hexdigest(),
            "last_use_indices":tuple(sorted(
                (tensor,index,where) for (tensor,index),where in last_use.items())),
            "source_scratch_slots":max(offsets[tensor]+size for tensor,size in sizes.items()),
            "evidence":"E0-only-independent-fanout-reduction-Phi-source-readiness",
            "original_GCC_calls":0,"original_vendor_calls":0,
            "limitations":[
                "Bounded 2..4 MatMul readers, single H producer, serial Add chain and Relu",
                "Logical source-event order, no executable C++ or hardware-cycle timing witness",
                "No physical RAM port/bank arbitration, bit-exact fixed-point or external firstness",
            ]}
