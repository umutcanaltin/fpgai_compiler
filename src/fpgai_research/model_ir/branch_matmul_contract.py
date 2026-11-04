"""Source-order, Phi-conditioned per-contribution tensor lifetime certificates.

Restricted *three MatMul* five-op diamond:
  X @ W0 -> H; H @ W1 -> A; H @ W2 -> B; A+B -> G; Relu(G).
The two consumers independently map reduction iterations into PE/SIMD/tile.
Every H[m,k] has Q distinct reads per consumer, not one abstract ACK.
Addresses are logical source-array slots, not hardware RAM banks or cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fpgai_research.architecture.resolver import coordinate
from hashlib import sha256
import json
from math import prod
import re

from fpgai_research.model_ir.contracts import FunctionalModel, ArchitecturalModel
from fpgai_research.model_ir.matmul_phi_order import mapped_matmul_order
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import (
    verify_refinement, functional_fingerprint, architecture_fingerprint,
)

SCHEMA="fpgai-research/cst-branch-matmul-per-contribution-e0/v1"


@dataclass(frozen=True)
class BranchEvent:
    kind: str             # COMMIT or READ
    tensor: str
    element: int
    consumer: str = ""
    output_element: int = -1
    phi_owner: tuple[int,int,int,int] | None = None  # (replica,PE,SIMD-lane,fold)


def _hash(value) -> str:
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def admit_branch(model: FunctionalModel):
    if not isinstance(model,FunctionalModel):
        raise TypeError("Typed FunctionalModel required")
    model.validate()
    if (len(model.operators)!=5 or len(model.tensors)!=9 or
            len(model.inputs)!=1 or len(model.outputs)!=1 or
            tuple(op.kind for op in model.operators)!=
            ("MatMul","MatMul","MatMul","Add","Relu") or
            any(op.attributes for op in model.operators)):
        raise ValueError("Only static MatMul -> (MatMul,MatMul) -> Add -> Relu supported")
    p,r0,r1,join,last=model.operators
    x=model.inputs[0]
    if any(len(op.outputs)!=1 for op in model.operators):
        raise ValueError("Single outputs only")
    h,a,b,g,y=[op.outputs[0] for op in model.operators]
    if (len(p.inputs)!=2 or p.inputs[0]!=x or
            len(r0.inputs)!=2 or r0.inputs[0]!=h or
            len(r1.inputs)!=2 or r1.inputs[0]!=h or
            join.inputs!=(a,b) or last.inputs!=(g,) or model.outputs!=(y,)):
        raise ValueError("Unsupported multi-reader port/tensor wiring")
    wx,w0,w1=[p.inputs[1],r0.inputs[1],r1.inputs[1]]
    if len({wx,w0,w1})!=3:
        raise ValueError("Independent constant weights required")
    tensors=(x,h,a,b,g,y)
    if any(model.tensor(name).data is not None for name in tensors):
        raise ValueError("Only three MatMul weights may be constant")
    if any(model.tensor(name).data is None for name in (wx,w0,w1)):
        raise ValueError("All three MatMul weights must be constant")
    if any(len(model.tensor(name).shape)!=2 for name in tensors+(wx,w0,w1)):
        raise ValueError("Only rank-two static tensors supported")
    m,k=model.tensor(x).shape
    kw,n=model.tensor(wx).shape
    if kw!=k or model.tensor(h).shape!=(m,n):
        raise ValueError("Producer MatMul shape mismatch")
    sizes=[]
    for w,t in ((w0,a),(w1,b)):
        if model.tensor(w).shape[0]!=n or model.tensor(t).shape[0]!=m:
            raise ValueError("Consumer MatMul shape mismatch")
        sizes.append(model.tensor(w).shape[1])
    if sizes[0]!=sizes[1] or sizes[0]<1:
        raise ValueError("Add requires matching MatMul output shapes")
    q=sizes[0]
    if any(model.tensor(t).shape!=(m,q) for t in (a,b,g,y)):
        raise ValueError("Add/Relu output shape mismatch")
    identities=[*tensors,wx,w0,w1,*(op.name for op in model.operators)]
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*",s) for s in identities):
        raise ValueError("Safe static C++ identifiers required")
    return (m,k,n,q),(h,a,b,g),(wx,w0,w1),model.operators


def _validate_arch(arch: ArchitecturalModel):
    if not isinstance(arch,ArchitecturalModel):
        raise TypeError("Resolved ArchitecturalModel required")
    dims,names,weights,ops=admit_branch(arch.functional)
    verify_refinement(lower(arch))
    formats={(p.effective.precision_bits,p.effective.integer_bits) for p in arch.points}
    if len(formats)!=1:
        raise ValueError("Mixed fixed-point formats require explicit cast lowering")
    for point in arch.points[3:]:
        a=point.effective
        if (a.parallelism,a.pes,a.simd,a.serialization)!=(1,1,1,1) or a.tile or a.pipeline_ii is not None:
            raise ValueError("Add/Relu mapped code unsupported; only scalar consumers")
    return dims,names,weights,ops


def branch_matmul_source_events(arch: ArchitecturalModel, *, max_events:int=100_000):
    (m,k,n,q),(h,a,b,g),weights,ops=_validate_arch(arch)
    if type(max_events) is not int or max_events<1:
        raise ValueError("Positive finite source events bound required")
    events=[]
    producer_iterations={}
    for stage,outname in enumerate((h,a,b)):
        iterations=mapped_matmul_order(arch.points[stage],max_iterations=max_events)
        producer_iterations[ops[stage].name]=iterations
        seen={}
        completed=set()
        for mi,ni,kk,*owner in iterations:
            key=(mi,ni)
            if key in completed:
                raise AssertionError("Producer contribution after COMMIT")
            if stage:
                events.append(BranchEvent("READ",h,mi*n+kk,ops[stage].name,
                                          mi*q+ni,tuple(owner)))
            used=seen.setdefault(key,set())
            if kk in used:
                raise AssertionError("Duplicate source contribution")
            used.add(kk)
            reduction_length=k if stage==0 else n
            if len(used)==reduction_length:
                completed.add(key)
                events.append(BranchEvent("COMMIT",outname,mi*(n if stage==0 else q)+ni))
        if len(completed)!=m*(n if stage==0 else q):
            raise AssertionError("Source MatMul failed output completeness")
    for idx in range(m*q):
        events.extend((
            BranchEvent("READ",a,idx,ops[3].name,idx),
            BranchEvent("READ",b,idx,ops[3].name,idx),
            BranchEvent("COMMIT",g,idx),
        ))
    for idx in range(m*q):
        events.append(BranchEvent("READ",g,idx,ops[4].name,idx))
    if len(events)>max_events:
        raise ValueError("Branch E0 source trace exceeds bounded event budget")
    return tuple(events),producer_iterations


def validate_branch_events(
    arch: ArchitecturalModel, offsets:dict[str,int],
    events:tuple[BranchEvent,...], *, max_events:int=100_000,
) -> dict:
    (m,k,n,q),(h,a,b,g),weights,ops=_validate_arch(arch)
    if (type(offsets) is not dict or set(offsets)!={h,a,b,g} or
            any(type(v) is not int or v<0 for v in offsets.values())):
        raise ValueError("Four exact nonnegative intermediate bases required")
    if (type(events) is not tuple or any(not isinstance(e,BranchEvent) for e in events)
            or type(max_events) is not int or max_events<1 or len(events)>max_events):
        raise ValueError("Bounded immutable typed BranchEvents required")
    sizes={h:m*n,a:m*q,b:m*q,g:m*q}
    expected={}
    for mi in range(m):
        for kk in range(n):
            key=(h,mi*n+kk)
            expected[key]={
                (op.name,mi*q+ni)
                for op in ops[1:3] for ni in range(q)
            }
    for tensor,op in ((a,ops[3]),(b,ops[3]),(g,ops[4])):
        for idx in range(m*q):
            expected[(tensor,idx)]={(op.name,idx)}
    seen_reads={}
    ready=set()
    live={}
    last_use={}
    def reject(reason,index,event,**details):
        return {"schema":SCHEMA,"status":"REJECTED","reason":reason,
                "event_index":index,
                "counterexample":{"event":asdict(event),**details},
                "original_vitis_calls":0}
    for i,event in enumerate(events):
        if event.tensor not in sizes or type(event.element) is not int or not 0<=event.element<sizes[event.tensor]:
            return reject("INVALID_ELEMENT",i,event)
        key=(event.tensor,event.element)
        addr=offsets[event.tensor]+event.element
        if event.kind=="COMMIT":
            if event.consumer or event.output_element!=-1 or event.phi_owner is not None:
                return reject("INVALID_PRODUCER_COMMIT",i,event)
            if key in ready:
                return reject("DUPLICATE_COMMIT",i,event)
            # Independent reduction-completeness obligation, not an
            # assumption borrowed from the event generator. For A/B,
            # every source H reduction contribution must have been READ
            # for this exact output before its COMMIT. For G, both
            # Add operand READs must precede the corresponding COMMIT.
            # The H producer's input reduction is not in this trace:
            # its completeness remains a separately scoped obligation.
            if event.tensor in (a,b):
                op_name=ops[1].name if event.tensor==a else ops[2].name
                row=event.element//q
                missing=[kk for kk in range(n)
                         if (op_name,event.element) not in
                         seen_reads.get((h,row*n+kk),set())]
                if missing:
                    return reject("PREMATURE_REDUCTION_COMMIT",i,event,
                                  operation=op_name,output=event.element,
                                  missing_reduction_indices=missing)
            elif event.tensor==g:
                add_op=ops[3].name
                missing=[tensor for tensor in (a,b)
                         if (add_op,event.element) not in
                         seen_reads.get((tensor,event.element),set())]
                if missing:
                    return reject("PREMATURE_ADD_COMMIT",i,event,
                                  operation=add_op,output=event.element,
                                  missing_inputs=missing)
            if addr in live:
                return reject("OVERWRITE_BEFORE_ALL_MAPPED_READS",i,event,
                              address=addr,live_value=list(live[addr]))
            ready.add(key)
            live[addr]=key
        elif event.kind=="READ":
            # A mapped READ certificate must carry the same physical iteration
            # owner as the resolved consumer operator, not a detached Phi hash.
            if event.tensor==h:
                if (type(event.phi_owner) is not tuple or len(event.phi_owner)!=4
                        or any(type(x) is not int or x<0 for x in event.phi_owner)):
                    return reject("MISSING_OR_INVALID_PHI_OWNER",i,event)
                if event.consumer not in (ops[1].name,ops[2].name):
                    return reject("UNKNOWN_READ_OWNER",i,event)
                stage=1 if event.consumer==ops[1].name else 2
                mi,kk=divmod(event.element,n)
                row,ni=divmod(event.output_element,q)
                if row!=mi or not 0<=ni<q:
                    return reject("WRONG_PHI_READ_COORDINATE",i,event)
                phi=coordinate(arch.points[stage],(mi,ni,kk))
                owner=(phi["replica"],phi["pe"],phi["lane"],
                       phi["reuse_step"][2])
                if event.phi_owner!=owner:
                    return reject("WRONG_PHI_READ_OWNER",i,event,
                                  required_owner=list(owner))
            elif event.phi_owner is not None:
                return reject("SPURIOUS_PHI_OWNER",i,event)
            token=(event.consumer,event.output_element)
            if token not in expected[key]:
                return reject("UNKNOWN_READ_OWNER",i,event)
            if key not in ready or live.get(addr)!=key:
                return reject("READ_BEFORE_READY_OR_STALE",i,event,address=addr)
            done=seen_reads.setdefault(key,set())
            if token in done:
                return reject("DUPLICATE_MAPPED_READ",i,event)
            done.add(token)
            if len(done)==len(expected[key]):
                last_use[key]=i
                del live[addr]
        else:
            return reject("UNKNOWN_EVENT_KIND",i,event)
    if ready!=set(expected) or any(seen_reads.get(key,set())!=tokens
                                  for key,tokens in expected.items()) or live:
        return {"schema":SCHEMA,"status":"REJECTED",
                "reason":"INCOMPLETE_CONTRIBUTIONS_OR_LAST_USE",
                "counterexample":{"ready":len(ready),"needed_values":len(expected),
                                  "complete_last_use":len(last_use),
                                  "live_addresses":len(live)},
                "original_vitis_calls":0}
    return {
        "schema":SCHEMA,"status":"VERIFIED_E0",
        "functional_sha256":functional_fingerprint(arch.functional),
        "architecture_sha256":architecture_fingerprint(arch),
        "logical_values":len(expected),
        "mapped_multi_reader_values":m*n,
        "actual_required_h_reads":2*m*n*q,
        "read_events":sum(e.kind=="READ" for e in events),
        "commit_events":sum(e.kind=="COMMIT" for e in events),
        "source_order_event_sha256":_hash([asdict(e) for e in events]),
        "last_use_sha256":_hash(sorted((a,b,i) for (a,b),i in last_use.items())),
        "source_scratch_slots":max(offsets[name]+sizes[name] for name in sizes),
        "original_vitis_calls":0,
        "evidence":"E0-exact-source-order-Phi-conditioned-contribution-READ-and-last-use",
        "limitations":[
            "Fixed source-loop order is not a scheduled hardware-cycle order",
            "Source scratch addresses, not physical RAM banks or port arbitration",
            "Only bounded three-MatMul diamond, not arbitrary tensor DAG",
            "H producer contribution completeness is outside this READ/COMMIT-only contract; no independent producer CONTRIBUTE events",
        ],
    }


def branch_contract(arch:ArchitecturalModel,offsets:dict[str,int],*,
                    max_events:int=100_000):
    events,iterations=branch_matmul_source_events(arch,max_events=max_events)
    result=validate_branch_events(arch,offsets,events,max_events=max_events)
    if result["status"]!="VERIFIED_E0":
        raise ValueError("BRANCH_ALIAS_UNSAFE: "+result["reason"]+" "+
                         str(result.get("counterexample")))
    result["phi_iterations"]={name:len(seq) for name,seq in iterations.items()}
    result["phi_iteration_sha256"]={name:_hash(seq)
                                    for name,seq in iterations.items()}
    return result
