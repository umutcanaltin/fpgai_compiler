"""Prospective R2b E0 alternative: actual Phi-conditioned reader issue order.

This is a new FINITE logical schedule/contract, not a generated C++ source,
HLS schedule, physical FIFO, bank-port or proof of novel compilation.
The original sequential GCC/Vivado study is immutable. A new E1 codegen
and independently executed C++ trace must be added before an R2b
source-level causality claim can be made.

Every producer reduction completes before the first reader starts. The
two readers then issue one mapped contribution alternately, using one
unchanged work-conserving PE-group issue rule for every Phi intervention.
Phi can now change the actual element-read order (unlike a PE-owner
label mutation). Add/Relu remain scalar, following both readers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.branch_matmul_contract import (
    BranchEvent, branch_matmul_source_events, validate_branch_events,
    admit_branch, _validate_arch,
)
from fpgai_research.model_ir.contracts import ArchitecturalModel


@dataclass(frozen=True)
class InterleavedBranchSchedule:
    events: tuple[BranchEvent,...]
    dimensions: tuple[int,int,int,int]
    source_schedule: str = "complete_producer_then_Phi_reader_round_robin/v1"


def _mapped_reader_issue(point, *, m:int,n:int,q:int,
                         max_events:int)->tuple[tuple[int,int,int,tuple[int,int,int,int]],...]:
    """One unchanged algorithm: per PE-group, each reduction step across PEs.

    Disallow tile/serialization/pipeline in this bounded prototype rather
    than silently ignore physical decisions. Other optional architecture
    features need a separately checked schedule grammar.
    """
    options=point.effective
    if (options.tile or options.serialization!=1
            or options.pipeline_ii is not None
            or options.parallelism!=1):
        raise ValueError("R2b v1 supports scalar/PE/SIMD only; no tile/fold/pipeline")
    width=options.pes
    if not 1<=width<=q or not 1<=options.simd<=n:
        raise ValueError("Unsupported bounded Phi issue geometry")
    if m*n*q>max_events:
        raise ValueError("Reader source schedule exceeds bounded event budget")
    order=[]
    for mi in range(m):
        for group in range(0,q,width):
            for kk in range(n):
                for ni in range(group,min(group+width,q)):
                    phi=coordinate(point,(mi,ni,kk))
                    owner=(phi["replica"],phi["pe"],phi["lane"],
                           phi["reuse_step"][2])
                    order.append((mi,ni,kk,owner))
    if len(order)!=m*n*q or len({(i,j,k) for i,j,k,_ in order})!=m*n*q:
        raise AssertionError("Phi issue schedule is missing or duplicating a reduction")
    return tuple(order)


def interleaved_branch_events(arch:ArchitecturalModel,*,
                              max_events:int=100_000)->InterleavedBranchSchedule:
    if type(max_events) is not int or max_events<1:
        raise ValueError("Positive integer event bound required")
    (m,k,n,q),(h,a,b,g),_weights,ops=_validate_arch(arch)
    # Producer source semantics stay EXACTLY those of original serial
    # branch study. The alternative changes only consumer issue order.
    serial,_=branch_matmul_source_events(arch,max_events=max_events)
    producer=tuple(e for e in serial if e.kind=="COMMIT" and e.tensor==h)
    if len(producer)!=m*n:
        raise AssertionError("Original producer COMMIT is incomplete")
    readers=[
        _mapped_reader_issue(arch.points[i],m=m,n=n,q=q,
                             max_events=max_events)
        for i in (1,2)
    ]
    if len(readers[0])!=len(readers[1]):
        raise AssertionError("Reader completeness mismatch")
    events=list(producer)
    count=[Counter(),Counter()]
    for index in range(len(readers[0])):
        # BOTH reader schedules are fully issued and source-order interleaved.
        # A different Phi may produce a different source READ ordering.
        for stage in (0,1):
            mi,ni,kk,owner=readers[stage][index]
            count[stage][(mi,ni)]+=1
            events.append(BranchEvent("READ",h,mi*n+kk,
                                      ops[stage+1].name,mi*q+ni,owner))
            if count[stage][(mi,ni)]==n:
                events.append(BranchEvent("COMMIT",(a,b)[stage],
                                          mi*q+ni))
    if (any(count[stage][(mi,ni)]!=n
            for stage in (0,1) for mi in range(m) for ni in range(q))):
        raise AssertionError("Reader output reduction not complete")
    for idx in range(m*q):
        events.extend((BranchEvent("READ",a,idx,ops[3].name,idx),
                       BranchEvent("READ",b,idx,ops[3].name,idx),
                       BranchEvent("COMMIT",g,idx)))
    for idx in range(m*q):
        events.append(BranchEvent("READ",g,idx,ops[4].name,idx))
    if len(events)>max_events:
        raise ValueError("Bounded interleaved schedule exceeded event budget")
    return InterleavedBranchSchedule(tuple(events),(m,k,n,q))


def release_positions(schedule:InterleavedBranchSchedule, tensor:str)->tuple[tuple[int,int],...]:
    """Source-event last-use ordinals, not logical owner hashes or cycles.

    Relies only on the event sequence, independent of the certificate
    emitter's SHA/owner representation; never promote this to E1 without
    an actual generated-source execution.
    """
    last={}
    for i,e in enumerate(schedule.events):
        if e.kind=="READ" and e.tensor==tensor:
            last[e.element]=i
    return tuple(sorted(last.items()))


def prove_interleaved_source_order(arch:ArchitecturalModel,
                                   offsets:dict[str,int],*,
                                   max_events:int=100_000)->dict:
    plan=interleaved_branch_events(arch,max_events=max_events)
    result=validate_branch_events(
        arch,offsets,plan.events,max_events=max_events)
    if result["status"]!="VERIFIED_E0":
        raise ValueError("Interleaved Phi source-order unsafe: "+
                         str(result.get("reason"))+" "+
                         str(result.get("counterexample")))
    # Distinct oracle: direct rank/owner arithmetic, never generator
    # iteration order or resolver.coordinate. A merely legal but
    # Phi-nonconforming event order is NOT a certified mapped schedule.
    from fpgai_research.model_ir.independent_phi_issue_reference import (
        validate_phi_issue_conformance,
    )
    independent=validate_phi_issue_conformance(arch,plan.events)
    if independent["status"]!="VERIFIED_E0":
        raise ValueError("Independent Phi source issue conformance failed: "+
                         str(independent))
    result["independent_phi_issue_conformance"]=independent
    (m,k,n,q)=plan.dimensions
    h=arch.functional.operators[0].outputs[0]
    result["schedule_policy"]=plan.source_schedule
    result["h_last_read_ordinals"]=list(release_positions(plan,h))
    result["mapping_event_effect_source_executed"]=False
    result["original_gcc_calls"]=0
    result["original_vendor_calls"]=0
    result["evidence_level"]="E0-LOGICAL-NOT-GCC-OR-HARDWARE"
    return result
