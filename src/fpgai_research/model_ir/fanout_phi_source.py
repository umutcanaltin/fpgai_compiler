"""Phi-conditioned, source-order fan-out contract for a bounded static diamond.

Per MatMul reduction owner projection follows `architecture.resolver.coordinate`.
The source-order contract mirrors the shared MatMul lowerer's actual
tile/replica/PE/fold enumeration; a *separate executed GCC trace* validates
that this is how C++ really ran.  The two Relu readers and Add remain scalar.

No general DAG source compiler, multiported RAM, clock timing, or original
Vitis evidence is implied. A spatial owner is not a scheduling cycle.
"""
from __future__ import annotations

from hashlib import sha256
import json

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.fanout_lifetime import (
    LifetimeEvent, verify_fanout_events,
)
from fpgai_research.model_ir.semantics import architecture_fingerprint

SCHEMA="fpgai-research/cst-fanout-mapped-source-order-e0/v1"


def phi_producer_iteration_order(arch: ArchitecturalModel, *, max_iterations: int = 100_000):
    """All (m,n,k,replica,PE,lane,fold) iterations of mapped source MatMul.

    Mirror the *real* shared `_emit_row_matmul` spatial/reduction loops, while
    checking each owner against the independently resolved Phi coordinate.
    """
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Positive finite mapped contribution budget required")
    from fpgai_research.codegen.hls.fanout_diamond import _admit
    model=arch.functional
    (m,k,n),_names,_readers,_weight=_admit(model)
    if tuple(p.operator for p in arch.points) != tuple(op.name for op in model.operators):
        raise ValueError("Resolved points do not match diamond graph identities")
    p=arch.points[0]
    if p.kind!="MatMul" or p.iteration_shape != (m,n,k):
        raise ValueError("Producer Phi geometry differs from Functional MatMul")
    a=p.effective
    tn=(a.tile or p.iteration_shape)[1]
    tk=(a.tile or p.iteration_shape)[2]
    width=a.parallelism*a.pes
    if m*n*k > max_iterations:
        raise ValueError("Mapped source reduction iterations exceed finite bound")
    result=[]
    for mi in range(m):
        for nb in range(0,n,tn):
            for ob in range(0,tn,width*a.serialization):
                for fold in range(a.serialization):
                    for replica in range(a.parallelism):
                        for pe in range(a.pes):
                            ni=nb+ob+fold*width+replica*a.pes+pe
                            if ni>=n or ni>=nb+tn:
                                continue
                            for kb in range(0,k,tk):
                                for ko in range(0,tk,a.simd):
                                    for lane in range(a.simd):
                                        kk=kb+ko+lane
                                        if kk>=k or ko+lane>=tk:
                                            continue
                                        owner=coordinate(p,(mi,ni,kk))
                                        if (owner["replica"],owner["pe"],owner["lane"],
                                            owner["reuse_step"][2]) != (
                                                replica,pe,lane,fold):
                                            raise ValueError("Mapped loop owner disagrees with Phi")
                                        result.append((mi,ni,kk,replica,pe,lane,fold))
    if len(result)!=m*n*k or len({tuple(x[:3]) for x in result})!=m*n*k:
        raise ValueError("Mapped MatMul contributions do not cover the full tensor uniquely")
    return tuple(result)


def mapped_source_order_events(arch: ArchitecturalModel, *, max_events:int=100_000):
    """Actual expected source COMMIT/CONSUME sequence for mapped producer."""
    from fpgai_research.codegen.hls.fanout_diamond import _admit
    from fpgai_research.codegen.hls.fanout_diamond import source_order_events
    model=arch.functional
    (m,k,n),(h,a,b,g),_names,_=_admit(model)
    iterations=phi_producer_iteration_order(arch,max_iterations=max_events)
    commit_order=[]
    seen=set()
    # The actual source finishes all reductions for one (m,n) before COMMIT.
    for mi,ni,_kk,_replica,_pe,_lane,_fold in iterations:
        pair=(mi,ni)
        if pair not in seen:
            commit_order.append(pair)
            seen.add(pair)
    events=[LifetimeEvent("COMMIT",h,mi*n+ni) for mi,ni in commit_order]
    normal=source_order_events(model)
    events.extend(x for x in normal if not (x.kind=="COMMIT" and x.tensor==h))
    if len(events)>max_events:
        raise ValueError("Mapped source events exceed finite bound")
    if len(commit_order)!=m*n:
        raise ValueError("Producer mapped commit count mismatch")
    return tuple(events)


def mapped_fanout_contract(arch: ArchitecturalModel, offsets: dict[str,int], *,
                           max_events:int=100_000) -> dict:
    """E0 correctness/identity for the exact source-order logical DAG schedule."""
    from fpgai_research.codegen.hls.fanout_diamond import _admit
    from fpgai_research.model_ir.lowering import lower
    from fpgai_research.model_ir.semantics import verify_refinement
    if not isinstance(arch,ArchitecturalModel):
        raise TypeError("Mapped fanout contract requires ArchitecturalModel")
    model=arch.functional
    _admit(model)
    verify_refinement(lower(arch))
    if len({(p.effective.precision_bits,p.effective.integer_bits) for p in arch.points})!=1:
        raise ValueError("Mixed numerical formats cannot be lowered")
    for point in arch.points[1:]:
        if (point.effective.parallelism,point.effective.pes,
                point.effective.simd,point.effective.serialization)!=(1,1,1,1):
            raise ValueError("Fanout consumer spatial mapping not implemented")
        if point.effective.tile or point.effective.pipeline_ii is not None:
            raise ValueError("Fanout scalar consumer tile/pipeline lowering unavailable")
    if type(max_events) is not int or max_events<1:
        raise ValueError("Positive finite E0 bound required")
    iterations=phi_producer_iteration_order(arch,max_iterations=max_events)
    events=mapped_source_order_events(arch,max_events=max_events)
    proof=verify_fanout_events(model,offsets,events,max_events=max_events)
    if proof["status"]!="VERIFIED":
        raise ValueError("MAPPED_FANOUT_ALIAS_UNSAFE: "+str(proof["reason"]))
    payload=json.dumps(iterations,separators=(",",":"))
    return {
        "schema":SCHEMA,"status":"VERIFIED_E0",
        "architecture_sha256":architecture_fingerprint(arch),
        "producer_iteration_count":len(iterations),
        "logical_values":proof["logical_values"],
        "multi_reader_values":proof["multi_reader_values"],
        "source_event_sha256":proof["event_sha256"],
        "producer_phi_iteration_sha256":sha256(payload.encode()).hexdigest(),
        "native_vitis_calls":0,
        "evidence":"E0-bounded-mapped-MatMul-owner-plus-two-scalar-readers",
        "limitations":[
            "Producer Phi iteration ownership, not independently timed hardware",
            "Consumer branches are scalar, not spatially/temporally independently mapped",
            "Emitted GCC source refinement is a separate E1 step",
            "No generic DAG, port banking, Vitis HLS or RTL evidence",
        ],
    }
