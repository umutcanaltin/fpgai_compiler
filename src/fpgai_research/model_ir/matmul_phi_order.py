"""Bounded exact source-loop iteration order of an independently resolved MatMul Phi.

A portable spatial assignment plus deterministic *host source loop* order,
NOT the hardware execution order, issued II or clock-cycle schedule.
"""
from __future__ import annotations

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturePoint


def mapped_matmul_order(point: ArchitecturePoint, *, max_iterations:int=100_000):
    if not isinstance(point,ArchitecturePoint) or point.kind!="MatMul":
        raise TypeError("Expected resolved MatMul ArchitecturePoint")
    if type(max_iterations) is not int or max_iterations<1:
        raise ValueError("Positive finite MatMul iteration budget required")
    if len(point.iteration_shape)!=3:
        raise ValueError("Static MatMul must have (M,N,K) iteration geometry")
    m,n,k=point.iteration_shape
    if m*n*k>max_iterations:
        raise ValueError("Mapped MatMul iterations exceed finite bound")
    a=point.effective
    tm,tn,tk=a.tile or (m,n,k)
    width=a.parallelism*a.pes
    if min(tm,tn,tk,width,a.simd,a.serialization)<1:
        raise ValueError("Invalid nonpositive mapped loop factor")
    emitted=[]
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
                                        expected=coordinate(point,(mi,ni,kk))
                                        if (expected["replica"],expected["pe"],expected["lane"],
                                            expected["reuse_step"][2]) != (replica,pe,lane,fold):
                                            raise ValueError("Source loop assignment violates Phi")
                                        emitted.append((mi,ni,kk,replica,pe,lane,fold))
    if (len(emitted)!=m*n*k or
            len({(mi,ni,kk) for mi,ni,kk,*_ in emitted})!=m*n*k):
        raise ValueError("MatMul lowering lost or duplicated a logical reduction")
    return tuple(emitted)
