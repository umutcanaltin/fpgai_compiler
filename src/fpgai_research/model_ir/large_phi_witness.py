"""Deterministic bounded *witnesses* for large source-level Phi domains.

Computes emitted HLS loop counters directly from sampled iteration
indices, without enumerating the full reduction domain. This is
differential source-model testing and NOT an exhaustive proof, a
compiler-executed trace or any native Vitis/Vivado evidence.

For independent execution of actual generated C++, use the existing
validation.cpp_phi_trace.trace_cpp_phi on *bounded* models.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from random import Random

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturePoint

SCHEMA="fpgai-research/large-phi-boundary-witness/v1"
_SUPPORTED=("MatMul","Gemm","Conv")


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      allow_nan=False).encode("utf-8")


def _positions(size, steps):
    result={0,size-1,size//2}
    for step in steps:
        if step<=0:
            continue
        blocks=(size-1)//step
        for block in (0,1,2,blocks-1,blocks,blocks+1):
            if block<0:
                continue
            edge=block*step
            result.update(v for v in (edge-1,edge,edge+1)
                          if 0<=v<size)
    return sorted(result)


def _sample_indices(shape, steps, *, count, seed):
    options=[_positions(n,s) for n,s in zip(shape,steps)]
    origin=tuple(0 for _ in shape)
    upper=tuple(n-1 for n in shape)
    candidates={origin,upper}
    # Check axial tile and SIMD boundaries, including one step on both
    # sides. Also vary two axes at once to expose interacting guards.
    for axis,choices in enumerate(options):
        for value in choices:
            for base in (origin,upper):
                index=list(base)
                index[axis]=value
                candidates.add(tuple(index))
    for i in range(len(shape)):
        j=(i+1)%len(shape)
        for offset in range(max(len(options[i]),len(options[j]))):
            index=list(origin)
            index[i]=options[i][offset%len(options[i])]
            index[j]=options[j][offset%len(options[j])]
            candidates.add(tuple(index))
    # Deterministic seeded interior witnesses; no large tensor
    # allocation and no Python hash randomization dependency.
    rng=Random(seed)
    for _ in range(count*3):
        candidates.add(tuple(rng.randrange(size) for size in shape))
    selected=sorted(candidates)
    if len(selected)>count:
        # Retain axis/boundary points before spreading interior samples
        # deterministically. The selected set is stable across machines.
        mandatory={origin,upper}
        for axis,choices in enumerate(options):
            for value in choices:
                index=list(origin);index[axis]=value
                mandatory.add(tuple(index))
                index=list(upper);index[axis]=value
                mandatory.add(tuple(index))
        if len(mandatory)>=count:
            selected=sorted(mandatory)[:count]
        else:
            remaining=[x for x in selected if x not in mandatory]
            rng.shuffle(remaining)
            selected=sorted(mandatory | set(remaining[:count-len(mandatory)]))
    return selected


def _hls_counters(point, index):
    """Inverse of the currently emitted MatMul/Conv nested loops.

    We derive tile bases, ob/fold, PE/replica and ki/lane *without*
    calling coordinate(), which is the independently checked Phi oracle.
    """
    shape=point.iteration_shape
    a=point.effective
    tile=a.tile or shape
    width=a.parallelism*a.pes
    stride=width*a.serialization
    if point.kind in ("MatMul","Gemm"):
        mi,ni,kk=index
        tm,tn,tk=tile
        mb=(mi//tm)*tm
        nb=(ni//tn)*tn
        kb=(kk//tk)*tk
        local=ni-nb
        ob=(local//stride)*stride
        fold=(local-ob)//width
        slot=(local-ob)%width
        replica=slot//a.pes
        pe=slot%a.pes
        ki=kb+((kk-kb)//a.simd)*a.simd
        lane=kk-ki
        return {
            "tile":(mb//tm,nb//tn,kb//tk) if a.tile else (0,),
            "replica":replica,"pe":pe,"lane":lane,
            "reuse_step":(mi-mb,ob//stride,fold,(ki-kb)//a.simd),
            "precision":a.precision_bits,"pipeline_ii_request":a.pipeline_ii,
        }
    b,oc,oh,ow,ic,ky,kx=index
    _tb,toc,toh,tow,*_=tile
    cb=(oc//toc)*toc
    hb=(oh//toh)*toh
    wb=(ow//tow)*tow
    local=oc-cb
    ob=(local//stride)*stride
    fold=(local-ob)//width
    slot=(local-ob)%width
    replica=slot//a.pes
    pe=slot%a.pes
    icb=(ic//a.simd)*a.simd
    lane=ic-icb
    return {
        "tile":(b//tile[0],cb//toc,hb//toh,wb//tow,0,0,0)
                  if a.tile else (0,),
        "replica":replica,"pe":pe,"lane":lane,
        "reuse_step":(b,oh-hb,ow-wb,ob//stride,fold,icb//a.simd,ky,kx),
        "precision":a.precision_bits,"pipeline_ii_request":a.pipeline_ii,
    }


def sample_large_hls_phi(point:ArchitecturePoint, *, max_samples:int=256)->dict:
    """Return bounded deterministic witnesses; never claim global coverage."""
    if not isinstance(point,ArchitecturePoint):
        raise TypeError("A typed Architectural IR point is required")
    if type(max_samples) is not int or not 1<=max_samples<=4096:
        raise ValueError("max_samples must be an integer between 1 and 4096")
    if point.kind not in _SUPPORTED:
        return {"schema":SCHEMA,"operator":point.operator,
                "status":"UNSUPPORTED_KIND","sampled_iterations":0,
                "exhaustive":False}
    shape=point.iteration_shape
    a=point.effective
    tile=a.tile or shape
    if len(shape)!=(3 if point.kind in ("MatMul","Gemm") else 7):
        raise ValueError("Invalid supported compute rank")
    if len(tile)!=len(shape) or any(type(v) is not int or v<1 for v in (*shape,*tile)):
        raise ValueError("Nonpositive or mismatched iteration/tile geometry")
    if point.kind=="Conv" and any(tile[j]!=shape[j] for j in (0,4,5,6)):
        raise ValueError("Unsupported Conv reduction/batch tiling")
    if any(type(v) is not int or v<1 for v in
           (a.parallelism,a.pes,a.simd,a.serialization)):
        raise ValueError("Malformed spatial or temporal mapping")
    spatial=a.parallelism*a.pes*a.serialization
    steps=[[v] for v in tile]
    steps[1].extend((a.parallelism*a.pes,spatial))
    if point.kind in ("MatMul","Gemm"):
        steps[2].append(a.simd)
    else:
        steps[4].append(a.simd)
    payload={
        "operator":point.operator,"kind":point.kind,
        "shape":shape,"tile":tile,
        "replicas":a.parallelism,"pes":a.pes,"simd":a.simd,
        "fold":a.serialization,"ii":a.pipeline_ii,
    }
    fingerprint=sha256(_canonical(payload)).hexdigest()
    indices=_sample_indices(shape,steps,count=max_samples,
                            seed=int(fingerprint[:16],16))
    for index in indices:
        expected=coordinate(point,index)
        emitted=_hls_counters(point,index)
        for field in ("tile","replica","pe","lane","reuse_step",
                      "precision","pipeline_ii_request"):
            if emitted[field]!=expected[field]:
                raise AssertionError(
                    f"MODELIR-LARGE-PHI: {point.operator} {index} "
                    f"{field} HLS={emitted[field]} Phi={expected[field]}"
                )
    total=prod(shape)
    return {
        "schema":SCHEMA,"operator":point.operator,
        "status":"SAMPLED_COUNTERS_MATCH_PHI",
        "expected_iterations":total,"sampled_iterations":len(indices),
        "sample_limit":max_samples,
        "sample_coordinates_sha256":sha256(_canonical(indices)).hexdigest(),
        "mapping_contract_sha256":fingerprint,
        "all_samples_matched":True,
        "exhaustive":False,
        "vendor_tools_executed":False,
        "actual_cpp_executed":False,
        "proof_limitation":(
            "Selected tile/tail/boundary and deterministic interior "
            "iterations are checked against the source-loop inverse. "
            "Unvisited iterations remain unverified; this is not a "
            "complete refinement proof or a GCC-executed HLS trace."
        ),
    }
