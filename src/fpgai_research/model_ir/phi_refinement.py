"""E0 bounded refinement check: Phi versus the emitted HLS iteration loops.

Enumerates two *independently expressed* mappings:
  - formal architecture coordinate(point, logical_iteration)
  - the nested MatMul/Gemm or Conv HLS loop counters in emitter.py

Each logical iteration must occur exactly once and have the same tile,
replica, PE, SIMD lane, temporal reuse step and pipeline request.

This is NOT a fixed-point arithmetic proof, a guarantee of native HLS
scheduling, or a statement about routed RTL equivalence.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturePoint
from fpgai_research.model_ir.large_phi_witness import sample_large_hls_phi
from fpgai_research.model_ir.division_certificate import certify_phi_partition

SCHEMA="fpgai-research/hls-phi-refinement-e0/v1"
SUPPORTED=("MatMul","Gemm","Conv")


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def verify_hls_phi(point: ArchitecturePoint, *, max_iterations: int=20_000) -> dict:
    """Check independently enumerated emitted-loop iterations against Phi.

    The bound is on reduction *iterations*, not just distinct output
    writes. A large or unsupported case returns UNVERIFIED rather than
    accepting it. Semantic mismatches raise and cannot be reported as PASS.
    """
    if not isinstance(point,ArchitecturePoint):
        raise TypeError("Typed ArchitecturePoint required")
    if type(max_iterations) is not int or max_iterations<1:
        raise ValueError("Positive integer iteration bound required")
    shape=point.iteration_shape
    kind=point.kind
    arch=point.effective
    base={"schema":SCHEMA,"operator":point.operator,"kind":kind,
          "evidence":"E0-bounded-source-loop-to-Phi-refinement-only",
          "native_tools_executed":False,"bit_exactness_proven":False,
          "rtl_schedule_proven":False}
    if kind not in SUPPORTED:
        return {**base,"status":"UNSUPPORTED_KIND","expected_iterations":None,
                "verified_iterations":None,"mapping_sha256":None}
    if len(shape)!=(3 if kind in ("MatMul","Gemm") else 7) or any(
            type(n) is not int or n<1 for n in shape):
        raise ValueError("Invalid statically shaped compute operator")
    tile=arch.tile or shape
    if len(tile)!=len(shape) or any(type(n) is not int or n<1 for n in tile):
        raise ValueError("Invalid tile geometry")
    if kind=="Conv" and any(tile[i]!=shape[i] for i in (0,4,5,6)):
        raise ValueError("Unsupported convolution batch/reduction tiling")
    p,pe,lane_count,fold=arch.parallelism,arch.pes,arch.simd,arch.serialization
    if any(type(n) is not int or n<1 for n in (p,pe,lane_count,fold)):
        raise ValueError("Invalid spatial/reduction/fold geometry")
    expected=prod(shape)
    if expected>max_iterations:
        # Boundary sampling is diagnostic evidence, never a substitute
        # for the exhaustive source-to-Phi refinement obligation.
        witnesses=sample_large_hls_phi(point)
        certificate=certify_phi_partition(point)
        return {**base,"status":"BOUND_EXCEEDED",
                "expected_iterations":expected,"verified_iterations":None,
                "mapping_sha256":None,
                "sampled_refinement":witnesses,
                "algebraic_partition":certificate}
    width=p*pe
    seen=set()
    signatures=[]
    def match(index, *, tile_id, replica, pe_id, lane, reuse_step):
        if index in seen:
            raise AssertionError(
                f"MODELIR-PHI-REFINEMENT: duplicate iteration {point.operator} {index}")
        seen.add(index)
        phi=coordinate(point,index)
        emitted={
            "tile":tile_id,"replica":replica,"pe":pe_id,
            "lane":lane,"reuse_step":reuse_step,
            "pipeline_ii_request":arch.pipeline_ii,
        }
        for key,actual in emitted.items():
            if phi[key]!=actual:
                raise AssertionError(
                    f"MODELIR-PHI-REFINEMENT: {point.operator} {index} "
                    f"{key} emitted={actual} Phi={phi[key]}")
        if phi["precision"]!=arch.precision_bits:
            raise AssertionError("MODELIR-PHI-REFINEMENT: precision disagreement")
        signatures.append((index,tile_id,replica,pe_id,lane,reuse_step))

    if kind in ("MatMul","Gemm"):
        m,n,k=shape
        tm,tn,tk=tile
        for mb in range(0,m,tm):
            for nb in range(0,n,tn):
                for ob in range(0,tn,width*fold):
                    for f in range(fold):
                        for replica in range(p):
                            for pe_id in range(pe):
                                ni=nb+ob+f*width+replica*pe+pe_id
                                if ni>=n or ni>=nb+tn:
                                    continue
                                for mi in range(mb,min(mb+tm,m)):
                                    for kb in range(0,k,tk):
                                        for ki in range(kb,min(k,kb+tk),lane_count):
                                            for lane in range(lane_count):
                                                kk=ki+lane
                                                if kk>=k or kk>=kb+tk:
                                                    continue
                                                index=(mi,ni,kk)
                                                match(index,
                                                    tile_id=((mb//tm,nb//tn,kb//tk)
                                                             if arch.tile else (0,)),
                                                    replica=replica,pe_id=pe_id,lane=lane,
                                                    reuse_step=(mi-mb,
                                                                ob//(width*fold),f,
                                                                (ki-kb)//lane_count))
    else:
        batch,co,ho,wo,ci,kh,kw=shape
        _tb,toc,toh,tow,*_=tile
        for b in range(batch):
            for cb in range(0,co,toc):
                for hb in range(0,ho,toh):
                    for wb in range(0,wo,tow):
                        for ob in range(0,toc,width*fold):
                            for f in range(fold):
                                for replica in range(p):
                                    for pe_id in range(pe):
                                        oc=cb+ob+f*width+replica*pe+pe_id
                                        if oc>=co or oc>=cb+toc:
                                            continue
                                        for oh in range(hb,min(hb+toh,ho)):
                                            for ow in range(wb,min(wb+tow,wo)):
                                                for icb in range(0,ci,lane_count):
                                                    for ky in range(kh):
                                                        for kx in range(kw):
                                                            for lane in range(lane_count):
                                                                ic=icb+lane
                                                                if ic>=ci:
                                                                    continue
                                                                index=(b,oc,oh,ow,ic,ky,kx)
                                                                match(index,
                                                                    tile_id=((b//tile[0],cb//toc,hb//toh,wb//tow,0,0,0)
                                                                             if arch.tile else (0,)),
                                                                    replica=replica,pe_id=pe_id,lane=lane,
                                                                    reuse_step=(b,oh-hb,ow-wb,
                                                                                ob//(width*fold),f,
                                                                                icb//lane_count,ky,kx))
    if len(seen)!=expected:
        raise AssertionError(
            f"MODELIR-PHI-REFINEMENT: {point.operator} "
            f"mapped {len(seen)}/{expected} iterations")
    return {**base,"status":"EXHAUSTIVE_PHI_REFINEMENT",
            "expected_iterations":expected,"verified_iterations":len(seen),
            "mapping_sha256":_digest(sorted(signatures)),
            "coverage_complete":True,"collisions":0,
            "limitations":(
                "Proves bounded logical iteration-to-emitted-loop-counter "
                "consistency for MatMul/Gemm/Conv only. It does not prove "
                "HLS RTL equivalence, arithmetic rounding, optimization "
                "quality, synthesis schedule, or board execution."
            )}
