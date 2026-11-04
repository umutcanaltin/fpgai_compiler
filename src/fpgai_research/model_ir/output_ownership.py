"""Bounded E0 ownership check of the actual MatMul/Gemm and Conv HLS loop nests.

This reproduces the output-writing iteration order in codegen/hls/emitter.py,
not the reduction arithmetic. In particular, P replicas and PE lanes
address outputs through slot = replica * pe_count + pe. The enumeration
checks unique coverage including partial tiles and folding tails.
It NEVER asserts bit-exactness, equal RTL, native scheduling or throughput.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
from math import prod

SCHEMA = "fpgai-research/lowered-output-ownership-e0/v1"
SUPPORTED = ("MatMul", "Gemm", "Conv")


def _digest(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def enumerate_output_ownership(operator: dict, *, max_outputs: int = 100_000) -> dict:
    """Check every output coordinate has exactly one lowered HLS writer.

    Input is one hash-verified progressive_ir LoweredOperator record in
    its public, weight-free representation. A bound-exceeded check is
    *unverified*, not a pass. All writer coordinates below replicate
    the currently emitted MatMul/Gemm and Conv loops, including the
    guarded tails. Other kinds are explicitly out of scope.
    """
    if type(max_outputs) is not int or max_outputs < 1:
        raise ValueError("A positive integer enumeration bound is required")
    name = operator.get("operator")
    kind = operator.get("kind")
    if not isinstance(name, str) or not name:
        raise ValueError("An operator identity is required")
    base = {"schema": SCHEMA, "operator": name, "kind": kind,
            "evidence": "E0-bounded-HLS-output-ownership-only",
            "native_tools_executed": False,
            "bit_exactness_proven": False,
            "equivalent_rtl_proven": False}
    if kind not in SUPPORTED:
        return {**base, "status": "UNSUPPORTED_KIND", "expected_outputs": None,
                "visited_outputs": None, "normalized_ownership_sha256": None}
    mechanisms = operator.get("lowered_mechanisms")
    if not isinstance(mechanisms, dict):
        raise ValueError("Lowered mechanisms missing")
    shape = operator.get("iteration_shape")
    tile = mechanisms.get("tile_shape")
    if (not isinstance(shape, (list, tuple)) or
        not isinstance(tile, (list, tuple)) or len(shape) != len(tile) or
        len(shape) != (3 if kind in ("MatMul","Gemm") else 7) or
        any(type(v) is not int or v < 1 for v in (*shape, *tile))):
        raise ValueError("Invalid static iteration/tile geometry")
    p, e, fold = (mechanisms.get(k) for k in
                   ("replica_count", "pe_count", "fold_factor"))
    if any(type(v) is not int or v < 1 for v in (p,e,fold)):
        raise ValueError("Invalid spatial replica/PE or temporal fold count")
    channels = shape[1]
    expected = prod(shape[:2]) if kind in ("MatMul","Gemm") else prod(shape[:4])
    if expected > max_outputs:
        return {**base, "status": "BOUND_EXCEEDED",
                "expected_outputs": expected, "visited_outputs": None,
                "normalized_ownership_sha256": None}
    tile_channels = tile[1]
    capacity = p * e
    # Each normalized owner contains only the flat spatial slot, not
    # the PE/replica factorization. So equivalent channel ownership is
    # provable independently of differing generated loop structure.
    records = {}
    def assign(output, owner):
        if output in records:
            raise AssertionError(
                f"MODELIR-OWNERSHIP: duplicate write to {name} output {output}")
        records[output] = owner

    if kind in ("MatMul","Gemm"):
        m, n, _k = shape
        tm, tn, _tk = tile
        for mb in range(0,m,tm):
            for nb in range(0,n,tn):
                for ob in range(0,tn,capacity*fold):
                    for f in range(fold):
                        for replica in range(p):
                            for pe in range(e):
                                ni = nb+ob+f*capacity+replica*e+pe
                                if ni >= n or ni >= nb+tn:
                                    continue
                                for mi in range(mb,min(mb+tm,m)):
                                    assign((mi,ni), (mb,nb,ob,f,replica*e+pe))
    else:
        bsz,co,ho,wo,*_ = shape
        _tb,toc,toh,tow,*_ = tile
        for b in range(bsz):
            for cb in range(0,co,toc):
                for hb in range(0,ho,toh):
                    for wb in range(0,wo,tow):
                        for ob in range(0,toc,capacity*fold):
                            for f in range(fold):
                                for replica in range(p):
                                    for pe in range(e):
                                        oc=cb+ob+f*capacity+replica*e+pe
                                        if oc >= co or oc >= cb+toc:
                                            continue
                                        for y in range(hb,min(hb+toh,ho)):
                                            for x in range(wb,min(wb+tow,wo)):
                                                assign((b,oc,y,x),(cb,hb,wb,ob,f,replica*e+pe))
    if len(records)!=expected:
        raise AssertionError(
            f"MODELIR-OWNERSHIP: {name} writes {len(records)}/{expected} outputs")
    signature = _digest([
        [list(coord),list(records[coord])] for coord in sorted(records)
    ])
    return {**base, "status": "EXHAUSTIVE_COVERAGE",
            "expected_outputs": expected, "visited_outputs":len(records),
            "spatial_slots":capacity, "replica_count":p,"pe_count":e,
            "fold_factor":fold,"normalized_ownership_sha256":signature,
            "coverage_complete":True,"collisions":0,
            "limitations":(
                "Checks one-writer output ownership of emitted static loop nests "
                "only, not reduction arithmetic, fixed-point rounding, memory "
                "hazards, physical scheduling or hardware equivalence."
            )}
