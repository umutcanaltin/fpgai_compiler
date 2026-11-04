"""Compositional arithmetic certificate for static Phi loop partitions.

This certifies only the integer-domain indexing argument: positive tile,
replica, PE, fold and SIMD divisors give unique Euclidean quotient/remainder
decompositions for logical iteration coordinates. It is an algebraic
derivation of the declared source-loop *scheme*, not machine-checked
verification of emitted C++, fixed-point arithmetic, Vitis scheduling or RTL.

It requires constant work and memory, even for very large dimensions.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod

from fpgai_research.model_ir.contracts import ArchitecturePoint

SCHEMA = "fpgai-research/phi-euclidean-partition/v1"
_SUPPORTED = ("MatMul", "Gemm", "Conv")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _split(size: int, divisor: int) -> dict:
    quotient, tail = divmod(size, divisor)
    # Euclidean division is the algebraic invariant; this check is not
    # an exhaustive iteration over individual coordinates.
    if not (quotient * divisor + tail == size and 0 <= tail < divisor):
        raise ArithmeticError("EUCLIDEAN_DIVISION_INVARIANT_FAILED")
    return {"domain_size": size, "divisor": divisor,
            "full_blocks": quotient, "tail_size": tail,
            "block_count": quotient + int(tail > 0)}


def certify_phi_partition(point: ArchitecturePoint) -> dict:
    """Produce an O(rank) arithmetic index-partition derivation.

    Unique quotient/remainder decomposition is the proof *argument* for
    the specified nested loop bounds. No AST proof ties these formulas
    to a particular compiled C++ translation unit.
    """
    if not isinstance(point, ArchitecturePoint):
        raise TypeError("A typed ArchitecturePoint is required")
    kind = point.kind
    base = {"schema": SCHEMA, "operator": point.operator, "kind": kind,
            "evidence": "E0-algebraic-index-partition-derivation-only",
            "model_only": True, "actual_cpp_executed": False,
            "fixed_point_equivalence_proven": False,
            "rtl_equivalence_proven": False, "mechanically_verified": False}
    if kind not in _SUPPORTED:
        return {**base, "status": "UNSUPPORTED_KIND"}
    a = point.effective
    shape = point.iteration_shape
    rank = 3 if kind in ("MatMul", "Gemm") else 7
    tile = a.tile or shape
    if (len(shape) != rank or len(tile) != rank
            or any(type(v) is not int or v < 1 for v in (*shape, *tile))):
        raise ValueError("Invalid positive static iteration/tile geometry")
    if kind == "Conv" and any(tile[i] != shape[i] for i in (0, 4, 5, 6)):
        raise ValueError("Convolution batch/reduction tiling not materialized")
    p, pe, lanes, folds = (a.parallelism, a.pes, a.simd, a.serialization)
    if any(type(v) is not int or v < 1 for v in (p, pe, lanes, folds)):
        raise ValueError("Invalid positive spatial/temporal division factors")
    width = p * pe
    stride = width * folds
    # For every output-channel coordinate j, write j = t*tile_index+r;
    # r = stride*group+width*fold+pe*replica+pe_index.
    # divmod applied successively uniquely reconstructs all counters.
    output_size = shape[1]
    output_tile = tile[1]
    output_tile_split = _split(output_size, output_tile)
    full_block_slots = _split(output_tile, stride)
    last_tile_size = output_tile_split["tail_size"] or output_tile
    last_block_slots = _split(last_tile_size, stride)

    reduce_axis = 2 if kind in ("MatMul", "Gemm") else 4
    reduce_tile = tile[reduce_axis] if kind in ("MatMul", "Gemm") else shape[reduce_axis]
    reduction = {
        "axis": reduce_axis,
        "reduction_tile": _split(shape[reduce_axis], reduce_tile),
        "lane_groups_full_tile": _split(reduce_tile, lanes),
        "lane_groups_last_tile": _split(
            shape[reduce_axis] % reduce_tile or reduce_tile, lanes),
    }
    axes = [
        {"axis": axis, "partition": _split(extent, tile_extent)}
        for axis, (extent, tile_extent) in enumerate(zip(shape, tile))
    ]
    return {
        **base, "status": "ALGEBRAIC_PARTITION_DERIVED",
        "logical_iteration_count": prod(shape),
        "iteration_rank": rank,
        "shape": list(shape), "tile": list(tile),
        "mapping_sha256": _digest({
            "kind": kind, "shape": shape, "tile": tile,
            "precision": a.precision_bits, "integer_bits": a.integer_bits,
            "parallelism": p, "pes": pe, "simd": lanes,
            "serialization": folds, "pipeline_ii": a.pipeline_ii}),
        "output_channel_partition": {
            "domain": output_tile_split,
            "full_tile_fold_groups": full_block_slots,
            "last_tile_fold_groups": last_block_slots,
            "replicas": p, "pes": pe, "spatial_slots": width,
            "fold_factor": folds, "group_stride": stride,
        },
        "reduction_partition": reduction,
        "tile_axis_partitions": axes,
        "algebraic_obligations": [
            "For each positive t and x>=0, x=t*(x//t)+(x%t), 0<=x%t<t",
            "Channel-local r=group*(P*PE*F)+fold*(P*PE)+replica*PE+pe",
            "0<=fold<F, 0<=replica<P, 0<=pe<PE; guarded final group",
            "Each reduction-local coordinate has a unique SIMD-group/lane split",
            "Other output indices have unique tile quotient/local remainder",
        ],
        "claims": {
            "integer_partition_uniqueness": True,
            "global_loop_execution_proven": False,
            "actual_cpp_emission_verified": False,
            "numerical_semantics_proven": False,
        },
        "limitations": (
            "An O(rank) mathematical Euclidean-division argument for the "
            "declared positive-integer static mapping contracts. Neither "
            "a mechanized theorem-prover proof nor a source/AST/RTL proof. "
            "It does not turn BOUND_EXCEEDED iteration checks into exhaustive "
            "compiler-execution verification."
        ),
    }
