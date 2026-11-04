"""Resolve static ONNX Reshape vectors under allowzero=0 semantics.

Dynamic shapes, unknown dimensions and zero-sized tensors fail closed.
"""
from __future__ import annotations
from math import prod

def resolve_static_reshape(input_shape: tuple[int, ...], requested: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(requested, (tuple, list)) or not 1 <= len(requested) <= 8:
        raise ValueError("Reshape requires a rank-1 constant shape of length 1..8")
    if not all(type(v) is int for v in requested):
        raise ValueError("Reshape dimensions must be exact integers")
    if any(v < -1 for v in requested) or requested.count(-1) > 1:
        raise ValueError("Reshape permits at most one inferred -1 dimension")
    out = list(requested)
    for i, d in enumerate(out):
        if d == 0:
            if i >= len(input_shape):
                raise ValueError("Reshape zero-copy dimension exceeds input rank")
            out[i] = input_shape[i]
    total = prod(input_shape)
    specified = prod(v for v in out if v != -1)
    if specified <= 0 or total % specified:
        raise ValueError("Reshape does not preserve element count")
    out = tuple(total // specified if v == -1 else v for v in out)
    if any(v <= 0 for v in out) or prod(out) != total:
        raise ValueError("Reshape output geometry is invalid")
    return out
