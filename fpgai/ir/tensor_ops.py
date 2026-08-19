from __future__ import annotations

from math import prod
from typing import Any, Iterable

import numpy as np


def normalize_axis(axis: int, rank: int) -> int:
    value = int(axis)
    if value < 0:
        value += int(rank)
    if value < 0 or value >= int(rank):
        raise ValueError(f"IRTENSOR001: axis {axis} is invalid for rank {rank}")
    return value


def axis_geometry(shape: Iterable[int], axis: int) -> tuple[int, int, int]:
    dims = tuple(int(x) for x in shape)
    resolved = normalize_axis(axis, len(dims))
    return int(prod(dims[:resolved]) if resolved else 1), dims[resolved], int(prod(dims[resolved + 1:]) if resolved + 1 < len(dims) else 1)


def _constant_list(graph: Any, name: str | None) -> list[int] | None:
    if not name:
        return None
    value = (getattr(graph, "constants", {}) or {}).get(str(name))
    if value is None:
        return None
    return [int(x) for x in np.asarray(value).reshape(-1).tolist()]


def resolve_slice_spec(graph: Any, op: Any, input_shape: Iterable[int]) -> dict[str, int]:
    shape = tuple(int(x) for x in input_shape)
    attrs = dict(getattr(op, "attrs", {}) or {})
    inputs = list(getattr(op, "inputs", []) or [])
    starts = attrs.get("starts") or (_constant_list(graph, inputs[1]) if len(inputs) > 1 else None)
    ends = attrs.get("ends") or (_constant_list(graph, inputs[2]) if len(inputs) > 2 else None)
    axes = attrs.get("axes") or (_constant_list(graph, inputs[3]) if len(inputs) > 3 else None)
    steps = attrs.get("steps") or (_constant_list(graph, inputs[4]) if len(inputs) > 4 else None)
    if starts is None or ends is None:
        raise ValueError("IRTENSOR002: Slice requires static starts and ends")
    starts = [int(x) for x in starts]
    ends = [int(x) for x in ends]
    axes = [int(x) for x in (axes if axes is not None else range(len(starts)))]
    steps = [int(x) for x in (steps if steps is not None else [1] * len(starts))]
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        raise ValueError("IRTENSOR003: Slice starts/ends/axes/steps lengths must match")
    if len(starts) != 1:
        raise ValueError("IRTENSOR004: current static HLS Slice contract supports exactly one sliced axis")
    if steps[0] != 1:
        raise ValueError("IRTENSOR005: current static HLS Slice contract requires step=1")
    axis = normalize_axis(axes[0], len(shape))
    dim = shape[axis]
    start = starts[0] + dim if starts[0] < 0 else starts[0]
    end = ends[0] + dim if ends[0] < 0 else ends[0]
    start = min(max(start, 0), dim)
    end = min(max(end, 0), dim)
    if end < start:
        end = start
    return {"axis": axis, "start": int(start), "end": int(end), "length": int(end - start)}


def slice_output_shape(graph: Any, op: Any, input_shape: Iterable[int]) -> tuple[int, ...]:
    shape = list(int(x) for x in input_shape)
    spec = resolve_slice_spec(graph, op, shape)
    shape[spec["axis"]] = spec["length"]
    return tuple(shape)


def resolve_resize_shape(graph: Any, op: Any, input_shape: Iterable[int]) -> tuple[int, ...]:
    shape = tuple(int(x) for x in input_shape)
    inputs = list(getattr(op, "inputs", []) or [])
    attrs = dict(getattr(op, "attrs", {}) or {})
    sizes = attrs.get("sizes")
    if sizes is None and len(inputs) > 3:
        sizes = _constant_list(graph, inputs[3])
    if sizes is not None:
        out = tuple(int(x) for x in sizes)
        if len(out) != len(shape) or any(x <= 0 for x in out):
            raise ValueError("IRTENSOR006: Resize sizes must be positive and match input rank")
        return out
    scales = attrs.get("scales")
    if scales is None and len(inputs) > 2:
        value = (getattr(graph, "constants", {}) or {}).get(str(inputs[2]))
        if value is not None:
            scales = [float(x) for x in np.asarray(value).reshape(-1).tolist()]
    if scales is None:
        raise ValueError("IRTENSOR007: Resize requires static sizes or scales")
    values = tuple(float(x) for x in scales)
    if len(values) != len(shape) or any(x <= 0 for x in values):
        raise ValueError("IRTENSOR008: Resize scales must be positive and match input rank")
    return tuple(max(1, int(round(dim * scale))) for dim, scale in zip(shape, values))


def gather_output_shape(data_shape: Iterable[int], index_shape: Iterable[int], axis: int) -> tuple[int, ...]:
    data = tuple(int(x) for x in data_shape)
    indices = tuple(int(x) for x in index_shape)
    resolved = normalize_axis(axis, len(data))
    return data[:resolved] + indices + data[resolved + 1:]
