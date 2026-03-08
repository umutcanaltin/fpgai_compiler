from __future__ import annotations

from typing import Any, Dict, List, Set

from fpgai.engine.models import MemoryPlan, TensorPlacement, LayerDescriptor, CompilePlan


def _shape_to_numel(shape) -> int:
    if not shape:
        return 0
    total = 1
    for d in shape:
        if d in (None, -1):
            return 0
        total *= int(d)
    return total


def _dtype_nbytes(dtype: Any) -> int:
    if dtype is None:
        return 4
    s = str(dtype).lower()
    if "float64" in s or "double" in s or "int64" in s:
        return 8
    if "float16" in s or "half" in s or "int16" in s:
        return 2
    if "int8" in s or "uint8" in s:
        return 1
    return 4


def _tensor_nbytes(graph: Any, tensor_name: str) -> int:
    spec = getattr(graph, "tensors", {}).get(tensor_name)
    if spec is not None:
        if isinstance(spec, dict):
            shape = spec.get("shape", ())
            dtype = spec.get("dtype", None)
        else:
            shape = getattr(spec, "shape", ())
            dtype = getattr(spec, "dtype", None)
        return _shape_to_numel(shape) * _dtype_nbytes(dtype)

    const = getattr(graph, "constants", {}).get(tensor_name)
    if const is not None:
        shape = getattr(const, "shape", ())
        dtype = getattr(const, "dtype", None)
        return _shape_to_numel(shape) * _dtype_nbytes(dtype)

    return 0


def _choose_param_region(size_bytes: int) -> str:
    if size_bytes <= 32 * 1024:
        return "BRAM"
    if size_bytes <= 256 * 1024:
        return "URAM"
    return "DDR"


def _choose_activation_region(size_bytes: int) -> str:
    if size_bytes <= 64 * 1024:
        return "BRAM"
    return "DDR"


def _collect_producers_consumers(descriptors: List[LayerDescriptor]):
    producers: Dict[str, str] = {}
    consumers: Dict[str, List[str]] = {}

    for desc in descriptors:
        for out in desc.outputs:
            producers[out] = desc.node_name
        for inp in desc.inputs:
            consumers.setdefault(inp, []).append(desc.node_name)

    return producers, consumers


def make_memory_plan(graph: Any, descriptors: List[LayerDescriptor], compile_plan: CompilePlan) -> MemoryPlan:
    placements: List[TensorPlacement] = []
    totals: Dict[str, int] = {}

    producers, consumers = _collect_producers_consumers(descriptors)
    placed: Set[str] = set()

    # ---------------------------------------------------------
    # 1) Graph inputs -> HOST
    # ---------------------------------------------------------
    for name in getattr(graph, "inputs", []):
        size_bytes = _tensor_nbytes(graph, name)
        placements.append(
            TensorPlacement(
                tensor_name=name,
                kind="input",
                region="HOST",
                layout="raw",
                offset=None,
                size_bytes=size_bytes,
                double_buffer=False,
                producer=None,
                consumer=",".join(consumers.get(name, [])) if consumers.get(name) else None,
            )
        )
        placed.add(name)
        totals["HOST"] = totals.get("HOST", 0) + size_bytes

    # ---------------------------------------------------------
    # 2) Graph outputs -> HOST
    # ---------------------------------------------------------
    for name in getattr(graph, "outputs", []):
        if name in placed:
            continue
        size_bytes = _tensor_nbytes(graph, name)
        placements.append(
            TensorPlacement(
                tensor_name=name,
                kind="output",
                region="HOST",
                layout="raw",
                offset=None,
                size_bytes=size_bytes,
                double_buffer=False,
                producer=producers.get(name),
                consumer=None,
            )
        )
        placed.add(name)
        totals["HOST"] = totals.get("HOST", 0) + size_bytes

    # ---------------------------------------------------------
    # 3) Parameters -> BRAM / URAM / DDR
    # ---------------------------------------------------------
    for desc in descriptors:
        for pname in desc.param_names:
            if pname in placed:
                continue
            size_bytes = _tensor_nbytes(graph, pname)
            region = _choose_param_region(size_bytes)

            placements.append(
                TensorPlacement(
                    tensor_name=pname,
                    kind="weight",
                    region=region,
                    layout="raw",
                    offset=None,
                    size_bytes=size_bytes,
                    double_buffer=(region == "DDR"),
                    producer=None,
                    consumer=desc.node_name,
                )
            )
            placed.add(pname)
            totals[region] = totals.get(region, 0) + size_bytes

    # ---------------------------------------------------------
    # 4) Intermediate activations / temps
    # ---------------------------------------------------------
    all_tensor_names = set(getattr(graph, "tensors", {}).keys())

    for tname in sorted(all_tensor_names):
        if tname in placed:
            continue
        if tname in getattr(graph, "inputs", []):
            continue
        if tname in getattr(graph, "outputs", []):
            continue
        if tname in getattr(graph, "constants", {}):
            continue

        size_bytes = _tensor_nbytes(graph, tname)
        region = _choose_activation_region(size_bytes)

        placements.append(
            TensorPlacement(
                tensor_name=tname,
                kind="activation",
                region=region,
                layout="raw",
                offset=None,
                size_bytes=size_bytes,
                double_buffer=False,
                producer=producers.get(tname),
                consumer=",".join(consumers.get(tname, [])) if consumers.get(tname) else None,
            )
        )
        placed.add(tname)
        totals[region] = totals.get(region, 0) + size_bytes

    return MemoryPlan(
        placements=placements,
        total_bytes_by_region=totals,
        notes={
            "planner": "memory_heuristic_v1",
            "param_rule": "small->BRAM, medium->URAM, large->DDR",
            "activation_rule": "small->BRAM, large->DDR",
        },
    )