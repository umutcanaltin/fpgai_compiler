"""Bounded exhaustive semantic checker for Phi: coverage and collision freedom."""
from itertools import product
from math import prod
from fpgai_research.model_ir.contracts import ArchitecturePoint
from fpgai_research.architecture.resolver import coordinate

def verify_mapping(point: ArchitecturePoint, max_iterations: int = 100_000) -> dict:
    if max_iterations < 1:
        raise ValueError("A positive iteration bound is required")
    shape = point.iteration_shape
    count = prod(shape)
    if count > max_iterations:
        raise ValueError(f"Iteration domain {count} exceeds oracle bound {max_iterations}")
    seen = set()
    replicas, pes, lanes, tiles = set(), set(), set(), set()
    for index in product(*(range(d) for d in shape)):
        value = coordinate(point, index)
        replicas.add(value["replica"])
        pes.add(value["pe"])
        lanes.add(value["lane"])
        tiles.add(value["tile"])
        key = (value["tile"], value["replica"], value["pe"], value["lane"], value["reuse_step"])
        if key in seen:
            raise AssertionError(f"Spatial/temporal collision: {point.operator}, {index}")
        seen.add(key)
    arch = point.effective
    if len(replicas) > arch.parallelism or len(pes) > arch.pes or len(lanes) > arch.simd:
        raise AssertionError("Capacity violation")
    return {"operator": point.operator, "iterations": count, "distinct_assignments": len(seen),
            "tiles": len(tiles), "observed_replicas": len(replicas),
            "observed_pes": len(pes), "observed_lanes": len(lanes),
            "evidence": "E0-semantic-enumeration"}
