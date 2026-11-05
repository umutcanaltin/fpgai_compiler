"""Source-only, explicit hardware RL experiment regime contracts.

A general-network campaign requires multiple *distinct ONNX computational
graph structures*, not only different filenames, weights or training seeds.
The structural digest intentionally excludes numeric initializer values but
keeps operator topology, tensor dimensions, kernel metadata and connectivity.
No vendor calls, case allocation, or training labels occur in this module.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite

NETWORK_REGIMES = ("auto", "same", "general")
CLOCK_REGIMES = ("search", "fixed", "task_fixed")


def onnx_structure_sha256(path) -> str:
    """Fingerprint operator graph/shape, excluding initializer *values*.

    This is a conservative structural discriminator, not proof of independent
    training/test data, semantic equivalence, or architecture-family holdout.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    def shape(value):
        t = value.type
        if not t.HasField("tensor_type"):
            return ("non_tensor", t.WhichOneof("value"))
        dims = []
        for d in t.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                dims.append("?")  # Symbolic label itself is not a shape change.
        return (int(t.tensor_type.elem_type), dims)

    weights = {
        v.name: (int(v.data_type), tuple(int(i) for i in v.dims))
        for v in graph.initializer
    }
    inputs = [v for v in graph.input if v.name not in weights]
    input_indices = {v.name: i for i, v in enumerate(inputs)}
    producers = {}

    def source(tensor):
        if not tensor:
            return ("optional",)
        if tensor in weights:
            return ("parameter", weights[tensor])
        if tensor in input_indices:
            return ("input", input_indices[tensor])
        if tensor in producers:
            return ("operator", producers[tensor])
        raise ValueError("ONNX contains unresolved tensor or non-topological nodes")

    def attribute(a):
        t = int(a.type)
        if t == onnx.AttributeProto.INT:
            value = int(a.i)
        elif t == onnx.AttributeProto.FLOAT:
            value = float(a.f)
            if not isfinite(value):
                raise ValueError("Nonfinite ONNX attribute is not a research graph")
        elif t == onnx.AttributeProto.STRING:
            value = a.s.decode("utf-8", errors="replace")
        elif t == onnx.AttributeProto.INTS:
            value = list(int(v) for v in a.ints)
        elif t == onnx.AttributeProto.FLOATS:
            value = list(float(v) for v in a.floats)
        elif t == onnx.AttributeProto.TENSOR:
            value = (int(a.t.data_type), tuple(int(v) for v in a.t.dims))
        else:
            # Reject unsupported nested graphs/attributes rather than
            # incorrectly certifying two distinct models as structurally same.
            raise ValueError("General-network graph has unsupported ONNX attribute")
        return (a.name, t, value)

    nodes = []
    for i, node in enumerate(graph.node):
        dependencies = tuple(source(name) for name in node.input)
        nodes.append((node.domain, node.op_type, dependencies,
                      tuple(sorted(attribute(a) for a in node.attribute)),
                      len(node.output)))
        for j, name in enumerate(node.output):
            if name in producers or name in input_indices or name in weights:
                raise ValueError("ONNX graph has duplicate tensor producers")
            producers[name] = (i, j)
    payload = {
        "inputs": [shape(v) for v in inputs],
        "nodes": nodes,
        "outputs": [(source(v.name), shape(v)) for v in graph.output],
    }
    return sha256(json.dumps(payload, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_experiment_regimes(schedule, *, network_regime: str,
                                clock_regime: str) -> dict:
    """Validate four preregistered same/general × fixed/search settings."""
    if network_regime not in NETWORK_REGIMES or clock_regime not in CLOCK_REGIMES:
        raise ValueError("Unknown network or clock experiment regime")
    if not schedule:
        raise ValueError("An experiment needs a nonempty bounded task schedule")
    model_ids = {task.onnx_sha256 for task in schedule}
    if network_regime == "same" and len(model_ids) != 1:
        raise ValueError("Same-network regime requires exactly one immutable ONNX model")
    structure_by_task = {}
    if network_regime == "general":
        if len(model_ids) < 2:
            raise ValueError("General-network regime requires at least two distinct ONNX models")
        for task in schedule:
            if task.id not in structure_by_task:
                structure_by_task[task.id] = onnx_structure_sha256(task.onnx)
        if len(set(structure_by_task.values())) < 2:
            raise ValueError("General-network regime needs distinct ONNX graph architectures, not only weights")
    unique_clocks = {task.initial_clock_ns for task in schedule}
    if clock_regime == "fixed" and len(unique_clocks) != 1:
        raise ValueError("Fixed-clock regime requires one identical initial_clock_ns across all scheduled tasks")
    if clock_regime == "task_fixed" and len(unique_clocks) < 2:
        raise ValueError("Task-fixed clock transfer requires at least two different initial clock targets")
    clock = next(iter(unique_clocks)) if clock_regime == "fixed" else None
    by_task = ({task.id: task.initial_clock_ns for task in schedule}
               if clock_regime == "task_fixed" else {})
    return {
        "network_regime": network_regime,
        "clock_regime": clock_regime,
        "distinct_onnx_model_count": len(model_ids),
        "distinct_onnx_graph_count": (
            len(set(structure_by_task.values())) if structure_by_task else None
        ),
        "onnx_graph_sha256_by_task": dict(sorted(structure_by_task.items())),
        "fixed_clock_ns": clock,
        "fixed_clock_ns_by_task": dict(sorted(by_task.items())),
        "clock_action_bounds_ns": (
            [clock, clock] if clock is not None else
            None if clock_regime == "task_fixed" else [2.0, 32.0]
        ),
    }
