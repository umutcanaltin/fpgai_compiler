from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from fpgai.engine.models import CompilePlan, LayerDescriptor, MemoryPlan, TensorPlacement


def _shape_numel(shape) -> int:
    if not shape:
        return 0
    n = 1
    for d in shape:
        try:
            n *= max(1, int(d))
        except Exception:
            n *= 1
    return n


def _infer_tensor_bytes_from_shape(shape, default_elem_bytes: int = 2) -> int:
    n = _shape_numel(shape)
    return int(n * default_elem_bytes)


def _safe_first(seq, default=None):
    if not seq:
        return default
    return seq[0]


def _compile_notes(compile_plan: CompilePlan) -> Dict[str, Any]:
    return compile_plan.notes or {}


def _region_order_for_weight(policy_name: str, notes: Dict[str, Any]) -> List[str]:
    prefs = notes.get("weight_region_preference")
    if isinstance(prefs, list) and prefs:
        return [str(x).upper() for x in prefs]
    if policy_name == "Latency-First":
        return ["BRAM", "URAM", "DDR"]
    if policy_name in ("Fit-First", "BRAM-Saver"):
        return ["DDR", "URAM", "BRAM"]
    return ["URAM", "BRAM", "DDR"]


def _region_order_for_activation(policy_name: str, notes: Dict[str, Any]) -> List[str]:
    prefs = notes.get("activation_region_preference")
    if isinstance(prefs, list) and prefs:
        return [str(x).upper() for x in prefs]
    if policy_name == "Latency-First":
        return ["BRAM", "URAM", "DDR"]
    if policy_name == "BRAM-Saver":
        return ["DDR", "URAM", "BRAM"]
    return ["BRAM", "URAM", "DDR"]


def _pick_weight_region(
    size_bytes: int,
    layer_weight_mode: str,
    policy_name: str,
    notes: Dict[str, Any],
) -> str:
    if layer_weight_mode == "embedded":
        return "BRAM" if size_bytes <= 64 * 1024 else "URAM"
    if layer_weight_mode == "stream":
        order = _region_order_for_weight(policy_name, notes)
        if size_bytes <= 64 * 1024 and "BRAM" in order[:2]:
            return "BRAM"
        if size_bytes <= 512 * 1024 and "URAM" in order[:2]:
            return "URAM"
        return "DDR"
    return "DDR"


def _pick_activation_region(size_bytes: int, policy_name: str, notes: Dict[str, Any]) -> str:
    order = _region_order_for_activation(policy_name, notes)

    # latency-first tries to keep more on chip
    if order and order[0] == "BRAM":
        if size_bytes <= 128 * 1024:
            return "BRAM"
        if size_bytes <= 1024 * 1024 and "URAM" in order:
            return "URAM"
        return "DDR"

    # BRAM-saver / fit-first spill earlier
    if size_bytes <= 32 * 1024 and "BRAM" in order:
        return "BRAM"
    if size_bytes <= 512 * 1024 and "URAM" in order:
        return "URAM"
    return "DDR"


def _add_total(totals: Dict[str, int], region: str, size_bytes: int) -> None:
    totals[region] = int(totals.get(region, 0)) + int(size_bytes)


def _find_descriptor(node_name: str, descriptors: List[LayerDescriptor]) -> Optional[LayerDescriptor]:
    for d in descriptors:
        if d.node_name == node_name:
            return d
    return None


def make_memory_plan(g, descriptors: List[LayerDescriptor], compile_plan: CompilePlan) -> MemoryPlan:
    placements: List[TensorPlacement] = []
    totals: Dict[str, int] = {}

    cnotes = _compile_notes(compile_plan)
    policy_name = str(cnotes.get("parallel_policy", "Balanced"))
    allow_double_buffer = bool(cnotes.get("allow_double_buffer", False))

    graph_inputs: List[str] = list(getattr(g, "inputs", []) or [])
    graph_outputs: List[str] = list(getattr(g, "outputs", []) or [])
    graph_params: Set[str] = set(getattr(g, "params", {}).keys()) if getattr(g, "params", None) else set()

    seen: Set[str] = set()

    # External input tensors
    for name in graph_inputs:
        try:
            tensor = g.get_tensor(name)
            shape = getattr(tensor, "shape", None)
        except Exception:
            shape = None
        size_bytes = _infer_tensor_bytes_from_shape(shape)
        region = "HOST"
        placements.append(
            TensorPlacement(
                tensor_name=name,
                kind="input",
                region=region,
                layout="raw",
                size_bytes=size_bytes,
                double_buffer=False,
                producer=None,
                consumer=None,
                notes={
                    "policy_name": policy_name,
                    "reason": "graph_input",
                },
            )
        )
        _add_total(totals, region, size_bytes)
        seen.add(name)

    # Parameters / activations per layer
    for lp in compile_plan.layer_plans:
        desc = _find_descriptor(lp.node_name, descriptors)

        # Weights / params
        for pname in (desc.param_names if desc is not None else []):
            if pname in seen:
                continue
            size_bytes = int(desc.param_bytes // max(1, len(desc.param_names))) if desc and desc.param_names else 0
            region = _pick_weight_region(
                size_bytes=size_bytes,
                layer_weight_mode=lp.weight_mode,
                policy_name=policy_name,
                notes=cnotes,
            )
            placements.append(
                TensorPlacement(
                    tensor_name=pname,
                    kind="weight",
                    region=region,
                    layout="tiled" if lp.weight_mode != "embedded" else "raw",
                    size_bytes=size_bytes,
                    double_buffer=allow_double_buffer and lp.weight_mode in ("stream", "ddr"),
                    producer=None,
                    consumer=lp.node_name,
                    notes={
                        "policy_name": policy_name,
                        "weight_mode": lp.weight_mode,
                        "partition_factor": lp.notes.get("partition_factor"),
                        "partition_mode": lp.notes.get("partition_mode"),
                        "reason": "layer_parameter",
                    },
                )
            )
            _add_total(totals, region, size_bytes)
            seen.add(pname)

        # Primary output activation of the layer
        out_name = None
        out_shape = None
        if desc is not None and desc.outputs:
            out_name = desc.outputs[0]
            out_shape = _safe_first(desc.output_shapes, None)

        if out_name and out_name not in seen and out_name not in graph_outputs:
            size_bytes = max(
                int(desc.activation_bytes_out if desc is not None else 0),
                _infer_tensor_bytes_from_shape(out_shape),
            )
            region = _pick_activation_region(size_bytes, policy_name, cnotes)
            placements.append(
                TensorPlacement(
                    tensor_name=out_name,
                    kind="activation",
                    region=region,
                    layout="tiled" if lp.tile else "raw",
                    size_bytes=size_bytes,
                    double_buffer=allow_double_buffer and lp.buffering == "double",
                    producer=lp.node_name,
                    consumer=None,
                    notes={
                        "policy_name": policy_name,
                        "tile": lp.tile,
                        "unroll": lp.unroll,
                        "buffering": lp.buffering,
                        "activation_mode": lp.activation_mode,
                        "reason": "layer_output_activation",
                    },
                )
            )
            _add_total(totals, region, size_bytes)
            seen.add(out_name)

    # Final outputs
    for name in graph_outputs:
        if name in seen:
            continue
        try:
            tensor = g.get_tensor(name)
            shape = getattr(tensor, "shape", None)
        except Exception:
            shape = None
        size_bytes = _infer_tensor_bytes_from_shape(shape)
        region = "HOST"
        placements.append(
            TensorPlacement(
                tensor_name=name,
                kind="output",
                region=region,
                layout="raw",
                size_bytes=size_bytes,
                double_buffer=False,
                producer=None,
                consumer=None,
                notes={
                    "policy_name": policy_name,
                    "reason": "graph_output",
                },
            )
        )
        _add_total(totals, region, size_bytes)
        seen.add(name)

    return MemoryPlan(
        placements=placements,
        total_bytes_by_region=totals,
        notes={
            "planner": "policy_memory_v2",
            "policy_name": policy_name,
            "allow_double_buffer": allow_double_buffer,
            "weight_region_preference": cnotes.get("weight_region_preference"),
            "activation_region_preference": cnotes.get("activation_region_preference"),
        },
    )