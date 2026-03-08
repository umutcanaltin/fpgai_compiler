from __future__ import annotations

from typing import Any, Dict, List

from fpgai.engine.models import CompilePlan, LayerDescriptor, LayerPlan


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _numerics_defaults(cfg) -> Dict[str, Any]:
    raw = cfg.raw
    return _cfg_get(raw, "numerics.defaults", {}) or {}


def _default_precision_info(cfg) -> Dict[str, Any]:
    numerics = _numerics_defaults(cfg)

    act = numerics.get("activation", {}) or {}
    weight = numerics.get("weight", {}) or {}

    act_type = str(act.get("type", "float")).lower()
    weight_type = str(weight.get("type", "float")).lower()

    # pick a single high-level mode for now
    if act_type.startswith("ap_fixed") or weight_type.startswith("ap_fixed") or act_type == "ap_fixed" or weight_type == "ap_fixed":
        precision_mode = "fixed"
    else:
        precision_mode = "float"

    return {
        "precision_mode": precision_mode,
        "act_bits": int(act["total_bits"]) if "total_bits" in act else None,
        "weight_bits": int(weight["total_bits"]) if "total_bits" in weight else None,
        "act_int_bits": int(act["int_bits"]) if "int_bits" in act else None,
        "weight_int_bits": int(weight["int_bits"]) if "int_bits" in weight else None,
        "act_type": act_type,
        "weight_type": weight_type,
    }


def _choose_weight_mode(desc: LayerDescriptor, global_weights_mode: str) -> str:
    # normalize aliases
    if global_weights_mode == "dma_ddr":
        global_weights_mode = "ddr"

    if global_weights_mode in ("embedded", "stream", "ddr"):
        if global_weights_mode == "embedded" and desc.param_bytes > 64 * 1024:
            return "ddr"
        return global_weights_mode

    if desc.param_bytes == 0:
        return "embedded"
    if desc.param_bytes <= 32 * 1024:
        return "embedded"
    if desc.param_bytes <= 256 * 1024:
        return "stream"
    return "ddr"


def _plan_conv(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str) -> LayerPlan:
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={"oh": 8, "ow": 8, "oc": 4},
        unroll={"ic": 1, "oc": 2},
        pipeline_ii=1,
        weight_mode=weights_mode,
        activation_mode="stream",
        buffering="double" if weights_mode in ("stream", "ddr") else "single",
        backend_kernel=desc.backend_kernel or "conv",
    )


def _plan_dense(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str) -> LayerPlan:
    out_features = int(desc.attrs.get("out_features", 1) or 1)
    in_features = int(desc.attrs.get("in_features", 1) or 1)

    out_tile = min(out_features, 16)
    in_tile = min(in_features, 64)

    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={"in": in_tile, "out": out_tile},
        unroll={"in": 2, "out": 2},
        pipeline_ii=1,
        weight_mode=weights_mode,
        activation_mode="stream",
        buffering="double" if weights_mode in ("stream", "ddr") else "single",
        backend_kernel=desc.backend_kernel or "dense",
    )


def _plan_pool(desc: LayerDescriptor, precision: Dict[str, Any]) -> LayerPlan:
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={"oh": 8, "ow": 8},
        unroll={},
        pipeline_ii=1,
        weight_mode="embedded",
        activation_mode="stream",
        buffering="single",
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
    )


def _plan_elementwise(desc: LayerDescriptor, precision: Dict[str, Any]) -> LayerPlan:
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={},
        unroll={},
        pipeline_ii=1,
        weight_mode="embedded",
        activation_mode="stream",
        buffering="single",
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
    )


def _plan_generic(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str) -> LayerPlan:
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={},
        unroll={},
        pipeline_ii=1,
        weight_mode=weights_mode,
        activation_mode="stream",
        buffering="single",
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
    )


def make_compile_plan(cfg, descriptors: List[LayerDescriptor]) -> CompilePlan:
    raw = cfg.raw
    precision = _default_precision_info(cfg)

    target_board = str(_cfg_get(raw, "targets.platform.board", "unknown"))
    target_part = str(_cfg_get(raw, "targets.platform.part", "unknown"))
    clock_mhz = float(_cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200.0))
    global_weights_mode = str(_cfg_get(raw, "data_movement.ps_pl.weights.mode", "embedded")).lower()

    layer_plans: List[LayerPlan] = []
    execution_order: List[str] = []

    for desc in descriptors:
        execution_order.append(desc.node_name)
        weights_mode = _choose_weight_mode(desc, global_weights_mode)

        if desc.op_type == "Conv":
            plan = _plan_conv(desc, precision, weights_mode)
        elif desc.op_type == "Dense":
            plan = _plan_dense(desc, precision, weights_mode)
        elif desc.op_type in ("MaxPool", "AvgPool"):
            plan = _plan_pool(desc, precision)
        elif desc.op_type in ("Relu", "LeakyRelu", "Sigmoid", "Softmax", "Add", "Reshape", "Flatten"):
            plan = _plan_elementwise(desc, precision)
        else:
            plan = _plan_generic(desc, precision, weights_mode)

        layer_plans.append(plan)

    total_param_bytes = sum(d.param_bytes for d in descriptors)
    total_activation_in = sum(d.activation_bytes_in for d in descriptors)
    total_activation_out = sum(d.activation_bytes_out for d in descriptors)

    return CompilePlan(
        target_board=target_board,
        target_part=target_part,
        clock_mhz=clock_mhz,
        execution_order=execution_order,
        layer_plans=layer_plans,
        global_resource_budget={
            "total_param_bytes": total_param_bytes,
            "total_activation_bytes_in": total_activation_in,
            "total_activation_bytes_out": total_activation_out,
        },
        notes={
            "planner": "heuristic_v1",
            "global_weights_mode_requested": global_weights_mode,
            "precision_mode": precision["precision_mode"],
            "act_bits": precision["act_bits"],
            "weight_bits": precision["weight_bits"],
            "act_int_bits": precision["act_int_bits"],
            "weight_int_bits": precision["weight_int_bits"],
        },
    )