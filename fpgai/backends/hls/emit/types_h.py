from __future__ import annotations

from typing import Any, Dict
from fpgai.ir.graph import Graph


def _spec_to_ap(spec) -> str:
    if not isinstance(spec, dict):
        return "ap_fixed<16,6>"
    tb = int(spec.get("total_bits", 16))
    ib = int(spec.get("int_bits", 6))
    return f"ap_fixed<{tb},{ib}>"


def _deep_get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _default_precision(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "activation": _deep_get(raw_cfg, "numerics.defaults.activation", {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}),
        "weight": _deep_get(raw_cfg, "numerics.defaults.weight", {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}),
        "bias": _deep_get(raw_cfg, "numerics.defaults.bias", {"type": "ap_fixed", "total_bits": 24, "int_bits": 10}),
        "accum": _deep_get(raw_cfg, "numerics.defaults.accum", {"type": "ap_fixed", "total_bits": 24, "int_bits": 10}),
    }


def _macro_int(raw_cfg: Dict[str, Any], path: str, default: int) -> int:
    v = _deep_get(raw_cfg, path, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _layer_plan_map(compile_plan) -> Dict[str, Dict[str, Any]]:
    if compile_plan is None:
        return {}

    if hasattr(compile_plan, "layer_plans"):
        out: Dict[str, Dict[str, Any]] = {}
        for lp in compile_plan.layer_plans:
            if hasattr(lp, "to_dict"):
                d = lp.to_dict()
            elif isinstance(lp, dict):
                d = lp
            else:
                d = {}
            if d.get("node_name"):
                out[d["node_name"]] = d
        return out

    if isinstance(compile_plan, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for lp in compile_plan.get("layer_plans", []):
            if isinstance(lp, dict) and lp.get("node_name"):
                out[lp["node_name"]] = lp
        return out

    return {}


def emit_types_h(
    graph: Graph,
    *,
    top_name: str,
    raw_cfg: Dict[str, Any] | None = None,
    compile_plan: Any = None,
) -> str:
    raw_cfg = raw_cfg or {}
    dflt = _default_precision(raw_cfg)
    plan_notes = getattr(compile_plan, "notes", {}) if compile_plan is not None else {}

    grad_act = _deep_get(raw_cfg, "numerics.training.grad_activation", _deep_get(raw_cfg, "numerics.training.grad", dflt["activation"]))
    grad_wgt = _deep_get(raw_cfg, "numerics.training.grad_weight", dflt["weight"])
    grad_bias = _deep_get(raw_cfg, "numerics.training.grad_bias", dflt["bias"])
    update_acc = _deep_get(raw_cfg, "numerics.training.update_accum", dflt["accum"])
    optimizer_state = _deep_get(raw_cfg, "numerics.training.optimizer_state", dflt["accum"])
    loss_t = _deep_get(raw_cfg, "numerics.training.loss", dflt["accum"])

    pe = int(plan_notes.get("parallel_pe", _deep_get(raw_cfg, "optimization.parallel.pe", 1)))
    simd = int(plan_notes.get("parallel_simd", _deep_get(raw_cfg, "optimization.parallel.simd", 1)))
    partition_factor = int(plan_notes.get("parallel_partition_factor", _deep_get(raw_cfg, "optimization.parallel.partition_factor", 1)))
    unroll_factor = int(plan_notes.get("parallel_unroll_factor", _deep_get(raw_cfg, "optimization.parallel.unroll_factor", 1)))
    pipeline_style = str(plan_notes.get("parallel_pipeline_style", _deep_get(raw_cfg, "optimization.parallel.pipeline_style", "balanced"))).lower()

    pipe_ii = 1 if pipeline_style != "conservative" else 2
    pipe_ii = _macro_int(raw_cfg, "hls.pipeline_ii", pipe_ii)

    dense_out_unroll = max(1, _macro_int(raw_cfg, "hls.dense.out_unroll", pe))
    dense_in_unroll = max(1, _macro_int(raw_cfg, "hls.dense.in_unroll", simd))
    dense_bwd_out_unroll = max(1, _macro_int(raw_cfg, "hls.dense.backward_out_unroll", dense_out_unroll))
    dense_bwd_in_unroll = max(1, _macro_int(raw_cfg, "hls.dense.backward_in_unroll", dense_in_unroll))
    dense_upd_unroll = max(1, _macro_int(raw_cfg, "hls.dense.update_unroll", unroll_factor))
    dense_part_in = max(1, _macro_int(raw_cfg, "hls.dense.partition_input", partition_factor))
    dense_part_out = max(1, _macro_int(raw_cfg, "hls.dense.partition_output", partition_factor))
    dense_part_w = max(1, _macro_int(raw_cfg, "hls.dense.partition_weights", partition_factor))
    dense_part_g = max(1, _macro_int(raw_cfg, "hls.dense.partition_grads", partition_factor))
    act_unroll = max(1, _macro_int(raw_cfg, "hls.activation.unroll", unroll_factor))
    conv_oc_unroll = max(1, _macro_int(raw_cfg, "hls.conv.oc_unroll", pe))
    conv_ic_unroll = max(1, _macro_int(raw_cfg, "hls.conv.ic_unroll", simd))

    layer_plan_map = _layer_plan_map(compile_plan)

    lines = []
    lines.append("#pragma once")
    lines.append("#include <ap_fixed.h>")
    lines.append("")
    lines.append(f"#define FPGAI_PIPELINE_II {pipe_ii}")
    lines.append(f"#define FPGAI_DENSE_OUT_UNROLL {dense_out_unroll}")
    lines.append(f"#define FPGAI_DENSE_IN_UNROLL {dense_in_unroll}")
    lines.append(f"#define FPGAI_DENSE_BWD_OUT_UNROLL {dense_bwd_out_unroll}")
    lines.append(f"#define FPGAI_DENSE_BWD_IN_UNROLL {dense_bwd_in_unroll}")
    lines.append(f"#define FPGAI_DENSE_UPD_UNROLL {dense_upd_unroll}")
    lines.append(f"#define FPGAI_DENSE_PARTITION_INPUT {dense_part_in}")
    lines.append(f"#define FPGAI_DENSE_PARTITION_OUTPUT {dense_part_out}")
    lines.append(f"#define FPGAI_DENSE_PARTITION_WEIGHTS {dense_part_w}")
    lines.append(f"#define FPGAI_DENSE_PARTITION_GRADS {dense_part_g}")
    lines.append(f"#define FPGAI_ACT_UNROLL {act_unroll}")
    lines.append(f"#define FPGAI_CONV_OC_UNROLL {conv_oc_unroll}")
    lines.append(f"#define FPGAI_CONV_IC_UNROLL {conv_ic_unroll}")
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")
    lines.append(f"typedef {_spec_to_ap(dflt['activation'])} act_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['weight'])} wgt_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['bias'])} bias_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['accum'])} acc_t;")
    lines.append("")
    lines.append(f"typedef {_spec_to_ap(grad_act)} grad_act_t;")
    lines.append(f"typedef {_spec_to_ap(grad_wgt)} grad_wgt_t;")
    lines.append(f"typedef {_spec_to_ap(grad_bias)} grad_bias_t;")
    lines.append(f"typedef {_spec_to_ap(update_acc)} upd_t;")
    lines.append(f"typedef {_spec_to_ap(optimizer_state)} opt_t;")
    lines.append(f"typedef {_spec_to_ap(loss_t)} loss_t;")
    lines.append("")

    for idx, op in enumerate(graph.ops):
        p = op.attrs.get("precision", dflt)
        tag = op.attrs.get("precision_tag", f"op{idx}")
        lp = layer_plan_map.get(op.name, {})
        lines.append(f"// layer: {op.name} ({op.op_type})")
        if lp:
            lines.append(f"//   planner_precision_mode: {lp.get('precision_mode')}")
            lines.append(f"//   planner_act_bits: {lp.get('act_bits')}")
            lines.append(f"//   planner_weight_bits: {lp.get('weight_bits')}")
            lines.append(f"//   planner_unroll: {lp.get('unroll')}")
            lines.append(f"//   planner_tile: {lp.get('tile')}")
            lines.append(f"//   planner_pipeline_ii: {lp.get('pipeline_ii')}")
            lines.append(f"//   planner_weight_mode: {lp.get('weight_mode')}")
            lines.append(f"//   planner_activation_mode: {lp.get('activation_mode')}")
        lines.append(f"typedef {_spec_to_ap(p['activation'])} {tag}_act_t;")
        lines.append(f"typedef {_spec_to_ap(p['weight'])} {tag}_wgt_t;")
        lines.append(f"typedef {_spec_to_ap(p['bias'])} {tag}_bias_t;")
        lines.append(f"typedef {_spec_to_ap(p['accum'])} {tag}_acc_t;")
        lines.append("")

    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)