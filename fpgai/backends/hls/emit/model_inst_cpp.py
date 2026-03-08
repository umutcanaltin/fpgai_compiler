from __future__ import annotations

from typing import Any, Dict, Optional

from fpgai.ir.graph import Graph


def _plan_map(compile_plan: Any) -> Dict[str, Dict[str, Any]]:
    if compile_plan is None:
        return {}

    if hasattr(compile_plan, "layer_plans"):
        out = {}
        for lp in compile_plan.layer_plans:
            if hasattr(lp, "to_dict"):
                d = lp.to_dict()
            elif isinstance(lp, dict):
                d = lp
            else:
                d = {"node_name": getattr(lp, "node_name", None)}
            name = d.get("node_name")
            if name:
                out[name] = d
        return out

    if isinstance(compile_plan, dict):
        layer_plans = compile_plan.get("layer_plans", [])
        out = {}
        for lp in layer_plans:
            if isinstance(lp, dict) and lp.get("node_name"):
                out[lp["node_name"]] = lp
        return out

    return {}


def _dense_sizes(graph: Graph, op) -> tuple[int, int]:
    IN: Optional[int] = None
    OUT: Optional[int] = None

    if hasattr(op, "attrs") and isinstance(op.attrs, dict):
        IN = op.attrs.get("in_features", op.attrs.get("in"))
        OUT = op.attrs.get("out_features", op.attrs.get("out"))

    if IN is None or OUT is None:
        x_name = op.inputs[0]
        y_name = op.outputs[0]
        x_t = graph.get_tensor(x_name)
        y_t = graph.get_tensor(y_name)

        if x_t and x_t.shape:
            IN = int(x_t.shape[-1])
        if y_t and y_t.shape:
            OUT = int(y_t.shape[-1])

    if IN is None or OUT is None:
        raise ValueError(f"Cannot infer Dense sizes for op {op.name}")

    return int(IN), int(OUT)


def emit_model_inst_cpp(graph: Graph, compile_plan: Any = None) -> str:
    """
    Emits explicit template instantiations for Dense ops so CSIM linking is stable.

    Now optionally accepts compile_plan and emits planning annotations as comments.
    This keeps generation stable while starting to thread plan information into
    backend code generation.
    """
    plan_by_name = _plan_map(compile_plan)

    lines: list[str] = []
    lines.append('#include "dense.h"')
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")

    for op in graph.ops:
        if op.op_type != "Dense":
            continue

        IN, OUT = _dense_sizes(graph, op)
        lp = plan_by_name.get(op.name, {})

        if lp:
            lines.append(f"// plan for {op.name}")
            lines.append(f"//   precision_mode: {lp.get('precision_mode')}")
            lines.append(f"//   act_bits      : {lp.get('act_bits')}")
            lines.append(f"//   weight_bits   : {lp.get('weight_bits')}")
            lines.append(f"//   tile          : {lp.get('tile', {})}")
            lines.append(f"//   unroll        : {lp.get('unroll', {})}")
            lines.append(f"//   pipeline_ii   : {lp.get('pipeline_ii')}")
            lines.append(f"//   weight_mode   : {lp.get('weight_mode')}")
            lines.append(f"//   buffering     : {lp.get('buffering')}")

        lines.append(
            f"template void dense_out_in<{IN},{OUT}>("
            f"const act_t[{IN}], act_t[{OUT}], "
            f"const wgt_t[{OUT}][{IN}], const bias_t[{OUT}]"
            f");"
        )
        lines.append("")

    lines.append("} // namespace fpgai")
    lines.append("")

    return "\n".join(lines)