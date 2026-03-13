from __future__ import annotations

from fpgai.ir.graph import Graph


def _tensor_total(graph: Graph, name: str) -> int:
    t = graph.get_tensor(name)
    if t is not None and getattr(t, "shape", None):
        total = 1
        for d in t.shape:
            total *= int(d)
        return total

    if name in graph.constants:
        total = 1
        for d in graph.constants[name].shape:
            total *= int(d)
        return total

    raise ValueError(f"Cannot resolve tensor size for parameter: {name}")


def emit_params_h_stub(graph: Graph, weights_mode: str = "embedded") -> str:
    runtime_weights = str(weights_mode).lower() in ("stream", "ddr")

    lines = []
    lines.append("#pragma once")
    lines.append('#include "fpgai_types.h"')
    lines.append("namespace fpgai {")
    lines.append("")

    weight_idx = 0
    for op_idx, op in enumerate(graph.ops):
        if op.op_type not in ["Conv", "Dense", "Gemm"]:
            continue

        tag = op.attrs.get("precision_tag", f"op{op_idx}")
        w_t = f"{tag}_wgt_t"
        b_t = f"{tag}_bias_t"

        w_prefix = f"extern {w_t}" if runtime_weights else f"extern const {w_t}"
        b_prefix = f"extern {b_t}" if runtime_weights else f"extern const {b_t}"

        w_name = None
        b_name = None

        if op.op_type in ("Dense", "Gemm"):
            w_name = op.attrs.get("weight")
            b_name = op.attrs.get("bias")
            if w_name is None and len(op.inputs) > 1:
                w_name = op.inputs[1]
            if b_name is None and len(op.inputs) > 2:
                b_name = op.inputs[2]
        elif op.op_type == "Conv":
            if len(op.inputs) > 1:
                w_name = op.inputs[1]
            if len(op.inputs) > 2:
                b_name = op.inputs[2]

        if w_name:
            w_total = _tensor_total(graph, w_name)
            lines.append(f"{w_prefix} W{weight_idx}[{w_total}];")

        if b_name:
            b_total = _tensor_total(graph, b_name)
            lines.append(f"{b_prefix} B{weight_idx}[{b_total}];")
        else:
            lines.append(f"{b_prefix} B{weight_idx}[1];")

        lines.append("")
        weight_idx += 1

    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)