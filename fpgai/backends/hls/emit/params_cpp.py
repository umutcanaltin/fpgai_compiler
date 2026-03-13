from __future__ import annotations

from fpgai.ir.graph import Graph
import numpy as np


def _fmt_values(data: np.ndarray, chunk_size: int = 10) -> list[str]:
    s_vals = [f"{float(x):.6f}" for x in data.reshape(-1)]
    lines: list[str] = []
    for i in range(0, len(s_vals), chunk_size):
        chunk = ", ".join(s_vals[i:i + chunk_size])
        suffix = "," if (i + chunk_size < len(s_vals)) else ""
        lines.append(f"  {chunk}{suffix}")
    return lines


def emit_params_cpp(graph: Graph) -> str:
    lines = []
    lines.append('#include "fpgai_params.h"')
    lines.append("namespace fpgai {")
    lines.append("")

    weight_idx = 0
    for op_idx, op in enumerate(graph.ops):
        if op.op_type not in ["Conv", "Dense", "Gemm"]:
            continue

        tag = op.attrs.get("precision_tag", f"op{op_idx}")
        w_t = f"{tag}_wgt_t"
        b_t = f"{tag}_bias_t"

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

        if w_name and w_name in graph.constants:
            w_data = graph.constants[w_name].astype(np.float32).flatten()
            lines.append(f"const {w_t} W{weight_idx}[{w_data.size}] = {{")
            lines.extend(_fmt_values(w_data))
            lines.append("};")
        else:
            lines.append(f"const {w_t} W{weight_idx}[1] = {{ 0 }};")

        if b_name and b_name in graph.constants:
            b_data = graph.constants[b_name].astype(np.float32).flatten()
            lines.append(f"const {b_t} B{weight_idx}[{b_data.size}] = {{")
            lines.extend(_fmt_values(b_data))
            lines.append("};")
        else:
            lines.append(f"const {b_t} B{weight_idx}[1] = {{ 0 }};")

        lines.append("")
        weight_idx += 1

    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)