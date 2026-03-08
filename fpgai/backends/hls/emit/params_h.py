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
    """
    Emit parameter declarations for generated HLS code.

    - embedded mode:
        extern const wgt_t W0[...];
        extern const bias_t B0[...];

    - stream / ddr mode:
        extern wgt_t W0[...];
        extern bias_t B0[...];

    Arrays are emitted as flat 1D buffers to keep declarations compatible
    across Conv and Dense. Dense call sites can cast flat storage to 2D.
    """
    runtime_weights = str(weights_mode).lower() in ("stream", "ddr")

    if runtime_weights:
        w_prefix = "    extern wgt_t"
        b_prefix = "    extern bias_t"
    else:
        w_prefix = "    extern const wgt_t"
        b_prefix = "    extern const bias_t"

    lines = []
    lines.append("#pragma once")
    lines.append('#include "fpgai_types.h"')
    lines.append("namespace fpgai {")
    lines.append("")

    weight_idx = 0

    for op in graph.ops:
        if op.op_type not in ["Conv", "Dense"]:
            continue

        w_name = None
        b_name = None

        if op.op_type == "Dense":
            w_name = op.attrs.get("weight")
            b_name = op.attrs.get("bias")
        elif op.op_type == "Conv":
            if len(op.inputs) > 1:
                w_name = op.inputs[1]
            if len(op.inputs) > 2:
                b_name = op.inputs[2]

        # Weights
        if w_name:
            w_total = _tensor_total(graph, w_name)
            lines.append(f"{w_prefix} W{weight_idx}[{w_total}];")

        # Bias
        if b_name:
            b_total = _tensor_total(graph, b_name)
            lines.append(f"{b_prefix} B{weight_idx}[{b_total}];")
        else:
            # dummy bias symbol if missing
            lines.append(f"{b_prefix} B{weight_idx}[1];")

        weight_idx += 1

    lines.append("")
    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)