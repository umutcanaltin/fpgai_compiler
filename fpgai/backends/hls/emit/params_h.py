from __future__ import annotations

from typing import List
import numpy as np


def _numel_from_graph_named(graph, tensor_name: str) -> int:
    if tensor_name is None:
        return 0

    if hasattr(graph, "constants") and tensor_name in graph.constants:
        return int(np.asarray(graph.constants[tensor_name]).size)

    if hasattr(graph, "params") and tensor_name in graph.params:
        return int(np.asarray(graph.params[tensor_name]).size)

    try:
        t = graph.get_tensor(tensor_name)
    except Exception:
        t = None

    if t is not None:
        shape = getattr(t, "shape", None)
        if shape is not None:
            n = 1
            for d in shape:
                n *= int(d)
            return int(n)

        data = getattr(t, "data", None)
        if data is not None:
            return int(np.asarray(data).size)

    return 0


def emit_params_h(graph, *, weights_mode: str = "embedded") -> str:
    lines: List[str] = []
    lines.append("#pragma once")
    lines.append('#include "fpgai_types.h"')
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")

    if weights_mode == "embedded":
        param_op_idx = 0

        for op in graph.ops:
            if op.op_type == "Conv":
                w_n = 0
                b_n = 0

                if len(op.inputs) > 1:
                    w_n = _numel_from_graph_named(graph, op.inputs[1])
                if len(op.inputs) > 2:
                    b_n = _numel_from_graph_named(graph, op.inputs[2])

                if w_n > 0:
                    lines.append(f"extern const wgt_t W{param_op_idx}[{w_n}];")
                if b_n > 0:
                    lines.append(f"extern const bias_t B{param_op_idx}[{b_n}];")

                param_op_idx += 1

            elif op.op_type == "Dense":
                in_f = int(op.attrs.get("in_features") or 0)
                out_f = int(op.attrs.get("out_features") or 0)

                w_n = out_f * in_f if in_f > 0 and out_f > 0 else 0
                b_n = out_f if out_f > 0 else 0

                if w_n > 0:
                    lines.append(f"extern const wgt_t W{param_op_idx}[{w_n}];")
                if b_n > 0:
                    lines.append(f"extern const bias_t B{param_op_idx}[{b_n}];")

                param_op_idx += 1

    lines.append("")
    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)


# backward-compatible alias
def emit_params_h_stub(graph, *, weights_mode: str = "embedded") -> str:
    return emit_params_h(graph, weights_mode=weights_mode)