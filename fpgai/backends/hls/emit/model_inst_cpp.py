from __future__ import annotations

from fpgai.ir.graph import Graph


def emit_model_inst_cpp(graph: Graph) -> str:
    """
    Emits explicit template instantiations for Dense ops so CSIM linking is stable.

    Assumes Dense ops have attrs:
      - in_features / out_features OR inferred from tensors
      - layout == "out_in"
    """
    lines: list[str] = []
    lines.append('#include "dense.h"')
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")

    for op in graph.ops:
        if op.op_type != "Dense":
            continue

        # try to obtain shapes from op attrs (your importer prints these)
        IN = None
        OUT = None

        # common patterns you showed:
        # op.attrs: in, out, layout, w, b, etc.
        if hasattr(op, "attrs") and isinstance(op.attrs, dict):
            IN = op.attrs.get("in_features", op.attrs.get("in"))
            OUT = op.attrs.get("out_features", op.attrs.get("out"))

        # Fallback: infer from tensor shapes if not in attrs
        if IN is None or OUT is None:
            # inputs: x -> last dim
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

        # explicit instantiation
        lines.append(
            f"template void dense_out_in<{int(IN)},{int(OUT)}>("
            f"const act_t[{int(IN)}], act_t[{int(OUT)}], "
            f"const wgt_t[{int(OUT)}][{int(IN)}], const bias_t[{int(OUT)}]"
            f");"
        )

    lines.append("")
    lines.append("} // namespace fpgai")
    lines.append("")

    return "\n".join(lines)
