from __future__ import annotations
from fpgai.ir.graph import Graph
from fpgai.ir.ops import Op

def insert_activations(g: Graph, *, kind: str, alpha: float = 0.1, except_last: bool = True) -> Graph:
    if kind not in {"leakyrelu", "none"}:
        raise ValueError(f"Unsupported activation insertion kind: {kind}")
    if kind == "none":
        return g

    # Insert LeakyRelu after Dense, skip last Dense by default
    dense_indices = [i for i, op in enumerate(g.ops) if op.op_type == "Dense"]
    if not dense_indices:
        return g
    last_dense = dense_indices[-1]

    new_ops = []
    rename = {}

    for i, op in enumerate(g.ops):
        op.inputs = [rename.get(x, x) for x in op.inputs]
        new_ops.append(op)

        if op.op_type != "Dense":
            continue
        if except_last and i == last_dense:
            continue
        if not op.outputs:
            continue

        out0 = op.outputs[0]
        act_out = f"{out0}_leakyrelu"

        new_ops.append(Op(
            name=f"{op.name}_leakyrelu",
            op_type="LeakyRelu",
            inputs=[out0],
            outputs=[act_out],
            attrs={"alpha": float(alpha)},
        ))
        rename[out0] = act_out

    g.ops = new_ops
    return g
