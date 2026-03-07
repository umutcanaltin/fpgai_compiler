from __future__ import annotations

from fpgai.ir.graph import Graph
from fpgai.ir.ops import Op


def assign_stable_names(g: Graph) -> Graph:
    """
    Assign stable names like dense0, act0, conv0, pool0...
    Keeps original op.name in attrs["source_name"].
    """
    dense_i = 0
    conv_i = 0
    act_i = 0
    pool_i = 0
    other_i = 0

    new_ops = []
    for op in g.ops:
        attrs = dict(op.attrs)
        attrs.setdefault("source_name", op.name)

        if op.op_type == "Dense":
            name = f"dense{dense_i}"
            dense_i += 1
        elif op.op_type == "Conv":
            name = f"conv{conv_i}"
            conv_i += 1
        elif op.op_type in {"Relu", "LeakyRelu", "Sigmoid", "Softmax"}:
            name = f"act{act_i}"
            act_i += 1
        elif op.op_type in {"MaxPool", "AvgPool"}:
            name = f"pool{pool_i}"
            pool_i += 1
        else:
            name = f"op{other_i}"
            other_i += 1

        new_ops.append(Op(name=name, op_type=op.op_type, inputs=list(op.inputs), outputs=list(op.outputs), attrs=attrs))

    g.ops = new_ops
    return g
