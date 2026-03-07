from __future__ import annotations

from typing import Dict, List, Set

from fpgai.ir.ops import Op


def fuse_matmul_add_to_dense(ops: List[Op], *, params: Set[str]) -> List[Op]:
    producer: Dict[str, int] = {}
    for i, op in enumerate(ops):
        for out in op.outputs:
            if out:
                producer[out] = i

    used = [False] * len(ops)
    out_ops: List[Op] = []

    for i, op in enumerate(ops):
        if used[i]:
            continue

        if op.op_type != "Add":
            out_ops.append(op)
            continue

        if len(op.inputs) < 2:
            out_ops.append(op)
            continue

        a, b = op.inputs[0], op.inputs[1]
        bias = None
        pre = None

        if b in params and a in producer:
            bias = b
            pre = a
        elif a in params and b in producer:
            bias = a
            pre = b

        if bias is None or pre is None:
            out_ops.append(op)
            continue

        j = producer.get(pre)
        if j is None:
            out_ops.append(op)
            continue

        pre_op = ops[j]
        if pre_op.op_type != "MatMul" or len(pre_op.inputs) < 2:
            out_ops.append(op)
            continue

        x, w = pre_op.inputs[0], pre_op.inputs[1]
        fused = Op(
            name=f"{pre_op.name}_fused",
            op_type="Dense",
            inputs=[x],
            outputs=list(op.outputs),
            attrs={"weight": w, "bias": bias, "layout": "out_in"},
        )

        used[j] = True
        used[i] = True
        out_ops.append(fused)

    return out_ops
