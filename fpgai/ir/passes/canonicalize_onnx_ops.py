from __future__ import annotations

from typing import List
from fpgai.ir.graph import Graph
from fpgai.ir.ops import Op


def _is_torch_linear(op_type: str) -> bool:
    return "torch" in op_type and "linear" in op_type.lower() and "Linear" in op_type


def canonicalize_onnx_ops(g: Graph) -> Graph:
    """
    Convert frontend-specific / exporter-specific ONNX nodes into canonical FPGAI ops.

    Current rules:
      - torch Linear nodes (custom op_type string) -> Dense

    Returns:
      A new Graph with ops rewritten (params/tensors preserved).
    """
    new_ops: List[Op] = []

    for op in g.ops:
        if _is_torch_linear(op.op_type):
            # Expect inputs: [X, W, B] and outputs: [Y]
            # Your example: in=['l_tensor_x_', 'fc0.weight', 'fc0.bias'] out=['fc0_1']
            if len(op.inputs) < 2:
                raise ValueError(f"Torch Linear op has too few inputs: {op}")

            x = op.inputs[0]
            w = op.inputs[1]
            b = op.inputs[2] if len(op.inputs) >= 3 else None

            attrs = dict(op.attrs)
            attrs["weight"] = w
            if b is not None:
                attrs["bias"] = b

            new_ops.append(
                Op(
                    name=op.name,
                    op_type="Dense",
                    inputs=[x],
                    outputs=list(op.outputs),
                    attrs=attrs,
                )
            )
        else:
            # Keep as-is for now (later we canonicalize Gemm/Conv/etc)
            new_ops.append(op)

    g.ops = new_ops
    return g
