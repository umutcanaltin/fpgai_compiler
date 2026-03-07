from __future__ import annotations

from fpgai.ir.ops import Op


def _is_torch_linear(op_type: str) -> bool:
    s = op_type.lower()
    return ("torch" in s) and ("linear" in s)


def canonicalize_op(op: Op) -> Op:
    # Torch Linear custom op -> Dense
    if _is_torch_linear(op.op_type):
        x = op.inputs[0] if len(op.inputs) > 0 else None
        w = op.inputs[1] if len(op.inputs) > 1 else None
        b = op.inputs[2] if len(op.inputs) > 2 else None
        if x and w:
            attrs = dict(op.attrs)
            attrs["weight"] = w
            if b:
                attrs["bias"] = b
            attrs.setdefault("layout", "out_in")
            return Op(name=op.name, op_type="Dense", inputs=[x], outputs=list(op.outputs), attrs=attrs)

    # Gemm -> Dense
    if op.op_type == "Gemm":
        x = op.inputs[0] if len(op.inputs) > 0 else None
        w = op.inputs[1] if len(op.inputs) > 1 else None
        b = op.inputs[2] if len(op.inputs) > 2 else None
        if x and w:
            attrs = dict(op.attrs)
            attrs["weight"] = w
            if b:
                attrs["bias"] = b
            attrs.setdefault("layout", "out_in")
            return Op(name=op.name, op_type="Dense", inputs=[x], outputs=list(op.outputs), attrs=attrs)

    return op
