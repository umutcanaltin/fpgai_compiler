from __future__ import annotations

from fpgai.ir.ops import Op


def _is_torch_linear(op_type: str) -> bool:
    s = op_type.lower()
    return ("torch" in s) and ("linear" in s)


def canonicalize_op(op: Op) -> Op:
    # Preserve source-domain information on imported nodes and normalize
    # well-defined ONNX Runtime contrib operators into generic FPGAI IR ops.
    # This is intentionally operator/domain based, never model-name based.
    attrs0 = dict(op.attrs or {})
    domain = str(attrs0.get("_onnx_domain", "ai.onnx") or "ai.onnx")

    if domain == "com.microsoft" and op.op_type == "RotaryEmbedding":
        attrs = dict(attrs0)
        rotary_dim = attrs.pop("rotary_embedding_dim", attrs.get("rotary_dim", 0))
        if int(rotary_dim or 0) > 0:
            attrs["rotary_dim"] = int(rotary_dim)
        attrs["interleaved"] = bool(int(attrs.get("interleaved", 0)))
        attrs["canonicalized_from"] = "com.microsoft::RotaryEmbedding"
        return Op(name=op.name, op_type="RotaryEmbedding", inputs=list(op.inputs), outputs=list(op.outputs), attrs=attrs)

    if domain == "com.microsoft" and op.op_type == "GroupQueryAttention":
        attrs = dict(attrs0)
        # ORT names the KV-head attribute kv_num_heads; FPGAI uses
        # num_kv_heads everywhere else. Keep both source metadata and one
        # canonical field so architecture/training/runtime passes agree.
        attrs["num_heads"] = int(attrs.get("num_heads", 1))
        attrs["num_kv_heads"] = int(attrs.get("kv_num_heads", attrs.get("num_kv_heads", attrs["num_heads"])))
        attrs["do_rotary"] = bool(int(attrs.get("do_rotary", 0)))
        attrs["rotary_interleaved"] = bool(int(attrs.get("rotary_interleaved", 0)))
        attrs.setdefault("causal", True)
        attrs.setdefault("execution_mode", "serialized")
        attrs["canonicalized_from"] = "com.microsoft::GroupQueryAttention"
        return Op(name=op.name, op_type="GroupQueryAttention", inputs=list(op.inputs), outputs=list(op.outputs), attrs=attrs)

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
