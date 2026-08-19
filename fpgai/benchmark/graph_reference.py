"""Small deterministic reference executor used by maintained compiler examples/tests.

This is intentionally outside the HLS backend: backend project emission must not own
model-family semantics.
"""

from __future__ import annotations

from typing import Mapping
import numpy as np

from fpgai.ir import Graph


def deterministic_graph_inputs(graph: Graph) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(graph.inputs):
        shape = tuple(int(x) for x in graph.get_tensor(name).shape)
        count = int(np.prod(shape))
        base = np.linspace(-0.75 + 0.15 * index, 0.75 - 0.1 * index, count, dtype=np.float32)
        result[str(name)] = base.reshape(shape)
    return result


def execute_graph_reference(graph: Graph, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    values: dict[str, np.ndarray] = {str(k): np.asarray(v, dtype=np.float32) for k, v in inputs.items()}
    for name, value in (getattr(graph, "constants", {}) or {}).items():
        values[str(name)] = np.asarray(value, dtype=np.float32)

    for op in graph.ops:
        ins = [values[str(x)] for x in op.inputs]
        attrs = op.attrs or {}
        if op.op_type == "Transpose":
            perm = tuple(int(x) for x in attrs.get("perm", ()))
            out = np.transpose(ins[0], axes=perm or None)
        elif op.op_type == "MatMul":
            out = np.matmul(ins[0], ins[1])
        elif op.op_type == "Mul":
            out = ins[0] * ins[1]
        elif op.op_type == "Add":
            out = ins[0] + ins[1]
        elif op.op_type == "Sub":
            out = ins[0] - ins[1]
        elif op.op_type == "Div":
            out = ins[0] / ins[1]
        elif op.op_type == "Sqrt":
            out = np.sqrt(ins[0])
        elif op.op_type == "Pow":
            out = np.power(ins[0], ins[1])
        elif op.op_type == "ReduceMean":
            axes = attrs.get("axes", attrs.get("dimensions", None))
            axis = None if axes is None else tuple(int(x) for x in axes)
            out = np.mean(ins[0], axis=axis, keepdims=bool(attrs.get("keepdims", 1)))
        elif op.op_type == "ReduceSum":
            axes = attrs.get("axes", attrs.get("dimensions", None))
            axis = None if axes is None else tuple(int(x) for x in axes)
            out = np.sum(ins[0], axis=axis, keepdims=bool(attrs.get("keepdims", 1)))
        elif op.op_type in {"Identity", "Cast", "Squeeze", "Unsqueeze", "Reshape", "Flatten"}:
            if op.op_type == "Squeeze":
                axes = attrs.get("axes")
                out = np.squeeze(ins[0], axis=None if axes is None else tuple(int(x) for x in axes))
            elif op.op_type == "Unsqueeze":
                out = ins[0]
                for axis in sorted(int(x) for x in attrs.get("axes", [])):
                    out = np.expand_dims(out, axis)
            elif op.op_type in {"Reshape", "Flatten"}:
                target = graph.get_tensor(op.outputs[0]).shape
                out = np.reshape(ins[0], target)
            else:
                out = np.array(ins[0], copy=True)
        elif op.op_type == "SiLU":
            x = np.asarray(ins[0], dtype=np.float32)
            out = x / (1.0 + np.exp(-x))
        elif op.op_type == "Softmax":
            axis = int(attrs.get("axis", -1))
            shifted = ins[0] - np.max(ins[0], axis=axis, keepdims=True)
            exp = np.exp(shifted)
            out = exp / np.sum(exp, axis=axis, keepdims=True)
        elif op.op_type == "CausalMask":
            diagonal = int(attrs.get("diagonal", 0))
            masked_value = float(attrs.get("masked_value", -32.0))
            out = np.array(ins[0], copy=True)
            rows, cols = out.shape[-2], out.shape[-1]
            mask = np.arange(cols)[None, :] > (np.arange(rows)[:, None] + diagonal)
            out.reshape((-1, rows, cols))[:, mask] = masked_value
        elif op.op_type == "LayerNormalization":
            axis = int(attrs.get("axis", -1))
            epsilon = float(attrs.get("epsilon", 1e-5))
            x, scale, bias = ins[0], ins[1], ins[2]
            axes = tuple(range(axis if axis >= 0 else x.ndim + axis, x.ndim))
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.mean((x - mean) ** 2, axis=axes, keepdims=True)
            out = (x - mean) / np.sqrt(var + epsilon) * scale + bias
        elif op.op_type == "RMSNorm":
            axis = int(attrs.get("axis", -1))
            epsilon = float(attrs.get("epsilon", 1e-5))
            x, scale = ins[0], ins[1]
            axes = tuple(range(axis if axis >= 0 else x.ndim + axis, x.ndim))
            out = x / np.sqrt(np.mean(x * x, axis=axes, keepdims=True) + epsilon) * scale
        elif op.op_type == "RotaryEmbedding":
            x, cos_table, sin_table = ins[0], ins[1], ins[2]
            flat = np.asarray(x, dtype=np.float32).reshape((-1, x.shape[-1]))
            cols = flat.shape[-1]
            rotary_dim = int(attrs.get("rotary_dim", cols))
            if rotary_dim != cols or cols % 2:
                raise ValueError("REF001: RoPE reference currently requires even full last dimension")
            offset = int(attrs.get("position_offset", 0))
            c_all = np.asarray(cos_table, dtype=np.float32).reshape((-1, cols // 2))
            s_all = np.asarray(sin_table, dtype=np.float32).reshape((-1, cols // 2))
            if offset < 0 or offset + flat.shape[0] > c_all.shape[0] or c_all.shape != s_all.shape:
                raise ValueError("REF002: RoPE position_offset exceeds cosine/sine table capacity")
            c = c_all[offset:offset + flat.shape[0]]
            si = s_all[offset:offset + flat.shape[0]]
            out2 = np.empty_like(flat)
            x0 = flat[:, 0::2]
            x1 = flat[:, 1::2]
            out2[:, 0::2] = x0 * c - x1 * si
            out2[:, 1::2] = x0 * si + x1 * c
            out = out2.reshape(x.shape)
        elif op.op_type == "MultiHeadAttention":
            q, k, v = [np.asarray(x, dtype=np.float32) for x in ins[:3]]
            if q.ndim == 3 and q.shape[0] == 1:
                q2, k2, v2 = q[0], k[0], v[0]
                restore_batch = True
            else:
                q2, k2, v2 = q, k, v
                restore_batch = False
            heads = int(attrs.get("num_heads", 1))
            seq, model = q2.shape
            head_dim = model // heads
            scale = float(attrs.get("scale", 1.0 / np.sqrt(float(head_dim))))
            causal = bool(attrs.get("causal", True))
            result = np.empty_like(q2)
            for head in range(heads):
                sl = slice(head * head_dim, (head + 1) * head_dim)
                scores = np.matmul(q2[:, sl], k2[:, sl].T) * scale
                if causal:
                    scores = np.where(
                        np.arange(seq)[None, :] > np.arange(seq)[:, None],
                        float(attrs.get("masked_value", -32.0)),
                        scores,
                    )
                shifted = scores - np.max(scores, axis=-1, keepdims=True)
                probs = np.exp(shifted)
                probs /= np.sum(probs, axis=-1, keepdims=True)
                result[:, sl] = np.matmul(probs, v2[:, sl])
            out = result[None, ...] if restore_batch else result
        else:
            raise ValueError(f"REF000: unsupported reference op {op.op_type!r}")
        values[str(op.outputs[0])] = np.asarray(out, dtype=np.float32)

    return np.asarray(values[str(graph.outputs[0])], dtype=np.float32)


__all__ = ["deterministic_graph_inputs", "execute_graph_reference"]
