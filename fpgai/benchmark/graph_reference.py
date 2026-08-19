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


def execute_graph_reference_trace(graph: Graph, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute the functional FPGAI IR and return every materialized tensor.

    This reference is intentionally architecture-neutral: architecture-only lowering
    must preserve these values within the configured numeric tolerance.  Returning a
    trace makes the same owner usable for model-, layer-, and intermediate-level
    compiler validation and standalone block reference capture.
    """
    values: dict[str, np.ndarray] = {str(k): np.asarray(v, dtype=np.float32) for k, v in inputs.items()}
    for name, value in (getattr(graph, "constants", {}) or {}).items():
        values[str(name)] = np.asarray(value, dtype=np.float32)

    for op in graph.ops:
        ins = [values[str(x)] for x in op.inputs]
        attrs = op.attrs or {}
        if op.op_type == "Transpose":
            perm = tuple(int(x) for x in attrs.get("perm", ()))
            out = np.transpose(ins[0], axes=perm or None)
        elif op.op_type == "Broadcast":
            target_spec = graph.get_tensor(op.outputs[0])
            if target_spec is None or not getattr(target_spec, "shape", None):
                raise ValueError(f"REF005: Broadcast output tensor metadata is missing for {op.outputs[0]!r}")
            target = tuple(int(v) for v in target_spec.shape)
            source = np.asarray(ins[0], dtype=np.float32)
            dims = attrs.get("broadcast_dimensions", attrs.get("dimensions", None))
            if dims is None:
                out = np.broadcast_to(source, target)
            else:
                dims = tuple(int(v) for v in dims)
                if len(dims) != source.ndim or len(set(dims)) != len(dims) or any(v < 0 or v >= len(target) for v in dims):
                    raise ValueError(
                        f"REF006: Broadcast dimensions {dims!r} are incompatible with source rank {source.ndim} and target {target!r}"
                    )
                reshape = [1] * len(target)
                for source_axis, target_axis in enumerate(dims):
                    reshape[target_axis] = int(source.shape[source_axis])
                out = np.broadcast_to(source.reshape(tuple(reshape)), target)
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
        elif op.op_type == "Relu":
            out = np.maximum(ins[0], 0.0)
        elif op.op_type == "LeakyRelu":
            alpha = float(attrs.get("alpha", 0.01))
            out = np.where(ins[0] >= 0.0, ins[0], alpha * ins[0])
        elif op.op_type == "Sigmoid":
            out = 1.0 / (1.0 + np.exp(-ins[0]))
        elif op.op_type == "Tanh":
            out = np.tanh(ins[0])
        elif op.op_type == "Dense":
            from fpgai.engine.training_graph_utils import resolve_dense_arrays
            weights, bias, input_features, output_features = resolve_dense_arrays(graph, op)
            x = np.asarray(ins[0], dtype=np.float32).reshape((-1, input_features))
            y = np.matmul(x, np.asarray(weights, dtype=np.float32).T) + np.asarray(bias, dtype=np.float32)
            target = tuple(int(v) for v in graph.get_tensor(op.outputs[0]).shape)
            out = y.reshape(target) if int(np.prod(target)) == int(y.size) else y
        elif op.op_type == "Conv":
            from fpgai.engine.training_graph_utils import resolve_conv_arrays
            weights, bias, weight_shape = resolve_conv_arrays(graph, op)
            x = np.asarray(ins[0], dtype=np.float32)
            # ONNX Conv is NCHW. FPGAI also accepts CHW for batch-one graphs.
            restore_batch = x.ndim == 4
            if x.ndim == 3:
                x = x[None, ...]
            if x.ndim != 4:
                raise ValueError(f"REF003: Conv reference requires CHW/NCHW input, got shape {x.shape}")
            n, cin, hin, win = x.shape
            cout, wcin, kh, kw = tuple(int(v) for v in weight_shape)
            groups = int(attrs.get("group", attrs.get("groups", 1)))
            if groups != 1 or wcin != cin:
                raise ValueError("REF004: Conv reference currently supports group=1 with matching input channels")
            strides = attrs.get("strides", [1, 1])
            dilations = attrs.get("dilations", [1, 1])
            pads = attrs.get("pads", [0, 0, 0, 0])
            sh, sw = int(strides[0]), int(strides[1])
            dh, dw = int(dilations[0]), int(dilations[1])
            if len(pads) == 2:
                pt = pb = int(pads[0]); pl = pr = int(pads[1])
            else:
                pt, pl, pb, pr = (int(v) for v in pads[:4])
            xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
            hout = (hin + pt + pb - dh * (kh - 1) - 1) // sh + 1
            wout = (win + pl + pr - dw * (kw - 1) - 1) // sw + 1
            y = np.empty((n, cout, hout, wout), dtype=np.float32)
            for bn in range(n):
                for co in range(cout):
                    for oy in range(hout):
                        for ox in range(wout):
                            total = float(bias[co])
                            for ci in range(cin):
                                for ky in range(kh):
                                    for kx in range(kw):
                                        iy = oy * sh + ky * dh
                                        ix = ox * sw + kx * dw
                                        total += float(xp[bn, ci, iy, ix]) * float(weights[co, ci, ky, kx])
                            y[bn, co, oy, ox] = total
            out = y if restore_batch else y[0]
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

    return {name: np.asarray(value, dtype=np.float32) for name, value in values.items()}


def execute_graph_reference(graph: Graph, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    trace = execute_graph_reference_trace(graph, inputs)
    return np.asarray(trace[str(graph.outputs[0])], dtype=np.float32)


__all__ = ["deterministic_graph_inputs", "execute_graph_reference", "execute_graph_reference_trace"]
