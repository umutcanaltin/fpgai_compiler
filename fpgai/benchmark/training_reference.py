from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np


@dataclass
class TrainingReferenceResult:
    out_dir: Path
    grads_flat_path: Path
    weights_before_flat_path: Path
    weights_after_flat_path: Path
    summary_json: Path
    summary_txt: Path
    loss_before: float
    loss_after: float
    layerwise_dir: Path


def _shape_wo_batch(shape) -> Tuple[int, ...]:
    if shape is None:
        return tuple()
    shp = tuple(int(x) for x in shape)
    if len(shp) > 1 and shp[0] == 1:
        return shp[1:]
    return shp


def _flat_size(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 1
    v = 1
    for d in shape:
        v *= int(d)
    return int(v)


def _as_chw(shape3: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shape3) != 3:
        raise ValueError(f"Expected 3D shape, got {shape3}")
    return int(shape3[0]), int(shape3[1]), int(shape3[2])


def _write_f32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.float32).reshape(-1).tofile(path)


def _as_numpy_numeric(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, (str, bytes)):
        return None
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.dtype.kind in ("U", "S", "O"):
        return None
    return arr.astype(np.float32, copy=False)


def _read_named_array(graph, name: str):
    if name is None:
        return None
    if hasattr(graph, "constants") and name in graph.constants:
        return np.asarray(graph.constants[name], dtype=np.float32)
    if hasattr(graph, "params") and name in graph.params:
        return np.asarray(graph.params[name], dtype=np.float32)
    t = graph.get_tensor(name)
    if t is not None:
        for attr in ("data", "initializer", "value", "values"):
            if hasattr(t, attr):
                v = getattr(t, attr)
                if v is not None:
                    arr = _as_numpy_numeric(v)
                    if arr is not None:
                        return arr
    return None


def _flatten_from_graph_named(graph, tensor_name: str) -> Optional[np.ndarray]:
    arr = _read_named_array(graph, tensor_name)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _resolve_attr_candidate(graph, v):
    arr = _as_numpy_numeric(v)
    if arr is not None and arr.size > 0:
        return arr.reshape(-1), "direct_numeric_attr"
    if isinstance(v, str):
        arr = _flatten_from_graph_named(graph, v)
        if arr is not None:
            return arr.reshape(-1), f"graph_ref('{v}')"
    return None, None


def _candidate_attr_arrays(graph, op, expected_w: int, expected_b: int):
    attrs = getattr(op, "attrs", {}) or {}
    candidates = []

    likely_weight_keys = [
        "weights", "weight", "kernel", "W", "w",
        "weight_data", "weights_data", "kernel_data",
        "weight_values", "weights_values", "kernel_values",
        "initializer", "params",
    ]
    likely_bias_keys = [
        "bias", "biases", "B", "b", "bias_data", "bias_values",
        "mean", "var", "scale", "gamma", "beta",
    ]

    for k in likely_weight_keys:
        if k in attrs:
            arr, src = _resolve_attr_candidate(graph, attrs[k])
            if arr is not None and arr.size > 0:
                candidates.append(("weight", k, arr, src))

    for k in likely_bias_keys:
        if k in attrs:
            arr, src = _resolve_attr_candidate(graph, attrs[k])
            if arr is not None and arr.size > 0:
                candidates.append(("bias", k, arr, src))

    for k, v in attrs.items():
        arr, src = _resolve_attr_candidate(graph, v)
        if arr is None or arr.size == 0:
            continue
        if expected_w > 0 and arr.size == expected_w:
            candidates.append(("weight", k, arr, src))
        if expected_b > 0 and arr.size == expected_b:
            candidates.append(("bias", k, arr, src))

    return candidates


def _pick_dense_from_attrs(graph, op, expected_w: int, expected_b: int):
    weight_arr = None
    bias_arr = None
    for kind, _key, arr, _src in _candidate_attr_arrays(graph, op, expected_w, expected_b):
        if kind == "weight" and weight_arr is None:
            weight_arr = arr
        elif kind == "bias" and bias_arr is None:
            bias_arr = arr
    return weight_arr, bias_arr


def _dense_expected_shapes(op) -> Tuple[int, int, int, int]:
    in_f = int(op.attrs.get("in_features") or 0)
    out_f = int(op.attrs.get("out_features") or 0)
    expected_w = out_f * in_f if in_f > 0 and out_f > 0 else 0
    expected_b = out_f if out_f > 0 else 0
    return in_f, out_f, expected_w, expected_b


def _get_tensor_shape(graph, name: str) -> Tuple[int, ...]:
    t = graph.get_tensor(name)
    if t is not None and getattr(t, "shape", None):
        return _shape_wo_batch(t.shape)
    if hasattr(graph, "tensors") and name in getattr(graph, "tensors", {}):
        ts = graph.tensors[name]
        if getattr(ts, "shape", None):
            return _shape_wo_batch(ts.shape)
    return tuple()


def _infer_conv_output_shape(
    xshape: Tuple[int, int, int],
    wshape: Tuple[int, int, int, int],
    stride: int,
    pad: int,
) -> Tuple[int, int, int]:
    c_in, h_in, w_in = _as_chw(xshape)
    c_out, _c_w, k_h, k_w = tuple(int(v) for v in wshape)
    if k_h != k_w:
        raise RuntimeError(f"Only square conv kernels are supported in training reference, got {wshape}")
    h_out = ((h_in + 2 * pad - k_h) // stride) + 1
    w_out = ((w_in + 2 * pad - k_w) // stride) + 1
    return int(c_out), int(h_out), int(w_out)


def _infer_pool_output_shape(
    xshape: Tuple[int, int, int],
    k: int,
    stride: int,
) -> Tuple[int, int, int]:
    c, h, w = _as_chw(xshape)
    h_out = ((h - k) // stride) + 1
    w_out = ((w - k) // stride) + 1
    return int(c), int(h_out), int(w_out)


def _dense_input_preflatten_shape(graph, op) -> Tuple[int, ...]:
    xname = op.inputs[0]
    producer = None
    for prev in graph.ops:
        if prev.outputs and prev.outputs[0] == xname:
            producer = prev
            break

    if producer is None:
        return _get_tensor_shape(graph, xname)

    if producer.op_type not in ("Flatten", "Reshape"):
        return _get_tensor_shape(graph, xname)

    if not producer.inputs:
        return _get_tensor_shape(graph, xname)

    src_shape = _get_tensor_shape(graph, producer.inputs[0])
    if src_shape:
        return src_shape

    return _get_tensor_shape(graph, xname)


def _remap_dense_weights_chw_to_hwc_if_needed(
    W: np.ndarray,
    logical_input_shape: Tuple[int, ...],
) -> np.ndarray:
    if len(logical_input_shape) != 3:
        return W

    c, h, w = (int(logical_input_shape[0]), int(logical_input_shape[1]), int(logical_input_shape[2]))
    n = c * h * w
    if W.shape[1] != n:
        return W

    chw_to_hwc = np.empty((n,), dtype=np.int64)
    chw_idx = 0
    for cc in range(c):
        for hh in range(h):
            for ww in range(w):
                hwc_idx = (hh * w + ww) * c + cc
                chw_to_hwc[chw_idx] = hwc_idx
                chw_idx += 1

    W_hwc = np.empty_like(W)
    W_hwc[:, chw_to_hwc] = W
    return W_hwc


def _resolve_dense_arrays(graph, op) -> Tuple[np.ndarray, np.ndarray, int, int]:
    in_f, out_f, expected_w, expected_b = _dense_expected_shapes(op)

    w_arr = None
    b_arr = None

    if len(op.inputs) > 1:
        w_arr = _flatten_from_graph_named(graph, op.inputs[1])
    if len(op.inputs) > 2:
        b_arr = _flatten_from_graph_named(graph, op.inputs[2])

    if w_arr is None or (b_arr is None and expected_b > 0):
        wa, ba = _pick_dense_from_attrs(graph, op, expected_w, expected_b)
        if w_arr is None and wa is not None:
            w_arr = wa
        if b_arr is None and ba is not None:
            b_arr = ba

    if w_arr is None:
        raise RuntimeError(f"Dense weights not found for training reference op '{op.name}'")

    xshape_flat = _get_tensor_shape(graph, op.inputs[0])
    xshape_logical = _dense_input_preflatten_shape(graph, op)
    yshape = _get_tensor_shape(graph, op.outputs[0])

    if in_f <= 0:
        in_f = _flat_size(xshape_flat)
    if out_f <= 0:
        out_f = _flat_size(yshape)

    W = np.asarray(w_arr, dtype=np.float32).reshape(out_f, in_f)
    W = _remap_dense_weights_chw_to_hwc_if_needed(W, xshape_logical)

    B = (
        np.zeros((out_f,), dtype=np.float32)
        if b_arr is None
        else np.asarray(b_arr, dtype=np.float32).reshape(out_f)
    )
    return W, B, in_f, out_f


def _resolve_conv_arrays(graph, op):
    w_arr = None
    b_arr = None

    if len(op.inputs) > 1:
        w_arr = _flatten_from_graph_named(graph, op.inputs[1])
    if len(op.inputs) > 2:
        b_arr = _flatten_from_graph_named(graph, op.inputs[2])

    if w_arr is None:
        attrs = getattr(op, "attrs", {}) or {}
        for key in ("weights", "weight", "kernel", "W", "w", "weight_data", "kernel_data"):
            if key in attrs:
                arr, _ = _resolve_attr_candidate(graph, attrs[key])
                if arr is not None:
                    w_arr = arr
                    break

    if w_arr is None:
        raise RuntimeError(f"Conv weight tensor not found for op '{op.name}'")

    ws = None
    if len(op.inputs) > 1:
        w_t = graph.get_tensor(op.inputs[1])
        if w_t is not None and getattr(w_t, "shape", None):
            ws = tuple(int(v) for v in w_t.shape)

    if ws is None and hasattr(graph, "constants") and len(op.inputs) > 1 and op.inputs[1] in graph.constants:
        ws = tuple(int(v) for v in graph.constants[op.inputs[1]].shape)

    if ws is None:
        attrs = getattr(op, "attrs", {}) or {}
        for key in ("kernel_shape_full", "weight_shape", "weights_shape"):
            if key in attrs and isinstance(attrs[key], (list, tuple)) and len(attrs[key]) == 4:
                ws = tuple(int(v) for v in attrs[key])
                break

    if ws is None:
        xshape = _get_tensor_shape(graph, op.inputs[0])
        yshape = _get_tensor_shape(graph, op.outputs[0])
        if xshape and yshape:
            c_in, _, _ = _as_chw(xshape)
            c_out, _, _ = _as_chw(yshape)
            k = int((getattr(op, "attrs", {}) or {}).get("kernel_shape", [3, 3])[0])
            ws = (c_out, c_in, k, k)

    if ws is None:
        raise RuntimeError(f"Conv weight shape unavailable for op '{op.name}'")

    out_c = int(ws[0])
    if b_arr is None:
        b_arr = np.zeros((out_c,), dtype=np.float32)

    return np.asarray(w_arr, dtype=np.float32).reshape(ws), np.asarray(b_arr, dtype=np.float32).reshape(-1), ws


def _resolve_bn_arrays(graph, op, c: int):
    def _get_or_default(idx, n, default):
        if len(op.inputs) > idx:
            arr = _flatten_from_graph_named(graph, op.inputs[idx])
            if arr is not None:
                return np.asarray(arr, dtype=np.float32).reshape(n)

        attrs = getattr(op, "attrs", {}) or {}
        attr_keys = {
            1: ("gamma", "scale", "weight", "weights"),
            2: ("beta", "bias", "biases"),
            3: ("mean", "running_mean"),
            4: ("var", "variance", "running_var"),
        }.get(idx, ())
        for k in attr_keys:
            if k in attrs:
                arr, _ = _resolve_attr_candidate(graph, attrs[k])
                if arr is not None:
                    return np.asarray(arr, dtype=np.float32).reshape(n)

        return np.full((n,), default, dtype=np.float32)

    gamma = _get_or_default(1, c, 1.0)
    beta = _get_or_default(2, c, 0.0)
    mean = _get_or_default(3, c, 0.0)
    var = _get_or_default(4, c, 1.0)
    return gamma, beta, mean, var


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    z = x - np.max(x)
    e = np.exp(z)
    s = np.sum(e)
    if s == 0:
        return np.zeros_like(x)
    return e / s


def _dense_forward(x: np.ndarray, W: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (W @ x + B).astype(np.float32)


def _dense_weight_grad(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.outer(dy, x).astype(np.float32)


def _dense_bias_grad(dy: np.ndarray) -> np.ndarray:
    return dy.astype(np.float32)


def _dense_backward_input(dy: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W.T @ dy).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(np.float32)


def _relu_backward_from_output(y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return (dy * (y > 0).astype(np.float32)).astype(np.float32)


def _leaky_relu(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(x > 0, x, alpha * x).astype(np.float32)


def _leaky_relu_backward_from_input(x: np.ndarray, dy: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(x > 0, dy, alpha * dy).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _sigmoid_backward_from_output(y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return (dy * y * (1.0 - y)).astype(np.float32)


def _softmax_backward(y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float32)
    for i in range(y.size):
        acc = 0.0
        for j in range(y.size):
            jac = y[i] * (1.0 - y[i]) if i == j else -y[i] * y[j]
            acc += jac * dy[j]
        out[i] = acc
    return out.astype(np.float32)


def _reshape_copy(x: np.ndarray) -> np.ndarray:
    return x.copy().astype(np.float32)


def _add_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b).astype(np.float32)


def _conv_forward_hwc(
    x: np.ndarray,
    xshape: Tuple[int, int, int],
    W: np.ndarray,
    B: np.ndarray,
    stride: int,
    pad: int,
    yshape: Tuple[int, int, int],
) -> np.ndarray:
    c_in, h_in, w_in = xshape
    c_out, h_out, w_out = yshape
    K = W.shape[2]
    y = np.zeros((h_out * w_out * c_out,), dtype=np.float32)

    for oh in range(h_out):
        for ow in range(w_out):
            for oc in range(c_out):
                acc = float(B[oc])
                for ic in range(c_in):
                    for kh in range(K):
                        for kw in range(K):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if 0 <= ih < h_in and 0 <= iw < w_in:
                                x_idx = (ih * w_in + iw) * c_in + ic
                                acc += float(x[x_idx]) * float(W[oc, ic, kh, kw])
                y[(oh * w_out + ow) * c_out + oc] = acc
    return y


def _conv_backward_input_hwc(
    dy: np.ndarray,
    W: np.ndarray,
    xshape: Tuple[int, int, int],
    yshape: Tuple[int, int, int],
    stride: int,
    pad: int,
) -> np.ndarray:
    c_in, h_in, w_in = xshape
    c_out, h_out, w_out = yshape
    K = W.shape[2]
    dx = np.zeros((h_in * w_in * c_in,), dtype=np.float32)

    for oh in range(h_out):
        for ow in range(w_out):
            for oc in range(c_out):
                gy = float(dy[(oh * w_out + ow) * c_out + oc])
                for ic in range(c_in):
                    for kh in range(K):
                        for kw in range(K):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if 0 <= ih < h_in and 0 <= iw < w_in:
                                x_idx = (ih * w_in + iw) * c_in + ic
                                dx[x_idx] += gy * float(W[oc, ic, kh, kw])
    return dx.astype(np.float32)


def _conv_weight_grad_hwc(
    x: np.ndarray,
    dy: np.ndarray,
    xshape: Tuple[int, int, int],
    yshape: Tuple[int, int, int],
    Wshape: Tuple[int, int, int, int],
    stride: int,
    pad: int,
) -> np.ndarray:
    c_in, h_in, w_in = xshape
    c_out, h_out, w_out = yshape
    _, _, K, _ = Wshape
    dW = np.zeros(Wshape, dtype=np.float32)

    for oc in range(c_out):
        for ic in range(c_in):
            for kh in range(K):
                for kw in range(K):
                    acc = 0.0
                    for oh in range(h_out):
                        for ow in range(w_out):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if 0 <= ih < h_in and 0 <= iw < w_in:
                                x_idx = (ih * w_in + iw) * c_in + ic
                                y_idx = (oh * w_out + ow) * c_out + oc
                                acc += float(x[x_idx]) * float(dy[y_idx])
                    dW[oc, ic, kh, kw] = acc
    return dW.reshape(-1).astype(np.float32)


def _conv_bias_grad_hwc(dy: np.ndarray, yshape: Tuple[int, int, int]) -> np.ndarray:
    c_out, h_out, w_out = yshape
    db = np.zeros((c_out,), dtype=np.float32)
    for oh in range(h_out):
        for ow in range(w_out):
            for oc in range(c_out):
                db[oc] += dy[(oh * w_out + ow) * c_out + oc]
    return db.astype(np.float32)


def _maxpool_forward_hwc(x: np.ndarray, xshape: Tuple[int, int, int], k: int, stride: int, yshape: Tuple[int, int, int]) -> np.ndarray:
    c_in, h_in, w_in = xshape
    c_out, h_out, w_out = yshape
    y = np.zeros((h_out * w_out * c_out,), dtype=np.float32)
    for oh in range(h_out):
        for ow in range(w_out):
            for c in range(c_in):
                best = x[((oh * stride) * w_in + (ow * stride)) * c_in + c]
                for kh in range(k):
                    for kw in range(k):
                        ih = oh * stride + kh
                        iw = ow * stride + kw
                        idx = (ih * w_in + iw) * c_in + c
                        if x[idx] > best:
                            best = x[idx]
                y[(oh * w_out + ow) * c_out + c] = best
    return y.astype(np.float32)


def _avgpool_forward_hwc(x: np.ndarray, xshape: Tuple[int, int, int], k: int, stride: int, yshape: Tuple[int, int, int]) -> np.ndarray:
    c_in, h_in, w_in = xshape
    c_out, h_out, w_out = yshape
    y = np.zeros((h_out * w_out * c_out,), dtype=np.float32)
    for oh in range(h_out):
        for ow in range(w_out):
            for c in range(c_in):
                acc = 0.0
                for kh in range(k):
                    for kw in range(k):
                        ih = oh * stride + kh
                        iw = ow * stride + kw
                        idx = (ih * w_in + iw) * c_in + c
                        acc += float(x[idx])
                y[(oh * w_out + ow) * c_out + c] = acc / float(k * k)
    return y.astype(np.float32)


def _maxpool_backward_hwc(x: np.ndarray, y: np.ndarray, dy: np.ndarray, xshape: Tuple[int, int, int], k: int, stride: int, yshape: Tuple[int, int, int]) -> np.ndarray:
    c_in, h_in, w_in = xshape
    c_out, h_out, w_out = yshape
    dx = np.zeros((h_in * w_in * c_in,), dtype=np.float32)
    for oh in range(h_out):
        for ow in range(w_out):
            for c in range(c_in):
                out_idx = (oh * w_out + ow) * c_out + c
                pooled = y[out_idx]
                routed = False
                for kh in range(k):
                    for kw in range(k):
                        ih = oh * stride + kh
                        iw = ow * stride + kw
                        in_idx = (ih * w_in + iw) * c_in + c
                        if (not routed) and (x[in_idx] == pooled):
                            dx[in_idx] += dy[out_idx]
                            routed = True
    return dx.astype(np.float32)


def _avgpool_backward_hwc(dy: np.ndarray, xshape: Tuple[int, int, int], k: int, stride: int, yshape: Tuple[int, int, int]) -> np.ndarray:
    c_in, h_in, w_in = xshape
    c_out, h_out, w_out = yshape
    dx = np.zeros((h_in * w_in * c_in,), dtype=np.float32)
    scale = 1.0 / float(k * k)
    for oh in range(h_out):
        for ow in range(w_out):
            for c in range(c_in):
                out_idx = (oh * w_out + ow) * c_out + c
                g = float(dy[out_idx]) * scale
                for kh in range(k):
                    for kw in range(k):
                        ih = oh * stride + kh
                        iw = ow * stride + kw
                        in_idx = (ih * w_in + iw) * c_in + c
                        dx[in_idx] += g
    return dx.astype(np.float32)


def _bn_forward_hwc(x: np.ndarray, shape: Tuple[int, int, int], gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    c, h, w = shape
    hw = h * w
    mean = np.zeros((c,), dtype=np.float32)
    var = np.zeros((c,), dtype=np.float32)
    xhat = np.zeros_like(x, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.float32)

    for ch in range(c):
        vals = np.array([x[hw_i * c + ch] for hw_i in range(hw)], dtype=np.float32)
        m = float(np.mean(vals))
        v = float(np.mean((vals - m) ** 2))
        mean[ch] = m
        var[ch] = v
        inv_std = 1.0 / np.sqrt(v + eps)
        for hw_i in range(hw):
            idx = hw_i * c + ch
            xn = (x[idx] - m) * inv_std
            xhat[idx] = xn
            y[idx] = gamma[ch] * xn + beta[ch]

    cache = {"mean": mean, "var": var, "xhat": xhat, "eps": float(eps)}
    return y.astype(np.float32), cache


def _bn_param_grad_hwc(dy: np.ndarray, xhat: np.ndarray, shape: Tuple[int, int, int]):
    c, h, w = shape
    hw = h * w
    dgamma = np.zeros((c,), dtype=np.float32)
    dbeta = np.zeros((c,), dtype=np.float32)

    for ch in range(c):
        dg = 0.0
        db = 0.0
        for hw_i in range(hw):
            idx = hw_i * c + ch
            dg += float(dy[idx]) * float(xhat[idx])
            db += float(dy[idx])
        dgamma[ch] = dg
        dbeta[ch] = db

    return dgamma, dbeta


def _bn_backward_input_exact_hwc(
    dy: np.ndarray,
    xhat: np.ndarray,
    var: np.ndarray,
    shape: Tuple[int, int, int],
    gamma: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    c, h, w = shape
    hw = h * w
    dx = np.zeros_like(dy, dtype=np.float32)

    for ch in range(c):
        inv_std = 1.0 / float(np.sqrt(float(var[ch]) + float(eps)))
        sum_dy = 0.0
        sum_dy_xhat = 0.0

        for hw_i in range(hw):
            idx = hw_i * c + ch
            g = float(dy[idx])
            xh = float(xhat[idx])
            sum_dy += g
            sum_dy_xhat += g * xh

        mean_dy = sum_dy / float(hw)
        mean_dy_xhat = sum_dy_xhat / float(hw)
        scale = float(gamma[ch]) * inv_std

        for hw_i in range(hw):
            idx = hw_i * c + ch
            xh = float(xhat[idx])
            dx[idx] = scale * (float(dy[idx]) - mean_dy - xh * mean_dy_xhat)

    return dx.astype(np.float32)


def _forward_pass(graph, params_state: Dict[str, Dict[str, np.ndarray]], x_input: np.ndarray, layerwise_dir: Path):
    vals: Dict[str, np.ndarray] = {}
    caches: Dict[str, Dict[str, Any]] = {}
    inferred_shapes: Dict[str, Tuple[int, ...]] = {}

    input_name = graph.inputs[0]
    vals[input_name] = x_input.astype(np.float32)

    input_shape = _get_tensor_shape(graph, input_name)
    if input_shape:
        inferred_shapes[input_name] = input_shape

    def get_shape(name: str) -> Tuple[int, ...]:
        if name in inferred_shapes and inferred_shapes[name]:
            return inferred_shapes[name]
        shp = _get_tensor_shape(graph, name)
        if shp:
            inferred_shapes[name] = shp
            return shp
        return tuple()

    for op in graph.ops:
        xname = op.inputs[0]
        yname = op.outputs[0]
        x = vals[xname]
        y = None
        cache: Dict[str, Any] = {}

        if op.op_type == "Dense":
            tag = op.name
            W = params_state[tag]["W"]
            B = params_state[tag]["B"]
            y = _dense_forward(x, W, B)
            cache["x"] = x
            inferred_shapes[yname] = (int(B.size),)

        elif op.op_type == "Conv":
            tag = op.name
            W = params_state[tag]["W"]
            B = params_state[tag]["B"]
            xshape = get_shape(xname)
            if not xshape:
                raise RuntimeError(f"Conv input shape unavailable for op '{op.name}' input '{xname}'")
            stride = int(op.attrs.get("strides", [1, 1])[0])
            pad = int(op.attrs.get("pads", [0, 0, 0, 0])[0])

            yshape = get_shape(yname)
            if not yshape:
                yshape = _infer_conv_output_shape(_as_chw(xshape), tuple(int(v) for v in W.shape), stride, pad)

            y = _conv_forward_hwc(x, _as_chw(xshape), W, B, stride, pad, _as_chw(yshape))
            cache["x"] = x
            cache["xshape"] = _as_chw(xshape)
            cache["yshape"] = _as_chw(yshape)
            cache["stride"] = stride
            cache["pad"] = pad
            cache["Wshape"] = W.shape
            inferred_shapes[yname] = tuple(int(v) for v in yshape)

        elif op.op_type == "Relu":
            y = _relu(x)
            xshape = get_shape(xname)
            if xshape:
                inferred_shapes[yname] = xshape

        elif op.op_type == "LeakyRelu":
            alpha = float((getattr(op, "attrs", {}) or {}).get("alpha", 0.01))
            y = _leaky_relu(x, alpha)
            cache["alpha"] = alpha
            cache["x"] = x
            xshape = get_shape(xname)
            if xshape:
                inferred_shapes[yname] = xshape

        elif op.op_type == "Sigmoid":
            y = _sigmoid(x)
            xshape = get_shape(xname)
            if xshape:
                inferred_shapes[yname] = xshape

        elif op.op_type == "Softmax":
            y = _softmax(x)
            xshape = get_shape(xname)
            if xshape:
                inferred_shapes[yname] = xshape

        elif op.op_type in ("Flatten", "Reshape"):
            y = _reshape_copy(x)
            inferred_shapes[yname] = (int(y.size),)

        elif op.op_type == "Add":
            rhs = vals[op.inputs[1]]
            y = _add_vec(x, rhs)
            xshape = get_shape(xname)
            if xshape:
                inferred_shapes[yname] = xshape
            else:
                inferred_shapes[yname] = (int(y.size),)

        elif op.op_type == "MaxPool":
            xshape = get_shape(xname)
            if not xshape:
                raise RuntimeError(f"MaxPool input shape unavailable for op '{op.name}' input '{xname}'")
            k = int(op.attrs.get("kernel_shape", [2, 2])[0])
            stride = int(op.attrs.get("strides", [2, 2])[0])
            yshape = get_shape(yname)
            if not yshape:
                yshape = _infer_pool_output_shape(_as_chw(xshape), k, stride)
            y = _maxpool_forward_hwc(x, _as_chw(xshape), k, stride, _as_chw(yshape))
            cache["x"] = x
            cache["xshape"] = _as_chw(xshape)
            cache["yshape"] = _as_chw(yshape)
            cache["k"] = k
            cache["stride"] = stride
            inferred_shapes[yname] = tuple(int(v) for v in yshape)

        elif op.op_type == "AvgPool":
            xshape = get_shape(xname)
            if not xshape:
                raise RuntimeError(f"AvgPool input shape unavailable for op '{op.name}' input '{xname}'")
            k = int(op.attrs.get("kernel_shape", [2, 2])[0])
            stride = int(op.attrs.get("strides", [2, 2])[0])
            yshape = get_shape(yname)
            if not yshape:
                yshape = _infer_pool_output_shape(_as_chw(xshape), k, stride)
            y = _avgpool_forward_hwc(x, _as_chw(xshape), k, stride, _as_chw(yshape))
            cache["xshape"] = _as_chw(xshape)
            cache["yshape"] = _as_chw(yshape)
            cache["k"] = k
            cache["stride"] = stride
            inferred_shapes[yname] = tuple(int(v) for v in yshape)

        elif op.op_type == "BatchNormalization":
            tag = op.name
            gamma = params_state[tag]["gamma"]
            beta = params_state[tag]["beta"]
            shape = get_shape(yname)
            if not shape:
                shape = get_shape(xname)
            if not shape:
                raise RuntimeError(f"BatchNormalization shape unavailable for op '{op.name}'")
            y, bn_cache = _bn_forward_hwc(x, _as_chw(shape), gamma, beta)
            cache["shape"] = _as_chw(shape)
            cache.update(bn_cache)
            inferred_shapes[yname] = tuple(int(v) for v in shape)

        else:
            raise RuntimeError(f"Unsupported training reference op: {op.op_type}")

        vals[yname] = y.astype(np.float32)
        caches[op.name] = cache
        _write_f32(layerwise_dir / f"{op.name}__fwd.bin", y.astype(np.float32))

    return vals, caches


def _mse_loss_and_grad(pred: np.ndarray, target: np.ndarray):
    diff = pred.astype(np.float32) - target.astype(np.float32)
    loss = float(np.mean(diff * diff))
    grad = (2.0 / float(pred.size)) * diff
    return loss, grad.astype(np.float32)


def _is_final_softmax_mse_case(graph, raw_cfg: Dict[str, Any]) -> bool:
    loss_type = str((((raw_cfg.get("training", {}) or {}).get("loss", {}) or {}).get("type", "mse")).lower())
    if loss_type != "mse":
        return False
    if not getattr(graph, "ops", None):
        return False
    last_op = graph.ops[-1]
    return bool(last_op.op_type == "Softmax" and last_op.outputs and graph.outputs and last_op.outputs[0] == graph.outputs[0])


def run_training_reference_step(
    *,
    graph,
    raw_cfg: Dict[str, Any],
    out_dir: Path,
    x_input: np.ndarray,
    target: np.ndarray,
) -> TrainingReferenceResult:
    ref_dir = Path(out_dir) / "training_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    layerwise_dir = ref_dir / "layerwise"
    layerwise_dir.mkdir(parents=True, exist_ok=True)

    lr = float((((raw_cfg.get("training", {}) or {}).get("optimizer", {}) or {}).get("learning_rate", 0.01)))
    bypass_final_softmax_backward = _is_final_softmax_mse_case(graph, raw_cfg)

    params_state: Dict[str, Dict[str, np.ndarray]] = {}
    trainable_order: List[Tuple[str, str]] = []

    for op in graph.ops:
        if op.op_type == "Dense":
            W, B, _in_f, _out_f = _resolve_dense_arrays(graph, op)
            params_state[op.name] = {"W": W.copy(), "B": B.copy()}
            trainable_order.append(("dense", op.name))

        elif op.op_type == "Conv":
            W, B, _ws = _resolve_conv_arrays(graph, op)
            params_state[op.name] = {"W": W.copy(), "B": B.reshape(-1).copy()}
            trainable_order.append(("conv", op.name))

        elif op.op_type == "BatchNormalization":
            shape = _get_tensor_shape(graph, op.outputs[0])
            if not shape:
                shape = _get_tensor_shape(graph, op.inputs[0])
            if not shape:
                raise RuntimeError(f"BatchNormalization shape unavailable for op '{op.name}'")
            c, _, _ = _as_chw(shape)
            gamma, beta, _mean, _var = _resolve_bn_arrays(graph, op, c)
            params_state[op.name] = {
                "gamma": gamma.copy(),
                "beta": beta.copy(),
            }
            trainable_order.append(("bn", op.name))

    weights_before_chunks: List[np.ndarray] = []
    for kind, name in trainable_order:
        ps = params_state[name]
        if kind in ("dense", "conv"):
            weights_before_chunks.append(ps["W"].reshape(-1).copy())
            weights_before_chunks.append(ps["B"].reshape(-1).copy())
            _write_f32(layerwise_dir / f"{name}__weights_before.bin", np.concatenate([ps["W"].reshape(-1), ps["B"].reshape(-1)]).astype(np.float32))
        else:
            weights_before_chunks.append(ps["gamma"].reshape(-1).copy())
            weights_before_chunks.append(ps["beta"].reshape(-1).copy())
            _write_f32(layerwise_dir / f"{name}__weights_before.bin", np.concatenate([ps["gamma"].reshape(-1), ps["beta"].reshape(-1)]).astype(np.float32))

    vals, caches = _forward_pass(graph, params_state, x_input.astype(np.float32), layerwise_dir)
    pred = vals[graph.outputs[0]]
    loss_before, d_output = _mse_loss_and_grad(pred, target.astype(np.float32))

    grads_by_tensor: Dict[str, np.ndarray] = {graph.outputs[0]: d_output.astype(np.float32)}
    grad_chunks_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for op_idx in range(len(graph.ops) - 1, -1, -1):
        op = graph.ops[op_idx]
        xname = op.inputs[0]
        yname = op.outputs[0]
        y = vals[yname]
        dy = grads_by_tensor[yname]

        if op.op_type == "Dense":
            ps = params_state[op.name]
            dW = _dense_weight_grad(caches[op.name]["x"], dy)
            dB = _dense_bias_grad(dy)
            dx = _dense_backward_input(dy, ps["W"])
            grad_chunks_map[op.name] = (dW.reshape(-1), dB.reshape(-1))
            grads_by_tensor[xname] = dx
            _write_f32(layerwise_dir / f"{op.name}__param_grad_w.bin", dW.reshape(-1).astype(np.float32))
            _write_f32(layerwise_dir / f"{op.name}__param_grad_b.bin", dB.reshape(-1).astype(np.float32))

        elif op.op_type == "Conv":
            ps = params_state[op.name]
            cache = caches[op.name]
            dW = _conv_weight_grad_hwc(
                cache["x"], dy, cache["xshape"], cache["yshape"], cache["Wshape"], cache["stride"], cache["pad"]
            )
            dB = _conv_bias_grad_hwc(dy, cache["yshape"])
            dx = _conv_backward_input_hwc(dy, ps["W"], cache["xshape"], cache["yshape"], cache["stride"], cache["pad"])
            grad_chunks_map[op.name] = (dW.reshape(-1), dB.reshape(-1))
            grads_by_tensor[xname] = dx
            _write_f32(layerwise_dir / f"{op.name}__param_grad_w.bin", dW.reshape(-1).astype(np.float32))
            _write_f32(layerwise_dir / f"{op.name}__param_grad_b.bin", dB.reshape(-1).astype(np.float32))

        elif op.op_type == "Relu":
            dx = _relu_backward_from_output(y, dy)
            grads_by_tensor[xname] = dx

        elif op.op_type == "LeakyRelu":
            dx = _leaky_relu_backward_from_input(caches[op.name]["x"], dy, caches[op.name]["alpha"])
            grads_by_tensor[xname] = dx

        elif op.op_type == "Sigmoid":
            dx = _sigmoid_backward_from_output(y, dy)
            grads_by_tensor[xname] = dx

        elif op.op_type == "Softmax":
            is_final_op = bool(op_idx == len(graph.ops) - 1 and yname == graph.outputs[0])
            if bypass_final_softmax_backward and is_final_op:
                grads_by_tensor[xname] = dy.copy().astype(np.float32)
            else:
                dx = _softmax_backward(y, dy)
                grads_by_tensor[xname] = dx

        elif op.op_type in ("Flatten", "Reshape"):
            grads_by_tensor[xname] = dy.copy().astype(np.float32)

        elif op.op_type == "Add":
            grads_by_tensor[xname] = dy.copy().astype(np.float32)
            rhs = op.inputs[1]
            grads_by_tensor[rhs] = grads_by_tensor.get(rhs, np.zeros_like(dy, dtype=np.float32)) + dy.astype(np.float32)

        elif op.op_type == "MaxPool":
            cache = caches[op.name]
            dx = _maxpool_backward_hwc(cache["x"], y, dy, cache["xshape"], cache["k"], cache["stride"], cache["yshape"])
            grads_by_tensor[xname] = dx

        elif op.op_type == "AvgPool":
            cache = caches[op.name]
            dx = _avgpool_backward_hwc(dy, cache["xshape"], cache["k"], cache["stride"], cache["yshape"])
            grads_by_tensor[xname] = dx

        elif op.op_type == "BatchNormalization":
            ps = params_state[op.name]
            cache = caches[op.name]
            dgamma, dbeta = _bn_param_grad_hwc(dy, cache["xhat"], cache["shape"])
            dx = _bn_backward_input_exact_hwc(
                dy,
                cache["xhat"],
                cache["var"],
                cache["shape"],
                ps["gamma"],
                eps=float(cache.get("eps", 1e-5)),
            )
            grad_chunks_map[op.name] = (dgamma.reshape(-1), dbeta.reshape(-1))
            grads_by_tensor[xname] = dx
            _write_f32(layerwise_dir / f"{op.name}__param_grad_gamma.bin", dgamma.reshape(-1).astype(np.float32))
            _write_f32(layerwise_dir / f"{op.name}__param_grad_beta.bin", dbeta.reshape(-1).astype(np.float32))

        else:
            raise RuntimeError(f"Unsupported training reference op: {op.op_type}")

        _write_f32(layerwise_dir / f"{op.name}__bwd_in.bin", grads_by_tensor[xname].astype(np.float32))

    grads_chunks: List[np.ndarray] = []
    for kind, name in trainable_order:
        g0, g1 = grad_chunks_map[name]
        grads_chunks.append(g0.reshape(-1))
        grads_chunks.append(g1.reshape(-1))

    grads_flat = np.concatenate(grads_chunks, axis=0).astype(np.float32) if grads_chunks else np.zeros((0,), dtype=np.float32)
    weights_before_flat = np.concatenate(weights_before_chunks, axis=0).astype(np.float32) if weights_before_chunks else np.zeros((0,), dtype=np.float32)

    for kind, name in trainable_order:
        ps = params_state[name]
        g0, g1 = grad_chunks_map[name]
        if kind in ("dense", "conv"):
            ps["W"] = (ps["W"].reshape(-1) - lr * g0.reshape(-1)).reshape(ps["W"].shape).astype(np.float32)
            ps["B"] = (ps["B"].reshape(-1) - lr * g1.reshape(-1)).reshape(ps["B"].shape).astype(np.float32)
        else:
            ps["gamma"] = (ps["gamma"].reshape(-1) - lr * g0.reshape(-1)).reshape(ps["gamma"].shape).astype(np.float32)
            ps["beta"] = (ps["beta"].reshape(-1) - lr * g1.reshape(-1)).reshape(ps["beta"].shape).astype(np.float32)

    weights_after_chunks: List[np.ndarray] = []
    for kind, name in trainable_order:
        ps = params_state[name]
        if kind in ("dense", "conv"):
            weights_after_chunks.append(ps["W"].reshape(-1).copy())
            weights_after_chunks.append(ps["B"].reshape(-1).copy())
            _write_f32(layerwise_dir / f"{name}__weights_after.bin", np.concatenate([ps["W"].reshape(-1), ps["B"].reshape(-1)]).astype(np.float32))
        else:
            weights_after_chunks.append(ps["gamma"].reshape(-1).copy())
            weights_after_chunks.append(ps["beta"].reshape(-1).copy())
            _write_f32(layerwise_dir / f"{name}__weights_after.bin", np.concatenate([ps["gamma"].reshape(-1), ps["beta"].reshape(-1)]).astype(np.float32))

    weights_after_flat = np.concatenate(weights_after_chunks, axis=0).astype(np.float32) if weights_after_chunks else np.zeros((0,), dtype=np.float32)

    vals_after, _ = _forward_pass(graph, params_state, x_input.astype(np.float32), layerwise_dir / "after_step")
    pred_after = vals_after[graph.outputs[0]]
    loss_after, _ = _mse_loss_and_grad(pred_after, target.astype(np.float32))

    grads_flat_path = ref_dir / "grads_ref.bin"
    weights_before_flat_path = ref_dir / "weights_before_ref.bin"
    weights_after_flat_path = ref_dir / "weights_after_ref.bin"
    summary_json = ref_dir / "summary.json"
    summary_txt = ref_dir / "summary.txt"

    _write_f32(grads_flat_path, grads_flat)
    _write_f32(weights_before_flat_path, weights_before_flat)
    _write_f32(weights_after_flat_path, weights_after_flat)

    payload = {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "num_grad_words": int(grads_flat.size),
        "num_weight_words": int(weights_before_flat.size),
        "layerwise_dir": str(layerwise_dir),
        "bypass_final_softmax_backward": bool(bypass_final_softmax_backward),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_txt.write_text(
        "\n".join(
            [
                "=========== FPGAI Training Reference ===========",
                f"loss_before : {loss_before}",
                f"loss_after  : {loss_after}",
                f"grad_words  : {int(grads_flat.size)}",
                f"param_words : {int(weights_before_flat.size)}",
                f"layerwise   : {layerwise_dir}",
                f"bypass_final_softmax_backward : {bool(bypass_final_softmax_backward)}",
                "================================================",
            ]
        ),
        encoding="utf-8",
    )

    return TrainingReferenceResult(
        out_dir=ref_dir,
        grads_flat_path=grads_flat_path,
        weights_before_flat_path=weights_before_flat_path,
        weights_after_flat_path=weights_after_flat_path,
        summary_json=summary_json,
        summary_txt=summary_txt,
        loss_before=loss_before,
        loss_after=loss_after,
        layerwise_dir=layerwise_dir,
    )