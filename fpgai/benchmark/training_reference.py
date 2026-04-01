from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np

from fpgai.ir.graph import Graph
from fpgai.util.fs import write_text


@dataclass(frozen=True)
class TrainingReferenceResult:
    loss_before: float
    loss_after: float
    grads_flat_path: Path
    weights_before_flat_path: Path
    weights_after_flat_path: Path
    summary_json: Path
    summary_txt: Path


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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


def _flatten_from_graph_named(graph: Graph, tensor_name: str) -> Optional[np.ndarray]:
    if tensor_name is None:
        return None

    if hasattr(graph, "constants") and tensor_name in graph.constants:
        arr = _as_numpy_numeric(graph.constants[tensor_name])
        if arr is not None:
            return arr.reshape(-1)

    if hasattr(graph, "params") and tensor_name in graph.params:
        arr = _as_numpy_numeric(graph.params[tensor_name])
        if arr is not None:
            return arr.reshape(-1)

    try:
        t = graph.get_tensor(tensor_name)
    except Exception:
        t = None

    if t is not None:
        data = getattr(t, "data", None)
        if data is not None:
            arr = _as_numpy_numeric(data)
            if arr is not None:
                return arr.reshape(-1)

        for attr_name in ("initializer", "value", "values"):
            if hasattr(t, attr_name):
                arr = _as_numpy_numeric(getattr(t, attr_name))
                if arr is not None:
                    return arr.reshape(-1)

    return None


def _array_from_graph_named(graph: Graph, tensor_name: str) -> Optional[np.ndarray]:
    arr = _flatten_from_graph_named(graph, tensor_name)
    if arr is not None:
        return arr.copy()

    try:
        t = graph.get_tensor(tensor_name)
    except Exception:
        t = None

    if t is not None and getattr(t, "shape", None):
        shape = tuple(int(x) for x in t.shape)
        flat = _flatten_from_graph_named(graph, tensor_name)
        if flat is not None and int(np.prod(shape)) == flat.size:
            return flat.reshape(shape).astype(np.float32)

    return None


def _resolve_attr_candidate(graph: Graph, v):
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


def _resolve_dense_arrays(graph: Graph, op) -> Tuple[np.ndarray, np.ndarray, int, int, str, Optional[str]]:
    in_f, out_f, expected_w, expected_b = _dense_expected_shapes(op)

    w_arr = None
    b_arr = None
    w_name = None
    b_name = None

    if len(op.inputs) > 1:
        w_name = op.inputs[1]
        w_arr = _flatten_from_graph_named(graph, w_name)

    if len(op.inputs) > 2:
        b_name = op.inputs[2]
        b_arr = _flatten_from_graph_named(graph, b_name)

    if w_arr is None or (b_arr is None and expected_b > 0):
        wa, ba = _pick_dense_from_attrs(graph, op, expected_w, expected_b)
        if w_arr is None and wa is not None:
            w_arr = wa
            w_name = f"{op.name}__attr_weight"
        if b_arr is None and ba is not None:
            b_arr = ba
            b_name = f"{op.name}__attr_bias"

    if w_arr is None:
        raise RuntimeError(
            f"Dense weights not found for training op '{op.name}'. "
            f"op.inputs={list(getattr(op, 'inputs', []))}, "
            f"op.attrs keys={sorted(list((getattr(op, 'attrs', {}) or {}).keys()))}"
        )

    if expected_w > 0:
        if w_arr.size != expected_w:
            raise RuntimeError(
                f"Dense weight size mismatch for '{op.name}': got {w_arr.size}, expected {expected_w}"
            )
        W = w_arr.reshape(out_f, in_f).astype(np.float32)
    else:
        if w_arr.ndim == 2:
            W = w_arr.astype(np.float32)
            out_f, in_f = int(W.shape[0]), int(W.shape[1])
        else:
            raise RuntimeError(f"Cannot infer Dense weight shape for '{op.name}'")

    if expected_b > 0:
        if b_arr is None:
            B = np.zeros((out_f,), dtype=np.float32)
        else:
            if b_arr.size != expected_b:
                raise RuntimeError(
                    f"Dense bias size mismatch for '{op.name}': got {b_arr.size}, expected {expected_b}"
                )
            B = b_arr.reshape(out_f).astype(np.float32)
    else:
        B = np.zeros((out_f,), dtype=np.float32)

    return W, B, in_f, out_f, w_name, b_name


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    return p.astype(np.float32)


def _softmax_backward(y: np.ndarray, gy: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    gy = gy.astype(np.float32)
    dot = float(np.sum(y * gy))
    return (y * (gy - dot)).astype(np.float32)


def _mse_loss_and_grad(pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    diff = pred - target
    loss = float(np.mean(diff * diff))
    grad = (2.0 / diff.size) * diff
    return loss, grad.astype(np.float32)


def _cross_entropy_loss_and_grad(probs: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    p = probs.astype(np.float64)
    p = p / np.sum(p)
    loss = float(-np.sum(target * np.log(p + 1e-12)))
    grad = (p - target).astype(np.float32)
    return loss, grad


def _conv2d_nchw(x, w, b, strides=(1, 1), pads=(0, 0, 0, 0)):
    n, c_in, h, w_in = x.shape
    c_out, _, kh, kw = w.shape
    sh, sw = strides
    pt, pl, pb, pr = pads

    xpad = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    hout = (xpad.shape[2] - kh) // sh + 1
    wout = (xpad.shape[3] - kw) // sw + 1

    y = np.zeros((n, c_out, hout, wout), dtype=np.float32)
    for nn in range(n):
        for co in range(c_out):
            for ho in range(hout):
                for wo in range(wout):
                    acc = float(b[co]) if b is not None else 0.0
                    hs = ho * sh
                    ws = wo * sw
                    for ci in range(c_in):
                        for i in range(kh):
                            for j in range(kw):
                                acc += float(xpad[nn, ci, hs + i, ws + j]) * float(w[co, ci, i, j])
                    y[nn, co, ho, wo] = acc
    return y, xpad


def _conv2d_backward_nchw(x, xpad, w, gy, strides=(1, 1), pads=(0, 0, 0, 0)):
    n, c_in, h, w_in = x.shape
    c_out, _, kh, kw = w.shape
    sh, sw = strides
    pt, pl, pb, pr = pads

    dxpad = np.zeros_like(xpad, dtype=np.float32)
    dw = np.zeros_like(w, dtype=np.float32)
    db = np.zeros((c_out,), dtype=np.float32)

    hout, wout = gy.shape[2], gy.shape[3]

    for nn in range(n):
        for co in range(c_out):
            for ho in range(hout):
                for wo in range(wout):
                    g = float(gy[nn, co, ho, wo])
                    db[co] += g
                    hs = ho * sh
                    ws = wo * sw
                    for ci in range(c_in):
                        for i in range(kh):
                            for j in range(kw):
                                dw[co, ci, i, j] += g * float(xpad[nn, ci, hs + i, ws + j])
                                dxpad[nn, ci, hs + i, ws + j] += g * float(w[co, ci, i, j])

    dx = dxpad[:, :, pt:pt + h, pl:pl + w_in].astype(np.float32)
    return dx, dw.astype(np.float32), db.astype(np.float32)


def _pool2d_forward_nchw(x, kernel, strides, pads, mode):
    n, c, h, w = x.shape
    kh, kw = kernel
    sh, sw = strides
    pt, pl, pb, pr = pads
    xpad = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    hout = (xpad.shape[2] - kh) // sh + 1
    wout = (xpad.shape[3] - kw) // sw + 1

    y = np.zeros((n, c, hout, wout), dtype=np.float32)
    mask = np.zeros_like(xpad, dtype=np.float32) if mode == "max" else None

    for nn in range(n):
        for cc in range(c):
            for ho in range(hout):
                for wo in range(wout):
                    hs = ho * sh
                    ws = wo * sw
                    window = xpad[nn, cc, hs:hs + kh, ws:ws + kw]
                    if mode == "max":
                        m = np.max(window)
                        y[nn, cc, ho, wo] = m
                        idx = np.unravel_index(np.argmax(window), window.shape)
                        mask[nn, cc, hs + idx[0], ws + idx[1]] += 1.0
                    else:
                        y[nn, cc, ho, wo] = float(np.mean(window))
    return y, xpad, mask


def _pool2d_backward_nchw(x, xpad, gy, kernel, strides, pads, mode, mask=None):
    n, c, h, w = x.shape
    kh, kw = kernel
    sh, sw = strides
    pt, pl, pb, pr = pads
    dxpad = np.zeros_like(xpad, dtype=np.float32)

    hout, wout = gy.shape[2], gy.shape[3]
    for nn in range(n):
        for cc in range(c):
            for ho in range(hout):
                for wo in range(wout):
                    g = float(gy[nn, cc, ho, wo])
                    hs = ho * sh
                    ws = wo * sw
                    if mode == "max":
                        window_mask = mask[nn, cc, hs:hs + kh, ws:ws + kw]
                        dxpad[nn, cc, hs:hs + kh, ws:ws + kw] += g * window_mask
                    else:
                        dxpad[nn, cc, hs:hs + kh, ws:ws + kw] += g / float(kh * kw)

    dx = dxpad[:, :, pt:pt + h, pl:pl + w].astype(np.float32)
    return dx


def _bn_forward(x, scale, bias, mean, var, eps):
    shape = (1, -1) + (1,) * (x.ndim - 2)
    y = ((x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)) * scale.reshape(shape) + bias.reshape(shape)
    return y.astype(np.float32)


def _bn_backward(x, gy, scale, mean, var, eps):
    shape = (1, -1) + (1,) * (x.ndim - 2)
    invstd = 1.0 / np.sqrt(var.reshape(shape) + eps)
    dx = gy * scale.reshape(shape) * invstd
    dscale = np.sum(gy * ((x - mean.reshape(shape)) * invstd), axis=tuple(i for i in range(x.ndim) if i != 1))
    dbias = np.sum(gy, axis=tuple(i for i in range(x.ndim) if i != 1))
    return dx.astype(np.float32), dscale.astype(np.float32), dbias.astype(np.float32)


def _read_conv_attrs(op):
    attrs = getattr(op, "attrs", {}) or {}
    strides = tuple(attrs.get("strides", [1, 1]))
    pads = attrs.get("pads", [0, 0, 0, 0])
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    return tuple(int(x) for x in strides), tuple(int(x) for x in pads)


def _read_pool_attrs(op):
    attrs = getattr(op, "attrs", {}) or {}
    kernel = attrs.get("kernel_shape", [2, 2])
    strides = attrs.get("strides", kernel)
    pads = attrs.get("pads", [0, 0, 0, 0])
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    return tuple(int(x) for x in kernel), tuple(int(x) for x in strides), tuple(int(x) for x in pads)


def run_training_reference_step(
    *,
    graph: Graph,
    raw_cfg: Dict[str, Any],
    out_dir: Path,
    x_input: np.ndarray,
    target: np.ndarray,
) -> TrainingReferenceResult:
    out_dir = Path(out_dir)
    ref_dir = out_dir / "training_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    lr = float(_cfg_get(raw_cfg, "training.optimizer.learning_rate", 0.01))
    loss_type = str(_cfg_get(raw_cfg, "training.loss.type", "mse")).lower()

    vals: Dict[str, np.ndarray] = {}
    grads: Dict[str, np.ndarray] = {}
    cache: Dict[str, Dict[str, Any]] = {}

    input_name = graph.inputs[0]
    output_name = graph.outputs[0]
    vals[input_name] = np.asarray(x_input, dtype=np.float32)

    trainable_order: List[Tuple[str, List[np.ndarray], List[np.ndarray]]] = []

    for op in graph.ops:
        xname = op.inputs[0]
        yname = op.outputs[0]
        x = vals[xname]

        if op.op_type in ("Flatten", "Reshape"):
            vals[yname] = x.reshape(-1).astype(np.float32)
            cache[op.name] = {"op_type": op.op_type, "xname": xname, "yname": yname, "in_shape": x.shape}

        elif op.op_type == "Dense":
            W, B, _in_f, _out_f, _w_name, _b_name = _resolve_dense_arrays(graph, op)
            xv = x.reshape(-1)
            y = (W @ xv + B).astype(np.float32)
            vals[yname] = y
            cache[op.name] = {"op_type": "Dense", "xname": xname, "yname": yname, "x": xv.copy(), "W": W.copy(), "B": B.copy()}
            trainable_order.append((op.name, [W.copy(), B.copy()], []))

        elif op.op_type == "Conv":
            W = _array_from_graph_named(graph, op.inputs[1]).astype(np.float32)
            B = _array_from_graph_named(graph, op.inputs[2]).astype(np.float32) if len(op.inputs) > 2 else np.zeros((W.shape[0],), dtype=np.float32)
            x4 = x.astype(np.float32)
            if x4.ndim == 3:
                x4 = x4[None, ...]
            strides, pads = _read_conv_attrs(op)
            y, xpad = _conv2d_nchw(x4, W, B, strides=strides, pads=pads)
            vals[yname] = y.astype(np.float32)
            cache[op.name] = {"op_type": "Conv", "xname": xname, "yname": yname, "x": x4.copy(), "xpad": xpad.copy(), "W": W.copy(), "B": B.copy(), "strides": strides, "pads": pads}
            trainable_order.append((op.name, [W.copy(), B.copy()], []))

        elif op.op_type == "Relu":
            y = np.maximum(x, 0.0).astype(np.float32)
            vals[yname] = y
            cache[op.name] = {"op_type": "Relu", "xname": xname, "yname": yname, "y": y.copy()}

        elif op.op_type == "LeakyRelu":
            alpha = float((getattr(op, "attrs", {}) or {}).get("alpha", 0.01))
            y = np.where(x > 0, x, alpha * x).astype(np.float32)
            vals[yname] = y
            cache[op.name] = {"op_type": "LeakyRelu", "xname": xname, "yname": yname, "x": x.copy(), "alpha": alpha}

        elif op.op_type == "Sigmoid":
            y = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
            vals[yname] = y
            cache[op.name] = {"op_type": "Sigmoid", "xname": xname, "yname": yname, "y": y.copy()}

        elif op.op_type == "Softmax":
            y = _softmax(x.reshape(-1))
            vals[yname] = y
            cache[op.name] = {"op_type": "Softmax", "xname": xname, "yname": yname, "y": y.copy()}

        elif op.op_type == "Add":
            rhs_name = op.inputs[1]
            rhs = vals[rhs_name] if rhs_name in vals else _array_from_graph_named(graph, rhs_name)
            y = (x + rhs).astype(np.float32)
            vals[yname] = y
            cache[op.name] = {"op_type": "Add", "xname": xname, "rhs_name": rhs_name, "yname": yname, "rhs_shape": np.shape(rhs)}

        elif op.op_type == "BatchNormalization":
            scale = _array_from_graph_named(graph, op.inputs[1]).astype(np.float32)
            bias = _array_from_graph_named(graph, op.inputs[2]).astype(np.float32)
            mean = _array_from_graph_named(graph, op.inputs[3]).astype(np.float32)
            var = _array_from_graph_named(graph, op.inputs[4]).astype(np.float32)
            eps = float((getattr(op, "attrs", {}) or {}).get("epsilon", 1e-5))
            y = _bn_forward(x.astype(np.float32), scale, bias, mean, var, eps)
            vals[yname] = y
            cache[op.name] = {"op_type": "BatchNormalization", "xname": xname, "yname": yname, "x": x.copy(), "scale": scale.copy(), "bias": bias.copy(), "mean": mean.copy(), "var": var.copy(), "eps": eps}
            trainable_order.append((op.name, [scale.copy(), bias.copy()], []))

        elif op.op_type == "MaxPool":
            kernel, strides, pads = _read_pool_attrs(op)
            x4 = x.astype(np.float32)
            if x4.ndim == 3:
                x4 = x4[None, ...]
            y, xpad, mask = _pool2d_forward_nchw(x4, kernel, strides, pads, mode="max")
            vals[yname] = y.astype(np.float32)
            cache[op.name] = {"op_type": "MaxPool", "xname": xname, "yname": yname, "x": x4.copy(), "xpad": xpad.copy(), "mask": mask.copy(), "kernel": kernel, "strides": strides, "pads": pads}

        elif op.op_type == "AvgPool":
            kernel, strides, pads = _read_pool_attrs(op)
            x4 = x.astype(np.float32)
            if x4.ndim == 3:
                x4 = x4[None, ...]
            y, xpad, _mask = _pool2d_forward_nchw(x4, kernel, strides, pads, mode="avg")
            vals[yname] = y.astype(np.float32)
            cache[op.name] = {"op_type": "AvgPool", "xname": xname, "yname": yname, "x": x4.copy(), "xpad": xpad.copy(), "kernel": kernel, "strides": strides, "pads": pads}

        else:
            raise RuntimeError(f"Unsupported training-reference op_type={op.op_type} in extended reference")

    pred = vals[output_name].reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)

    if loss_type == "mse":
        loss_before, dy = _mse_loss_and_grad(pred, target)
    elif loss_type == "cross_entropy":
        loss_before, dy = _cross_entropy_loss_and_grad(pred, target)
    else:
        raise RuntimeError(f"Unsupported loss type: {loss_type}")

    grads[output_name] = dy
    trainable_grads_rev: List[List[np.ndarray]] = []

    for op in reversed(graph.ops):
        rec = cache[op.name]
        xname = rec["xname"]
        yname = rec["yname"]
        gy = grads[yname]

        if rec["op_type"] in ("Flatten", "Reshape"):
            grads[xname] = gy.reshape(rec["in_shape"]).astype(np.float32)

        elif rec["op_type"] == "Relu":
            grads[xname] = (gy * (rec["y"] > 0).astype(np.float32)).astype(np.float32)

        elif rec["op_type"] == "LeakyRelu":
            x = rec["x"]
            alpha = rec["alpha"]
            grads[xname] = (gy * np.where(x > 0, 1.0, alpha)).astype(np.float32)

        elif rec["op_type"] == "Sigmoid":
            y = rec["y"]
            grads[xname] = (gy * y * (1.0 - y)).astype(np.float32)

        elif rec["op_type"] == "Softmax":
            grads[xname] = _softmax_backward(rec["y"], gy.reshape(-1))

        elif rec["op_type"] == "Add":
            grads[xname] = gy.astype(np.float32)
            rhs_name = rec["rhs_name"]
            if rhs_name in vals:
                grads[rhs_name] = gy.astype(np.float32)

        elif rec["op_type"] == "Dense":
            x = rec["x"].reshape(-1)
            W = rec["W"]
            gyf = gy.reshape(-1)
            dW = np.outer(gyf, x).astype(np.float32)
            dB = gyf.astype(np.float32)
            dx = (W.T @ gyf).astype(np.float32)
            trainable_grads_rev.append([dW, dB])
            grads[xname] = dx

        elif rec["op_type"] == "Conv":
            gy4 = gy.astype(np.float32)
            if gy4.ndim == 3:
                gy4 = gy4[None, ...]
            dx, dW, dB = _conv2d_backward_nchw(
                rec["x"], rec["xpad"], rec["W"], gy4,
                strides=rec["strides"], pads=rec["pads"]
            )
            trainable_grads_rev.append([dW, dB])
            grads[xname] = dx.astype(np.float32)

        elif rec["op_type"] == "BatchNormalization":
            dx, dscale, dbias = _bn_backward(rec["x"], gy, rec["scale"], rec["mean"], rec["var"], rec["eps"])
            trainable_grads_rev.append([dscale, dbias])
            grads[xname] = dx.astype(np.float32)

        elif rec["op_type"] == "MaxPool":
            gy4 = gy.astype(np.float32)
            if gy4.ndim == 3:
                gy4 = gy4[None, ...]
            dx = _pool2d_backward_nchw(
                rec["x"], rec["xpad"], gy4,
                rec["kernel"], rec["strides"], rec["pads"],
                mode="max", mask=rec["mask"]
            )
            grads[xname] = dx.astype(np.float32)

        elif rec["op_type"] == "AvgPool":
            gy4 = gy.astype(np.float32)
            if gy4.ndim == 3:
                gy4 = gy4[None, ...]
            dx = _pool2d_backward_nchw(
                rec["x"], rec["xpad"], gy4,
                rec["kernel"], rec["strides"], rec["pads"],
                mode="avg", mask=None
            )
            grads[xname] = dx.astype(np.float32)

    trainable_grads_rev.reverse()

    weights_before: List[np.ndarray] = []
    grads_flattened: List[np.ndarray] = []
    weights_after: List[np.ndarray] = []

    for (_op_name, params_before, _), param_grads in zip(trainable_order, trainable_grads_rev):
        for p_before, dp in zip(params_before, param_grads):
            p_before_f = np.asarray(p_before, dtype=np.float32).reshape(-1)
            dp_f = np.asarray(dp, dtype=np.float32).reshape(-1)
            weights_before.append(p_before_f)
            grads_flattened.append(dp_f)
            weights_after.append((p_before_f - lr * dp_f).astype(np.float32))

    if not weights_before:
        raise RuntimeError("No trainable parameters found in training reference")

    weights_before_flat = np.concatenate(weights_before, axis=0).astype(np.float32)
    grads_flat = np.concatenate(grads_flattened, axis=0).astype(np.float32)
    weights_after_flat = np.concatenate(weights_after, axis=0).astype(np.float32)

    grads_path = ref_dir / "grads_ref.bin"
    w_before_path = ref_dir / "weights_before_ref.bin"
    w_after_path = ref_dir / "weights_after_ref.bin"

    grads_flat.tofile(grads_path)
    weights_before_flat.tofile(w_before_path)
    weights_after_flat.tofile(w_after_path)

    payload = {
        "loss_before": float(loss_before),
        "loss_after": None,
        "grads_ref_bin": str(grads_path),
        "weights_before_ref_bin": str(w_before_path),
        "weights_after_ref_bin": str(w_after_path),
        "num_grad_words": int(grads_flat.size),
        "num_weight_words": int(weights_before_flat.size),
    }

    summary_json = ref_dir / "summary.json"
    summary_txt = ref_dir / "summary.txt"
    write_text(summary_json, json.dumps(payload, indent=2))
    write_text(
        summary_txt,
        "\n".join(
            [
                "=========== FPGAI Training Reference ===========",
                f"loss_before      : {loss_before}",
                f"num_grad_words   : {grads_flat.size}",
                f"num_weight_words : {weights_before_flat.size}",
                "================================================",
            ]
        ),
    )

    return TrainingReferenceResult(
        loss_before=float(loss_before),
        loss_after=float("nan"),
        grads_flat_path=grads_path,
        weights_before_flat_path=w_before_path,
        weights_after_flat_path=w_after_path,
        summary_json=summary_json,
        summary_txt=summary_txt,
    )