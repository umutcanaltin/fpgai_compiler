from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


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
    return arr


def _flatten_from_graph_named(graph, tensor_name: str) -> Optional[np.ndarray]:
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


def _numel_from_graph_named(graph, tensor_name: str) -> int:
    if tensor_name is None:
        return 0

    if hasattr(graph, "constants") and tensor_name in graph.constants:
        return int(np.asarray(graph.constants[tensor_name]).size)

    if hasattr(graph, "params") and tensor_name in graph.params:
        return int(np.asarray(graph.params[tensor_name]).size)

    try:
        t = graph.get_tensor(tensor_name)
    except Exception:
        t = None

    if t is not None:
        shape = getattr(t, "shape", None)
        if shape is not None:
            n = 1
            for d in shape:
                n *= int(d)
            return int(n)

        data = getattr(t, "data", None)
        if data is not None:
            return int(np.asarray(data).size)

    return 0


def _fmt_array(name: str, ctype: str, arr: np.ndarray) -> str:
    flat = np.asarray(arr).reshape(-1)
    vals = []
    for v in flat:
        if np.issubdtype(flat.dtype, np.integer):
            vals.append(str(int(v)))
        else:
            vals.append(repr(float(v)))
    return f"const {ctype} {name}[{flat.size}] = {{ {', '.join(vals)} }};"


def _normalize_len(arr: np.ndarray, expected_n: int) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    if expected_n <= 0:
        return flat
    if flat.size == expected_n:
        return flat
    if flat.size > expected_n:
        return flat[:expected_n]
    padded = np.zeros((expected_n,), dtype=flat.dtype)
    padded[: flat.size] = flat
    return padded


def _dense_expected_shapes(op) -> Tuple[int, int, int, int]:
    in_f = int(op.attrs.get("in_features") or 0)
    out_f = int(op.attrs.get("out_features") or 0)
    expected_w = out_f * in_f if in_f > 0 and out_f > 0 else 0
    expected_b = out_f if out_f > 0 else 0
    return in_f, out_f, expected_w, expected_b


def _resolve_attr_candidate(graph, v):
    # Case 1: attr directly contains numeric array
    arr = _as_numpy_numeric(v)
    if arr is not None and arr.size > 0:
        return arr.reshape(-1), "direct_numeric_attr"

    # Case 2: attr is a tensor-name string
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
        "bias", "biases", "B", "b",
        "bias_data", "bias_values",
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
    weight_src = None
    bias_src = None

    for kind, key, arr, src in _candidate_attr_arrays(graph, op, expected_w, expected_b):
        if kind == "weight" and weight_arr is None:
            weight_arr = arr
            weight_src = f"op.attrs['{key}'] -> {src}"
        elif kind == "bias" and bias_arr is None:
            bias_arr = arr
            bias_src = f"op.attrs['{key}'] -> {src}"

    return weight_arr, bias_arr, weight_src, bias_src


def emit_params_cpp(graph) -> str:
    lines: List[str] = []
    lines.append('#include "fpgai_params.h"')
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")

    param_op_idx = 0

    for op in graph.ops:
        if op.op_type == "Conv":
            w_arr = None
            b_arr = None
            w_n = 0
            b_n = 0

            if len(op.inputs) > 1:
                w_name = op.inputs[1]
                w_arr = _flatten_from_graph_named(graph, w_name)
                w_n = _numel_from_graph_named(graph, w_name)

            if len(op.inputs) > 2:
                b_name = op.inputs[2]
                b_arr = _flatten_from_graph_named(graph, b_name)
                b_n = _numel_from_graph_named(graph, b_name)

            if w_n > 0:
                if w_arr is None:
                    raise ValueError(f"Conv weights not found for op '{op.name}' tensor '{op.inputs[1]}'")
                lines.append(_fmt_array(f"W{param_op_idx}", "wgt_t", _normalize_len(w_arr, w_n)))

            if b_n > 0:
                if b_arr is None:
                    raise ValueError(f"Conv bias not found for op '{op.name}' tensor '{op.inputs[2]}'")
                lines.append(_fmt_array(f"B{param_op_idx}", "bias_t", _normalize_len(b_arr, b_n)))

            lines.append("")
            param_op_idx += 1

        elif op.op_type == "Dense":
            in_f, out_f, expected_w, expected_b = _dense_expected_shapes(op)

            w_arr = None
            b_arr = None
            w_src = None
            b_src = None

            if len(op.inputs) > 1:
                w_name = op.inputs[1]
                w_arr = _flatten_from_graph_named(graph, w_name)
                if w_arr is not None:
                    w_src = f"graph tensor '{w_name}'"

            if len(op.inputs) > 2:
                b_name = op.inputs[2]
                b_arr = _flatten_from_graph_named(graph, b_name)
                if b_arr is not None:
                    b_src = f"graph tensor '{b_name}'"

            if w_arr is None or b_arr is None:
                wa, ba, wa_src, ba_src = _pick_dense_from_attrs(graph, op, expected_w, expected_b)
                if w_arr is None and wa is not None:
                    w_arr = wa
                    w_src = wa_src
                if b_arr is None and ba is not None:
                    b_arr = ba
                    b_src = ba_src

            if expected_w > 0:
                if w_arr is None:
                    attr_keys = sorted(list((getattr(op, "attrs", {}) or {}).keys()))
                    raise ValueError(
                        f"Dense weights not found for op '{op.name}'. "
                        f"Expected {expected_w} values. "
                        f"op.inputs={list(getattr(op, 'inputs', []))}, "
                        f"op.attrs keys={attr_keys}"
                    )
                w_arr = _normalize_len(w_arr, expected_w)
                lines.append(_fmt_array(f"W{param_op_idx}", "wgt_t", w_arr))

            if expected_b > 0:
                if b_arr is None:
                    attr_keys = sorted(list((getattr(op, "attrs", {}) or {}).keys()))
                    raise ValueError(
                        f"Dense bias not found for op '{op.name}'. "
                        f"Expected {expected_b} values. "
                        f"op.inputs={list(getattr(op, 'inputs', []))}, "
                        f"op.attrs keys={attr_keys}"
                    )
                b_arr = _normalize_len(b_arr, expected_b)
                lines.append(_fmt_array(f"B{param_op_idx}", "bias_t", b_arr))

            lines.append(f"// Dense source W{param_op_idx}: {w_src}")
            lines.append(f"// Dense source B{param_op_idx}: {b_src}")
            lines.append("")
            param_op_idx += 1

    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)