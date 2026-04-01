from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from fpgai.ir.graph import Graph


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


def _resolve_dense_arrays(graph: Graph, op) -> Tuple[np.ndarray, np.ndarray, int, int]:
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
        raise RuntimeError(
            f"Dense weights not found for training top op '{op.name}'. "
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
            raise RuntimeError(f"Cannot infer Dense shape for '{op.name}'")

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

    return W, B, in_f, out_f


def _strip_batch(shape):
    if shape is None:
        return tuple()
    shape = tuple(int(x) for x in shape)
    if len(shape) > 1 and shape[0] == 1:
        return tuple(shape[1:])
    return shape


def _tensor_shape(graph: Graph, name: str) -> Tuple[int, ...]:
    t = graph.get_tensor(name)
    if t is not None and getattr(t, "shape", None):
        return _strip_batch(t.shape)
    if name in getattr(graph, "constants", {}):
        return tuple(int(x) for x in np.asarray(graph.constants[name]).shape)
    return tuple()


def _flat_size(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 1
    return int(np.prod(shape))


def _dense_layers(graph: Graph):
    out = []
    dense_idx = 0
    for op in graph.ops:
        if op.op_type != "Dense":
            continue
        W, B, in_f, out_f = _resolve_dense_arrays(graph, op)
        out.append((dense_idx, op, W, B, in_f, out_f))
        dense_idx += 1
    return out


def emit_top_train_cpp(
    *,
    graph: Graph,
    top_name: str,
    weights_mode: str,
    training_cfg: Dict[str, Any],
) -> str:
    supported = {"Dense", "Relu", "Sigmoid", "Softmax", "Flatten", "Reshape"}
    bad = [op.op_type for op in graph.ops if op.op_type not in supported]
    if bad:
        raise RuntimeError(
            f"First PL training pipeline only supports {sorted(supported)}; found unsupported ops: {sorted(set(bad))}"
        )

    loss_type = str(training_cfg.get("loss", {}).get("type", "mse")).lower()
    lr = float(training_cfg.get("optimizer", {}).get("learning_rate", 0.01))
    if loss_type not in {"mse", "cross_entropy"}:
        raise RuntimeError("top_train_cpp currently supports only mse or cross_entropy")

    input_name = graph.inputs[0]
    output_name = graph.outputs[0]
    input_words = _flat_size(_tensor_shape(graph, input_name))
    output_words = _flat_size(_tensor_shape(graph, output_name))
    dense_layers = _dense_layers(graph)

    lines: list[str] = []
    lines.append('#include "fpgai_types.h"')
    lines.append('#include "fpgai_params.h"')
    if weights_mode in ("stream", "ddr"):
        lines.append('#include "weights_runtime.h"')
    lines.append("#include <hls_stream.h>")
    lines.append("#include <ap_axi_sdata.h>")
    lines.append("#include <math.h>")
    lines.append("")
    lines.append("typedef ap_axis<32,0,0,0> axis_t;")
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("static inline float bits_to_float(ap_uint<32> bits) { union { unsigned int i; float f; } u; u.i = bits.to_uint(); return u.f; }")
    lines.append("static inline ap_uint<32> float_to_bits(float v) { union { unsigned int i; float f; } u; u.f = v; return (ap_uint<32>)u.i; }")
    lines.append("static inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }")
    lines.append("static inline void softmax_f(const act_t* x, act_t* y, int n) {")
    lines.append("  float maxv = (float)x[0];")
    lines.append("  for (int i = 1; i < n; ++i) if ((float)x[i] > maxv) maxv = (float)x[i];")
    lines.append("  float sumexp = 0.0f;")
    lines.append("  for (int i = 0; i < n; ++i) { y[i] = (act_t)expf((float)x[i] - maxv); sumexp += (float)y[i]; }")
    lines.append("  for (int i = 0; i < n; ++i) y[i] = (act_t)((float)y[i] / sumexp);")
    lines.append("}")
    lines.append("static inline void softmax_backward_f(const act_t* y, const grad_act_t* gy, grad_act_t* gx, int n) {")
    lines.append("  float dot = 0.0f;")
    lines.append("  for (int i = 0; i < n; ++i) dot += (float)y[i] * (float)gy[i];")
    lines.append("  for (int i = 0; i < n; ++i) gx[i] = (grad_act_t)((float)y[i] * ((float)gy[i] - dot));")
    lines.append("}")
    lines.append("static inline void push_f32(hls::stream<axis_t>& s, float v, bool last=false) { axis_t p; p.data = float_to_bits(v); p.keep = -1; p.strb = -1; p.last = last ? 1 : 0; s.write(p); }")
    lines.append("static inline float pop_f32(hls::stream<axis_t>& s) { axis_t p = s.read(); return bits_to_float(p.data); }")
    lines.append("}")

    for dense_idx, _op, W, B, in_f, out_f in dense_layers:
        lines.append(f"static wgt_t WM{dense_idx}[{out_f * in_f}];")
        lines.append(f"static bias_t BM{dense_idx}[{out_f}];")
        lines.append(f"static bool W_INIT_{dense_idx} = false;")

        if weights_mode == "embedded":
            flatW = ", ".join(repr(float(v)) for v in W.reshape(-1))
            flatB = ", ".join(repr(float(v)) for v in B.reshape(-1))
            lines.append(f"static const float WM_INIT_{dense_idx}[{out_f * in_f}] = {{ {flatW} }};")
            lines.append(f"static const float BM_INIT_{dense_idx}[{out_f}] = {{ {flatB} }};")
            lines.append(f"static inline void init_dense_{dense_idx}() {{")
            lines.append(f"  if (W_INIT_{dense_idx}) return;")
            lines.append(f"  for (int i = 0; i < {out_f * in_f}; ++i) WM{dense_idx}[i] = (wgt_t)WM_INIT_{dense_idx}[i];")
            lines.append(f"  for (int i = 0; i < {out_f}; ++i) BM{dense_idx}[i] = (bias_t)BM_INIT_{dense_idx}[i];")
            lines.append(f"  W_INIT_{dense_idx} = true;")
            lines.append("}")
        else:
            lines.append(f"static inline void init_dense_{dense_idx}() {{")
            lines.append(f"  if (W_INIT_{dense_idx}) return;")
            lines.append(f"  for (int i = 0; i < {out_f * in_f}; ++i) WM{dense_idx}[i] = 0;")
            lines.append(f"  for (int i = 0; i < {out_f}; ++i) BM{dense_idx}[i] = 0;")
            lines.append(f"  W_INIT_{dense_idx} = true;")
            lines.append("}")

    lines.append("")
    lines.append(f'extern "C" void {top_name}(hls::stream<axis_t>& in, hls::stream<axis_t>& out, hls::stream<axis_t>& aux, int mode) {{')
    lines.append("#pragma HLS INTERFACE axis port=in")
    lines.append("#pragma HLS INTERFACE axis port=out")
    lines.append("#pragma HLS INTERFACE axis port=aux")
    lines.append("#pragma HLS INTERFACE s_axilite port=mode bundle=control")
    lines.append("#pragma HLS INTERFACE s_axilite port=return bundle=control")

    for dense_idx, _op, _W, _B, _in_f, _out_f in dense_layers:
        lines.append(f"  init_dense_{dense_idx}();")

    if weights_mode in ("stream", "ddr"):
        lines.append("  if (mode == 0) {")
        for dense_idx, _op, _W, _B, in_f, out_f in dense_layers:
            lines.append(f"    for (int i = 0; i < {out_f * in_f}; ++i) WM{dense_idx}[i] = (wgt_t)pop_f32(aux);")
            lines.append(f"    for (int i = 0; i < {out_f}; ++i) BM{dense_idx}[i] = (bias_t)pop_f32(aux);")
            lines.append(f"    W_INIT_{dense_idx} = true;")
        lines.append("    return;")
        lines.append("  }")

    lines.append("  if (mode != 2) { return; }")
    lines.append(f"  act_t input_buf[{input_words}];")
    lines.append(f"  act_t target_buf[{output_words}];")
    lines.append(f"  for (int i = 0; i < {input_words}; ++i) input_buf[i] = (act_t)pop_f32(in);")
    lines.append(f"  for (int i = 0; i < {output_words}; ++i) target_buf[i] = (act_t)pop_f32(aux);")

    tensor_name_to_size: Dict[str, int] = {input_name: input_words}
    tensor_name_to_cxx: Dict[str, str] = {input_name: "input_buf"}

    for op in graph.ops:
        out_name = op.outputs[0]
        sz = _flat_size(_tensor_shape(graph, out_name))
        tensor_name_to_size[out_name] = sz
        cname = f"buf_{len(tensor_name_to_cxx)}"
        tensor_name_to_cxx[out_name] = cname
        lines.append(f"  act_t {cname}[{sz}];")

    grad_name_to_cxx: Dict[str, str] = {}
    for name, sz in tensor_name_to_size.items():
        gname = f"grad_{len(grad_name_to_cxx)}"
        grad_name_to_cxx[name] = gname
        lines.append(f"  grad_act_t {gname}[{sz}];")

    for dense_idx, _op, _W, _B, in_f, out_f in dense_layers:
        lines.append(f"  grad_wgt_t dW{dense_idx}[{out_f * in_f}];")
        lines.append(f"  grad_bias_t dB{dense_idx}[{out_f}];")
        lines.append(f"  wgt_t W_before_{dense_idx}[{out_f * in_f}];")
        lines.append(f"  bias_t B_before_{dense_idx}[{out_f}];")

    dense_counter = 0
    for op in graph.ops:
        x_name = op.inputs[0]
        y_name = op.outputs[0]
        xbuf = tensor_name_to_cxx[x_name]
        ybuf = tensor_name_to_cxx[y_name]

        if op.op_type in ("Flatten", "Reshape"):
            out_sz = tensor_name_to_size[y_name]
            lines.append(f"  for (int i = 0; i < {out_sz}; ++i) {ybuf}[i] = {xbuf}[i];")

        elif op.op_type == "Dense":
            _idx, _op, _W, _B, in_f, out_f = dense_layers[dense_counter]
            lines.append(f"  for (int o = 0; o < {out_f}; ++o) {{")
            lines.append(f"    acc_t acc = (acc_t)BM{dense_counter}[o];")
            lines.append(f"    for (int i = 0; i < {in_f}; ++i) acc += (acc_t){xbuf}[i] * (acc_t)WM{dense_counter}[o*{in_f}+i];")
            lines.append(f"    {ybuf}[o] = (act_t)acc;")
            lines.append("  }")
            dense_counter += 1

        elif op.op_type == "Relu":
            sz = tensor_name_to_size[y_name]
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {ybuf}[i] = ({xbuf}[i] > 0) ? {xbuf}[i] : (act_t)0;")

        elif op.op_type == "Sigmoid":
            sz = tensor_name_to_size[y_name]
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {ybuf}[i] = (act_t)sigmoid_f((float){xbuf}[i]);")

        elif op.op_type == "Softmax":
            sz = tensor_name_to_size[y_name]
            lines.append(f"  softmax_f({xbuf}, {ybuf}, {sz});")

    final_buf = tensor_name_to_cxx[output_name]
    final_grad = grad_name_to_cxx[output_name]

    if loss_type == "mse":
        lines.append(f"  for (int i = 0; i < {output_words}; ++i) {final_grad}[i] = (grad_act_t)((2.0f / {output_words}) * ((float){final_buf}[i] - (float)target_buf[i]));")
    else:
        lines.append(f"  for (int i = 0; i < {output_words}; ++i) {final_grad}[i] = (grad_act_t)((float){final_buf}[i] - (float)target_buf[i]);")

    dense_counter = len(dense_layers) - 1
    for op in reversed(graph.ops):
        x_name = op.inputs[0]
        y_name = op.outputs[0]
        xbuf = tensor_name_to_cxx[x_name]
        ybuf = tensor_name_to_cxx[y_name]
        gx = grad_name_to_cxx[x_name]
        gy = grad_name_to_cxx[y_name]

        if op.op_type in ("Flatten", "Reshape"):
            sz = tensor_name_to_size[x_name]
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {gx}[i] = {gy}[i];")

        elif op.op_type == "Relu":
            sz = tensor_name_to_size[y_name]
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {gx}[i] = ((float){ybuf}[i] > 0.0f) ? {gy}[i] : (grad_act_t)0;")

        elif op.op_type == "Sigmoid":
            sz = tensor_name_to_size[y_name]
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {gx}[i] = (grad_act_t)((float){gy}[i] * (float){ybuf}[i] * (1.0f - (float){ybuf}[i]));")

        elif op.op_type == "Softmax":
            sz = tensor_name_to_size[y_name]
            lines.append(f"  softmax_backward_f({ybuf}, {gy}, {gx}, {sz});")

        elif op.op_type == "Dense":
            _idx, _op, _W, _B, in_f, out_f = dense_layers[dense_counter]
            lines.append(f"  for (int o = 0; o < {out_f}; ++o) {{")
            lines.append(f"    dB{dense_counter}[o] = (grad_bias_t){gy}[o];")
            lines.append(f"    B_before_{dense_counter}[o] = BM{dense_counter}[o];")
            lines.append("  }")
            lines.append(f"  for (int k = 0; k < {out_f * in_f}; ++k) W_before_{dense_counter}[k] = WM{dense_counter}[k];")
            lines.append(f"  for (int o = 0; o < {out_f}; ++o) {{")
            lines.append(f"    for (int i = 0; i < {in_f}; ++i) dW{dense_counter}[o*{in_f}+i] = (grad_wgt_t)((float){gy}[o] * (float){xbuf}[i]);")
            lines.append("  }")
            lines.append(f"  for (int i = 0; i < {in_f}; ++i) {{")
            lines.append("    upd_t acc = 0;")
            lines.append(f"    for (int o = 0; o < {out_f}; ++o) acc += (upd_t)W_before_{dense_counter}[o*{in_f}+i] * (upd_t){gy}[o];")
            lines.append(f"    {gx}[i] = (grad_act_t)acc;")
            lines.append("  }")
            lines.append(f"  for (int o = 0; o < {out_f}; ++o) BM{dense_counter}[o] = (bias_t)((float)BM{dense_counter}[o] - {lr}f * (float)dB{dense_counter}[o]);")
            lines.append(f"  for (int k = 0; k < {out_f * in_f}; ++k) WM{dense_counter}[k] = (wgt_t)((float)WM{dense_counter}[k] - {lr}f * (float)dW{dense_counter}[k]);")
            dense_counter -= 1

    total_words = 0
    for _dense_idx, _op, _W, _B, in_f, out_f in dense_layers:
        total_words += out_f * in_f + out_f
    last_index = total_words * 3 - 1
    emit_index = 0

    for dense_idx, _op, _W, _B, in_f, out_f in dense_layers:
        for i in range(out_f * in_f):
            last = "true" if emit_index == last_index else "false"
            lines.append(f"  push_f32(out, (float)dW{dense_idx}[{i}], {last});")
            emit_index += 1
        for i in range(out_f):
            last = "true" if emit_index == last_index else "false"
            lines.append(f"  push_f32(out, (float)dB{dense_idx}[{i}], {last});")
            emit_index += 1

    for dense_idx, _op, _W, _B, in_f, out_f in dense_layers:
        for i in range(out_f * in_f):
            last = "true" if emit_index == last_index else "false"
            lines.append(f"  push_f32(out, (float)W_before_{dense_idx}[{i}], {last});")
            emit_index += 1
        for i in range(out_f):
            last = "true" if emit_index == last_index else "false"
            lines.append(f"  push_f32(out, (float)B_before_{dense_idx}[{i}], {last});")
            emit_index += 1

    for dense_idx, _op, _W, _B, in_f, out_f in dense_layers:
        for i in range(out_f * in_f):
            last = "true" if emit_index == last_index else "false"
            lines.append(f"  push_f32(out, (float)WM{dense_idx}[{i}], {last});")
            emit_index += 1
        for i in range(out_f):
            last = "true" if emit_index == last_index else "false"
            lines.append(f"  push_f32(out, (float)BM{dense_idx}[{i}], {last});")
            emit_index += 1

    lines.append("}")
    return "\n".join(lines)