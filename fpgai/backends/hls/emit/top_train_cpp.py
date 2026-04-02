from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import re

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


def _sanitize(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
    if not s:
        s = "t"
    if s[0].isdigit():
        s = "_" + s
    return s


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
            f"inputs={list(getattr(op, 'inputs', []))} attrs_keys={list((getattr(op, 'attrs', {}) or {}).keys())}"
        )

    if in_f <= 0 or out_f <= 0:
        if len(op.inputs) > 0 and len(op.outputs) > 0:
            xshape = _shape_wo_batch(graph.get_tensor(op.inputs[0]).shape)
            yshape = _shape_wo_batch(graph.get_tensor(op.outputs[0]).shape)
            in_f = _flat_size(xshape)
            out_f = _flat_size(yshape)

    W = np.asarray(w_arr, dtype=np.float32).reshape(out_f, in_f)
    if b_arr is None:
        B = np.zeros((out_f,), dtype=np.float32)
    else:
        B = np.asarray(b_arr, dtype=np.float32).reshape(out_f)

    return W, B, in_f, out_f


def emit_top_train_cpp(
    *,
    graph: Graph,
    top_name: str,
    weights_mode: str,
    training_cfg: Dict[str, Any],
) -> str:
    supported = {
        "Dense", "Relu", "LeakyRelu", "Sigmoid", "Softmax",
        "Flatten", "Reshape", "Add"
    }
    bad = [op.op_type for op in graph.ops if op.op_type not in supported]
    if bad:
        raise RuntimeError(
            f"Current template refactor step supports {sorted(supported)}; found unsupported ops: {sorted(set(bad))}"
        )

    loss_type = str(training_cfg.get("loss", {}).get("type", "mse")).lower()
    lr = float(training_cfg.get("optimizer", {}).get("learning_rate", 0.01))

    phase_overrides = training_cfg.get("phase_overrides", {}) or {}
    forward_parallel = (phase_overrides.get("forward", {}) or {}).get("parallel", {}) or {}
    update_parallel = (phase_overrides.get("update", {}) or {}).get("parallel", {}) or {}

    dense_pe = int(forward_parallel.get("dense_pe", 1))
    dense_simd = int(forward_parallel.get("dense_simd", 1))
    dense_update_unroll = int(update_parallel.get("weight_update_unroll", 1))

    input_name = graph.inputs[0]
    output_name = graph.outputs[0]

    tensor_name_to_shape: Dict[str, Tuple[int, ...]] = {}
    tensor_name_to_size: Dict[str, int] = {}
    tensor_name_to_cxx: Dict[str, str] = {}
    grad_name_to_cxx: Dict[str, str] = {}

    for tname, tspec in graph.tensors.items():
        shp = _shape_wo_batch(tspec.shape)
        tensor_name_to_shape[tname] = shp
        tensor_name_to_size[tname] = _flat_size(shp)
        sname = _sanitize(tname)
        tensor_name_to_cxx[tname] = f"buf_{sname}"
        grad_name_to_cxx[tname] = f"grad_{sname}"

    input_size = tensor_name_to_size[input_name]
    output_size = tensor_name_to_size[output_name]

    lines: List[str] = []
    lines.append('#include <hls_stream.h>')
    lines.append('#include <ap_axi_sdata.h>')
    lines.append('#include <math.h>')
    lines.append('#include "layers/dense.h"')
    lines.append("")
    lines.append("typedef ap_axis<32,0,0,0> axis_t;")
    lines.append("typedef float act_t;")
    lines.append("typedef float grad_t;")
    lines.append("typedef float weight_t;")
    lines.append("typedef float grad_weight_t;")
    lines.append("typedef float bias_t;")
    lines.append("typedef float grad_bias_t;")
    lines.append("typedef float acc_t;")
    lines.append("")
    lines.append("using namespace fpgai;")
    lines.append("")
    lines.append("static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }")
    lines.append("static void softmax_f(const act_t* x, act_t* y, int n) {")
    lines.append("  float mx = x[0];")
    lines.append("  for (int i = 1; i < n; ++i) if ((float)x[i] > mx) mx = (float)x[i];")
    lines.append("  float s = 0.0f;")
    lines.append("  for (int i = 0; i < n; ++i) { y[i] = expf((float)x[i] - mx); s += (float)y[i]; }")
    lines.append("  float inv = (s != 0.0f) ? (1.0f / s) : 0.0f;")
    lines.append("  for (int i = 0; i < n; ++i) y[i] = (act_t)((float)y[i] * inv);")
    lines.append("}")
    lines.append("static void softmax_backward_f(const act_t* y, const grad_t* gy, grad_t* gx, int n) {")
    lines.append("  for (int i = 0; i < n; ++i) {")
    lines.append("    float acc = 0.0f;")
    lines.append("    for (int j = 0; j < n; ++j) {")
    lines.append("      float jac = (i == j) ? ((float)y[i] * (1.0f - (float)y[i])) : (-(float)y[i] * (float)y[j]);")
    lines.append("      acc += jac * (float)gy[j];")
    lines.append("    }")
    lines.append("    gx[i] = (grad_t)acc;")
    lines.append("  }")
    lines.append("}")
    lines.append("static inline unsigned int f32_to_u32(float v) { union { float f; unsigned int u; } x; x.f = v; return x.u; }")
    lines.append("static inline float u32_to_f32(unsigned int v) { union { float f; unsigned int u; } x; x.u = v; return x.f; }")
    lines.append("static inline void write_f32(hls::stream<axis_t>& s, float v, bool last=false) { axis_t p; p.data=f32_to_u32(v); p.keep=-1; p.strb=-1; p.last=last?1:0; s.write(p); }")
    lines.append("static inline float read_f32(hls::stream<axis_t>& s) { axis_t p = s.read(); return u32_to_f32((unsigned int)p.data); }")
    lines.append("")

    for tname, size in tensor_name_to_size.items():
        lines.append(f"static act_t {tensor_name_to_cxx[tname]}[{size}];")
        lines.append(f"static grad_t {grad_name_to_cxx[tname]}[{size}];")

    param_specs = []

    for op in graph.ops:
        if op.op_type == "Dense":
            W, B, in_f, out_f = _resolve_dense_arrays(graph, op)
            ws = _sanitize(op.name)
            lines.append(f"static weight_t W_{ws}[{W.size}] = {{ {', '.join(f'{float(x):.8f}f' for x in W.reshape(-1))} }};")
            lines.append(f"static bias_t B_{ws}[{B.size}] = {{ {', '.join(f'{float(x):.8f}f' for x in B.reshape(-1))} }};")
            lines.append(f"static grad_weight_t dW_{ws}[{W.size}];")
            lines.append(f"static grad_bias_t dB_{ws}[{B.size}];")
            param_specs.append((op, ws, in_f, out_f, W.size, B.size))

    lines.append("")
    lines.append(f'extern "C" void {top_name}(')
    lines.append("  hls::stream<axis_t>& in,")
    lines.append("  hls::stream<axis_t>& out,")
    lines.append("  hls::stream<axis_t>& aux,")
    lines.append("  int mode")
    lines.append(") {")
    lines.append("#pragma HLS INTERFACE axis port=in")
    lines.append("#pragma HLS INTERFACE axis port=out")
    lines.append("#pragma HLS INTERFACE axis port=aux")
    lines.append("#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL")
    lines.append("#pragma HLS INTERFACE s_axilite port=return bundle=CTRL")
    lines.append("")

    if weights_mode in ("stream", "ddr"):
        lines.append("  if (mode == 0) {")
        for _op, ws, _in_f, _out_f, w_size, b_size in param_specs:
            lines.append(f"    for (int i = 0; i < {w_size}; ++i) W_{ws}[i] = read_f32(aux);")
            lines.append(f"    for (int i = 0; i < {b_size}; ++i) B_{ws}[i] = read_f32(aux);")
        lines.append("    return;")
        lines.append("  }")
        lines.append("")

    lines.append(f"  for (int i = 0; i < {input_size}; ++i) {tensor_name_to_cxx[input_name]}[i] = read_f32(in);")
    lines.append(f"  static act_t target_buf[{output_size}];")
    lines.append(f"  for (int i = 0; i < {output_size}; ++i) target_buf[i] = read_f32(in);")
    lines.append("")

    for tname, size in tensor_name_to_size.items():
        lines.append(f"  for (int i = 0; i < {size}; ++i) {grad_name_to_cxx[tname]}[i] = 0;")
    for _op, ws, _in_f, _out_f, w_size, b_size in param_specs:
        lines.append(f"  for (int i = 0; i < {w_size}; ++i) dW_{ws}[i] = 0;")
        lines.append(f"  for (int i = 0; i < {b_size}; ++i) dB_{ws}[i] = 0;")
    lines.append("")

    for op in graph.ops:
        xname = op.inputs[0]
        yname = op.outputs[0]
        xbuf = tensor_name_to_cxx[xname]
        ybuf = tensor_name_to_cxx[yname]
        sz = tensor_name_to_size[yname]

        if op.op_type == "Dense":
            ws = _sanitize(op.name)
            _W, _B, in_f, out_f = _resolve_dense_arrays(graph, op)
            lines.append(
                f"  dense_forward<act_t, weight_t, bias_t, acc_t, {in_f}, {out_f}, {dense_pe}, {dense_simd}>("
                f"{xbuf}, {ybuf}, W_{ws}, B_{ws});"
            )
        elif op.op_type == "Relu":
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {ybuf}[i] = ({xbuf}[i] > 0) ? {xbuf}[i] : (act_t)0;")
        elif op.op_type == "LeakyRelu":
            alpha = float((getattr(op, 'attrs', {}) or {}).get("alpha", 0.01))
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {ybuf}[i] = ({xbuf}[i] > 0) ? {xbuf}[i] : (act_t)({alpha:.8f}f * (float){xbuf}[i]);")
        elif op.op_type == "Sigmoid":
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {ybuf}[i] = (act_t)sigmoid_f((float){xbuf}[i]);")
        elif op.op_type == "Softmax":
            lines.append(f"  softmax_f({xbuf}, {ybuf}, {sz});")
        elif op.op_type in ("Flatten", "Reshape"):
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {ybuf}[i] = {xbuf}[i];")
        elif op.op_type == "Add":
            rhs = tensor_name_to_cxx[op.inputs[1]]
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {ybuf}[i] = (act_t)((float){xbuf}[i] + (float){rhs}[i]);")

    final_buf = tensor_name_to_cxx[output_name]
    final_grad = grad_name_to_cxx[output_name]

    if loss_type == "mse":
        lines.append(f"  for (int i = 0; i < {output_size}; ++i) {final_grad}[i] = (grad_t)(2.0f * ((float){final_buf}[i] - (float)target_buf[i]) / {max(output_size,1)}.0f);")
    else:
        lines.append(f"  for (int i = 0; i < {output_size}; ++i) {final_grad}[i] = (grad_t)(((float){final_buf}[i]) - ((float)target_buf[i]));")
    lines.append("")

    for op in reversed(graph.ops):
        xname = op.inputs[0]
        yname = op.outputs[0]
        xbuf = tensor_name_to_cxx[xname]
        ybuf = tensor_name_to_cxx[yname]
        gx = grad_name_to_cxx[xname]
        gy = grad_name_to_cxx[yname]
        sz = tensor_name_to_size[yname]

        if op.op_type == "Dense":
            ws = _sanitize(op.name)
            _W, _B, in_f, out_f = _resolve_dense_arrays(graph, op)
            lines.append(
                f"  dense_backward_params<act_t, grad_t, grad_weight_t, grad_bias_t, {in_f}, {out_f}, {dense_pe}, {dense_simd}>("
                f"{xbuf}, {gy}, dW_{ws}, dB_{ws});"
            )
            lines.append(
                f"  dense_backward_input<act_t, weight_t, grad_t, acc_t, {in_f}, {out_f}, {dense_pe}, {dense_simd}>("
                f"{gy}, {gx}, W_{ws});"
            )
        elif op.op_type == "Relu":
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {gx}[i] += ((float){ybuf}[i] > 0.0f) ? {gy}[i] : (grad_t)0;")
        elif op.op_type == "LeakyRelu":
            alpha = float((getattr(op, 'attrs', {}) or {}).get("alpha", 0.01))
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {gx}[i] += ((float){xbuf}[i] > 0.0f) ? {gy}[i] : (grad_t)({alpha:.8f}f * (float){gy}[i]);")
        elif op.op_type == "Sigmoid":
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {gx}[i] += (grad_t)((float){gy}[i] * (float){ybuf}[i] * (1.0f - (float){ybuf}[i]));")
        elif op.op_type == "Softmax":
            lines.append(f"  softmax_backward_f({ybuf}, {gy}, {gx}, {sz});")
        elif op.op_type in ("Flatten", "Reshape"):
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {gx}[i] += {gy}[i];")
        elif op.op_type == "Add":
            rhsg = grad_name_to_cxx[op.inputs[1]]
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {gx}[i] += {gy}[i];")
            lines.append(f"  for (int i = 0; i < {sz}; ++i) {rhsg}[i] += {gy}[i];")

    for _op, ws, in_f, out_f, _w_size, _b_size in param_specs:
        lines.append(
            f"  dense_update_sgd<weight_t, bias_t, grad_weight_t, grad_bias_t, {in_f}, {out_f}, {dense_update_unroll}>("
            f"W_{ws}, B_{ws}, dW_{ws}, dB_{ws}, {lr:.8f}f);"
        )

    total_param_words = sum(w_size + b_size for _op, _ws, _in_f, _out_f, w_size, b_size in param_specs)

    for _op, ws, _in_f, _out_f, w_size, b_size in param_specs:
        lines.append(f"  for (int i = 0; i < {w_size}; ++i) write_f32(out, dW_{ws}[i], false);")
        lines.append(f"  for (int i = 0; i < {b_size}; ++i) write_f32(out, dB_{ws}[i], false);")

    for _op, ws, _in_f, _out_f, w_size, b_size in param_specs:
        lines.append(f"  for (int i = 0; i < {w_size}; ++i) write_f32(out, W_{ws}[i] + {lr:.8f}f * dW_{ws}[i], false);")
        lines.append(f"  for (int i = 0; i < {b_size}; ++i) write_f32(out, B_{ws}[i] + {lr:.8f}f * dB_{ws}[i], false);")

    emitted = 0
    for _op, ws, _in_f, _out_f, w_size, b_size in param_specs:
        for i in range(w_size):
            emitted += 1
            last = emitted == total_param_words
            lines.append(f"  write_f32(out, W_{ws}[{i}], {'true' if last else 'false'});")
        for i in range(b_size):
            emitted += 1
            last = emitted == total_param_words
            lines.append(f"  write_f32(out, B_{ws}[{i}], {'true' if last else 'false'});")

    if total_param_words == 0:
        lines.append("  write_f32(out, 0.0f, true);")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)