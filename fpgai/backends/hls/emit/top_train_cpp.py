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


def _as_chw(shape3: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shape3) != 3:
        raise ValueError(f"Expected 3D shape, got {shape3}")
    c, h, w = shape3
    return int(c), int(h), int(w)


def _get_tensor_shape(graph: Graph, name: str) -> Tuple[int, ...]:
    try:
        t = graph.get_tensor(name)
    except Exception:
        t = None
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
        raise RuntimeError(f"Only square conv kernels are supported in training top, got {wshape}")
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


def _dense_input_preflatten_shape(graph: Graph, op) -> Tuple[int, ...]:
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
        raise RuntimeError(f"Dense weights not found for training top op '{op.name}'")

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


def _resolve_conv_arrays(graph: Graph, op):
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
        try:
            w_t = graph.get_tensor(op.inputs[1])
        except Exception:
            w_t = None
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

    return np.asarray(w_arr, dtype=np.float32).reshape(-1), np.asarray(b_arr, dtype=np.float32).reshape(-1), ws


def _resolve_bn_arrays(graph: Graph, op, c: int):
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


def _build_tensor_shapes(graph: Graph) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, int]]:
    inferred_shapes: Dict[str, Tuple[int, ...]] = {}

    if hasattr(graph, "tensors"):
        for tname, tspec in graph.tensors.items():
            shp = _shape_wo_batch(getattr(tspec, "shape", None))
            if shp:
                inferred_shapes[tname] = shp

    for name in list(getattr(graph, "inputs", [])) + list(getattr(graph, "outputs", [])):
        if name not in inferred_shapes:
            shp = _get_tensor_shape(graph, name)
            if shp:
                inferred_shapes[name] = shp

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
        xshape = get_shape(xname)

        yshape = get_shape(yname)
        if yshape:
            inferred_shapes[yname] = yshape
            continue

        if op.op_type in ("Relu", "LeakyRelu", "Sigmoid", "Softmax", "Add", "BatchNormalization"):
            if xshape:
                inferred_shapes[yname] = xshape

        elif op.op_type in ("Flatten", "Reshape"):
            if xshape:
                inferred_shapes[yname] = (_flat_size(xshape),)

        elif op.op_type == "Dense":
            try:
                _W, B, _in_f, _out_f = _resolve_dense_arrays(graph, op)
                inferred_shapes[yname] = (int(B.size),)
            except Exception:
                pass

        elif op.op_type == "Conv":
            try:
                _Wflat, _Bflat, ws = _resolve_conv_arrays(graph, op)
                stride = int(op.attrs.get("strides", [1, 1])[0])
                pad = int(op.attrs.get("pads", [0, 0, 0, 0])[0])
                if xshape:
                    inferred_shapes[yname] = _infer_conv_output_shape(_as_chw(xshape), tuple(int(v) for v in ws), stride, pad)
            except Exception:
                pass

        elif op.op_type in ("MaxPool", "AvgPool"):
            try:
                k = int(op.attrs.get("kernel_shape", [2, 2])[0])
                stride = int(op.attrs.get("strides", [2, 2])[0])
                if xshape:
                    inferred_shapes[yname] = _infer_pool_output_shape(_as_chw(xshape), k, stride)
            except Exception:
                pass

    all_tensor_names = set()
    all_tensor_names.update(getattr(graph, "inputs", []))
    all_tensor_names.update(getattr(graph, "outputs", []))
    if hasattr(graph, "tensors"):
        all_tensor_names.update(graph.tensors.keys())
    for op in graph.ops:
        all_tensor_names.update(op.inputs)
        all_tensor_names.update(op.outputs)

    tensor_name_to_shape: Dict[str, Tuple[int, ...]] = {}
    tensor_name_to_size: Dict[str, int] = {}

    for tname in sorted(all_tensor_names):
        shp = inferred_shapes.get(tname, tuple())
        if not shp:
            shp = _get_tensor_shape(graph, tname)
        if not shp:
            shp = (1,)
        tensor_name_to_shape[tname] = shp
        tensor_name_to_size[tname] = _flat_size(shp)

    return tensor_name_to_shape, tensor_name_to_size


def _build_tensor_aliases(
    graph: Graph,
    tensor_name_to_size: Dict[str, int],
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    all_names = set(tensor_name_to_size.keys())
    produced = {op.outputs[0] for op in graph.ops if op.outputs}
    inputs = set(getattr(graph, "inputs", []))
    outputs = set(getattr(graph, "outputs", []))

    canonical: Dict[str, str] = {name: name for name in all_names}

    prev_output: Optional[str] = None
    for op in graph.ops:
        if op.inputs:
            xname = op.inputs[0]
            if (
                prev_output is not None
                and xname not in produced
                and xname not in inputs
                and xname not in outputs
                and tensor_name_to_size.get(xname, -1) == tensor_name_to_size.get(prev_output, -2)
            ):
                canonical[xname] = canonical.get(prev_output, prev_output)
        if op.outputs:
            yname = op.outputs[0]
            if yname not in canonical:
                canonical[yname] = yname
            prev_output = yname

    root_to_cxx: Dict[str, str] = {}
    root_to_grad: Dict[str, str] = {}
    roots_in_order: List[str] = []

    def _root(name: str) -> str:
        seen = set()
        cur = name
        while canonical.get(cur, cur) != cur and cur not in seen:
            seen.add(cur)
            cur = canonical[cur]
        return cur

    tensor_name_to_cxx: Dict[str, str] = {}
    grad_name_to_cxx: Dict[str, str] = {}

    for name in sorted(all_names):
        root = _root(name)
        if root not in root_to_cxx:
            root_to_cxx[root] = f"buf_{_sanitize(root)}"
            root_to_grad[root] = f"grad_{_sanitize(root)}"
            roots_in_order.append(root)
        tensor_name_to_cxx[name] = root_to_cxx[root]
        grad_name_to_cxx[name] = root_to_grad[root]

    return tensor_name_to_cxx, grad_name_to_cxx, roots_in_order


def _is_final_softmax_mse_case(graph: Graph, training_cfg: Dict[str, Any]) -> bool:
    loss_type = str(training_cfg.get("loss", {}).get("type", "mse")).lower()
    if loss_type != "mse":
        return False
    if not getattr(graph, "ops", None):
        return False
    last_op = graph.ops[-1]
    return bool(last_op.op_type == "Softmax" and last_op.outputs and graph.outputs and last_op.outputs[0] == graph.outputs[0])


def _needs_input_gradient(graph: Graph, op_idx: int) -> bool:
    op = graph.ops[op_idx]
    if not op.inputs:
        return False

    xname = op.inputs[0]
    if xname in getattr(graph, "inputs", []):
        return False

    producer_idx = None
    for i, prev in enumerate(graph.ops):
        if prev.outputs and prev.outputs[0] == xname:
            producer_idx = i
            break

    if producer_idx is None:
        return False

    producer = graph.ops[producer_idx]
    return producer.op_type in {
        "Dense", "Conv", "BatchNormalization",
        "Relu", "LeakyRelu", "Sigmoid", "Softmax",
        "Add", "MaxPool", "AvgPool", "Flatten", "Reshape"
    }


def emit_top_train_cpp(
    *,
    graph: Graph,
    top_name: str,
    weights_mode: str,
    training_cfg: Dict[str, Any],
    compile_plan: Any = None,
    memory_plan: Any = None,
    communication_plan: Any = None,
) -> str:
    supported = {
        "Dense", "Conv", "MaxPool", "AvgPool", "BatchNormalization",
        "Relu", "LeakyRelu", "Sigmoid", "Softmax",
        "Flatten", "Reshape", "Add"
    }
    bad = [op.op_type for op in graph.ops if op.op_type not in supported]
    if bad:
        raise RuntimeError(f"Unsupported training ops: {sorted(set(bad))}")

    loss_type = str(training_cfg.get("loss", {}).get("type", "mse")).lower()
    lr = float(training_cfg.get("optimizer", {}).get("learning_rate", 0.01))
    bypass_final_softmax_backward = _is_final_softmax_mse_case(graph, training_cfg)

    input_name = graph.inputs[0]
    output_name = graph.outputs[0]

    tensor_name_to_shape, tensor_name_to_size = _build_tensor_shapes(graph)
    tensor_name_to_cxx, grad_name_to_cxx, roots_in_order = _build_tensor_aliases(graph, tensor_name_to_size)

    input_size = tensor_name_to_size[input_name]
    output_size = tensor_name_to_size[output_name]

    lines: List[str] = []
    lines.append('#include <hls_stream.h>')
    lines.append('#include <ap_axi_sdata.h>')
    lines.append('#include <math.h>')
    lines.append('#include "fpgai_types.h"')
    lines.append('#include "layers/dense.h"')
    lines.append('#include "layers/conv.h"')
    lines.append('#include "layers/pool.h"')
    lines.append('#include "layers/activations.h"')
    lines.append('#include "layers/batchnorm.h"')
    lines.append("")
    lines.append("typedef ap_axis<32,0,0,0> axis_t;")
    lines.append("using namespace fpgai;")
    lines.append("")
    lines.append("static inline unsigned int f32_to_u32(float v) { union { float f; unsigned int u; } x; x.f = v; return x.u; }")
    lines.append("static inline float u32_to_f32(unsigned int v) { union { float f; unsigned int u; } x; x.u = v; return x.f; }")
    lines.append("static inline void write_f32(hls::stream<axis_t>& s, float v, bool last=false) { axis_t p; p.data=f32_to_u32(v); p.keep=-1; p.strb=-1; p.last=last?1:0; s.write(p); }")
    lines.append("static inline float read_f32(hls::stream<axis_t>& s) { axis_t p = s.read(); return u32_to_f32((unsigned int)p.data); }")
    lines.append("")
    lines.append("template<int N>")
    lines.append("static inline void emit_stream_block(hls::stream<axis_t>& out, const float* data, bool is_last_block) {")
    lines.append("#pragma HLS INLINE off")
    lines.append("  for (int i = 0; i < N; ++i) {")
    lines.append("#pragma HLS PIPELINE II=1")
    lines.append("    bool last = is_last_block && (i == N - 1);")
    lines.append("    write_f32(out, data[i], last);")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append('template<int C, int HW>')
    lines.append('static inline void fpgai_bn_backward_input_exact(')
    lines.append('  const grad_act_t gy[C * HW],')
    lines.append('  const wgt_t gamma[C],')
    lines.append('  const acc_t var[C],')
    lines.append('  const act_t xhat[C * HW],')
    lines.append('  grad_act_t gx[C * HW]')
    lines.append(') {')
    lines.append('  const float eps = 1e-5f;')
    lines.append('  for (int c = 0; c < C; ++c) {')
    lines.append('    float sum_dy = 0.0f;')
    lines.append('    float sum_dy_xhat = 0.0f;')
    lines.append('    for (int hw = 0; hw < HW; ++hw) {')
    lines.append('      const int idx = hw * C + c;')
    lines.append('      const float g = (float)gy[idx];')
    lines.append('      const float xh = (float)xhat[idx];')
    lines.append('      sum_dy += g;')
    lines.append('      sum_dy_xhat += g * xh;')
    lines.append('    }')
    lines.append('    const float mean_dy = sum_dy / (float)HW;')
    lines.append('    const float mean_dy_xhat = sum_dy_xhat / (float)HW;')
    lines.append('    const float inv_std = 1.0f / sqrtf((float)var[c] + eps);')
    lines.append('    const float scale = ((float)gamma[c]) * inv_std;')
    lines.append('    for (int hw = 0; hw < HW; ++hw) {')
    lines.append('      const int idx = hw * C + c;')
    lines.append('      const float xh = (float)xhat[idx];')
    lines.append('      gx[idx] = (grad_act_t)(scale * (((float)gy[idx]) - mean_dy - xh * mean_dy_xhat));')
    lines.append('    }')
    lines.append('  }')
    lines.append('}')
    lines.append("")

    for root in roots_in_order:
        size = tensor_name_to_size[root]
        cxx = tensor_name_to_cxx[root]
        gcxx = grad_name_to_cxx[root]
        lines.append(f"static act_t {cxx}[{size}];")
        lines.append(f"static grad_act_t {gcxx}[{size}];")

    param_specs = []

    for op in graph.ops:
        if op.op_type == "Dense":
            W, B, _in_f, _out_f = _resolve_dense_arrays(graph, op)
            tag = _sanitize(op.name)
            lines.append(f"static wgt_t W_{tag}[{W.size}] = {{ {', '.join(f'{float(x):.8f}f' for x in W.reshape(-1))} }};")
            lines.append(f"static bias_t B_{tag}[{B.size}] = {{ {', '.join(f'{float(x):.8f}f' for x in B.reshape(-1))} }};")
            lines.append(f"static grad_wgt_t dW_{tag}[{W.size}];")
            lines.append(f"static grad_bias_t dB_{tag}[{B.size}];")
            lines.append(f"static float OUT_grad_{tag}[{W.size + B.size}];")
            param_specs.append(("dense", op, tag, W.size, B.size))

        elif op.op_type == "Conv":
            W, B, _ws = _resolve_conv_arrays(graph, op)
            tag = _sanitize(op.name)
            lines.append(f"static wgt_t W_{tag}[{W.size}] = {{ {', '.join(f'{float(x):.8f}f' for x in W.reshape(-1))} }};")
            lines.append(f"static bias_t B_{tag}[{B.size}] = {{ {', '.join(f'{float(x):.8f}f' for x in B.reshape(-1))} }};")
            lines.append(f"static grad_wgt_t dW_{tag}[{W.size}];")
            lines.append(f"static grad_bias_t dB_{tag}[{B.size}];")
            lines.append(f"static float OUT_grad_{tag}[{W.size + B.size}];")
            param_specs.append(("conv", op, tag, W.size, B.size))

        elif op.op_type == "BatchNormalization":
            out_shape = tensor_name_to_shape.get(op.outputs[0], tuple()) or tensor_name_to_shape.get(op.inputs[0], tuple())
            c, h, w = _as_chw(out_shape)
            hw = h * w
            tag = _sanitize(op.name)
            gamma, beta, mean, var = _resolve_bn_arrays(graph, op, c)
            lines.append(f"static wgt_t BN_G_{tag}[{c}] = {{ {', '.join(f'{float(x):.8f}f' for x in gamma)} }};")
            lines.append(f"static bias_t BN_B_{tag}[{c}] = {{ {', '.join(f'{float(x):.8f}f' for x in beta)} }};")
            lines.append(f"static acc_t BN_M_{tag}[{c}] = {{ {', '.join(f'{float(x):.8f}f' for x in mean)} }};")
            lines.append(f"static acc_t BN_V_{tag}[{c}] = {{ {', '.join(f'{float(x):.8f}f' for x in var)} }};")
            lines.append(f"static act_t BN_XHAT_{tag}[{c * hw}];")
            lines.append(f"static grad_wgt_t dBN_G_{tag}[{c}];")
            lines.append(f"static grad_bias_t dBN_B_{tag}[{c}];")
            lines.append(f"static float OUT_grad_{tag}[{2 * c}];")
            param_specs.append(("bn", op, tag, c, c))

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
        for kind, _op, tag, w_size, b_size in param_specs:
            if kind in ("dense", "conv"):
                lines.append(f"    for (int i = 0; i < {w_size}; ++i) W_{tag}[i] = (wgt_t)read_f32(aux);")
                lines.append(f"    for (int i = 0; i < {b_size}; ++i) B_{tag}[i] = (bias_t)read_f32(aux);")
            elif kind == "bn":
                lines.append(f"    for (int i = 0; i < {w_size}; ++i) BN_G_{tag}[i] = (wgt_t)read_f32(aux);")
                lines.append(f"    for (int i = 0; i < {b_size}; ++i) BN_B_{tag}[i] = (bias_t)read_f32(aux);")
        lines.append("    return;")
        lines.append("  }")
        lines.append("")

    lines.append(f"  for (int i = 0; i < {input_size}; ++i) {tensor_name_to_cxx[input_name]}[i] = (act_t)read_f32(in);")
    lines.append(f"  static act_t target_buf[{output_size}];")
    lines.append(f"  for (int i = 0; i < {output_size}; ++i) target_buf[i] = (act_t)read_f32(aux);")
    lines.append("")

    for root in roots_in_order:
        size = tensor_name_to_size[root]
        lines.append(f"  for (int i = 0; i < {size}; ++i) {grad_name_to_cxx[root]}[i] = (grad_act_t)0;")
    for kind, _op, tag, w_size, b_size in param_specs:
        if kind in ("dense", "conv"):
            lines.append(f"  for (int i = 0; i < {w_size}; ++i) dW_{tag}[i] = (grad_wgt_t)0;")
            lines.append(f"  for (int i = 0; i < {b_size}; ++i) dB_{tag}[i] = (grad_bias_t)0;")
        elif kind == "bn":
            lines.append(f"  for (int i = 0; i < {w_size}; ++i) dBN_G_{tag}[i] = (grad_wgt_t)0;")
            lines.append(f"  for (int i = 0; i < {b_size}; ++i) dBN_B_{tag}[i] = (grad_bias_t)0;")
    lines.append("")

    for op in graph.ops:
        xname = op.inputs[0]
        yname = op.outputs[0]
        xbuf = tensor_name_to_cxx[xname]
        ybuf = tensor_name_to_cxx[yname]
        xshape = tensor_name_to_shape.get(xname, tuple())
        yshape = tensor_name_to_shape.get(yname, tuple())
        sz = tensor_name_to_size[yname]

        if op.op_type == "Dense":
            tag = _sanitize(op.name)
            _W, _B, in_f, out_f = _resolve_dense_arrays(graph, op)
            lines.append(f"  fpgai::dense_out_in<{in_f}, {out_f}>({xbuf}, {ybuf}, W_{tag}, B_{tag});")

        elif op.op_type == "Conv":
            tag = _sanitize(op.name)
            ws = _resolve_conv_arrays(graph, op)[2]
            stride = int(op.attrs.get('strides', [1, 1])[0])
            pad = int(op.attrs.get('pads', [0, 0, 0, 0])[0])
            if not yshape:
                yshape = _infer_conv_output_shape(_as_chw(xshape), tuple(int(v) for v in ws), stride, pad)
            c_in, h_in, w_in = _as_chw(xshape)
            c_out, h_out, w_out = _as_chw(yshape)
            k_h = int(ws[2])
            lines.append(f"  fpgai::conv2d<{h_in}, {w_in}, {c_in}, {h_out}, {w_out}, {c_out}, {k_h}, {stride}, {pad}>({xbuf}, {ybuf}, W_{tag}, B_{tag});")

        elif op.op_type == "Relu":
            lines.append(f"  fpgai::relu<{sz}>({xbuf}, {ybuf});")

        elif op.op_type == "LeakyRelu":
            alpha = float((getattr(op, 'attrs', {}) or {}).get('alpha', 0.01))
            lines.append(f"  fpgai::leaky_relu<{sz}>({xbuf}, {ybuf}, (act_t){alpha:.8f}f);")

        elif op.op_type == "Sigmoid":
            lines.append(f"  fpgai::sigmoid<{sz}>({xbuf}, {ybuf});")

        elif op.op_type == "Softmax":
            lines.append(f"  fpgai::softmax<{sz}>({xbuf}, {ybuf});")

        elif op.op_type in ('Flatten', 'Reshape'):
            lines.append(f"  fpgai::reshape_copy<{sz}>({xbuf}, {ybuf});")

        elif op.op_type == "Add":
            rhs = tensor_name_to_cxx[op.inputs[1]]
            lines.append(f"  fpgai::add_vec<{sz}>({xbuf}, {rhs}, {ybuf});")

        elif op.op_type == "MaxPool":
            k = int(op.attrs.get("kernel_shape", [2, 2])[0])
            stride = int(op.attrs.get("strides", [2, 2])[0])
            if not yshape:
                yshape = _infer_pool_output_shape(_as_chw(xshape), k, stride)
            c_in, h_in, w_in = _as_chw(xshape)
            _c_out, h_out, w_out = _as_chw(yshape)
            lines.append(f"  fpgai::maxpool2d<{h_in}, {w_in}, {c_in}, {k}, {stride}, {h_out}, {w_out}>({xbuf}, {ybuf});")

        elif op.op_type == "AvgPool":
            k = int(op.attrs.get("kernel_shape", [2, 2])[0])
            stride = int(op.attrs.get("strides", [2, 2])[0])
            if not yshape:
                yshape = _infer_pool_output_shape(_as_chw(xshape), k, stride)
            c_in, h_in, w_in = _as_chw(xshape)
            _c_out, h_out, w_out = _as_chw(yshape)
            lines.append(f"  fpgai::avgpool2d<{h_in}, {w_in}, {c_in}, {k}, {stride}, {h_out}, {w_out}>({xbuf}, {ybuf});")

        elif op.op_type == "BatchNormalization":
            tag = _sanitize(op.name)
            if not yshape:
                yshape = xshape
            c, h, w = _as_chw(yshape)
            hw = h * w
            lines.append(f"  fpgai::batchnorm_train_forward<{c}, {hw}>({xbuf}, {ybuf}, BN_G_{tag}, BN_B_{tag}, BN_M_{tag}, BN_V_{tag}, BN_XHAT_{tag});")

    final_buf = tensor_name_to_cxx[output_name]
    final_grad = grad_name_to_cxx[output_name]

    if loss_type == "mse":
        lines.append("  loss_t loss_value = (loss_t)0;")
        lines.append(f"  for (int i = 0; i < {output_size}; ++i) {{")
        lines.append(f"    acc_t diff = (acc_t){final_buf}[i] - (acc_t)target_buf[i];")
        lines.append(f"    {final_grad}[i] = (grad_act_t)diff;")
        lines.append("    loss_value += (loss_t)(diff * diff);")
        lines.append("  }")
    else:
        lines.append(f"  for (int i = 0; i < {output_size}; ++i) {final_grad}[i] = (grad_act_t)(((acc_t){final_buf}[i]) - ((acc_t)target_buf[i]));")
    lines.append("")

    for op_idx in range(len(graph.ops) - 1, -1, -1):
        op = graph.ops[op_idx]
        xname = op.inputs[0]
        yname = op.outputs[0]
        xbuf = tensor_name_to_cxx[xname]
        ybuf = tensor_name_to_cxx[yname]
        gx = grad_name_to_cxx[xname]
        gy = grad_name_to_cxx[yname]
        xshape = tensor_name_to_shape.get(xname, tuple())
        yshape = tensor_name_to_shape.get(yname, tuple())
        sz_in = tensor_name_to_size[xname]
        sz_out = tensor_name_to_size[yname]

        if op.op_type == "Dense":
            tag = _sanitize(op.name)
            _W, _B, in_f, out_f = _resolve_dense_arrays(graph, op)
            lines.append(f"  fpgai::dense_weight_grad<{in_f}, {out_f}>({xbuf}, {gy}, dW_{tag});")
            lines.append(f"  fpgai::dense_bias_grad<{out_f}>({gy}, dB_{tag});")
            lines.append(f"  fpgai::dense_backward_input<{in_f}, {out_f}>({gy}, W_{tag}, {gx});")

        elif op.op_type == "Conv":
            tag = _sanitize(op.name)
            ws = _resolve_conv_arrays(graph, op)[2]
            stride = int(op.attrs.get('strides', [1, 1])[0])
            pad = int(op.attrs.get('pads', [0, 0, 0, 0])[0])
            if not yshape:
                yshape = _infer_conv_output_shape(_as_chw(xshape), tuple(int(v) for v in ws), stride, pad)
            c_in, h_in, w_in = _as_chw(xshape)
            c_out, h_out, w_out = _as_chw(yshape)
            k_h = int(ws[2])

            lines.append(f"  fpgai::conv2d_weight_grad<{h_in}, {w_in}, {c_in}, {h_out}, {w_out}, {c_out}, {k_h}, {stride}, {pad}>({xbuf}, {gy}, dW_{tag});")
            lines.append(f"  fpgai::conv2d_bias_grad<{c_out}, {h_out}, {w_out}>({gy}, dB_{tag});")

            if _needs_input_gradient(graph, op_idx):
                lines.append(f"  fpgai::conv2d_backward_input<{h_in}, {w_in}, {c_in}, {h_out}, {w_out}, {c_out}, {k_h}, {stride}, {pad}>({gy}, W_{tag}, {gx});")
            else:
                lines.append(f"  for (int i = 0; i < {sz_in}; ++i) {gx}[i] = (grad_act_t)0;")

        elif op.op_type == "Relu":
            lines.append(f"  fpgai::relu_backward_from_output<{sz_out}>({ybuf}, {gy}, {gx});")

        elif op.op_type == "LeakyRelu":
            alpha = float((getattr(op, 'attrs', {}) or {}).get('alpha', 0.01))
            lines.append(f"  fpgai::leaky_relu_backward_from_input<{sz_out}>({xbuf}, {gy}, {gx}, (act_t){alpha:.8f}f);")

        elif op.op_type == "Sigmoid":
            lines.append(f"  fpgai::sigmoid_backward_from_output<{sz_out}>({ybuf}, {gy}, {gx});")

        elif op.op_type == "Softmax":
            is_final_op = (op_idx == len(graph.ops) - 1 and yname == output_name)
            if bypass_final_softmax_backward and is_final_op:
                lines.append(f"  for (int i = 0; i < {sz_out}; ++i) {gx}[i] += {gy}[i];")
            else:
                lines.append(f"  fpgai::softmax_backward<{sz_out}>({ybuf}, {gy}, {gx});")

        elif op.op_type in ('Flatten', 'Reshape'):
            lines.append(f"  for (int i = 0; i < {sz_out}; ++i) {gx}[i] += {gy}[i];")

        elif op.op_type == "Add":
            rhsg = grad_name_to_cxx[op.inputs[1]]
            lines.append(f"  fpgai::add_backward<{sz_out}>({gy}, {gx}, {rhsg});")

        elif op.op_type == "MaxPool":
            k = int(op.attrs.get("kernel_shape", [2, 2])[0])
            stride = int(op.attrs.get("strides", [2, 2])[0])
            if not yshape:
                yshape = _infer_pool_output_shape(_as_chw(xshape), k, stride)
            c_in, h_in, w_in = _as_chw(xshape)
            _c_out, h_out, w_out = _as_chw(yshape)
            lines.append(f"  fpgai::maxpool2d_backward<{h_in}, {w_in}, {c_in}, {k}, {stride}, {h_out}, {w_out}>({xbuf}, {ybuf}, {gy}, {gx});")

        elif op.op_type == "AvgPool":
            k = int(op.attrs.get("kernel_shape", [2, 2])[0])
            stride = int(op.attrs.get("strides", [2, 2])[0])
            if not yshape:
                yshape = _infer_pool_output_shape(_as_chw(xshape), k, stride)
            c_in, h_in, w_in = _as_chw(xshape)
            _c_out, h_out, w_out = _as_chw(yshape)
            lines.append(f"  fpgai::avgpool2d_backward<{h_in}, {w_in}, {c_in}, {k}, {stride}, {h_out}, {w_out}>({gy}, {gx});")

        elif op.op_type == "BatchNormalization":
            tag = _sanitize(op.name)
            if not yshape:
                yshape = xshape
            c, h, w = _as_chw(yshape)
            hw = h * w
            lines.append(f"  fpgai::batchnorm_param_grad<{c}, {hw}>({gy}, BN_XHAT_{tag}, dBN_G_{tag}, dBN_B_{tag});")
            lines.append(f"  fpgai_bn_backward_input_exact<{c}, {hw}>({gy}, BN_G_{tag}, BN_V_{tag}, BN_XHAT_{tag}, {gx});")

    for kind, _op, tag, w_size, b_size in param_specs:
        if kind in ("dense", "conv"):
            lines.append(f"  fpgai::sgd_update_wgt<{w_size}>(W_{tag}, dW_{tag}, (upd_t){lr:.8f}f);")
            lines.append(f"  fpgai::sgd_update_bias<{b_size}>(B_{tag}, dB_{tag}, (upd_t){lr:.8f}f);")
        elif kind == "bn":
            lines.append(f"  fpgai::sgd_update_wgt<{w_size}>(BN_G_{tag}, dBN_G_{tag}, (upd_t){lr:.8f}f);")
            lines.append(f"  fpgai::sgd_update_bias<{b_size}>(BN_B_{tag}, dBN_B_{tag}, (upd_t){lr:.8f}f);")

    for idx, (kind, _op, tag, w_size, b_size) in enumerate(param_specs):
        total = w_size + b_size
        if kind in ("dense", "conv"):
            lines.append(f"  for (int i = 0; i < {w_size}; ++i) OUT_grad_{tag}[i] = (float)dW_{tag}[i];")
            lines.append(f"  for (int i = 0; i < {b_size}; ++i) OUT_grad_{tag}[{w_size} + i] = (float)dB_{tag}[i];")
        else:
            lines.append(f"  for (int i = 0; i < {w_size}; ++i) OUT_grad_{tag}[i] = (float)dBN_G_{tag}[i];")
            lines.append(f"  for (int i = 0; i < {b_size}; ++i) OUT_grad_{tag}[{w_size} + i] = (float)dBN_B_{tag}[i];")
        is_last = "true" if idx == len(param_specs) - 1 else "false"
        lines.append(f"  emit_stream_block<{total}>(out, OUT_grad_{tag}, {is_last});")

    if not param_specs:
        lines.append("  write_f32(out, 0.0f, true);")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)