from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fpgai.engine.models import LayerDescriptor


def _shape_to_tuple(shape: Any) -> Tuple[int, ...]:
    if shape is None:
        return tuple()
    if isinstance(shape, tuple):
        return shape
    if isinstance(shape, list):
        out = []
        for x in shape:
            try:
                out.append(int(x))
            except Exception:
                out.append(-1)
        return tuple(out)
    return tuple()


def _numel(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 0
    total = 1
    for d in shape:
        if d is None or d == -1:
            return 0
        total *= int(d)
    return total


def _dtype_nbytes(dtype: Any) -> int:
    if dtype is None:
        return 4
    s = str(dtype).lower()
    if "float64" in s or "double" in s or "int64" in s:
        return 8
    if "float16" in s or "half" in s or "int16" in s:
        return 2
    if "int8" in s or "uint8" in s:
        return 1
    return 4


def _tensor_shape(graph: Any, tensor_name: str) -> Tuple[int, ...]:
    spec = getattr(graph, "tensors", {}).get(tensor_name)
    if spec is None:
        const = getattr(graph, "constants", {}).get(tensor_name)
        if const is not None and hasattr(const, "shape"):
            return _shape_to_tuple(const.shape)
        return tuple()

    if isinstance(spec, dict):
        return _shape_to_tuple(spec.get("shape"))

    shape = getattr(spec, "shape", None)
    return _shape_to_tuple(shape)


def _tensor_nbytes(graph: Any, tensor_name: str) -> int:
    spec = getattr(graph, "tensors", {}).get(tensor_name)
    if spec is not None:
        if isinstance(spec, dict):
            shape = _shape_to_tuple(spec.get("shape"))
            dtype = spec.get("dtype")
        else:
            shape = _shape_to_tuple(getattr(spec, "shape", None))
            dtype = getattr(spec, "dtype", None)
        return _numel(shape) * _dtype_nbytes(dtype)

    const = getattr(graph, "constants", {}).get(tensor_name)
    if const is not None:
        shape = _shape_to_tuple(getattr(const, "shape", None))
        dtype = getattr(const, "dtype", None)
        if dtype is None and hasattr(const, "dtype"):
            dtype = const.dtype
        return _numel(shape) * _dtype_nbytes(dtype)

    return 0


def _collect_param_names(graph: Any, op: Any) -> List[str]:
    constants = set(getattr(graph, "constants", {}).keys())
    params = set(getattr(graph, "params", {}).keys())
    known = constants | params
    out: List[str] = []

    for name in getattr(op, "inputs", []):
        if name in known and name not in out:
            out.append(name)

    attrs = dict(getattr(op, "attrs", {}) or {})
    for key in ("weight", "weights", "bias", "scale", "mean", "var"):
        val = attrs.get(key)
        if isinstance(val, str) and val in known and val not in out:
            out.append(val)

    return out


def _estimate_macs(op_type: str, attrs: Dict[str, Any], input_shapes: List[Tuple[int, ...]]) -> int:
    op_type = str(op_type)

    if op_type == "Dense":
        in_features = attrs.get("in_features", 0)
        out_features = attrs.get("out_features", 0)
        batch = 1
        if input_shapes and len(input_shapes[0]) >= 1 and input_shapes[0][0] not in (-1, None):
            batch = int(input_shapes[0][0])
        if in_features and out_features:
            return batch * int(in_features) * int(out_features)

    if op_type in ("MatMul", "Gemm"):
        if len(input_shapes) >= 2:
            a, b = input_shapes[0], input_shapes[1]
            if len(a) >= 2 and len(b) >= 2:
                m = a[-2] if a[-2] not in (-1, None) else 0
                k = a[-1] if a[-1] not in (-1, None) else 0
                n = b[-1] if b[-1] not in (-1, None) else 0
                return int(m) * int(k) * int(n)

    if op_type == "Conv":
        if input_shapes:
            x = input_shapes[0]
            if len(x) == 4:
                n, cin, h, w = x
                cout = int(attrs.get("out_channels", 0) or 0)
                if cout == 0:
                    w_name = attrs.get("weight")
                    if isinstance(w_name, str):
                        w_shape = input_shapes[1] if len(input_shapes) > 1 else tuple()
                        if len(w_shape) >= 1 and w_shape[0] not in (-1, None):
                            cout = int(w_shape[0])

                kshape = attrs.get("kernel_shape", [0, 0])
                kh = int(kshape[0]) if len(kshape) > 0 else 0
                kw = int(kshape[1]) if len(kshape) > 1 else 0
                if all(v not in (-1, None) for v in [n, cin, h, w, cout, kh, kw]):
                    return int(n) * int(h) * int(w) * int(cin) * int(cout) * int(kh) * int(kw)

    return 0


def _compute_hint(op_type: str, param_bytes: int, act_in: int, act_out: int, macs: int) -> str:
    traffic = param_bytes + act_in + act_out
    if macs == 0 and traffic == 0:
        return "unknown"
    if macs > max(1, traffic // 4):
        return "compute_bound"
    if traffic > max(1, macs // 4):
        return "memory_bound"
    return "balanced"


def analyze_graph(graph: Any) -> List[LayerDescriptor]:
    descriptors: List[LayerDescriptor] = []

    for op in getattr(graph, "ops", []):
        op_type = getattr(op, "op_type", "Unknown")
        node_name = getattr(op, "name", op_type)
        inputs = list(getattr(op, "inputs", []))
        outputs = list(getattr(op, "outputs", []))
        attrs = dict(getattr(op, "attrs", {}) or {})

        input_shapes = [_tensor_shape(graph, x) for x in inputs]
        output_shapes = [_tensor_shape(graph, x) for x in outputs]

        param_names = _collect_param_names(graph, op)
        param_bytes = sum(_tensor_nbytes(graph, p) for p in param_names)

        activation_inputs = [x for x in inputs if x not in param_names]
        activation_bytes_in = sum(_tensor_nbytes(graph, x) for x in activation_inputs)
        activation_bytes_out = sum(_tensor_nbytes(graph, x) for x in outputs)

        macs = _estimate_macs(op_type, attrs, input_shapes)
        compute_hint = _compute_hint(
            op_type=op_type,
            param_bytes=param_bytes,
            act_in=activation_bytes_in,
            act_out=activation_bytes_out,
            macs=macs,
        )

        desc = LayerDescriptor(
            node_name=node_name,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_names=param_names,
            param_bytes=param_bytes,
            activation_bytes_in=activation_bytes_in,
            activation_bytes_out=activation_bytes_out,
            macs=macs,
            attrs=attrs,
            compute_hint=compute_hint,
            backend_kernel=op_type.lower(),
        )
        descriptors.append(desc)

    return descriptors