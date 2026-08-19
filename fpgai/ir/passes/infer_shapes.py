from __future__ import annotations

from typing import Any, Iterable

from fpgai.ir.graph import Graph
from fpgai.ir.tensor_ops import gather_output_shape, resolve_resize_shape, slice_output_shape, normalize_axis


_PASSTHROUGH_OPS = {
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "SiLU",
    "Softmax",
    "Identity",
    "Cast",
    "BatchNormalization",
    "LayerNormalization",
    "RMSNorm",
    "Exp",
    "Rsqrt",
    "Sqrt",
    "Broadcast",
    "CausalMask",
    "RotaryEmbedding",
    "MultiHeadAttention",
    "GroupQueryAttention",
    "KVCacheUpdate",
    "PersistentStateRead",
    "PersistentStateReset",
}


def _known_shape(g: Graph, name: str) -> tuple[int, ...] | None:
    spec = g.get_tensor(name)
    if spec is None:
        return None
    raw_shape = getattr(spec, "shape", None)
    if raw_shape is None:
        return None
    shape = tuple(int(dim) for dim in raw_shape)
    # Empty tuple is a valid scalar tensor shape.
    if any(dim <= 0 for dim in shape):
        return None
    return shape


def _dtype(g: Graph, name: str) -> str:
    spec = g.get_tensor(name)
    return str(getattr(spec, "dtype", "float32")) if spec is not None else "float32"


def _broadcast_shape(a: Iterable[int], b: Iterable[int], *, op_name: str = "elementwise") -> tuple[int, ...]:
    left = list(int(x) for x in a)
    right = list(int(x) for x in b)
    rank = max(len(left), len(right))
    left = [1] * (rank - len(left)) + left
    right = [1] * (rank - len(right)) + right
    out: list[int] = []
    for da, db in zip(left, right):
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            raise ValueError(f"IRSHAPE001: incompatible {op_name} broadcast dimensions {da} and {db}")
    return tuple(out)



def _matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...] | None:
    # ONNX/Numpy-style rank >= 2 MatMul shape propagation for the attention path.
    if len(a) < 2 or len(b) < 2 or a[-1] != b[-2]:
        return None
    batch = _broadcast_shape(a[:-2], b[:-2], op_name="MatMul batch")
    return tuple(batch) + (a[-2], b[-1])


def _transpose_shape(shape: tuple[int, ...], perm: list[int] | tuple[int, ...] | None) -> tuple[int, ...] | None:
    if perm is None:
        perm = tuple(reversed(range(len(shape))))
    try:
        order = tuple(int(x) for x in perm)
    except Exception:
        return None
    if sorted(order) != list(range(len(shape))):
        return None
    return tuple(shape[i] for i in order)


def _reshape_target(g: Graph, op: Any) -> tuple[int, ...] | None:
    if len(getattr(op, "inputs", []) or []) < 2:
        return None
    shape_name = op.inputs[1]
    values = getattr(g, "constants", {}).get(shape_name)
    if values is None:
        return None
    try:
        return tuple(int(x) for x in values.reshape(-1).tolist())
    except Exception:
        return None


def _flatten_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...] | None:
    rank = len(shape)
    if axis < 0:
        axis += rank
    if axis < 0 or axis > rank:
        return None
    left = 1
    for dim in shape[:axis]:
        left *= dim
    right = 1
    for dim in shape[axis:]:
        right *= dim
    return (left, right)


def _set_if_missing(g: Graph, name: str, shape: tuple[int, ...] | None, dtype: str) -> bool:
    if shape is None:
        return False
    known = g.get_tensor(name)
    if known is not None and tuple(getattr(known, "shape", ()) or ()):
        return False
    g.add_tensor(name, tuple(int(x) for x in shape), dtype)
    return True


def infer_shapes(g: Graph) -> Graph:
    """Propagate static FPGAI IR shapes after ONNX/external-operator import.

    ONNX shape inference is best-effort for models that contain custom-domain
    operators. Once an approved external operator callback has populated its
    output tensors, this pass continues shape propagation through supported
    built-in FPGAI operators. It is intentionally conservative: operators whose
    shapes cannot be proven are left unresolved rather than guessed.
    """

    # Iterate to a fixed point because a custom operator can unlock downstream
    # standard operators that ONNX itself could not infer past the custom node.
    for _ in range(max(1, len(getattr(g, "ops", []) or []) + 1)):
        changed = False
        for op in getattr(g, "ops", []) or []:
            inputs = list(getattr(op, "inputs", []) or [])
            outputs = list(getattr(op, "outputs", []) or [])
            if not outputs:
                continue

            out_shape: tuple[int, ...] | None = None
            out_dtype = _dtype(g, inputs[0]) if inputs else "float32"

            if op.op_type in _PASSTHROUGH_OPS and inputs:
                out_shape = _known_shape(g, inputs[0])

                # GroupQueryAttention has three outputs in the ORT contrib
                # schema: attention output, present key, present value. The
                # first follows query shape; present caches follow the supplied
                # past-cache bounded shapes when available.
                if op.op_type == "GroupQueryAttention":
                    if outputs:
                        changed |= _set_if_missing(g, outputs[0], out_shape, out_dtype)
                    if len(outputs) > 1 and len(inputs) > 3:
                        changed |= _set_if_missing(g, outputs[1], _known_shape(g, inputs[3]), _dtype(g, inputs[3]))
                    if len(outputs) > 2 and len(inputs) > 4:
                        changed |= _set_if_missing(g, outputs[2], _known_shape(g, inputs[4]), _dtype(g, inputs[4]))
                    continue

            elif op.op_type == "PersistentStateLength" and inputs:
                out_shape = (1,)
                out_dtype = "int32"

            elif op.op_type in {"Add", "Mul", "Sub", "Div"} and len(inputs) >= 2:
                lhs = _known_shape(g, inputs[0])
                rhs = _known_shape(g, inputs[1])
                if lhs is not None and rhs is not None:
                    out_shape = _broadcast_shape(lhs, rhs, op_name=op.op_type)

            elif op.op_type in {"Squeeze", "Unsqueeze"} and inputs:
                src = _known_shape(g, inputs[0])
                if src is not None:
                    attrs = getattr(op, "attrs", {}) or {}
                    axes = attrs.get("axes")
                    if axes is None and len(inputs) > 1:
                        raw_axes = getattr(g, "constants", {}).get(inputs[1])
                        if raw_axes is not None:
                            axes = [int(x) for x in raw_axes.reshape(-1).tolist()]
                    axes = [int(x) for x in (axes or [])]
                    if op.op_type == "Squeeze":
                        if not axes:
                            out_shape = tuple(dim for dim in src if dim != 1)
                        else:
                            normalized = sorted({a + len(src) if a < 0 else a for a in axes})
                            if all(0 <= a < len(src) and src[a] == 1 for a in normalized):
                                out_shape = tuple(dim for i, dim in enumerate(src) if i not in normalized)
                    else:
                        rank = len(src) + len(axes)
                        normalized = sorted(a + rank if a < 0 else a for a in axes)
                        if len(set(normalized)) == len(normalized) and all(0 <= a < rank for a in normalized):
                            values = list(src)
                            for axis in normalized:
                                values.insert(axis, 1)
                            out_shape = tuple(values)

            elif op.op_type in {"ReduceMax", "ReduceSum", "ReduceMean"} and inputs:
                src = _known_shape(g, inputs[0])
                if src is not None:
                    attrs = getattr(op, "attrs", {}) or {}
                    raw_dims = attrs.get("axes", attrs.get("dimensions", None))
                    if raw_dims is None:
                        dims = list(range(len(src)))
                    elif isinstance(raw_dims, int):
                        dims = [int(raw_dims)]
                    else:
                        dims = [int(x) for x in raw_dims]
                    keepdims = bool(attrs.get("keepdims", True))
                    normalized = sorted({d + len(src) if d < 0 else d for d in dims})
                    if all(0 <= d < len(src) for d in normalized):
                        if keepdims:
                            out_shape = tuple(1 if i in normalized else dim for i, dim in enumerate(src))
                        else:
                            out_shape = tuple(dim for i, dim in enumerate(src) if i not in normalized)


            elif op.op_type == "Concat" and inputs:
                shapes = [_known_shape(g, name) for name in inputs]
                if all(shape is not None for shape in shapes):
                    rank = len(shapes[0])
                    axis = normalize_axis(int((getattr(op, "attrs", {}) or {}).get("axis", 0)), rank)
                    base = list(shapes[0])
                    if all(len(shape) == rank and all(shape[d] == base[d] for d in range(rank) if d != axis) for shape in shapes[1:]):
                        base[axis] = sum(shape[axis] for shape in shapes)
                        out_shape = tuple(base)

            elif op.op_type == "Slice" and inputs:
                src = _known_shape(g, inputs[0])
                if src is not None:
                    try:
                        out_shape = slice_output_shape(g, op, src)
                    except ValueError:
                        pass

            elif op.op_type == "Resize" and inputs:
                src = _known_shape(g, inputs[0])
                if src is not None:
                    try:
                        out_shape = resolve_resize_shape(g, op, src)
                    except ValueError:
                        pass

            elif op.op_type == "Gather" and len(inputs) >= 2:
                data = _known_shape(g, inputs[0])
                indices = _known_shape(g, inputs[1])
                if data is not None and indices is not None:
                    out_shape = gather_output_shape(data, indices, int((getattr(op, "attrs", {}) or {}).get("axis", 0)))

            elif op.op_type == "MatMul" and len(inputs) >= 2:
                lhs = _known_shape(g, inputs[0])
                rhs = _known_shape(g, inputs[1])
                if lhs is not None and rhs is not None:
                    out_shape = _matmul_shape(lhs, rhs)

            elif op.op_type == "Transpose" and inputs:
                src = _known_shape(g, inputs[0])
                if src is not None:
                    out_shape = _transpose_shape(src, getattr(op, "attrs", {}).get("perm"))

            elif op.op_type == "Flatten" and inputs:
                src = _known_shape(g, inputs[0])
                if src is not None:
                    out_shape = _flatten_shape(src, int(getattr(op, "attrs", {}).get("axis", 1)))

            elif op.op_type == "Reshape" and inputs:
                target = _reshape_target(g, op)
                if target is not None and all(dim > 0 for dim in target):
                    out_shape = target

            if out_shape is None:
                continue

            for output in outputs:
                changed = _set_if_missing(g, output, out_shape, out_dtype) or changed

        if not changed:
            break
    return g
