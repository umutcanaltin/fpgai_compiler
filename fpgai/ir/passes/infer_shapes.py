from __future__ import annotations

from typing import Any, Iterable

from fpgai.ir.graph import Graph


_PASSTHROUGH_OPS = {
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Softmax",
    "Identity",
    "BatchNormalization",
}


def _known_shape(g: Graph, name: str) -> tuple[int, ...] | None:
    spec = g.get_tensor(name)
    if spec is None:
        return None
    shape = tuple(int(dim) for dim in getattr(spec, "shape", ()) or ())
    if not shape or any(dim <= 0 for dim in shape):
        return None
    return shape


def _dtype(g: Graph, name: str) -> str:
    spec = g.get_tensor(name)
    return str(getattr(spec, "dtype", "float32")) if spec is not None else "float32"


def _broadcast_shape(a: Iterable[int], b: Iterable[int]) -> tuple[int, ...]:
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
            raise ValueError(f"IRSHAPE001: incompatible Add broadcast dimensions {da} and {db}")
    return tuple(out)


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

            elif op.op_type == "Add" and len(inputs) >= 2:
                lhs = _known_shape(g, inputs[0])
                rhs = _known_shape(g, inputs[1])
                if lhs is not None and rhs is not None:
                    out_shape = _broadcast_shape(lhs, rhs)

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
