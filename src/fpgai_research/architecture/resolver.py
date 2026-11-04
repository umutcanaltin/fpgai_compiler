"""Fail-closed, per-instance architecture resolution and iteration coordinates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional
from fpgai_research.model_ir.contracts import ArchitectureRequest, ArchitecturePoint, ArchitecturalModel, FunctionalModel


class ArchitectureError(ValueError):
    def __init__(self, operator: str, code: str, detail: str):
        self.operator, self.code, self.detail = operator, code, detail
        super().__init__(f"{operator}: {code}: {detail}")


@dataclass(frozen=True)
class TargetLimits:
    max_parallel_units: Optional[int] = None
    supported_widths: tuple[int, ...] = (8, 12, 16, 32)


def iteration_shape(model: FunctionalModel, op):
    t = model.tensor
    kind = op.kind
    # Manual IR construction and restored artifact snapshots must meet the
    # same supported ONNX semantics as the strict importer. Unknown attributes
    # must not silently disappear in the generated source.
    permitted = {
        "MatMul": (), "Gemm": ("alpha", "beta", "transA", "transB"),
        "Conv": ("strides", "pads", "group", "dilations", "kernel_shape", "auto_pad"),
        "MaxPool": ("kernel_shape", "strides", "pads", "auto_pad", "ceil_mode", "dilations", "storage_order"),
        "Relu": (), "Identity": (), "Sigmoid": (), "Add": (), "Mul": (),
        "Flatten": ("axis",), "Reshape": ("shape",), "Squeeze": ("axes",), "Unsqueeze": ("axes",), "Softmax": ("axis",), "Transpose": ("perm",),
        "GlobalAveragePool": (), "BatchNormalization": ("epsilon",),
        "Concat": ("axis",), "LeakyRelu": ("alpha",),
    }
    if kind not in permitted:
        raise ArchitectureError(op.name, "UNSUPPORTED_OPERATOR", kind)
    unknown = set(dict(op.attributes)) - set(permitted[kind])
    if unknown:
        raise ArchitectureError(op.name, "UNSUPPORTED_ATTR", f"Unsupported {kind} attributes: {sorted(unknown)}")
    if kind in ("Conv", "MaxPool"):
        auto_pad = op.attr("auto_pad", "NOTSET")
        if isinstance(auto_pad, bytes):
            auto_pad = auto_pad.decode("ascii")
        if auto_pad != "NOTSET":
            raise ArchitectureError(op.name, "UNSUPPORTED_ATTR", "Automatic padding unsupported")
    if kind == "Conv" and op.attr("kernel_shape") is not None:
        kernel = t(op.inputs[1]).shape[2:] if len(op.inputs) >= 2 else ()
        if tuple(op.attr("kernel_shape")) != kernel:
            raise ArchitectureError(op.name, "CONV_ATTR", "kernel_shape contradicts weight dimensions")
    if kind == "MaxPool" and (int(op.attr("ceil_mode", 0)) != 0
                              or int(op.attr("storage_order", 0)) != 0
                              or tuple(op.attr("dilations", (1,1))) != (1,1)):
        raise ArchitectureError(op.name, "MAXPOOL", "MaxPool requires default ceil/storage/dilation behavior")
    if kind == "Flatten" and int(op.attr("axis", 1)) != 1:
        raise ArchitectureError(op.name, "FLATTEN", "Only axis=1 Flatten is supported")
    if kind == "LeakyRelu" and not (0 <= float(op.attr("alpha", .01)) <= 1):
        raise ArchitectureError(op.name, "LEAKY", "LeakyRelu alpha must be in [0,1]")
    valid_arity = {"MatMul": (2,), "Gemm": (2, 3), "Conv": (2, 3),
                   "Relu": (1,), "Identity": (1,), "Sigmoid": (1,), "Reshape": (1,), "Squeeze": (1,), "Unsqueeze": (1,),
                   "Add": (2,), "Mul": (2,), "Flatten": (1,), "Softmax": (1,),
                   "Transpose": (1,), "GlobalAveragePool": (1,), "MaxPool": (1,),
                   "BatchNormalization": (5,), "LeakyRelu": (1,)}
    if kind in valid_arity and len(op.inputs) not in valid_arity[kind]:
        raise ArchitectureError(op.name, "ARITY", f"{kind} expects {valid_arity[kind]} inputs")
    if kind == "Gemm" and (float(op.attr("alpha", 1)) != 1.0
                           or float(op.attr("beta", 1)) != 1.0
                           or int(op.attr("transA", 0)) != 0
                           or int(op.attr("transB", 0)) not in (0, 1)):
        raise ArchitectureError(op.name, "GEMM_ATTR", "Only alpha=beta=1, transA=0, transB=0/1")
    if kind in ("MatMul", "Gemm"):
        if len(op.inputs) < 2:
            raise ArchitectureError(op.name, "INPUTS", "Missing matrix operand")
        a, b = t(op.inputs[0]).shape, t(op.inputs[1]).shape
        transb = op.kind == "Gemm" and int(op.attr("transB", 0)) == 1
        logical_k = b[1] if transb else b[0]
        logical_n = b[0] if transb else b[1]
        if len(a) != 2 or len(b) != 2 or a[1] != logical_k:
            raise ArchitectureError(op.name, "SHAPE", "Only matching 2D matrices are supported")
        if t(op.outputs[0]).shape != (a[0], logical_n):
            raise ArchitectureError(op.name, "SHAPE", "Output shape does not match matrices")
        if kind == "Gemm" and len(op.inputs) == 3:
            if t(op.inputs[2]).shape != (logical_n,):
                raise ArchitectureError(op.name, "BIAS", "Gemm bias must have shape (N,)")
        return (a[0], logical_n, a[1])
    if kind == "Conv":
        a, w = t(op.inputs[0]).shape, t(op.inputs[1]).shape
        o = t(op.outputs[0]).shape
        if len(a) != 4 or len(w) != 4 or len(o) != 4 or a[1] != w[1] or o[:2] != (a[0], w[0]):
            raise ArchitectureError(op.name, "SHAPE", "Expected NCHW Conv2D, groups=1")
        strides = tuple(op.attr("strides", (1, 1)))
        pads = tuple(op.attr("pads", (0, 0, 0, 0)))
        if len(strides) != 2 or len(pads) != 4 or any(s < 1 for s in strides) or any(p < 0 for p in pads):
            raise ArchitectureError(op.name, "CONV_ATTR", "Invalid stride/padding")
        if len(op.inputs) == 3 and t(op.inputs[2]).shape != (w[0],):
            raise ArchitectureError(op.name, "BIAS", "Convolution bias must be output-channel vector")
        if op.attr("group", 1) != 1 or tuple(op.attr("dilations", (1, 1))) != (1, 1):
            raise ArchitectureError(op.name, "CONV_ATTR", "Grouped/dilated convolution unsupported")
        oh = (a[2] + pads[0] + pads[2] - w[2]) // strides[0] + 1
        ow = (a[3] + pads[1] + pads[3] - w[3]) // strides[1] + 1
        if o[2:] != (oh, ow) or oh < 1 or ow < 1:
            raise ArchitectureError(op.name, "SHAPE", "Invalid convolution spatial output")
        return (o[0], o[1], o[2], o[3], w[1], w[2], w[3])
    if kind == "MaxPool":
        a = t(op.inputs[0]).shape
        if len(a) != 4 or tuple(op.attr("kernel_shape", ())) != (2, 2) or tuple(op.attr("strides", (2, 2))) != (2, 2) or tuple(op.attr("pads", (0, 0, 0, 0))) != (0, 0, 0, 0):
            raise ArchitectureError(op.name, "MAXPOOL", "Only unpadded 2x2 stride-2 NCHW pooling supported")
        if t(op.outputs[0]).shape != (a[0], a[1], a[2]//2, a[3]//2):
            raise ArchitectureError(op.name, "SHAPE", "MaxPool output shape mismatch")
        return t(op.outputs[0]).shape
    if kind == "Concat":
        if len(op.inputs) < 2:
            raise ArchitectureError(op.name, "ARITY", "Concat requires at least two inputs")
        shapes = [t(name).shape for name in op.inputs]
        if len(shapes[0]) not in (2, 4) or any(len(s) != len(shapes[0]) or s[0] != shapes[0][0] or s[2:] != shapes[0][2:] for s in shapes):
            raise ArchitectureError(op.name, "CONCAT", "Only rank-2/4 axis=1 Concat with matching spatial dimensions")
        target = (shapes[0][0], sum(s[1] for s in shapes), *shapes[0][2:])
        if int(op.attr("axis", 1)) != 1 or t(op.outputs[0]).shape != target:
            raise ArchitectureError(op.name, "CONCAT", "Invalid channel-concatenation shape or axis")
        return target
    if kind == "BatchNormalization":
        x = t(op.inputs[0]).shape
        if len(x) != 4 or t(op.outputs[0]).shape != x:
            raise ArchitectureError(op.name, "BATCHNORM", "Inference BatchNormalization requires NCHW")
        channels = x[1]
        if any(t(name).shape != (channels,) for name in op.inputs[1:]):
            raise ArchitectureError(op.name, "BATCHNORM", "Affine and statistics tensors must be channel vectors")
        if any(t(name).data is None for name in op.inputs[1:]):
            raise ArchitectureError(op.name, "BATCHNORM", "Only constant inference statistics are supported")
        if any(v < 0 for v in t(op.inputs[4]).data) or not (0.0 < float(op.attr("epsilon", 1e-5)) < float('inf')):
            raise ArchitectureError(op.name, "BATCHNORM", "Variance must be nonnegative and epsilon positive")
        return x
    if kind in ("Relu", "Identity", "Add", "Mul", "Sigmoid", "Flatten", "Reshape", "Squeeze", "Unsqueeze", "Softmax", "Transpose", "GlobalAveragePool", "LeakyRelu"):
        if kind == "Softmax" and (len(t(op.inputs[0]).shape) != 2 or int(op.attr("axis", -1)) not in (-1, 1)):
            raise ArchitectureError(op.name, "SOFTMAX", "Only last-axis 2D Softmax supported")
        if kind in ("Add", "Mul"):
            from fpgai_research.frontend.broadcast import broadcast_shape
            try:
                expected_broadcast = broadcast_shape(t(op.inputs[0]).shape, t(op.inputs[1]).shape)
            except ValueError as exc:
                raise ArchitectureError(op.name, "BROADCAST", str(exc)) from exc
            if t(op.outputs[0]).shape != expected_broadcast:
                raise ArchitectureError(op.name, "BROADCAST", "Declared output does not match broadcast shape")
        if kind in ("Relu", "Sigmoid", "Identity", "Softmax", "LeakyRelu") and t(op.inputs[0]).shape != t(op.outputs[0]).shape:
            raise ArchitectureError(op.name, "SHAPE", "Elementwise output must match input shape")
        if kind == "Flatten":
            dims = t(op.inputs[0]).shape
            if len(dims) < 2 or t(op.outputs[0]).shape != (dims[0], __import__("math").prod(dims[1:])):
                raise ArchitectureError(op.name, "SHAPE", "Invalid axis-1 Flatten output")
        if kind == "Reshape":
            from fpgai_research.frontend.static_reshape import resolve_static_reshape
            shape = resolve_static_reshape(t(op.inputs[0]).shape, op.attr("shape"))
            if t(op.outputs[0]).shape != shape:
                raise ArchitectureError(op.name, "RESHAPE", "Resolved shape differs from declared output")
        if kind in ("Squeeze", "Unsqueeze"):
            from fpgai_research.frontend.static_axes import squeeze_shape, unsqueeze_shape
            shape = t(op.inputs[0]).shape
            try:
                axes = op.attr("axes")
                expected = squeeze_shape(shape, axes) if kind == "Squeeze" else unsqueeze_shape(shape, axes)
            except (ValueError, TypeError) as exc:
                raise ArchitectureError(op.name, "AXES", str(exc)) from exc
            if t(op.outputs[0]).shape != expected:
                raise ArchitectureError(op.name, "AXES", "Declared tensor shape differs from ONNX static axes")
        if kind == "Transpose":
            dims = t(op.inputs[0]).shape
            if len(dims) != 2 or tuple(op.attr("perm", (1, 0))) != (1, 0) or t(op.outputs[0]).shape != (dims[1], dims[0]):
                raise ArchitectureError(op.name, "SHAPE", "Only 2D matrix transpose is supported")
        if kind == "GlobalAveragePool":
            dims = t(op.inputs[0]).shape
            if len(dims) != 4 or t(op.outputs[0]).shape != (dims[0], dims[1], 1, 1):
                raise ArchitectureError(op.name, "SHAPE", "GlobalAveragePool requires NCHW")
        return t(op.outputs[0]).shape
    raise ArchitectureError(op.name, "UNSUPPORTED_OPERATOR", kind)


def resolve(model: FunctionalModel, default: ArchitectureRequest,
            overrides: Optional[Mapping[str, ArchitectureRequest]] = None,
            limits: TargetLimits = TargetLimits()) -> ArchitecturalModel:
    model.validate()
    overrides = overrides or {}
    names = {op.name for op in model.operators}
    if set(overrides) - names:
        raise ArchitectureError("<model>", "UNKNOWN_SCOPE", repr(sorted(set(overrides) - names)))
    # JSON/YAML callers may construct dataclass requests with floats and bools.
    # Reject malformed contracts before lowering can emit invalid C++ loops.
    def validate_request(request, operator):
        if not isinstance(request, ArchitectureRequest):
            raise ArchitectureError(operator, "REQUEST_TYPE", "Expected ArchitectureRequest")
        ints = ("precision_bits", "integer_bits", "parallelism", "pes", "simd", "serialization")
        for name in ints:
            if type(getattr(request, name)) is not int:
                raise ArchitectureError(operator, "REQUEST_TYPE", f"{name} must be an integer, not {type(getattr(request, name)).__name__}")
        if not isinstance(request.tile, (tuple, list)) or any(type(x) is not int for x in request.tile):
            raise ArchitectureError(operator, "REQUEST_TYPE", "tile must be an integer array")
        if request.pipeline_ii is not None and type(request.pipeline_ii) is not int:
            raise ArchitectureError(operator, "REQUEST_TYPE", "pipeline_ii must be an integer or null")

    validate_request(default, "<model>")
    if not isinstance(limits, TargetLimits) or any(type(w) is not int for w in limits.supported_widths):
        raise ArchitectureError("<model>", "TARGET_TYPE", "Target widths must be integral")
    if limits.max_parallel_units is not None and (type(limits.max_parallel_units) is not int or limits.max_parallel_units < 1):
        raise ArchitectureError("<model>", "TARGET_TYPE", "Target parallelism limit must be a positive integer")
    for name, request in overrides.items():
        validate_request(request, name)
    points = []
    for op in model.operators:
        req = overrides.get(op.name, default)
        shape = iteration_shape(model, op)
        if req.precision_bits not in limits.supported_widths or not (1 <= req.integer_bits <= req.precision_bits):
            raise ArchitectureError(op.name, "PRECISION", "Unsupported fixed-point width/format")
        if min(req.parallelism, req.pes, req.simd, req.serialization) < 1:
            raise ArchitectureError(op.name, "NONPOSITIVE", "Spatial and temporal factors must be positive")
        if req.pipeline_ii is not None and req.pipeline_ii < 1:
            raise ArchitectureError(op.name, "PIPELINE", "II must be >= 1")
        if req.tile and (len(req.tile) != len(shape) or any(x < 1 for x in req.tile)):
            raise ArchitectureError(op.name, "TILE", "Tile extents must be positive and match iteration rank")
        if limits.max_parallel_units is not None and req.parallelism * req.pes * req.simd > limits.max_parallel_units:
            raise ArchitectureError(op.name, "CAPACITY", "Requested spatial width exceeds target limit")
        # Each factor is explicit; unsupported combinations are blocked, never rewritten.
        if op.kind not in ("MatMul", "Gemm", "Conv") and (req.parallelism, req.pes, req.simd, req.serialization) != (1, 1, 1, 1):
            raise ArchitectureError(op.name, "CAPABILITY", "Non-compute operator supports scalar mapping only")
        if op.kind not in ("MatMul", "Gemm", "Conv") and req.tile:
            raise ArchitectureError(op.name, "CAPABILITY", "Non-compute operator does not implement tiling")
        if op.kind == "Conv" and req.tile and any(req.tile[j] != shape[j] for j in (0, 4, 5, 6)):
            raise ArchitectureError(op.name, "CAPABILITY", "Conv currently tiles output-channel/height/width only")
        points.append(ArchitecturePoint(op.name, req, req, shape, "operator" if op.name in overrides else "model", op.kind))
    return ArchitecturalModel(model, tuple(points))


def coordinate(point: ArchitecturePoint, index: tuple[int, ...]) -> dict:
    """Portable iteration assignment mirrored by the MatMul/Conv HLS loops.

    Reduction coordinates select SIMD lanes. Output-channel work selects
    coarse replica, PE, and fold within a tile. No cycle-level II claim.
    """
    shape = point.iteration_shape
    if len(index) != len(shape) or any(i < 0 or i >= n for i, n in zip(index, shape)):
        raise IndexError(index)
    a = point.effective
    tile = tuple(i // t for i, t in zip(index, a.tile)) if a.tile else (0,)
    if point.kind in ("MatMul", "Gemm", "Conv"):
        out_axis, reduce_axis = (1, 2) if point.kind in ("MatMul", "Gemm") else (1, 4)
        block = a.tile[out_axis] if a.tile else shape[out_axis]
        local_output = index[out_axis] % block
        width = a.parallelism * a.pes
        output_group = local_output // (width * a.serialization)
        fold = (local_output // width) % a.serialization
        pe = local_output % a.pes
        replica = local_output // a.pes % a.parallelism
        # Source reduction loops restart SIMD lanes at every reduction tile.
        # A global modulo silently disagrees with HLS for e.g. Tk=2, SIMD=4.
        reduce_tile = a.tile[reduce_axis] if a.tile else shape[reduce_axis]
        lane = (index[reduce_axis] % reduce_tile) % a.simd
        if point.kind in ("MatMul", "Gemm"):
            tile_m = a.tile[0] if a.tile else shape[0]
            tile_k = a.tile[2] if a.tile else shape[2]
            reuse_step = (index[0] % tile_m, output_group, fold,
                          (index[2] % tile_k) // a.simd)
        else:
            th = a.tile[2] if a.tile else shape[2]
            tw = a.tile[3] if a.tile else shape[3]
            reuse_step = (index[0], index[2] % th, index[3] % tw,
                          output_group, fold, index[4] // a.simd,
                          index[5], index[6])
    else:
        replica = pe = lane = 0
        reuse_step = tuple(index)
    return {"precision": a.precision_bits, "tile": tile, "replica": replica,
            "pe": pe, "lane": lane, "reuse_step": reuse_step,
            "pipeline_ii_request": a.pipeline_ii}
