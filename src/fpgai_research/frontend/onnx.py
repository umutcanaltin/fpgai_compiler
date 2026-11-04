"""Strict static ONNX import. No fake support or silent operator substitution."""
from __future__ import annotations
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel
from fpgai_research.frontend.attributes import canonical_attributes


def load_onnx(path: str) -> FunctionalModel:
    try:
        import onnx
        from onnx import numpy_helper, shape_inference, TensorProto
    except ImportError as exc:
        raise RuntimeError("Install the optional ONNX dependencies: pip install -e '.[onnx]'") from exc
    proto = shape_inference.infer_shapes(onnx.load(path))
    onnx.checker.check_model(proto)
    if len(proto.graph.sparse_initializer):
        raise ValueError("Sparse initializers are not supported")
    graph = proto.graph
    # Shape operands are compile-time metadata, not float32 runtime tensors.
    static_shapes = {}
    constant_data = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.INT64:
            if arr.ndim != 1 or not 1 <= arr.size <= 8:
                raise ValueError(f"Only bounded rank-1 int64 shape vectors are supported: {init.name}")
            static_shapes[init.name] = tuple(int(v) for v in arr.tolist())
        elif init.data_type == TensorProto.FLOAT:
            constant_data[init.name] = arr
        else:
            raise ValueError(f"Unsupported ONNX initializer type: {init.name}")
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        if node.domain not in ("", "ai.onnx") or len(node.input) or len(node.output) != 1:
            raise ValueError("Only standard no-input single-output ONNX Constant nodes supported")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if set(attrs) != {"value"}:
            raise ValueError("Only tensor-valued ONNX Constant nodes supported")
        tensor = attrs["value"]
        arr = numpy_helper.to_array(tensor)
        name = node.output[0]
        if not name or name in static_shapes or name in constant_data:
            raise ValueError("Duplicate or empty ONNX Constant identity")
        if tensor.data_type == TensorProto.INT64:
            if arr.ndim != 1 or not 1 <= arr.size <= 8:
                raise ValueError("Only bounded int64 Constant shape vectors supported")
            static_shapes[name] = tuple(int(v) for v in arr.tolist())
        elif tensor.data_type == TensorProto.FLOAT:
            if any(d < 1 for d in arr.shape):
                raise ValueError("Empty Constant data is not supported")
            constant_data[name] = arr
        else:
            raise ValueError("Unsupported ONNX Constant tensor dtype")
    tensors = {}
    for info in list(graph.input) + list(graph.value_info) + list(graph.output):
        value = info.type.tensor_type
        if info.name in static_shapes:
            if value.elem_type != TensorProto.INT64:
                raise ValueError(f"Compile-time shape tensor has inconsistent type: {info.name}")
            continue
        if value.elem_type != TensorProto.FLOAT:
            raise ValueError(f"Only float32 runtime tensors/operators supported: {info.name}; dynamic shapes require static int64 constants")
        shape = []
        for dim in value.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value < 1:
                raise ValueError(f"Static positive dimensions required: {info.name}")
            shape.append(int(dim.dim_value))
        tensors[info.name] = Tensor(info.name, tuple(shape))
    import numpy as np
    for name, arr in constant_data.items():
        if any(d < 1 for d in arr.shape) or not np.all(np.isfinite(arr)):
            raise ValueError(f"Nonfinite or empty ONNX constant: {name}")
        tensors[name] = Tensor(name, tuple(int(x) for x in arr.shape), data=tuple(float(x) for x in arr.flat))
    ops = []
    allowed = {"MatMul", "Gemm", "Relu", "Identity", "Add", "Flatten", "Conv", "MaxPool", "Softmax", "Mul", "Sigmoid", "Transpose", "GlobalAveragePool", "BatchNormalization", "Concat", "LeakyRelu", "Reshape", "Squeeze", "Unsqueeze"}
    for index, node in enumerate(graph.node):
        if node.op_type == "Constant":
            continue
        if node.domain not in ("", "ai.onnx") or node.op_type not in allowed:
            raise ValueError(f"Unsupported ONNX operator: {node.domain}:{node.op_type}")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Reshape":
            from fpgai_research.frontend.static_reshape import resolve_static_reshape
            if len(node.input) != 2 or not node.input[0] or node.input[1] not in static_shapes:
                raise ValueError("Reshape requires float input and static int64 shape initializer/Constant")
            if set(attrs) - {"allowzero"} or int(attrs.get("allowzero",0)) != 0:
                raise ValueError("Only Reshape allowzero=0 semantics supported")
            desired = resolve_static_reshape(tensors[node.input[0]].shape, static_shapes[node.input[1]])
            if tensors[node.output[0]].shape != desired:
                raise ValueError("ONNX inferred Reshape geometry disagrees with its static shape")
            attrs = {"shape": tuple(static_shapes[node.input[1]])}
        elif node.op_type in ("Squeeze", "Unsqueeze"):
            from fpgai_research.frontend.static_axes import squeeze_shape, unsqueeze_shape
            if len(node.input) == 2:
                if node.input[1] not in static_shapes or set(attrs):
                    raise ValueError("Shape axes must be a compile-time int64 vector")
                axes = tuple(static_shapes[node.input[1]])
            elif len(node.input) == 1:
                if set(attrs) - {"axes"}:
                    raise ValueError("Unsupported static axes attributes")
                axes = tuple(attrs["axes"]) if "axes" in attrs else None
            else:
                raise ValueError("Squeeze/Unsqueeze need one input and optional static axes")
            if node.op_type == "Unsqueeze" and axes is None:
                raise ValueError("Unsqueeze requires static axes")
            shape = tensors[node.input[0]].shape
            expected = squeeze_shape(shape, axes) if node.op_type == "Squeeze" else unsqueeze_shape(shape, axes)
            if tensors[node.output[0]].shape != expected:
                raise ValueError("Squeeze/Unsqueeze shape does not match ONNX inference")
            attrs = {} if axes is None else {"axes": tuple(axes)}
        elif node.op_type == "Gemm":
            if any(k not in {"alpha", "beta", "transA", "transB"} for k in attrs):
                raise ValueError("Unsupported Gemm attribute")
            if (float(attrs.get("alpha", 1)) != 1.0 or float(attrs.get("beta", 1)) != 1.0
                    or int(attrs.get("transA", 0)) != 0 or int(attrs.get("transB", 0)) not in (0, 1)):
                raise ValueError("Research Gemm supports alpha=beta=1, transA=0, transB=0/1")
        elif node.op_type == "Flatten":
            if any(k != "axis" for k in attrs) or int(attrs.get("axis", 1)) != 1:
                raise ValueError("Flatten supports axis=1 only")
        elif node.op_type == "Conv":
            attrs = canonical_attributes("Conv", attrs)
        elif node.op_type == "MaxPool":
            attrs = canonical_attributes("MaxPool", attrs)
        elif node.op_type == "BatchNormalization":
            if set(attrs) - {"epsilon", "momentum", "training_mode"}:
                raise ValueError("Unsupported BatchNormalization attribute")
            if int(attrs.get("training_mode", 0)) != 0:
                raise ValueError("Training BatchNormalization is unsupported")
            attrs.pop("training_mode", None)
            attrs.pop("momentum", None)
            eps = float(attrs.get("epsilon", 1e-5))
            if not (0.0 < eps < float("inf")):
                raise ValueError("Invalid BatchNormalization epsilon")
            attrs = {"epsilon": eps}
        elif node.op_type == "Concat":
            if set(attrs) != {"axis"} or int(attrs["axis"]) != 1:
                raise ValueError("Only axis=1 Concat is supported")
            attrs = {"axis": 1}
        elif node.op_type == "LeakyRelu":
            if set(attrs) - {"alpha"}:
                raise ValueError("Unsupported LeakyRelu attribute")
            alpha = float(attrs.get("alpha", 0.01))
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("LeakyRelu alpha must be within [0,1]")
            attrs = {"alpha": alpha}
        elif node.op_type == "Transpose":
            if any(k != "perm" for k in attrs):
                raise ValueError("Unsupported Transpose attribute")
            if "perm" in attrs:
                attrs["perm"] = tuple(attrs["perm"])
        elif node.op_type == "Softmax":
            if any(k != "axis" for k in attrs):
                raise ValueError("Unsupported Softmax attribute")
        elif attrs:
            raise ValueError(f"Unsupported attributes on {node.op_type}")
        runtime_inputs = (node.input[0],) if node.op_type in ("Reshape", "Squeeze", "Unsqueeze") else tuple(node.input)
        ops.append(Operator(node.name or f"node_{index}", node.op_type,
                            runtime_inputs, tuple(node.output), tuple(sorted(attrs.items()))))
    constants = set(constant_data) | set(static_shapes)
    if any(name in static_shapes for name in (x.name for x in graph.output)):
        raise ValueError("Integer shape operands cannot be model outputs")
    result = FunctionalModel(tuple(tensors.values()), tuple(ops),
                             tuple(x.name for x in graph.input if x.name not in constants),
                             tuple(x.name for x in graph.output))
    result.validate()
    return result
