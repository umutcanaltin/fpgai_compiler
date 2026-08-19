from __future__ import annotations

from fpgai.layers.registry import layer_registry

from .operator_contract import OperatorCapabilities, OperatorContract, OperatorEntrypoints
from .operator_schema import AttributeContract, OnnxBinding, TensorPortContract

_ALIAS_TARGETS = {
    "Linear": "Dense",
    "Conv2D": "Conv",
    "DepthwiseConv2D": "Conv",
    "PointwiseConv2D": "Conv",
    "AveragePool": "AvgPool",
    "BatchNorm": "BatchNormalization",
}

_ONNX_BINDINGS = {
    "Dense": (OnnxBinding("ai.onnx", "Gemm"), OnnxBinding("ai.onnx", "MatMul")),
    "Conv": (OnnxBinding("ai.onnx", "Conv"),),
    "MaxPool": (OnnxBinding("ai.onnx", "MaxPool"),),
    "AvgPool": (OnnxBinding("ai.onnx", "AveragePool"),),
    "GlobalAveragePool": (OnnxBinding("ai.onnx", "GlobalAveragePool"),),
    "BatchNormalization": (OnnxBinding("ai.onnx", "BatchNormalization"),),
    "Relu": (OnnxBinding("ai.onnx", "Relu"),),
    "LeakyRelu": (OnnxBinding("ai.onnx", "LeakyRelu"),),
    "Sigmoid": (OnnxBinding("ai.onnx", "Sigmoid"),),
    "SiLU": (),
    "Softmax": (OnnxBinding("ai.onnx", "Softmax"),),
    "Flatten": (OnnxBinding("ai.onnx", "Flatten"),),
    "Reshape": (OnnxBinding("ai.onnx", "Reshape"),),
    "Add": (OnnxBinding("ai.onnx", "Add"),),
    "Mul": (OnnxBinding("ai.onnx", "Mul"),),
    "MatMul": (OnnxBinding("ai.onnx", "MatMul"),),
    "Transpose": (OnnxBinding("ai.onnx", "Transpose"),),
    "LayerNormalization": (OnnxBinding("ai.onnx", "LayerNormalization", opset_min=17),),
    "RMSNorm": (OnnxBinding("ai.onnx", "RMSNormalization", opset_min=23),),
    "GroupQueryAttention": (OnnxBinding("com.microsoft", "GroupQueryAttention"),),
    "Concat": (OnnxBinding("ai.onnx", "Concat"),),
    "Slice": (OnnxBinding("ai.onnx", "Slice"),),
    "Resize": (OnnxBinding("ai.onnx", "Resize"),),
    "Gather": (OnnxBinding("ai.onnx", "Gather"),),
    "Identity": (OnnxBinding("ai.onnx", "Identity"),),
    "Cast": (OnnxBinding("ai.onnx", "Cast"),),
    "Squeeze": (OnnxBinding("ai.onnx", "Squeeze"),),
    "Unsqueeze": (OnnxBinding("ai.onnx", "Unsqueeze"),),
    "Sub": (OnnxBinding("ai.onnx", "Sub"),),
    "Div": (OnnxBinding("ai.onnx", "Div"),),
    "Sqrt": (OnnxBinding("ai.onnx", "Sqrt"),),
    "Pow": (OnnxBinding("ai.onnx", "Pow"),),
    "ReduceMean": (OnnxBinding("ai.onnx", "ReduceMean"),),
    "ReduceSum": (OnnxBinding("ai.onnx", "ReduceSum"),),
}

_ATTRIBUTES = {
    "Conv": (
        AttributeContract("strides", "integer_list", default=[1, 1]),
        AttributeContract("pads", "integer_list", default=[0, 0, 0, 0]),
        AttributeContract("dilations", "integer_list", default=[1, 1]),
        AttributeContract("group", "integer", default=1),
    ),
    "MaxPool": (AttributeContract("kernel_shape", "integer_list", required=True),),
    "AvgPool": (AttributeContract("kernel_shape", "integer_list", required=True),),
    "LeakyRelu": (AttributeContract("alpha", "float", default=0.01),),
    "Softmax": (AttributeContract("axis", "integer", default=-1),),
    "Flatten": (AttributeContract("axis", "integer", default=1),),
    "Transpose": (AttributeContract("perm", "integer_list", required=False),),
    "LayerNormalization": (
        AttributeContract("axis", "integer", default=-1),
        AttributeContract("epsilon", "float", default=1e-5),
        AttributeContract("stash_type", "integer", default=1),
    ),
    "RMSNorm": (
        AttributeContract("axis", "integer", default=-1),
        AttributeContract("epsilon", "float", default=1e-5),
        AttributeContract("stash_type", "integer", default=1),
    ),
    "CausalMask": (
        AttributeContract("diagonal", "integer", default=0),
        AttributeContract("masked_value", "float", default=-32.0),
    ),
    "RotaryEmbedding": (
        AttributeContract("rotary_dim", "integer", required=True),
        AttributeContract("interleaved", "boolean", default=False),
        AttributeContract("position_offset", "integer", default=0),
    ),
    "MultiHeadAttention": (
        AttributeContract("num_heads", "integer", required=True),
        AttributeContract("num_kv_heads", "integer", required=False),
        AttributeContract("causal", "boolean", default=True),
        AttributeContract("scale", "float", required=False),
        AttributeContract("execution_mode", "string", default="auto"),
    ),
    "GroupQueryAttention": (
        AttributeContract("num_heads", "integer", required=True),
        AttributeContract("num_kv_heads", "integer", required=True),
        AttributeContract("scale", "float", required=False),
        AttributeContract("do_rotary", "boolean", default=False),
        AttributeContract("rotary_interleaved", "boolean", default=False),
        AttributeContract("causal", "boolean", default=True),
        AttributeContract("execution_mode", "string", default="serialized"),
    ),
    "KVCacheUpdate": (
        AttributeContract("sequence_axis", "integer", default=-2),
        AttributeContract("capacity", "integer", required=True),
        AttributeContract("update_policy", "string", default="append"),
    ),
    "TransformerBlock": (
        AttributeContract("num_heads", "integer", required=True),
        AttributeContract("causal", "boolean", default=True),
        AttributeContract("epsilon", "float", default=1e-5),
        AttributeContract("execution_mode", "string", default="auto"),
        AttributeContract("position_offset", "integer", default=0),
    ),
    "GatedMLP": (
        AttributeContract("activation", "string", default="silu"),
    ),
    "Concat": (AttributeContract("axis", "integer", required=True),),
    "Split": (
        AttributeContract("axis", "integer", default=0),
        AttributeContract("split", "integer_list", required=False),
    ),
    "Slice": (
        AttributeContract("starts", "integer_list", required=False),
        AttributeContract("ends", "integer_list", required=False),
        AttributeContract("axes", "integer_list", required=False),
        AttributeContract("steps", "integer_list", required=False),
    ),
    "Resize": (
        AttributeContract("mode", "string", default="nearest"),
        AttributeContract("coordinate_transformation_mode", "string", default="asymmetric"),
        AttributeContract("nearest_mode", "string", default="floor"),
    ),
    "Gather": (AttributeContract("axis", "integer", default=0),),
    "ReduceMean": (
        AttributeContract("axes", "integer_list", required=False),
        AttributeContract("keepdims", "integer", default=1),
    ),
    "ReduceSum": (
        AttributeContract("axes", "integer_list", required=False),
        AttributeContract("keepdims", "integer", default=1),
    ),
}


def _package_id(op_type: str) -> str:
    return "fpgai.operator." + op_type.lower().replace("_", "")


def builtin_operator_contracts(*, pipeline_mode: str = "inference") -> tuple[OperatorContract, ...]:
    registry = layer_registry(pipeline_mode=pipeline_mode)
    contracts: list[OperatorContract] = []
    for op_type, capability in sorted(registry.items()):
        canonical = _ALIAS_TARGETS.get(op_type, op_type)
        training_supported = bool(capability["training"]["supported"])
        aliases = tuple(alias for alias, target in _ALIAS_TARGETS.items() if target == op_type)
        contracts.append(
            OperatorContract(
                operator_id=_package_id(op_type),
                canonical_op_type=canonical,
                version=1,
                category=str(capability["category"]),
                inputs=(TensorPortContract("input", description="Primary operator input"),),
                outputs=(TensorPortContract("output", description="Primary operator output"),),
                attributes=_ATTRIBUTES.get(canonical, ()),
                onnx_bindings=_ONNX_BINDINGS.get(canonical, ()),
                aliases=aliases,
                capabilities=OperatorCapabilities(
                    inference=bool(capability["inference"]["supported"]),
                    training_forward=training_supported,
                    backward_input=training_supported,
                    parameter_gradients=training_supported and bool(capability["has_weights"]),
                    bias_gradients=training_supported and bool(capability["has_weights"]),
                    shape_inference=True,
                    type_inference=True,
                    numeric_reference=True,
                    canonicalization=canonical in {"Dense"},
                    resource_estimation=True,
                ),
                entrypoints=OperatorEntrypoints(
                    shape_inference="builtin:fpgai.ir.passes.infer_shapes",
                    type_inference="builtin:fpgai.ir.types",
                    legality="builtin:fpgai.ir.passes.validate",
                    numeric_reference="builtin:fpgai.benchmark",
                    canonicalization="builtin:fpgai.frontend.onnx.canonicalize" if canonical == "Dense" else None,
                    training_reference="builtin:fpgai.benchmark.training_reference" if training_supported else None,
                    gradient="builtin:fpgai.benchmark.training_reference" if training_supported else None,
                    resource_estimator="builtin:fpgai.analysis.resource_estimator",
                ),
                implementation_requirements={
                    "has_weights": bool(capability["has_weights"]),
                    "has_activation_output": bool(capability["has_activation_output"]),
                    "knobs": capability["knobs"],
                },
                notes=(str(capability["inference"]["detail"]), str(capability["training"]["detail"])),
            )
        )
    return tuple(contracts)


def get_builtin_operator_contract(op_type: str, *, pipeline_mode: str = "inference") -> OperatorContract:
    for contract in builtin_operator_contracts(pipeline_mode=pipeline_mode):
        if contract.canonical_op_type == op_type or op_type in contract.aliases or contract.operator_id == op_type:
            return contract
    raise KeyError(op_type)
