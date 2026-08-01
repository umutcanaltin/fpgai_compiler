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
    "Softmax": (OnnxBinding("ai.onnx", "Softmax"),),
    "Flatten": (OnnxBinding("ai.onnx", "Flatten"),),
    "Reshape": (OnnxBinding("ai.onnx", "Reshape"),),
    "Add": (OnnxBinding("ai.onnx", "Add"),),
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
