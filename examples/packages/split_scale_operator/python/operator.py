from __future__ import annotations

import numpy as np

from fpgai.operators import (
    AttributeContract,
    OnnxBinding,
    OperatorCapabilities,
    OperatorContract,
    OperatorEntrypoints,
    TensorPortContract,
)
from fpgai.operators.external import (
    ExternalOperatorDefinition,
    ImportedOperator,
    ReferenceExecutionResult,
    ShapeInferenceResult,
    TypeInferenceResult,
)


def _import(ctx):
    return ImportedOperator(
        "SplitScale",
        ctx.inputs,
        ctx.outputs,
        {"scale": float(ctx.attributes.get("scale", 2.0))},
    )


def _shape(ctx):
    shape = ctx.inputs[0].shape
    return ShapeInferenceResult((shape, shape))


def _type(ctx):
    dtype = ctx.input_dtypes[0]
    return TypeInferenceResult((dtype, dtype))


def _reference(ctx):
    x = np.asarray(ctx.inputs[0], dtype=np.float32)
    scale = float(ctx.attributes.get("scale", 2.0))
    return ReferenceExecutionResult((x.copy(), x * scale))


def define_operator():
    contract = OperatorContract(
        operator_id="community.operator.split_scale",
        canonical_op_type="SplitScale",
        version=1,
        category="elementwise",
        inputs=(TensorPortContract("input"),),
        outputs=(TensorPortContract("identity"), TensorPortContract("scaled")),
        attributes=(AttributeContract("scale", "float", False, 2.0),),
        onnx_bindings=(OnnxBinding("community.fpgai", "SplitScale", 1, None),),
        capabilities=OperatorCapabilities(
            inference=True,
            shape_inference=True,
            type_inference=True,
            numeric_reference=True,
        ),
        entrypoints=OperatorEntrypoints(
            shape_inference="_shape",
            type_inference="_type",
            numeric_reference="_reference",
        ),
        notes=("Maintained external multi-output validation operator.",),
    )
    return ExternalOperatorDefinition(1, contract, _import, _shape, _type, _reference)
