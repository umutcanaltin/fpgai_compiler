from __future__ import annotations
import numpy as np
from fpgai.operators import (
    AttributeContract, OnnxBinding, OperatorCapabilities, OperatorContract,
    OperatorEntrypoints, TensorPortContract,
)
from fpgai.operators.external import (
    ExternalOperatorDefinition, ImportedOperator, ReferenceExecutionResult,
    ShapeInferenceResult, TypeInferenceResult,
)

def _import(ctx):
    attrs={"scale":float(ctx.attributes.get("scale",1.0)),"bias":float(ctx.attributes.get("bias",0.0))}
    return ImportedOperator("ScaleBias",ctx.inputs,ctx.outputs,attrs)

def _shape(ctx): return ShapeInferenceResult((ctx.inputs[0].shape,))
def _type(ctx): return TypeInferenceResult((ctx.input_dtypes[0],))
def _reference(ctx):
    x=np.asarray(ctx.inputs[0]); return ReferenceExecutionResult((x*float(ctx.attributes.get("scale",1.0))+float(ctx.attributes.get("bias",0.0)),))

def define_operator():
    contract=OperatorContract(
        operator_id="community.operator.scale_bias", canonical_op_type="ScaleBias", version=1, category="elementwise",
        inputs=(TensorPortContract("input"),), outputs=(TensorPortContract("output"),),
        attributes=(AttributeContract("scale","float",False,1.0),AttributeContract("bias","float",False,0.0)),
        onnx_bindings=(OnnxBinding("community.fpgai","ScaleBias",1,None),),
        capabilities=OperatorCapabilities(inference=True,shape_inference=True,type_inference=True,numeric_reference=True),
        entrypoints=OperatorEntrypoints(shape_inference="_shape",type_inference="_type",numeric_reference="_reference"),
    )
    return ExternalOperatorDefinition(1,contract,_import,_shape,_type,_reference)
