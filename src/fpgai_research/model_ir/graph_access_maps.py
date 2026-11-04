"""Bounded rank-two per-output source access maps for graph CST edges.

Explicit map from each non-MatMul output index to the exact operand,
port and flat source index. Separate from the original C++ emitter:
the GCC tracer must check ACTUAL source-load indices against this map.
"""
from __future__ import annotations

from fpgai_research.model_ir.contracts import FunctionalModel, Operator


def scalar_output_accesses(model:FunctionalModel,op:Operator,out_index:int):
    """Ordered (port, tensor, flat source index) for exactly one output.

    Supports ReLU, rank-two ONNX broadcast Add, and rank-two axis-1 Concat.
    All extents are static/positive and validated by the graph admission.
    """
    if type(out_index) is not int:
        raise TypeError("Output element index must be an integer")
    rows,width=model.tensor(op.outputs[0]).shape
    if not 0<=out_index<rows*width:
        raise ValueError("Output index outside static scalar operation")
    row,col=divmod(out_index,width)
    if op.kind=="Relu":
        return ((0,op.inputs[0],out_index),)
    if op.kind=="Add":
        result=[]
        for port,name in enumerate(op.inputs):
            ir,ic=model.tensor(name).shape
            r=0 if ir==1 else row
            c=0 if ic==1 else col
            result.append((port,name,r*ic+c))
        return tuple(result)
    if op.kind=="Concat":
        begin=0
        for port,name in enumerate(op.inputs):
            ir,ic=model.tensor(name).shape
            if begin<=col<begin+ic:
                return ((port,name,row*ic+(col-begin)),)
            begin+=ic
        raise AssertionError("Concat output column has no source")
    raise ValueError("Unsupported scalar operator access-map semantics")


def validate_scalar_access_shape(model:FunctionalModel,op:Operator):
    """Fail-closed static shapes; do not silently support rank/axis changes."""
    if op.kind not in ("Relu","Add","Concat"):
        raise ValueError("Operator lacks scalar source map")
    out=model.tensor(op.outputs[0]).shape
    if len(out)!=2 or any(len(model.tensor(name).shape)!=2 for name in op.inputs):
        raise ValueError("Bounded scalar source map requires rank-two tensors")
    if op.kind=="Relu":
        if len(op.inputs)!=1 or model.tensor(op.inputs[0]).shape!=out:
            raise ValueError("Relu requires exact same-shape input")
    elif op.kind=="Add":
        if len(op.inputs)!=2:
            raise ValueError("Add requires exactly two inputs")
        for name in op.inputs:
            shape=model.tensor(name).shape
            if any(d not in (1,target) for d,target in zip(shape,out)):
                raise ValueError("Unsupported rank-two Add broadcast dimensions")
        if not any(max(model.tensor(name).shape[i] for name in op.inputs)==out[i]
                   for i in (0,1)):
            raise ValueError("Unqualified Add broadcast output")
        # This check alone must not accept (1,1) into arbitrary larger dims.
        if tuple(max(model.tensor(name).shape[i] for name in op.inputs)
                 for i in (0,1))!=out:
            raise ValueError("Add output is not exact broadcast result")
    else:
        if len(op.inputs)<2:
            raise ValueError("Concat requires two or more inputs")
        if op.attr("axis",1)!=1 or any(k!="axis" for k,_ in op.attributes):
            raise ValueError("Only rank-two axis=1 Concat is implemented")
        shapes=[model.tensor(name).shape for name in op.inputs]
        if any(shape[0]!=out[0] for shape in shapes) or sum(s[1] for s in shapes)!=out[1]:
            raise ValueError("Concat output shape differs from input-axis sum")
