"""R2z prespecified mixed CNN/MLP source-network fixtures.

These are deterministic, original nontrained numeric probes and
architecture-local requests. The existing FunctionalModel + Conv2D/
Gemm/MatMul emitters carry the functional semantics. Trained ONNX
accuracy and native Vitis/Vivado results require separate experiments.
"""
from __future__ import annotations

from dataclasses import replace
from math import prod

from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)

KINDS=("cnn_pool_dense","cnn_strided_dense","cnn_two_conv_softmax")


def _weights(name:str,shape:tuple[int,...],seed:int)->Tensor:
    return Tensor(name,shape,data=tuple(
        (((i*7+seed)%13)-6)*0.00625 for i in range(prod(shape))
    ))


def _bias(name:str,n:int,seed:int)->Tensor:
    return Tensor(name,(n,),data=tuple(
        ((i+seed)%7-3)*0.0125 for i in range(n)
    ))


def network_fixture(kind:str="cnn_pool_dense")->FunctionalModel:
    if kind not in KINDS:
        raise ValueError("R2z only the three prespecified network families")
    if kind=="cnn_pool_dense":
        tensors=(
            Tensor("x",(1,1,6,6)),
            _weights("w0",(2,1,3,3),1),_bias("b0",2,2),
            Tensor("c0",(1,2,6,6)),
            Tensor("r0",(1,2,6,6)),
            Tensor("p0",(1,2,3,3)),
            _weights("w1",(3,2,1,1),3),
            Tensor("c1",(1,3,3,3)),
            Tensor("flat",(1,27)),
            _weights("w2",(27,5),4),
            Tensor("d0",(1,5)),
            Tensor("r1",(1,5)),
            _weights("w3",(2,5),5),_bias("b3",2,6),
            Tensor("out",(1,2)),
        )
        ops=(
            Operator("conv0","Conv",("x","w0","b0"),("c0",),
                     (("pads",(1,1,1,1)),)),
            Operator("relu0","Relu",("c0",),("r0",)),
            Operator("pool0","MaxPool",("r0",),("p0",),
                     (("kernel_shape",(2,2)),("strides",(2,2)))),
            Operator("conv1","Conv",("p0","w1"),("c1",)),
            Operator("flatten","Flatten",("c1",),("flat",),(("axis",1),)),
            Operator("dense0","MatMul",("flat","w2"),("d0",)),
            Operator("relu1","Relu",("d0",),("r1",)),
            Operator("dense1","Gemm",("r1","w3","b3"),("out",),
                     (("transB",1),)),
        )
    elif kind=="cnn_strided_dense":
        tensors=(
            Tensor("x",(2,1,7,8)),
            _weights("w0",(3,1,3,3),4),
            Tensor("c0",(2,3,4,4)),
            Tensor("r0",(2,3,4,4)),
            Tensor("flat",(2,48)),
            _weights("w1",(48,4),6),_bias("b1",4,2),
            Tensor("d0",(2,4)),Tensor("r1",(2,4)),
            _weights("w2",(4,3),8),
            Tensor("out",(2,3)),
        )
        ops=(
            Operator("conv0","Conv",("x","w0"),("c0",),
                     (("strides",(2,2)),("pads",(1,1,1,1)))),
            Operator("relu0","Relu",("c0",),("r0",)),
            Operator("flatten","Flatten",("r0",),("flat",)),
            Operator("dense0","Gemm",("flat","w1","b1"),("d0",)),
            Operator("relu1","Relu",("d0",),("r1",)),
            Operator("dense1","MatMul",("r1","w2"),("out",)),
        )
    else:
        tensors=(
            Tensor("x",(1,2,5,7)),
            _weights("w0",(3,2,3,2),1),_bias("b0",3,3),
            Tensor("c0",(1,3,5,6)),
            _weights("w1",(2,3,2,3),4),
            Tensor("c1",(1,2,4,6)),
            Tensor("r0",(1,2,4,6)),
            Tensor("flat",(1,48)),
            _weights("w2",(3,48),6),_bias("b2",3,7),
            Tensor("d0",(1,3)),
            Tensor("out",(1,3)),
        )
        ops=(
            Operator("conv0","Conv",("x","w0","b0"),("c0",),
                     (("pads",(1,0,1,0)),)),
            Operator("conv1","Conv",("c0","w1"),("c1",),
                     (("pads",(0,1,0,1)),)),
            Operator("relu0","Relu",("c1",),("r0",)),
            Operator("flatten","Flatten",("r0",),("flat",)),
            Operator("dense0","Gemm",("flat","w2","b2"),("d0",),
                     (("transB",1),)),
            Operator("softmax","Softmax",("d0",),("out",)),
        )
    result=FunctionalModel(
        tensors=tensors,operators=ops,inputs=("x",),outputs=("out",))
    result.validate()
    return result


def network_architecture(kind:str,*,independent_phi:bool=True):
    model=network_fixture(kind)
    base=ArchitectureRequest()
    if not independent_phi:
        return resolve(model,base)
    overrides={}
    for op in model.operators:
        if op.kind=="Conv":
            shape=tuple(model.tensor(op.outputs[0]).shape)
            _,co,ho,wo=shape
            w=model.tensor(op.inputs[1]).shape
            overrides[op.name]=replace(
                base,pes=2,simd=2,serialization=2,
                tile=(shape[0],min(2,co),min(2,ho),min(3,wo),
                      w[1],w[2],w[3]),
            )
        elif op.kind in ("MatMul","Gemm"):
            a=model.tensor(op.outputs[0]).shape
            k=prod(model.tensor(op.inputs[0]).shape[1:])
            overrides[op.name]=replace(
                base,pes=2,simd=3,serialization=2,
                tile=(a[0],min(3,a[1]),k),
            )
    return resolve(model,base,overrides)
