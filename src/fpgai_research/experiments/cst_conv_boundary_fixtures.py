"""Deterministic static NCHW Conv fanout fixtures for CST source work."""
from __future__ import annotations
from fpgai_research.model_ir.contracts import FunctionalModel, Operator, Tensor


def conv_diamond(*,height=4,width=4,kernel=3,pad=1,stride=1,channels=2):
    for x in (height,width,kernel,channels,stride):
        if type(x) is not int or x<1 or x>8:
            raise ValueError("Bounded positive NCHW dimensions required")
    if type(pad) is not int or not 0<=pad<=3:
        raise ValueError("Bounded explicit convolution padding required")
    oh=(height+2*pad-kernel)//stride+1
    ow=(width+2*pad-kernel)//stride+1
    if min(oh,ow)<1:
        raise ValueError("Requested kernel has no valid output")
    x=Tensor("x",(1,1,height,width))
    w0=Tensor("w0",(channels,1,kernel,kernel),data=tuple(
        ((-1)**i)*(i+1)*.0125 for i in range(channels*kernel*kernel)))
    hidden=Tensor("h",(1,channels,oh,ow))
    a=Tensor("a",hidden.shape)
    w1=Tensor("w1",(channels,channels,1,1),data=tuple(
        ((-1)**i)*(i+1)*.03125 for i in range(channels*channels)))
    b=Tensor("b",hidden.shape)
    merge=Tensor("g",hidden.shape)
    out=Tensor("out",hidden.shape)
    ops=(
        Operator("producer","Conv",("x","w0"),("h",),
                 attributes=(("strides",(stride,stride)),
                             ("pads",(pad,pad,pad,pad)))),
        Operator("reader0","Relu",("h",),("a",)),
        Operator("reader1","Conv",("h","w1"),("b",)),
        Operator("merge","Add",("a","b"),("g",)),
        Operator("final","Relu",("g",),("out",)),
    )
    return FunctionalModel(tensors=(x,w0,hidden,a,w1,b,merge,out),
                           operators=ops,inputs=("x",),outputs=("out",))
