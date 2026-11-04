"""Prespecified mixed NCHW graph CST research fixtures.

Not random networks: keep every graph/shape/mapping in the evidence
denominator. Conv kernels are immutable, explicit, nontrained examples.
"""
from __future__ import annotations
from fpgai_research.model_ir.contracts import FunctionalModel, Operator, Tensor


def nchw_case(kind="branch_concat",*,height=4,width=4,channels=2):
    if (kind not in ("branch_concat","pool_fork","broadcast_fork") or
            any(type(x) is not int or not 2<=x<=8
                for x in (height,width,channels))):
        raise ValueError("Bounded NCHW graph preset required")
    if kind=="pool_fork" and (height%2 or width%2):
        raise ValueError("2x2 pool fork needs even spatial dimensions")
    def weight(name,oc,ic,kh,kw,scale):
        return Tensor(name,(oc,ic,kh,kw),data=tuple(
            ((-1)**i)*(i+1)*scale for i in range(oc*ic*kh*kw)))
    def out(name,c=channels,h=height,w=width):
        return Tensor(name,(1,c,h,w))
    tensors=[
        Tensor("x",(1,1,height,width)),
        weight("w0",channels,1,3,3,.00625),
        out("h"),
    ]
    ops=[Operator("producer","Conv",("x","w0"),("h",),
         attributes=(("pads",(1,1,1,1)),("strides",(1,1))))]
    if kind=="branch_concat":
        tensors.extend([
            out("a"),weight("w1",channels,channels,1,1,.0125),out("b"),
            out("cat",2*channels),
            weight("w2",channels,2*channels,1,1,.005),
            out("reduced"),out("merged"),out("out"),
        ])
        ops.extend([
            Operator("relu_branch","Relu",("h",),("a",)),
            Operator("conv_branch","Conv",("h","w1"),("b",)),
            Operator("concat","Concat",("a","b"),("cat",),attributes=(("axis",1),)),
            Operator("conv_rejoin","Conv",("cat","w2"),("reduced",)),
            Operator("merge","Add",("reduced","h"),("merged",)),
            Operator("last","Relu",("merged",),("out",)),
        ])
    elif kind=="pool_fork":
        sh,sw=height//2,width//2
        tensors.extend([
            out("pooled",channels,sh,sw),
            weight("w1",channels,channels,2,2,.01),
            out("down",channels,sh,sw),
            out("merged",channels,sh,sw),
            out("out",channels,sh,sw),
        ])
        ops.extend([
            Operator("pool_branch","MaxPool",("h",),("pooled",),
                     attributes=(("kernel_shape",(2,2)),("strides",(2,2)))),
            Operator("conv_branch","Conv",("h","w1"),("down",),
                     attributes=(("strides",(2,2)),)),
            Operator("merge","Add",("pooled","down"),("merged",)),
            Operator("last","Relu",("merged",),("out",)),
        ])
    else:
        tensors.extend([
            weight("w1",channels,channels,height,width,.001),
            out("narrow",channels,1,1),
            out("broadcast"),
            weight("w2",channels,channels,1,1,.0075),
            out("full"),out("merged"),out("out"),
        ])
        ops.extend([
            Operator("narrow_conv","Conv",("h","w1"),("narrow",)),
            Operator("broadcast_add","Add",("h","narrow"),("broadcast",)),
            Operator("full_conv","Conv",("h","w2"),("full",)),
            Operator("merge","Add",("broadcast","full"),("merged",)),
            Operator("last","Relu",("merged",),("out",)),
        ])
    return FunctionalModel(tuple(tensors),tuple(ops),inputs=("x",),outputs=("out",))
