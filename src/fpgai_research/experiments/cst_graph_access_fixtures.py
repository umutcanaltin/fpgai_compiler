"""Deterministic nonuniform rank-two graph access-map CST fixtures.

A width-changing Concat edge and a repeated-read broadcast Add edge
exercise per-input source-index maps; all coefficients are explicit and
all MatMul operators choose independent mappings in experiments.
"""
from __future__ import annotations

from fpgai_research.model_ir.contracts import FunctionalModel,Operator,Tensor


def access_graph(kind:str="broadcast",*,m:int=2,n:int=3,q:int=2):
    if (kind not in ("broadcast","concat") or
            any(type(v) is not int or not 1<=v<=8 for v in (m,n,q))):
        raise ValueError("Bounded rank-two access-map graph required")
    def matrix(name,rows,cols,scale):
        return Tensor(name,(rows,cols),data=tuple(
            ((-1)**i)*(i+1)*scale for i in range(rows*cols)))
    tensors=[Tensor("x",(m,n)),matrix("wp",n,n,.0625),Tensor("h",(m,n))]
    ops=[Operator("producer","MatMul",("x","wp"),("h",))]
    if kind=="broadcast":
        tensors.extend([
            matrix("wn",n,1,.035),Tensor("narrow",(m,1)),
            Tensor("merged",(m,n)),Tensor("out",(m,n)),
        ])
        ops.extend([
            Operator("reader","MatMul",("h","wn"),("narrow",)),
            Operator("broadcast_add","Add",("h","narrow"),("merged",)),
            Operator("final","Relu",("merged",),("out",)),
        ])
    else:
        tensors.extend([
            matrix("wq",n,q,.0375),Tensor("a",(m,n)),
            Tensor("b",(m,q)),Tensor("cat",(m,n+q)),
            matrix("wc",n+q,n,.02),Tensor("reduced",(m,n)),
            Tensor("out",(m,n)),
        ])
        ops.extend([
            Operator("first","Relu",("h",),("a",)),
            Operator("second","MatMul",("h","wq"),("b",)),
            Operator("combine","Concat",("a","b"),("cat",),
                     attributes=(("axis",1),)),
            Operator("reader","MatMul",("cat","wc"),("reduced",)),
            Operator("final","Relu",("reduced",),("out",)),
        ])
    return FunctionalModel(tensors=tuple(tensors),operators=tuple(ops),
                           inputs=("x",),outputs=("out",))
