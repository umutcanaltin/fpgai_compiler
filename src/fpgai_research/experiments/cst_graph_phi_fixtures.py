"""Prespecified bounded DAGs for graph-wide CST Phi source research.

These are deterministic, static and suitable for source-executed E0/E1
causality, not pretrained networks or production hardware benchmarks.
"""
from __future__ import annotations

from fpgai_research.model_ir.contracts import FunctionalModel,Operator,Tensor


def sample_graph(kind:str="nested",*,m:int=2,n:int=3):
    if type(m) is not int or type(n) is not int or not 1<=m<=8 or not 2<=n<=8:
        raise ValueError("Bounded static graph shape required")
    if kind not in ("mixed","triple_reader","nested"):
        raise ValueError("Unknown prespecified DAG shape")
    tensors=[
        Tensor("x",(m,n)),
        Tensor("w0",(n,n),data=tuple(((-1)**i)*(i+1)/80
                                     for i in range(n*n))),
    ]
    ops=[Operator("producer","MatMul",("x","w0"),("h",))]
    tensors.append(Tensor("h",(m,n)))
    def weight(label,scale):
        tensors.append(Tensor(label,(n,n),data=tuple(
            ((-1)**i)*(i+1)*scale for i in range(n*n))))
    def output(label):
        tensors.append(Tensor(label,(m,n)))
    if kind=="mixed":
        weight("w1",.025)
        output("a");output("b");output("c");output("d");output("out")
        ops+= [
            Operator("reader0","Relu",("h",),("a",)),
            Operator("reader1","MatMul",("h","w1"),("b",)),
            Operator("reader2","Relu",("b",),("c",)),
            Operator("merge","Add",("a","c"),("d",)),
            Operator("final","Relu",("d",),("out",)),
        ]
    elif kind=="triple_reader":
        weight("w1",.025)
        for name in ("a","b","c","d","e","out"):
            output(name)
        ops+= [
            Operator("reader0","MatMul",("h","w1"),("a",)),
            Operator("reader1","Relu",("h",),("b",)),
            Operator("reader2","Relu",("h",),("c",)),
            Operator("merge0","Add",("a","b"),("d",)),
            Operator("merge1","Add",("d","c"),("e",)),
            Operator("final","Relu",("e",),("out",)),
        ]
    else:
        weight("w1",.025);weight("w2",.045);weight("w3",.02)
        for name in ("a","b","g","c","d","e","out"):
            output(name)
        ops+= [
            Operator("reader0","MatMul",("h","w1"),("a",)),
            Operator("reader1","MatMul",("h","w2"),("b",)),
            Operator("merge0","Add",("a","b"),("g",)),
            Operator("reader2","MatMul",("g","w3"),("c",)),
            Operator("reader3","Relu",("g",),("d",)),
            Operator("merge1","Add",("c","d"),("e",)),
            Operator("final","Relu",("e",),("out",)),
        ]
    return FunctionalModel(tensors=tuple(tensors),operators=tuple(ops),
                           inputs=("x",),outputs=("out",))
