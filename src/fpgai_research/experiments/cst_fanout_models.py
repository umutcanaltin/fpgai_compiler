"""Bounded 2..4 MatMul-reader FunctionalModel fixtures for CST R2f.

Exact topological producer -> R independent consumers -> R-1 Add chain
-> Relu with unique constants and deterministic original float payloads.
These are source/Model IR E0 research fixtures, not measured hardware
or trained neural networks.
"""
from __future__ import annotations

from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel


def sample_fanout(readers:int=2,*,m:int=2,k:int=2,n:int=3,q:int=4)->FunctionalModel:
    if (type(readers) is not int or readers not in (2,3,4)
            or any(type(v) is not int or not 1<=v<=16 for v in (m,k,n,q))):
        raise ValueError("R2f admits only static 2..4-reader small fanout")
    def weight(rows,cols,scale):
        return tuple(((-1)**i)*(i+1)*scale for i in range(rows*cols))
    constants=[
        Tensor("wp",(k,n),data=weight(k,n,0.08)),
    ]
    constants.extend(
        Tensor(f"wc{i}",(n,q),data=weight(n,q,0.025+0.01*i))
        for i in range(readers)
    )
    intermediates=[Tensor("h",(m,n))]
    intermediates.extend(Tensor(f"a{i}",(m,q)) for i in range(readers))
    intermediates.extend(Tensor(f"g{i}",(m,q)) for i in range(readers-1))
    producer=Operator("producer","MatMul",("x","wp"),("h",))
    reader_ops=[
        Operator(f"reader{i}","MatMul",("h",f"wc{i}"),(f"a{i}",))
        for i in range(readers)
    ]
    previous="a0"
    adds=[]
    for i in range(readers-1):
        out=f"g{i}"
        adds.append(Operator(f"merge{i}","Add",
                             (previous,f"a{i+1}"),(out,)))
        previous=out
    final=Operator("final","Relu",(previous,),("out",))
    return FunctionalModel(
        tensors=(Tensor("x",(m,k)),*constants,
                 *intermediates,Tensor("out",(m,q))),
        operators=(producer,*reader_ops,*adds,final),
        inputs=("x",),outputs=("out",),
    )
