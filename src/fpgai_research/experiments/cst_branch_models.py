"""Deterministic bounded static two-MatMul-reader Functional graph fixture.

Used by source-pinned campaigns and independent mutation/numerical tests.
No hidden benchmark assets, model training, RL or vendor toolchain required.
"""
from __future__ import annotations

from fpgai_research.model_ir.contracts import FunctionalModel, Operator, Tensor


def sample_branch(m=2,k=2,n=3,q=2):
    if any(type(v) is not int or v<1 or v>16 for v in (m,k,n,q)):
        raise ValueError("Bounded static dimensions required")
    def weights(rows,columns,scale):
        return tuple(((-1)**i)*(i+1)*scale for i in range(rows*columns))
    return FunctionalModel(
        tensors=(Tensor("x",(m,k)),
                 Tensor("wp",(k,n),data=weights(k,n,0.08)),
                 Tensor("wc0",(n,q),data=weights(n,q,0.035)),
                 Tensor("wc1",(n,q),data=weights(n,q,0.045)),
                 Tensor("h",(m,n)),Tensor("a",(m,q)),
                 Tensor("b",(m,q)),Tensor("g",(m,q)),
                 Tensor("out",(m,q))),
        operators=(
            Operator("producer","MatMul",("x","wp"),("h",)),
            Operator("reader0","MatMul",("h","wc0"),("a",)),
            Operator("reader1","MatMul",("h","wc1"),("b",)),
            Operator("merge","Add",("a","b"),("g",)),
            Operator("final","Relu",("g",),("out",)),
        ),inputs=("x",),outputs=("out",),
    )
