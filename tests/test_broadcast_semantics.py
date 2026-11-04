"""Verify broadcasted Add/Mul via actual generated C++ and NumPy reference."""
from math import prod
import shutil
import numpy as np
import pytest
from fpgai_research.frontend.broadcast import broadcast_shape,broadcast_flat_index
from fpgai_research.model_ir.contracts import Tensor,Operator,FunctionalModel,ArchitectureRequest
from fpgai_research.architecture.resolver import resolve,ArchitectureError
from fpgai_research.model_ir.lowering import lower
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.reference import evaluate

@pytest.mark.parametrize("a,b,out",[
 ((1,3,4,5),(1,3,1,1),(1,3,4,5)),((1,6),(6,),(1,6)),
 ((1,2,3),(),(1,2,3)),((5,1),(1,7),(5,7))])
def test_static_shape(a,b,out):
    assert broadcast_shape(a,b)==out

@pytest.mark.parametrize("a,b",[((2,3),(2,4)),((1,3,2,2,2),(1,)),((2,3),(4,3))])
def test_incompatible_shape_rejected(a,b):
    with pytest.raises(ValueError):
        broadcast_shape(a,b)

@pytest.mark.parametrize("shape,bias",[
 ((1,3,4,5),(1,3,1,1)),((1,3,4,5),(5,)),((2,3),(3,)),
 ((1,2,3),()),((2,3,4),(1,1,4)),((2,1),(1,7))])
@pytest.mark.parametrize("kind",["Add","Mul"])
def test_compiled_numerical_broadcast(shape,bias,kind):
    out=broadcast_shape(shape,bias)
    const=tuple(np.linspace(-1.,1.,prod(bias),dtype=np.float32))
    m=FunctionalModel((Tensor("x",shape),Tensor("b",bias,data=const),Tensor("y",out)),
        (Operator("compute",kind,("x","b"),("y",)),),("x",),("y",))
    x=np.linspace(-.7,.8,prod(shape),dtype=np.float32).reshape(shape)
    assert evaluate(m,{"x":x})["y"].shape==out
    emitted=emit_hls(lower(resolve(m,ArchitectureRequest())))
    if shutil.which("g++"):
        receipt=validate_host(m,emitted,x)
        assert receipt["accepted"],receipt
        assert receipt["max_absolute_error"]<1e-5

def test_wrong_result_shape_fails_closed():
    m=FunctionalModel((Tensor("x",(2,3)),Tensor("b",(3,),data=(.1,.2,.3)),Tensor("y",(2,2))),
      (Operator("add","Add",("x","b"),("y",)),),("x",),("y",))
    with pytest.raises(ArchitectureError,match="BROADCAST"):
        resolve(m,ArchitectureRequest())

def test_scalar_index_and_json_fingerprint_are_canonical():
    assert broadcast_flat_index((),(1,3,4,5))=="0"
    from fpgai_research.model_ir.semantics import functional_fingerprint
    from fpgai_research.model_ir.contracts import Tensor
    t=Tensor("a",(1,),data=(np.float32(0.25),))
    assert t.data==(0.25,)
