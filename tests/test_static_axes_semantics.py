"""Squeeze/Unsqueeze preserve shape intent through compiler stages and GCC."""
import shutil
import numpy as np
import pytest
from fpgai_research.frontend.static_axes import squeeze_shape,unsqueeze_shape
from fpgai_research.model_ir.contracts import Tensor,Operator,FunctionalModel,ArchitectureRequest
from fpgai_research.architecture.resolver import resolve,ArchitectureError
from fpgai_research.model_ir.lowering import lower
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.validation.host import validate_host

@pytest.mark.parametrize("shape,axes,expected",[
 ((1,2,1,3),(0,2),(2,3)),((1,2,1,3),(-4,-2),(2,3)),
 ((1,2,1,3),None,(2,3)),((2,3),(),(2,3))])
def test_squeeze_shape(shape,axes,expected):
    assert squeeze_shape(shape,axes)==expected

@pytest.mark.parametrize("shape,axes,expected",[
 ((2,3),(0,2),(1,2,1,3)),((2,3),(-4,-2),(1,2,1,3)),((2,3),(1,),(2,1,3))])
def test_unsqueeze_shape(shape,axes,expected):
    assert unsqueeze_shape(shape,axes)==expected

@pytest.mark.parametrize("kind,shape,axes",[
 ("Squeeze",(2,3),(0,)),("Squeeze",(1,2,3),(3,)),
 ("Unsqueeze",(2,3),(0,0)),("Unsqueeze",(2,3),())])
def test_bad_axes_rejected(kind,shape,axes):
    with pytest.raises(ValueError):
        squeeze_shape(shape,axes) if kind=="Squeeze" else unsqueeze_shape(shape,axes)

def test_generated_host_cpp_roundtrip():
    model=FunctionalModel((Tensor("x",(1,2,1,3)),Tensor("sq",(2,3)),
      Tensor("us",(1,2,1,3)),Tensor("y",(1,2,1,3))),(
      Operator("squeeze","Squeeze",("x",),("sq",),(("axes",(0,2)),)),
      Operator("unsqueeze","Unsqueeze",("sq",),("us",),(("axes",(0,2)),)),
      Operator("copy","Identity",("us",),("y",))),("x",),("y",))
    emitted=emit_hls(lower(resolve(model,ArchitectureRequest())))
    if shutil.which("g++"):
        assert validate_host(model,emitted,np.arange(6,dtype=np.float32).reshape(1,2,1,3))["accepted"]

def test_wrong_result_shape_rejected():
    m=FunctionalModel((Tensor("x",(1,2,1,3)),Tensor("y",(2,1,3))),(
       Operator("squeeze","Squeeze",("x",),("y",),(("axes",(0,2)),)),),("x",),("y",))
    with pytest.raises(ArchitectureError,match="AXES"):
        resolve(m,ArchitectureRequest())
