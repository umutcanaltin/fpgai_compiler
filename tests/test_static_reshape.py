"""Static Reshape contracts and actual generated host C++ numeric evidence."""
import shutil
import numpy as np
import pytest
from fpgai_research.frontend.static_reshape import resolve_static_reshape
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel, ArchitectureRequest
from fpgai_research.architecture.resolver import resolve, ArchitectureError
from fpgai_research.model_ir.lowering import lower
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.reference import evaluate

@pytest.mark.parametrize('shape,desired,output',[
    ((1,2,3),(0,-1),(1,6)),((1,2,3),(-1,3),(2,3)),
    ((2,3,4),(0,0,-1),(2,3,4)),((1,4,4),(-1,2,2,2),(2,2,2,2))])
def test_static_geometry(shape,desired,output):
    assert resolve_static_reshape(shape,desired)==output

@pytest.mark.parametrize('shape,requested',[
    ((2,3),(-1,-1)),((2,3),(0,0,0)),((2,3),(4,2)),
    ((2,3),(-2,3)),((2,3),()),((2,3),(1.0,6)),((2,3),(True,6))])
def test_bad_shapes_rejected(shape,requested):
    with pytest.raises(ValueError):
        resolve_static_reshape(shape,requested)

def model():
    return FunctionalModel((Tensor('x',(1,2,3)),Tensor('r',(1,6)),
      Tensor('w',(6,2),data=tuple(float(i)/8 for i in range(12))),Tensor('y',(1,2))),
      (Operator('reshape','Reshape',('x',),('r',),(('shape',(0,-1)),)),
       Operator('mm','MatMul',('r','w'),('y',))),('x',),('y',))

def test_real_cpp_reshape_matmul():
    m=model();x=np.arange(6,dtype=np.float32).reshape((1,2,3))/8
    assert evaluate(m,{'x':x})['y'].shape==(1,2)
    emission=emit_hls(lower(resolve(m,ArchitectureRequest())))
    if shutil.which('g++'):
        receipt=validate_host(m,emission,x)
        assert receipt['accepted'],receipt

def test_inconsistent_declared_reshape_rejected():
    m=model()
    bad=Operator('reshape','Reshape',('x',),('r',),(('shape',(6,1)),))
    with pytest.raises(ArchitectureError,match='RESHAPE'):
        resolve(FunctionalModel(m.tensors,(bad,m.operators[1]),m.inputs,m.outputs),ArchitectureRequest())
