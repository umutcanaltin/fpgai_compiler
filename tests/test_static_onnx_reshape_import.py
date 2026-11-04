"""Exercise strict ONNX import using a minimal protobuf-shaped stub.

These tests check frontend behavior without claiming a real ONNX checker or
onnxruntime comparison; native integration still needs the onnx dependency.
"""
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from fpgai_research.frontend.onnx import load_onnx

FLOAT,INT64=1,7
class Dim:
    def __init__(self,n): self.dim_value=n
    def HasField(self,key): return key=='dim_value'
def info(name,shape,dtype=FLOAT):
    return NS(name=name,type=NS(tensor_type=NS(elem_type=dtype,shape=NS(dim=[Dim(d) for d in shape]))))
def tensor(name,value,dtype):
    return NS(name=name,data_type=dtype,value=np.array(value,dtype=np.float32 if dtype==FLOAT else np.int64))
def stub_onnx(monkeypatch,*,shape_source='Constant',allowzero=0):
    const=tensor('shape_tensor',[0,-1],INT64)
    nodes=[]; initializers=[]
    if shape_source=='Constant':
        nodes.append(NS(name='shape_constant',domain='',op_type='Constant',input=[],output=['shape_tensor'],attribute=[NS(name='value',value=const)]))
    elif shape_source=='initializer':
        initializers.append(const)
    nodes.append(NS(name='reshape',domain='',op_type='Reshape',input=['x','shape_tensor'],output=['y'],attribute=[NS(name='allowzero',value=allowzero)]))
    graph=NS(input=[info('x',(1,2,3))],value_info=[info('shape_tensor',(2,),INT64)],
             output=[info('y',(1,6))],sparse_initializer=[],initializer=initializers,node=nodes)
    stub=NS(load=lambda _:NS(graph=graph),checker=NS(check_model=lambda _:None),
            shape_inference=NS(infer_shapes=lambda v:v),numpy_helper=NS(to_array=lambda x:x.value),
            TensorProto=NS(FLOAT=FLOAT,INT64=INT64),
            helper=NS(get_attribute_value=lambda a:a.value))
    monkeypatch.setitem(sys.modules,'onnx',stub)

@pytest.mark.parametrize('kind',['initializer','Constant'])
def test_static_shape_not_runtime_port(monkeypatch,kind):
    stub_onnx(monkeypatch,shape_source=kind)
    model=load_onnx('fixture.onnx')
    assert model.inputs==('x',) and len(model.operators)==1
    assert model.operators[0].kind=='Reshape'
    assert model.operators[0].inputs==('x',)
    assert dict(model.operators[0].attributes)=={'shape':(0,-1)}
    assert all(t.name!='shape_tensor' for t in model.tensors)

@pytest.mark.parametrize('allowzero',[1,2])
def test_reject_allowzero_unsupported(monkeypatch,allowzero):
    stub_onnx(monkeypatch,allowzero=allowzero)
    with pytest.raises(ValueError,match='allowzero'):
        load_onnx('fixture.onnx')

def test_dynamic_shape_rejected(monkeypatch):
    stub_onnx(monkeypatch,shape_source='dynamic')
    with pytest.raises(ValueError,match='static int64'):
        load_onnx('fixture.onnx')
