"""Direct/saved Functional Model IR must not silently ignore ONNX attributes."""
from dataclasses import replace
import pytest
from fpgai_research.architecture.resolver import resolve, ArchitectureError
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel, ArchitectureRequest


def make(kind,attributes=()):
    tensors={
       'Conv':(Tensor('x',(1,1,5,5)),Tensor('w',(1,1,3,3),data=(.5,)*9),Tensor('y',(1,1,3,3))),
       'MaxPool':(Tensor('x',(1,1,6,6)),Tensor('y',(1,1,3,3))),
       'Flatten':(Tensor('x',(1,2,3)),Tensor('y',(1,6))),
       'Relu':(Tensor('x',(1,3)),Tensor('y',(1,3))),
       'LeakyRelu':(Tensor('x',(1,3)),Tensor('y',(1,3))),
    }[kind]
    inputs=('x','w') if kind=='Conv' else ('x',)
    return FunctionalModel(tensors,(Operator('op',kind,inputs,('y',),attributes),),('x',),('y',))


@pytest.mark.parametrize('kind,attributes,code', [
    ('Relu', (('alpha',.1),), 'UNSUPPORTED_ATTR'),
    ('Conv', (('kernel_shape',(5,5)),), 'CONV_ATTR'),
    ('Conv', (('auto_pad','SAME_UPPER'),), 'UNSUPPORTED_ATTR'),
    ('MaxPool', (('kernel_shape',(2,2)),('ceil_mode',1)), 'MAXPOOL'),
    ('MaxPool', (('kernel_shape',(2,2)),('dilations',(2,2))), 'MAXPOOL'),
    ('Flatten', (('axis',2),), 'FLATTEN'),
    ('LeakyRelu', (('alpha',-0.2),), 'LEAKY'),
])
def test_attributes_must_match_implemented_semantics(kind,attributes,code):
    with pytest.raises(ArchitectureError,match=code):
        resolve(make(kind,attributes),ArchitectureRequest())


def test_supported_conv_kernel_shape_is_accepted():
    model=make('Conv',(('kernel_shape',(3,3)),))
    assert len(resolve(model,ArchitectureRequest()).points)==1
