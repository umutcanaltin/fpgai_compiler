"""Reject undefined HLS arrays and unmaterialized direct graph outputs."""
import pytest
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel

def test_declared_undefined_tensor_is_rejected():
    m=FunctionalModel((Tensor('x',(1,2)),Tensor('dangling',(1,2)),Tensor('y',(1,2))),
       (Operator('id','Identity',('x',),('y',)),),('x',),('y',))
    with pytest.raises(ValueError,match='Declared tensor'):
        m.validate()

def test_direct_input_output_alias_is_rejected():
    m=FunctionalModel((Tensor('x',(1,2)),Tensor('y',(1,2))),
       (Operator('id','Identity',('x',),('y',)),),('x',),('x',))
    with pytest.raises(ValueError,match='written by an operator'):
        m.validate()

def test_constant_direct_output_rejected():
    m=FunctionalModel((Tensor('x',(1,2)),Tensor('c',(1,2),data=(0.,1.)),Tensor('y',(1,2))),
       (Operator('id','Identity',('x',),('y',)),),('x',),('c',))
    with pytest.raises(ValueError,match='written by an operator'):
        m.validate()

def test_no_operator_is_unsupported():
    m=FunctionalModel((Tensor('x',(1,2)),),(),('x',),('x',))
    with pytest.raises(ValueError,match='at least one operator'):
        m.validate()

def test_unused_initializer_is_safe():
    m=FunctionalModel((Tensor('x',(1,2)),Tensor('unused',(1,2),data=(0.,1.)),Tensor('y',(1,2))),
       (Operator('id','Identity',('x',),('y',)),),('x',),('y',))
    m.validate()
