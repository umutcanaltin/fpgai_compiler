"""Malformed serialized/user architecture fields must fail before HLS source emission."""
from dataclasses import replace
import pytest
from fpgai_research.architecture.resolver import resolve, TargetLimits, ArchitectureError
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel, ArchitectureRequest


def model():
    return FunctionalModel(
        (Tensor('x',(1,4)), Tensor('w',(4,3),data=(.1,)*12), Tensor('y',(1,3))),
        (Operator('mm','MatMul',('x','w'),('y',)),), ('x',), ('y',))


@pytest.mark.parametrize('field,value', [
    ('precision_bits', 16.), ('integer_bits', True), ('pes', False),
    ('parallelism', 2.), ('serialization', '2'), ('simd', 1.1),
    ('pipeline_ii', 2.5), ('tile', (1,3,1.5)), ('tile', (True,3,4)),
    ('tile', '1,3,4'),
])
def test_invalid_architecture_type_fails_closed(field,value):
    with pytest.raises(ArchitectureError,match='REQUEST_TYPE'):
        resolve(model(),ArchitectureRequest(),{'mm':replace(ArchitectureRequest(),**{field:value})})


def test_unused_default_and_target_types_are_rejected():
    with pytest.raises(ArchitectureError,match='REQUEST_TYPE'):
        resolve(model(),replace(ArchitectureRequest(),simd=1.5))
    with pytest.raises(ArchitectureError,match='TARGET_TYPE'):
        resolve(model(),ArchitectureRequest(),limits=TargetLimits(max_parallel_units=2.5))
    with pytest.raises(ArchitectureError,match='TARGET_TYPE'):
        resolve(model(),ArchitectureRequest(),limits=TargetLimits(supported_widths=(True,16)))


def test_boolean_tensor_dimension_rejected():
    with pytest.raises(ValueError,match='static dimensions'):
        Tensor('bad',(True,3))


def test_multigraph_unsupported_extra_output_rejected():
    f=model()
    bad=FunctionalModel(f.tensors, (Operator('mm','MatMul',('x','w'),('y','x')),),f.inputs,f.outputs)
    with pytest.raises(ValueError,match='single-output'):
        bad.validate()


def test_duplicate_attributes_and_model_outputs_rejected():
    f=model()
    op=Operator('mm','MatMul',('x','w'),('y',),(('custom',1),('custom',2)))
    with pytest.raises(ValueError,match='attributes'):
        FunctionalModel(f.tensors,(op,),f.inputs,f.outputs).validate()
    with pytest.raises(ValueError,match='output identity'):
        FunctionalModel(f.tensors,f.operators,f.inputs,('y','y')).validate()
