"""Phi assignment must match independently enumerated HLS source-loop work."""
from dataclasses import replace
import pytest
from fpgai_research.architecture.materialization_oracle import verify_source_mapping
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel, ArchitectureRequest
from fpgai_research.model_ir.lowering import lower


def gemm(m=2, n=5, k=7):
    return FunctionalModel((Tensor('x', (m, k)), Tensor('w', (k, n), data=(.1,) * (k*n)),
                            Tensor('y', (m, n))),
                           (Operator('mm', 'MatMul', ('x', 'w'), ('y',)),),('x',),('y',))


def conv():
    return FunctionalModel((Tensor('x',(2,3,5,7)), Tensor('w',(5,3,2,3),data=(.1,)*(5*3*2*3)),
                            Tensor('y',(2,5,4,5))),
                           (Operator('conv','Conv',('x','w'),('y',)),), ('x',), ('y',))


@pytest.mark.parametrize('arch_request', [
    ArchitectureRequest(),
    ArchitectureRequest(pes=2, parallelism=3, simd=2, serialization=3, tile=(1,3,4)),
    ArchitectureRequest(pes=3, parallelism=2, simd=4, serialization=2, tile=(2,2,2)),
    ArchitectureRequest(pes=1, parallelism=1, simd=9, serialization=7, tile=(1,1,1)),
])
def test_matmul_source_and_phi_agree(arch_request):
    arch = resolve(gemm(),ArchitectureRequest(),{'mm':arch_request})
    proof = verify_source_mapping(arch.points[0],lower(arch).operators[0])
    assert proof['logical_iterations'] == 2*5*7
    assert proof['source_level_assignments'] == 2*5*7


@pytest.mark.parametrize('arch_request', [
    ArchitectureRequest(),
    ArchitectureRequest(parallelism=2, pes=3, simd=2, serialization=2, tile=(2,3,2,3,3,2,3)),
    ArchitectureRequest(parallelism=3, pes=2, simd=4, serialization=3, tile=(2,1,3,2,3,2,3)),
])
def test_conv_source_and_phi_agree(arch_request):
    arch=resolve(conv(), ArchitectureRequest(), {'conv':arch_request})
    check=verify_source_mapping(arch.points[0],lower(arch).operators[0])
    assert check['logical_iterations']==2*5*4*5*3*2*3


def test_corrupted_materialization_is_detected():
    arch=resolve(gemm(),ArchitectureRequest(), {'mm':ArchitectureRequest(simd=2, pes=2)})
    lowering=lower(arch).operators[0]
    with pytest.raises(AssertionError,match='spatial assignment'):
        verify_source_mapping(arch.points[0],replace(lowering,simd_lanes=3))
    with pytest.raises(ValueError,match='oracle bound'):
        verify_source_mapping(arch.points[0],lowering,max_iterations=1)
