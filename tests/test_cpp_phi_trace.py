"""Host-compiled C++ must actually visit Phi assignments."""
import shutil
import pytest
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_phi_trace import trace_cpp_phi
from test_hls_loop_phi_oracle import gemm,conv

pytestmark = pytest.mark.skipif(shutil.which('g++') is None,reason='GCC not installed')


@pytest.mark.parametrize('arch_request', [
    ArchitectureRequest(),
    ArchitectureRequest(simd=4,pes=2,parallelism=3,serialization=2,tile=(1,2,2)),
    ArchitectureRequest(simd=9,serialization=5,tile=(1,1,1)),
])
def test_actual_generated_matmul_iterations(arch_request):
    arch=resolve(gemm(),ArchitectureRequest(),{'mm':arch_request})
    receipt=trace_cpp_phi(lower(arch))
    assert receipt['logical_iterations']==70
    assert receipt['operators']['mm']==70
    assert receipt['source_phi_agreement']
    assert not receipt['native_vitis_executed']


@pytest.mark.parametrize('arch_request', [
    ArchitectureRequest(),
    ArchitectureRequest(simd=2,pes=3,parallelism=2,serialization=2,tile=(2,3,2,3,3,2,3)),
])
def test_actual_generated_conv_iterations(arch_request):
    arch=resolve(conv(),ArchitectureRequest(),{'conv':arch_request})
    receipt=trace_cpp_phi(lower(arch),max_iterations=10_000)
    assert receipt['logical_iterations']==2*5*4*5*3*2*3
    assert receipt['operators']['conv']==receipt['logical_iterations']


def test_trace_budget_fail_closed_before_compilation():
    arch=resolve(gemm(),ArchitectureRequest())
    with pytest.raises(ValueError,match='exceeds bound'):
        trace_cpp_phi(lower(arch),max_iterations=1)
