"""Case preparation cannot silently reuse stale numerical or native evidence."""
import numpy as np
import pytest
from fpgai_research.model_ir.contracts import Tensor,Operator,FunctionalModel,ArchitectureRequest
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.lowering import lower
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package, make_vitis_tcl


def model():
    return FunctionalModel((Tensor('x',(1,2)),Tensor('w',(2,1),data=(1.,2.)),Tensor('y',(1,1))),
                           (Operator('mm','MatMul',('x','w'),('y',)),),('x',),('y',))


def test_refuse_preexisting_case_data(tmp_path):
    m=model()
    emitter=emit_hls(lower(resolve(m,ArchitectureRequest())))
    sample=np.array([[1.,2.]],dtype=np.float32)
    case=tmp_path/'case'
    case.mkdir()
    (case/'manifest.json').write_text('{"source":"old"}')
    with pytest.raises(FileExistsError, match='nonempty'):
        write_vitis_package(str(case),m,emitter,sample)
    assert (case/'manifest.json').read_text()=='{"source":"old"}'


def test_fresh_case_created_once(tmp_path):
    m=model()
    emitter=emit_hls(lower(resolve(m,ArchitectureRequest())))
    sample=np.array([[1.,2.]],dtype=np.float32)
    case=tmp_path/'case'
    write_vitis_package(str(case),m,emitter,sample)
    with pytest.raises(FileExistsError,match='nonempty'):
        write_vitis_package(str(case),m,emitter,sample)


@pytest.mark.parametrize('clock',[float('nan'),float('inf'),-2.])
def test_vitis_tcl_rejects_nonfinite_clock(clock):
    with pytest.raises(ValueError,match='finite'):
        make_vitis_tcl(clock_ns=clock)
