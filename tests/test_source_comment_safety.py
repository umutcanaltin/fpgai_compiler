"""Untrusted ONNX operator identifiers must not create preprocessor directives."""
import shutil
import numpy as np
from fpgai_research.model_ir.contracts import Tensor,Operator,FunctionalModel,ArchitectureRequest
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.lowering import lower
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.validation.host import validate_host

def test_hostile_operator_name_cannot_inject_source():
    name='mm\n#error injected_by_onnx\n'
    model=FunctionalModel((Tensor('x',(1,2)),Tensor('w',(2,1),data=(.125,.25)),Tensor('y',(1,1))),
       (Operator(name,'MatMul',('x','w'),('y',)),),('x',),('y',))
    emitted=emit_hls(lower(resolve(model,ArchitectureRequest())))
    assert '\n#error injected_by_onnx' not in emitted.cpp
    assert emitted.manifest['operators'][0]['operator']==name
    if shutil.which('g++'):
        assert validate_host(model,emitted,np.array([[1.,2.]],dtype=np.float32))['accepted']
