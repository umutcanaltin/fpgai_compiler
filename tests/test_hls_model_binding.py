"""Bind generated C++ and fixed-point C-simulation references to the same Model IR."""
import numpy as np
import pytest
from fpgai_research.model_ir.contracts import Tensor,Operator,FunctionalModel,ArchitectureRequest
from fpgai_research.model_ir.semantics import functional_fingerprint
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.lowering import lower
from fpgai_research.codegen.hls.emitter import Emission,emit_hls
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package

def model(values):
    return FunctionalModel((Tensor("x",(1,2)),Tensor("w",(2,1),data=values),Tensor("y",(1,1))),
         (Operator("mm","MatMul",("x","w"),("y",)),),("x",),("y",))

def test_correct_model_is_present_in_hls_manifest(tmp_path):
    m=model((.1,.2))
    e=emit_hls(lower(resolve(m,ArchitectureRequest())))
    assert e.manifest["functional_sha256"]==functional_fingerprint(m)
    package=write_vitis_package(str(tmp_path/"case"),m,e,np.ones((1,2),dtype=np.float32))
    assert package["functional_sha256"]==e.manifest["functional_sha256"]

def test_altered_weights_cannot_be_numerical_reference(tmp_path):
    m=model((.1,.2))
    e=emit_hls(lower(resolve(m,ArchitectureRequest())))
    target=tmp_path/"case"
    with pytest.raises(ValueError,match="different Functional Model IR"):
        write_vitis_package(str(target),model((.1,.3)),e,np.ones((1,2),dtype=np.float32))
    assert not target.exists()

def test_modified_cpp_with_old_manifest_rejected(tmp_path):
    m=model((.1,.2))
    e=emit_hls(lower(resolve(m,ArchitectureRequest())))
    corrupted=Emission(e.cpp+"\n// unexpected code\n",e.manifest)
    target=tmp_path/"case"
    with pytest.raises(ValueError,match="emission manifest"):
        write_vitis_package(str(target),m,corrupted,np.ones((1,2),dtype=np.float32))
    assert not target.exists()
