"""Executed host C++ and independent semantic-oracle regression tests."""
import numpy as np
import pytest
from fpgai_research import ArchitectureRequest, FunctionalModel, Operator, Tensor, resolve, lower
from fpgai_research.architecture.mapping import verify_mapping
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.validation.host import validate_host
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package

def matmul():
    return FunctionalModel((Tensor("x",(2,4)),
         Tensor("w",(4,3),data=tuple(float(i%9)/8 for i in range(12))),
         Tensor("y",(2,3))),
         (Operator("mm","MatMul",("x","w"),("y",)),),("x",),("y",))

@pytest.mark.parametrize("tile",((2,3,4),(1,2,3),(1,1,1)))
@pytest.mark.parametrize("simd",(1,2,3,4))
def test_matmul_space_time_and_executed_cpp(tile,simd):
    model=matmul()
    req=ArchitectureRequest(precision_bits=12,integer_bits=5,
             parallelism=2,pes=2,simd=simd,serialization=2,
             tile=tile,pipeline_ii=2)
    arch=resolve(model,ArchitectureRequest(),{"mm":req})
    assert verify_mapping(arch.points[0])["distinct_assignments"]==24
    lowered=lower(arch)
    assert lowered.operators[0].tail_policy=="bounds-guard"
    emission=emit_hls(lowered)
    assert emission.manifest["operators"][0]["lowered_mechanisms"]["simd_lanes"]==simd
    result=validate_host(model,emission,np.arange(8,dtype=np.float32).reshape(2,4))
    assert result["accepted"]
    assert result["strongest_evidence"]=="E1-plus-host-float-check"

def test_transb_gemm_and_bias_in_generated_cpp():
    model=FunctionalModel((Tensor("x",(1,3)),
      Tensor("w",(2,3),data=(1,2,3,4,5,6)),
      Tensor("b",(2,),data=(1,2)),Tensor("y",(1,2))),
      (Operator("fc","Gemm",("x","w","b"),("y",),(("transB",1),)),),("x",),("y",))
    arch=resolve(model,ArchitectureRequest(),{"fc":ArchitectureRequest(pes=2,simd=2)})
    assert verify_mapping(arch.points[0])["iterations"]==6
    assert validate_host(model,emit_hls(lower(arch)),np.array([[1,2,3]],dtype=np.float32))["accepted"]

def test_conv_relu_flatten_softmax_host():
    model=FunctionalModel((Tensor("x",(1,1,3,3)),
      Tensor("w",(1,1,2,2),data=(1,0,0,1)),
      Tensor("conv",(1,1,2,2)),Tensor("relu",(1,1,2,2)),
      Tensor("flat",(1,4)),Tensor("y",(1,4))),(
      Operator("conv","Conv",("x","w"),("conv",)),
      Operator("relu","Relu",("conv",),("relu",)),
      Operator("flatten","Flatten",("relu",),("flat",),(("axis",1),)),
      Operator("softmax","Softmax",("flat",),("y",))),("x",),("y",))
    contract=ArchitectureRequest(pes=2,simd=2,serialization=2,
                                 tile=(1,1,1,1,1,2,2))
    arch=resolve(model,ArchitectureRequest(),{"conv":contract})
    assert verify_mapping(arch.points[0])["iterations"]==16
    assert validate_host(model,emit_hls(lower(arch)),
                         np.arange(9,dtype=np.float32).reshape(1,1,3,3))["accepted"]

def test_paper_evidence_does_not_invent_resources(tmp_path):
    model=matmul()
    emission=emit_hls(lower(resolve(model,ArchitectureRequest())))
    write_vitis_package(str(tmp_path),model,emission,
                        np.arange(8,dtype=np.float32).reshape(2,4))
    report=summarize_case(str(tmp_path))
    assert report["evidence"]=="E1-HLS-source"
    assert report["metrics"]["lut"] is None
    (tmp_path/"model.cpp").write_text("modified")
    with pytest.raises(ValueError,match="hash"):
        summarize_case(str(tmp_path))

def test_explicit_rejection_of_noncompute_tiling():
    model=FunctionalModel((Tensor("x",(1,2)),Tensor("y",(1,2))),
         (Operator("relu","Relu",("x",),("y",)),),("x",),("y",))
    with pytest.raises(ValueError,match="CAPABILITY"):
        resolve(model,ArchitectureRequest(tile=(1,2)))
