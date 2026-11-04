"""Static-trip-count MatMul/Gemm tile lowering regression.

These are source/host tests, never proof of native Vitis HLS scheduling
or FPGA timing. The new lowering must be rerun under real Vitis for E3.
"""
from __future__ import annotations

import shutil
import numpy as np
import pytest

from fpgai_research.architecture.materialization_oracle import verify_source_mapping
from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.host import validate_host


def _model(kind,m,n,k):
    """Binary-exact weights, and one bias for transposed-Gemm semantics."""
    if kind=="Gemm":
        weight_shape=(n,k)
        attrs=(("transB",1),)
        inputs=("x","w","bias")
    else:
        weight_shape=(k,n)
        attrs=()
        inputs=("x","w")
    data=tuple((.25 if idx%2 else -.125) for idx in range(k*n))
    tensors=[Tensor("x",(m,k)),
             Tensor("w",weight_shape,data=data)]
    if kind=="Gemm":
        tensors.append(Tensor("bias",(n,),data=(.25,)*n))
    tensors.append(Tensor("y",(m,n)))
    return FunctionalModel(tuple(tensors),(
        Operator("compute",kind,inputs,("y",),attrs),),
        ("x",),("y",))


@pytest.mark.parametrize("kind",("MatMul","Gemm"))
@pytest.mark.parametrize("shape,tile,simd,replicas,pes,fold",[
    ((1,8,8),(1,4,4),1,1,1,1),
    ((2,5,7),(1,3,3),2,1,1,1),
    ((3,7,9),(2,3,4),3,2,2,2),
    ((2,5,7),(1,1,2),4,2,1,3),
    ((1,8,8),(1,2,2),2,1,2,1),
])
def test_tiled_generated_loop_bounds_are_static(
        kind,shape,tile,simd,replicas,pes,fold):
    model=_model(kind,*shape)
    request=ArchitectureRequest(
        precision_bits=16,integer_bits=6,
        tile=tile,simd=simd,parallelism=replicas,pes=pes,
        serialization=fold)
    arch=resolve(model,ArchitectureRequest(),
                 {"compute":request})
    emitted=emit_hls(lower(arch))
    cpp=emitted.cpp
    assert "const int row_limit" not in cpp
    assert "for (int mi=mb; mi<row_limit;" not in cpp
    assert "for (int ki=kb; ki<" not in cpp
    assert f"for (int mo=0; mo<{tile[0]}; ++mo)" in cpp
    assert f"for (int ko=0; ko<{tile[2]}; ko+={simd})" in cpp
    assert "const int kk=kb+ko+lane;" in cpp
    assert f"if (kk<{shape[2]} && ko+lane<{tile[2]})" in cpp
    assert f"if (mi<{shape[0]} && ni<{shape[1]} && ni<nb+{tile[1]})" in cpp
    assert "if (ni >=" not in cpp
    assert "partial[lane] = data_t(0);" in cpp
    assert "#pragma HLS ARRAY_PARTITION variable=partial complete" in cpp
    assert verify_source_mapping(arch.points[0],lower(arch).operators[0])[
        "logical_iterations"]==shape[0]*shape[1]*shape[2]


@pytest.mark.skipif(shutil.which("g++") is None,reason="host GCC not installed")
@pytest.mark.parametrize("kind",("MatMul","Gemm"))
@pytest.mark.parametrize("shape,tile,simd,replicas,pes,fold",[
    ((1,8,8),(1,4,4),1,1,1,1),
    ((2,5,7),(1,3,3),2,1,1,1),
    ((3,7,9),(2,3,4),3,2,2,2),
    ((2,5,7),(1,1,2),4,2,1,3),
    ((1,8,8),(1,2,2),2,1,2,1),
])
def test_nondivisible_tiling_retains_host_float_reference(
        kind,shape,tile,simd,replicas,pes,fold):
    model=_model(kind,*shape)
    request=ArchitectureRequest(
        precision_bits=16,integer_bits=6,tile=tile,
        simd=simd,parallelism=replicas,pes=pes,serialization=fold)
    emitted=emit_hls(lower(resolve(
        model,ArchitectureRequest(),{"compute":request})))
    m,n,k=shape
    inputs=(np.arange(m*k,dtype=np.float32)%3).reshape(m,k)*.25
    result=validate_host(model,emitted,inputs)
    assert result["accepted"] is True,result
    assert result["max_absolute_error"]<1e-5,result


def test_identity_and_requested_tiles_are_reflected_in_source():
    model=_model("MatMul",1,8,8)
    a=emit_hls(lower(resolve(model,ArchitectureRequest(),
        {"compute":ArchitectureRequest(tile=(1,4,4))})))
    b=emit_hls(lower(resolve(model,ArchitectureRequest(),
        {"compute":ArchitectureRequest(tile=(1,2,2))})))
    assert a.manifest["cpp_sha256"]!=b.manifest["cpp_sha256"]
    assert a.cpp!=b.cpp
    assert "for (int ko=0; ko<4;" in a.cpp
    assert "for (int ko=0; ko<2;" in b.cpp


def test_archived_legacy_v1_artifact_remains_exactly_regenerable():
    from fpgai_research.codegen.hls.emitter import (
        LOOP_CODEGEN_LEGACY,LOOP_CODEGEN_STATIC,
    )
    from fpgai_research.model_ir.artifacts import (
        _hash,export_ir_artifact,verify_ir_artifact,
    )
    model=_model("MatMul",1,8,8)
    architecture=resolve(model,ArchitectureRequest(),
                         {"compute":ArchitectureRequest(tile=(1,4,4))})
    lowered=lower(architecture)
    old=emit_hls(lowered,loop_codegen=LOOP_CODEGEN_LEGACY)
    new=emit_hls(lowered)
    assert "loop_codegen_version" not in old.manifest
    assert new.manifest["loop_codegen_version"]==LOOP_CODEGEN_STATIC
    assert old.manifest["cpp_sha256"]!=new.manifest["cpp_sha256"]
    assert "for (int ki=kb; ki<8 && ki<kb+4;" in old.cpp
    assert "for (int ko=0; ko<4;" in new.cpp
    artifact=export_ir_artifact(lowered)
    artifact["stages"]["hls"]["manifest"]=old.manifest
    artifact["stage_sha256"]["hls"]=_hash(artifact["stages"]["hls"])
    assert verify_ir_artifact(artifact)==lowered
    assert verify_ir_artifact(export_ir_artifact(lowered))==lowered


def test_saved_unknown_generator_version_fails_closed():
    from fpgai_research.model_ir.artifacts import (
        _hash,export_ir_artifact,verify_ir_artifact,
    )
    model=_model("Gemm",2,3,5)
    lowered=lower(resolve(model,ArchitectureRequest()))
    artifact=export_ir_artifact(lowered)
    artifact["stages"]["hls"]["manifest"]["loop_codegen_version"]="unknown-v999"
    artifact["stage_sha256"]["hls"]=_hash(artifact["stages"]["hls"])
    with pytest.raises(ValueError,match="Unknown saved HLS loop-codegen"):
        verify_ir_artifact(artifact)
    with pytest.raises(ValueError,match="Unsupported explicit HLS"):
        emit_hls(lowered,loop_codegen="not-a-real-version")
