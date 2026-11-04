"""Static-trip Conv tile lowering and exact historical source recovery.

Host-float code is not bit-exact fixed-point CSim; Vitis must separately
qualify the new v3 generator before any E3/E4 claim is made.
"""
from __future__ import annotations

import shutil
import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.architecture.materialization_oracle import verify_source_mapping
from fpgai_research.codegen.hls.emitter import (
    emit_hls,LOOP_CODEGEN_CONV_STATIC,LOOP_CODEGEN_STATIC,
)
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest,FunctionalModel,Operator,Tensor,
)
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.artifacts import (
    _hash,export_ir_artifact,verify_ir_artifact,
)
from fpgai_research.validation.host import validate_host


def _conv(batch,cin,hi,wi,cout,kh,kw):
    ho=hi-kh+1
    wo=wi-kw+1
    data=tuple((.25 if i%3==0 else -.125)
               for i in range(cout*cin*kh*kw))
    model=FunctionalModel((
        Tensor("input",(batch,cin,hi,wi)),
        Tensor("weight",(cout,cin,kh,kw),data=data),
        Tensor("output",(batch,cout,ho,wo))),(
            Operator("compute","Conv",("input","weight"),("output",)),),
            ("input",),("output",))
    model.validate()
    return model


PROBES=(
    ((1,2,8,8,4,3,3),(1,2,3,2,2,3,3),1,1,1,1),
    ((1,2,5,7,3,2,3),(1,2,3,2,2,2,3),2,1,1,1),
    ((1,3,7,8,5,3,2),(1,3,2,3,3,3,2),4,2,2,2),
    ((2,3,6,6,4,2,2),(2,1,3,2,3,2,2),3,2,1,2),
)


def _lower(shape,tile,simd,replicas,pes,serialization):
    model=_conv(*shape)
    request=ArchitectureRequest(
        tile=tile,simd=simd,parallelism=replicas,pes=pes,
        serialization=serialization)
    arch=resolve(model,ArchitectureRequest(),
                 {"compute":request})
    return model,lower(arch)


@pytest.mark.parametrize("shape,tile,simd,replicas,pes,fold",PROBES)
def test_conv_v3_tiled_loop_source_uses_static_spatial_trip_counts(
        shape,tile,simd,replicas,pes,fold):
    model,lo=_lower(shape,tile,simd,replicas,pes,fold)
    source=emit_hls(lo)
    assert source.manifest["loop_codegen_version"]==LOOP_CODEGEN_CONV_STATIC
    cpp=source.cpp
    batch,cin,hi,wi,cout,kh,kw=shape
    _,_,toh,tow,_,_,_=tile
    ho,wo=hi-kh+1,wi-kw+1
    assert f"for(int yh=0;yh<{toh};++yh)" in cpp
    assert f"for(int xw=0;xw<{tow};++xw)" in cpp
    assert f"if(oh<{ho})" in cpp
    assert f"if(ow<{wo})" in cpp
    assert f"if(oc<{cout} && oc<cb+{tile[1]})" in cpp
    assert "const int oh=hb+yh;" in cpp
    assert "const int ow=wb+xw;" in cpp
    assert "continue;" not in cpp
    assert "for(int oh=hb;" not in cpp
    assert "for(int ow=wb;" not in cpp
    assert "#pragma HLS ARRAY_PARTITION variable=partial complete" in cpp
    assert verify_source_mapping(lo.architectural.points[0],
                                 lo.operators[0])["coverage_and_uniqueness"]


@pytest.mark.skipif(shutil.which("g++") is None,reason="host GCC missing")
@pytest.mark.parametrize("shape,tile,simd,replicas,pes,fold",PROBES)
def test_conv_v3_guarded_tail_preserves_reference_numerics(
        shape,tile,simd,replicas,pes,fold):
    model,lo=_lower(shape,tile,simd,replicas,pes,fold)
    batch,cin,hi,wi,*_=shape
    values=(np.arange(batch*cin*hi*wi,dtype=np.float32)%3)
    samples=(values.reshape(batch,cin,hi,wi)*.25).astype(np.float32)
    check=validate_host(model,emit_hls(lo),samples)
    assert check["accepted"] is True,check
    assert check["max_absolute_error"]<1e-5,check


def test_conv_v2_historical_source_and_ir_still_replay_exactly():
    model,lo=_lower(*PROBES[0])
    old=emit_hls(lo,loop_codegen=LOOP_CODEGEN_STATIC)
    new=emit_hls(lo)
    assert old.manifest["loop_codegen_version"]==LOOP_CODEGEN_STATIC
    assert new.manifest["loop_codegen_version"]==LOOP_CODEGEN_CONV_STATIC
    assert old.manifest["cpp_sha256"]!=new.manifest["cpp_sha256"]
    assert "for(int oh=hb;oh<" in old.cpp
    assert "for(int ow=wb;ow<" in old.cpp
    assert "for(int yh=0;yh<" in new.cpp
    assert "for(int xw=0;xw<" in new.cpp
    archived=export_ir_artifact(lo)
    archived["stages"]["hls"]["manifest"]=old.manifest
    archived["stage_sha256"]["hls"]=_hash(archived["stages"]["hls"])
    assert verify_ir_artifact(archived)==lo
    assert verify_ir_artifact(export_ir_artifact(lo))==lo


def test_matmul_v2_codegen_default_does_not_move_to_conv_v3():
    m=FunctionalModel((
        Tensor("x",(1,2)),Tensor("w",(2,2),data=(.25,)*4),
        Tensor("y",(1,2))),(
        Operator("mm","MatMul",("x","w"),("y",)),),
        ("x",),("y",))
    lo=lower(resolve(m,ArchitectureRequest()))
    implicit=emit_hls(lo)
    explicit=emit_hls(lo,loop_codegen=LOOP_CODEGEN_STATIC)
    assert implicit==explicit
    assert implicit.manifest["loop_codegen_version"]==LOOP_CODEGEN_STATIC
