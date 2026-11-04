"""Algebraic mapping structure is stronger than samples, not native proof."""
from dataclasses import replace
from math import prod

import pytest

from fpgai_research.model_ir.contracts import ArchitecturePoint, ArchitectureRequest
from fpgai_research.model_ir.division_certificate import certify_phi_partition
from fpgai_research.model_ir.phi_refinement import verify_hls_phi
from fpgai_research.architecture.resolver import coordinate


def _point(kind="Gemm", *, shape=None, tile=(), p=3, pe=2, simd=4, fold=3):
    shape=shape or ((320,129,1024) if kind!="Conv" else
                    (2,31,40,37,16,3,3))
    a=ArchitectureRequest(parallelism=p,pes=pe,simd=simd,
                          serialization=fold,tile=tile)
    return ArchitecturePoint("compute",a,a,shape,"model",kind)


@pytest.mark.parametrize("kind,shape,tile", [
    ("MatMul",(320,129,1024),(17,13,31)),
    ("Gemm",(320,129,1024),()),
    ("Conv",(2,31,40,37,16,3,3),(2,7,9,8,16,3,3)),
])
def test_large_domains_receive_constant_size_algebraic_derivation(kind,shape,tile):
    point=_point(kind,shape=shape,tile=tile)
    certificate=certify_phi_partition(point)
    assert certificate["schema"]=="fpgai-research/phi-euclidean-partition/v1"
    assert certificate["status"]=="ALGEBRAIC_PARTITION_DERIVED"
    assert certificate["logical_iteration_count"]==prod(shape)
    assert certificate["claims"]["integer_partition_uniqueness"] is True
    assert certificate["claims"]["global_loop_execution_proven"] is False
    assert certificate["claims"]["actual_cpp_emission_verified"] is False
    assert certificate["mechanically_verified"] is False
    assert not certificate["actual_cpp_executed"]
    assert not certificate["rtl_equivalence_proven"]
    assert len(certificate["mapping_sha256"])==64
    assert certificate==certify_phi_partition(point)
    phi=verify_hls_phi(point,max_iterations=200)
    assert phi["status"]=="BOUND_EXCEEDED"
    assert phi["verified_iterations"] is None
    assert phi["mapping_sha256"] is None
    assert phi["algebraic_partition"]==certificate
    assert phi["sampled_refinement"]["exhaustive"] is False


@pytest.mark.parametrize("p,pe,fold",[(1,1,1),(2,1,3),(1,2,3),(3,4,5)])
@pytest.mark.parametrize("tile_n", [1,2,7,13,25])
def test_division_partition_recovers_every_logical_output_channel(p,pe,fold,tile_n):
    shape=(3,39,11)
    point=_point("MatMul",shape=shape,tile=(2,tile_n,4),p=p,pe=pe,fold=fold)
    certificate=certify_phi_partition(point)
    factor=p*pe
    stride=factor*fold
    assert certificate["output_channel_partition"]["group_stride"]==stride
    for output in range(shape[1]):
        nb=(output//tile_n)*tile_n
        local=output-nb
        group,residual=divmod(local,stride)
        time,slot=divmod(residual,factor)
        replica,pe_index=divmod(slot,pe)
        assert 0<=time<fold
        assert 0<=replica<p
        assert 0<=pe_index<pe
        reconstructed=nb+group*stride+time*factor+replica*pe+pe_index
        assert reconstructed==output
        mapped=coordinate(point,(1,output,3))
        assert (mapped["replica"],mapped["pe"])==(replica,pe_index)
        assert mapped["reuse_step"][1:3]==(group,time)


def test_no_iteration_count_enumeration_is_required():
    point=_point("Gemm",shape=(2**30,2**20,2**25))
    result=certify_phi_partition(point)
    assert result["logical_iteration_count"]==2**75
    assert result["status"]=="ALGEBRAIC_PARTITION_DERIVED"
    assert len(result["tile_axis_partitions"])==3


@pytest.mark.parametrize("corruption",[
    {"kind":"MatMul","shape":(0,10,3),"tile":()},
    {"kind":"Gemm","shape":(2,9,4),"tile":(2,0,4)},
    {"kind":"Conv","shape":(2,9,4,4,3,3,3),
     "tile":(1,4,2,2,3,3,3)},
])
def test_illegal_contracts_rejected(corruption):
    with pytest.raises(ValueError):
        certify_phi_partition(_point(**corruption))


def test_unsupported_operators_have_no_false_certificate():
    request=ArchitectureRequest()
    point=ArchitecturePoint("relu",request,request,(1,4),"model","Relu")
    receipt=certify_phi_partition(point)
    assert receipt["status"]=="UNSUPPORTED_KIND"
    assert "logical_iteration_count" not in receipt


def test_only_typed_points_allowed():
    with pytest.raises(TypeError):
        certify_phi_partition({"operator":"not a typed IR point"})


def test_provenance_changes_with_mapping_even_when_nominal_capacity_equal():
    a=certify_phi_partition(_point(p=2,pe=1))
    b=certify_phi_partition(_point(p=1,pe=2))
    assert a["output_channel_partition"]["spatial_slots"]==2
    assert b["output_channel_partition"]["spatial_slots"]==2
    assert a["mapping_sha256"]!=b["mapping_sha256"]
    assert a["logical_iteration_count"]==b["logical_iteration_count"]
