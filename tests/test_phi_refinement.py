"""Phi semantics must match independent HLS loop-index enumerations, E0 only."""
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.contracts import (
    ArchitecturePoint, ArchitectureRequest,
    Tensor, Operator, FunctionalModel,
)
from fpgai_research.model_ir import phi_refinement as phi


def _point(kind,shape,*,tile=(),p=1,pes=1,simd=1,fold=1):
    a=ArchitectureRequest(parallelism=p,pes=pes,simd=simd,
                          serialization=fold,tile=tile)
    return ArchitecturePoint("test_operator",a,a,shape,
                             "model",kind)


@pytest.mark.parametrize("kind",("MatMul","Gemm"))
@pytest.mark.parametrize("p,pes",((1,1),(2,1),(1,2),(2,3)))
@pytest.mark.parametrize("simd", (1,2,4))
@pytest.mark.parametrize("fold", (1,2))
@pytest.mark.parametrize("tile", ((),(2,3,2),(1,4,5)))
def test_dense_phi_matches_tiled_emitter_loop_indexing(kind,p,pes,simd,fold,tile):
    result=phi.verify_hls_phi(_point(
        kind,(3,7,5),tile=tile,p=p,pes=pes,simd=simd,fold=fold))
    assert result["status"]=="EXHAUSTIVE_PHI_REFINEMENT"
    assert result["expected_iterations"]==result["verified_iterations"]==105
    assert result["coverage_complete"] is True
    assert result["collisions"]==0
    assert not result["bit_exactness_proven"]
    assert not result["rtl_schedule_proven"]


@pytest.mark.parametrize("p,pes",((1,1),(2,1),(1,2),(2,3)))
@pytest.mark.parametrize("simd",(1,2,3,4))
@pytest.mark.parametrize("fold",(1,2,3))
def test_conv_phi_matches_hls_loop_indexing_with_partial_output_tiles(
        p,pes,simd,fold):
    shape=(2,5,3,4,3,2,2)
    tile=(2,2,2,3,3,2,2)
    result=phi.verify_hls_phi(_point(
        "Conv",shape,tile=tile,p=p,pes=pes,simd=simd,fold=fold))
    assert result["status"]=="EXHAUSTIVE_PHI_REFINEMENT"
    assert result["verified_iterations"]==2*5*3*4*3*2*2


def test_real_resolver_point_is_suitable_for_independent_phi_proof():
    model=FunctionalModel(
        (Tensor("x",(2,3)),
         Tensor("w",(3,7),data=tuple([.25]*21)),
         Tensor("y",(2,7))),
        (Operator("mm","MatMul",("x","w"),("y",)),),
        ("x",),("y",))
    req=ArchitectureRequest(parallelism=2,pes=2,simd=4,
                             serialization=2,tile=(1,4,2))
    point=resolve(model,ArchitectureRequest(),{"mm":req}).points[0]
    result=phi.verify_hls_phi(point)
    assert result["status"]=="EXHAUSTIVE_PHI_REFINEMENT"
    assert result["verified_iterations"]==42


def test_phi_mismatch_is_never_downgraded_to_success(monkeypatch):
    point=_point("MatMul",(2,3,4),tile=(1,2,2),p=2,simd=3)
    original=phi.coordinate
    def wrong(point,index):
        data=original(point,index)
        if index==(0,0,0):
            data["lane"]=42
        return data
    monkeypatch.setattr(phi,"coordinate",wrong)
    with pytest.raises(AssertionError,match="MODELIR-PHI-REFINEMENT"):
        phi.verify_hls_phi(point)


@pytest.mark.parametrize("shape,tile",[
    ((1,2,3),(1,2)),((1,2,3),(1,0,3)),
    ((0,2,3),(1,2,3)),((1,2),(1,2)),
])
def test_invalid_phi_geometry_rejected(shape,tile):
    with pytest.raises(ValueError):
        phi.verify_hls_phi(_point("MatMul",shape,tile=tile))


def test_unsupported_conv_reduction_tiling_rejected():
    point=_point("Conv",(1,3,2,2,4,2,2),
                 tile=(1,2,2,2,2,2,2))
    with pytest.raises(ValueError,match="Unsupported convolution"):
        phi.verify_hls_phi(point)


def test_large_domain_is_unverified_not_proven():
    point=_point("MatMul",(32,128,512),p=2,pes=2)
    result=phi.verify_hls_phi(point,max_iterations=1000)
    assert result["status"]=="BOUND_EXCEEDED"
    assert result["verified_iterations"] is None
    assert result["mapping_sha256"] is None
    sampled=result["sampled_refinement"]
    assert sampled["status"]=="SAMPLED_COUNTERS_MATCH_PHI"
    assert sampled["exhaustive"] is False
    assert 0<sampled["sampled_iterations"]<=256
    assert not sampled["actual_cpp_executed"]
    algebra= result["algebraic_partition"]
    assert algebra["status"]=="ALGEBRAIC_PARTITION_DERIVED"
    assert algebra["logical_iteration_count"]==32*128*512
    assert algebra["claims"]["integer_partition_uniqueness"] is True
    assert algebra["claims"]["actual_cpp_emission_verified"] is False
    assert algebra["mechanically_verified"] is False


def test_unsupported_operator_is_unverified():
    result=phi.verify_hls_phi(_point("Relu",(1,2),tile=()))
    assert result["status"]=="UNSUPPORTED_KIND"


@pytest.mark.parametrize("bound",[0,-1,True,1.5])
def test_invalid_iteration_bound_fails_closed(bound):
    with pytest.raises(ValueError):
        phi.verify_hls_phi(_point("MatMul",(1,2,3)),
                           max_iterations=bound)


def test_invalid_point_is_rejected():
    with pytest.raises(TypeError,match="ArchitecturePoint"):
        phi.verify_hls_phi({"operator":"not-a-typed-contract"})
