"""Exhaustive, bounded E0 writer-ownership proofs against emitted loop indexing.

These are SOURCE-level mapping checks; Vitis HLS/RTL are never invoked.
"""
import pytest

from fpgai_research.model_ir.output_ownership import enumerate_output_ownership


def _op(kind, dims, tile, p=1, pe=1, fold=1):
    return {
        "operator":"op1","kind":kind,"iteration_shape":dims,
        "lowered_mechanisms":{
            "tile_shape":tile,"replica_count":p,"pe_count":pe,
            "fold_factor":fold,
        },
    }


@pytest.mark.parametrize("kind",["MatMul","Gemm"])
@pytest.mark.parametrize("p,pe",[(1,1),(2,1),(1,2),(2,3),(3,2)])
@pytest.mark.parametrize("fold", [1,2,3])
def test_dense_nested_tiling_and_tail_write_every_output_once(kind,p,pe,fold):
    result=enumerate_output_ownership(
        _op(kind,[3,7,5],[2,3,4],p,pe,fold))
    assert result["status"]=="EXHAUSTIVE_COVERAGE"
    assert result["expected_outputs"]==result["visited_outputs"]==21
    assert result["collisions"]==0
    assert result["equivalent_rtl_proven"] is False
    assert result["bit_exactness_proven"] is False


@pytest.mark.parametrize("p,pe",[(1,1),(2,1),(1,2),(2,3),(3,2)])
@pytest.mark.parametrize("fold", [1,2,3])
def test_conv_nested_output_tiling_with_ragged_channels_and_spatial_tails(p,pe,fold):
    result=enumerate_output_ownership(
        _op("Conv",[2,5,3,4,2,2,2],[2,2,2,3,2,2,2],p,pe,fold))
    assert result["status"]=="EXHAUSTIVE_COVERAGE"
    assert result["expected_outputs"]==result["visited_outputs"]==120
    assert result["collisions"]==0


@pytest.mark.parametrize("kind,dims,tile",[
    ("MatMul",[2,7,5],[1,3,4]),
    ("Gemm",[1,11,8],[1,4,3]),
    ("Conv",[1,7,4,5,2,3,3],[1,3,3,2,2,3,3]),
])
def test_p_times_pe_equal_yields_identical_normalized_output_ownership(kind,dims,tile):
    signatures=[]
    for p,pe in ((2,1),(1,2)):
        proof=enumerate_output_ownership(_op(kind,dims,tile,p,pe,fold=2))
        assert proof["status"]=="EXHAUSTIVE_COVERAGE"
        signatures.append(proof["normalized_ownership_sha256"])
    assert signatures[0]==signatures[1]


def test_exhaustive_signature_sensitive_to_different_channel_capacity():
    one=enumerate_output_ownership(_op("MatMul",[1,7,3],[1,3,3],1,1))
    two=enumerate_output_ownership(_op("MatMul",[1,7,3],[1,3,3],2,1))
    assert one["normalized_ownership_sha256"]!=two["normalized_ownership_sha256"]


def test_larger_than_safe_bound_is_unverified_not_a_pass():
    result=enumerate_output_ownership(
        _op("Conv",[2,32,128,128,4,3,3],[2,4,16,16,4,3,3]),
        max_outputs=50_000)
    assert result["status"]=="BOUND_EXCEEDED"
    assert result["expected_outputs"]==2*32*128*128
    assert result["visited_outputs"] is None
    assert result["normalized_ownership_sha256"] is None


def test_unsupported_kinds_are_explicitly_not_proven():
    result=enumerate_output_ownership(_op("Softmax",[1,4],[1,4]))
    assert result["status"]=="UNSUPPORTED_KIND"
    assert result["normalized_ownership_sha256"] is None


@pytest.mark.parametrize("dims,tile,p,pe,fold",[
    ([1,7,2],[1,0,2],1,1,1),
    ([1,7,2],[1,7,2],0,1,1),
    ([1,7,2],[1,7,2],1,0,1),
    ([1,7,2],[1,7,2],1,1,0),
    ([1,7,2],[1,7],1,1,1),
    ([1,7,2],[1,7,2],1,True,1),
])
def test_invalid_geometries_fail_closed(dims,tile,p,pe,fold):
    with pytest.raises(ValueError):
        enumerate_output_ownership(_op("MatMul",dims,tile,p,pe,fold))


@pytest.mark.parametrize("bound",[0,-1,True,2.5])
def test_invalid_enumeration_bound_fails_closed(bound):
    with pytest.raises(ValueError):
        enumerate_output_ownership(_op("MatMul",[1,2,3],[1,2,3]),
                                   max_outputs=bound)
