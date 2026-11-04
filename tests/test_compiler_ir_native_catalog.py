"""Release tests for predeclared pilot-versus-paper original-HLS populations.

These validate *source preparation only*. Even if all cases match, no Vitis
CSim, CSynth, BRAM, device timing or causal performance claim is established.
"""
from __future__ import annotations

import pytest

from fpgai_research.experiments.cst_native_campaign import (
    prepare as tensor_prepare, verify_prepared,
)
from fpgai_research.experiments.cst_block_native_prepare import prepare as block_prepare


def test_mixed_size_tensor_catalog_prepares_real_matched_original_source(tmp_path):
    target=tmp_path/"tensor"
    manifest=tensor_prepare(output=target,max_cases=9,shape_catalog="paper")
    assert manifest["shape_catalog"]=="paper"
    assert len(manifest["cases"])==9
    _, rows=verify_prepared(target)
    assert len(rows)==9
    third=manifest["cases"][6:9]
    assert len({r["functional_sha256"] for r in third})==1
    assert len({r["cpp_sha256"] for r in third})==3
    assert all(r["model_shape"]==[8,16] for r in third)
    assert manifest["original_native_tool_executed_by_preparation"] is False


def test_mixed_size_block_catalog_prepares_real_three_schedule_triplet(tmp_path):
    target=tmp_path/"block"
    manifest=block_prepare(output=target,max_cases=9,shape_catalog="paper")
    assert manifest["shape_catalog"]=="paper"
    assert len(manifest["cases"])==9
    _, paired=verify_prepared(target)
    assert len(paired)==9
    third=manifest["cases"][6:9]
    assert len({r["functional_sha256"] for r in third})==1
    assert [r["row_block"] for r in third]==[1,2,8]
    assert len({r["cpp_sha256"] for r in third})==3
    assert all(r["operator_count"]==3 for r in third)
    assert third[0]["declared_scratch_elements"]*8 == third[2]["declared_scratch_elements"]
    assert manifest["original_native_tool_executed_by_preparation"] is False


@pytest.mark.parametrize("kind",["tensor","block"])
def test_unknown_shape_catalog_rejected_before_creating_source(tmp_path,kind):
    fn=tensor_prepare if kind=="tensor" else block_prepare
    target=tmp_path/"bad"
    with pytest.raises(ValueError,match="shape_catalog"):
        fn(output=target,max_cases=3,shape_catalog="impossible")
    assert not target.exists()
