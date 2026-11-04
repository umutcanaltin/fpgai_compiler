"""E0/E1 mapping-conditioned fanout producer, GCC source and negative controls.

Consumer branches remain scalar. This test is deliberately not RTL/HLS.
"""
from dataclasses import replace
from hashlib import sha256
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.fanout_diamond import (
    emit_fanout_diamond_hls, fanout_diamond_offsets,
)
from fpgai_research.experiments.cst_fanout_source_campaign import sample_diamond
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.fanout_phi_source import (
    mapped_fanout_contract, mapped_source_order_events,
    phi_producer_iteration_order,
)
from fpgai_research.model_ir.fanout_lifetime import verify_fanout_events
from fpgai_research.validation.cpp_fanout_trace import trace_fanout_cpp
from fpgai_research.validation.host import validate_host


def phi_arch(model, *, pes=1, simd=1, serialization=1, tile=()):
    base=ArchitectureRequest()
    override=replace(base,pes=pes,simd=simd,serialization=serialization,tile=tile)
    return resolve(model,base,{model.operators[0].name:override})


@pytest.mark.skipif(shutil.which("g++") is None, reason="Requires original GCC host")
@pytest.mark.parametrize("m,n,mapping",[
    (2,3,(1,1,1,())),
    (3,5,(2,2,1,())),
    (5,3,(2,2,2,(2,2,2))),
])
def test_actual_mapped_phi_producer_and_two_independent_readers(m,n,mapping):
    model=sample_diamond(m,n)
    arch=phi_arch(model,pes=mapping[0],simd=mapping[1],
                  serialization=mapping[2],tile=mapping[3])
    offsets=fanout_diamond_offsets(model,share_released=True)
    contract=mapped_fanout_contract(arch,offsets)
    assert contract["status"]=="VERIFIED_E0"
    assert contract["producer_iteration_count"]==m*n*2
    assert contract["multi_reader_values"]==m*n
    assert verify_fanout_events(model,offsets,mapped_source_order_events(arch))["status"]=="VERIFIED"
    emitted=emit_fanout_diamond_hls(model,offsets=offsets,architecture=arch)
    assert emitted.manifest["mapped_phi_enabled"] is True
    assert emitted.manifest["producer_phi_iteration_sha256"]==contract["producer_phi_iteration_sha256"]
    x=np.asarray([(-1)**i*(i+1)/4 for i in range(m*2)],
                 dtype=np.float32).reshape(m,2)
    result=validate_host(model,emitted,x)
    assert result["accepted"]
    executed=trace_fanout_cpp(model,emitted,architecture=arch)
    assert executed["status"]=="SOURCE_FANOUT_LIFETIME_VERIFIED"
    assert executed["mapped_phi_verified"] is True
    assert executed["phi_reduction_iterations_verified"]==m*n*2
    assert executed["phi_iteration_sha256"]==contract["producer_phi_iteration_sha256"]
    assert executed["multi_reader_values"]==m*n
    assert executed["native_vitis_calls"]==0


def test_changing_only_producer_phi_changes_projection_and_emitted_cpp():
    m=sample_diamond(3,5)
    a=phi_arch(m)
    b=phi_arch(m,pes=2,simd=2)
    sa=fanout_diamond_offsets(m,share_released=True)
    ca=mapped_fanout_contract(a,sa)
    cb=mapped_fanout_contract(b,sa)
    assert ca["producer_iteration_count"]==cb["producer_iteration_count"]
    assert ca["producer_phi_iteration_sha256"]!=cb["producer_phi_iteration_sha256"]
    ea=emit_fanout_diamond_hls(m,offsets=sa,architecture=a)
    eb=emit_fanout_diamond_hls(m,offsets=sa,architecture=b)
    assert ea.manifest["functional_sha256"]==eb.manifest["functional_sha256"]
    assert ea.manifest["cpp_sha256"]!=eb.manifest["cpp_sha256"]


def test_mixed_format_and_unlowered_consumer_mapping_reject():
    model=sample_diamond(2,3)
    base=ArchitectureRequest()
    mixed=resolve(model,base,{
        model.operators[0].name:replace(base,precision_bits=8,integer_bits=4)})
    with pytest.raises(ValueError,match="Mixed numerical"):
        mapped_fanout_contract(mixed,fanout_diamond_offsets(model,share_released=True))
    spatial=resolve(model,base,{model.operators[1].name:
                               replace(base,pipeline_ii=2)})
    with pytest.raises(ValueError,match="pipeline"):
        mapped_fanout_contract(spatial,fanout_diamond_offsets(model,share_released=True))


def test_wrong_architecture_or_insufficient_events_fail_closed():
    model=sample_diamond(2,2)
    arch=phi_arch(model,pes=2)
    other=sample_diamond(3,2)
    offsets=fanout_diamond_offsets(model,share_released=True)
    with pytest.raises(ValueError,match="exact Functional"):
        emit_fanout_diamond_hls(model,offsets=offsets,
                                architecture=phi_arch(other))
    with pytest.raises(ValueError,match="finite"):
        phi_producer_iteration_order(arch,max_iterations=1)
    with pytest.raises(ValueError,match="precision"):
        emit_fanout_diamond_hls(model,offsets=offsets,architecture=arch,
                                precision_bits=8,integer_bits=4)


@pytest.mark.skipif(shutil.which("g++") is None,reason="Actual GCC required")
def test_source_mutant_changes_phi_reduction_owner_or_order():
    model=sample_diamond(2,3)
    arch=phi_arch(model,pes=2,simd=2)
    emission=emit_fanout_diamond_hls(model,
        offsets=fanout_diamond_offsets(model,share_released=True),architecture=arch)
    assert "FPGAI_ITER 0" in emission.cpp
    mutant=emission.cpp.replace("FPGAI_ITER 0", "FPGAI_ITER 1")
    changed=Emission(mutant,{**emission.manifest,
                             "cpp_sha256":sha256(mutant.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="Wrong producer operator"):
        trace_fanout_cpp(model,changed,architecture=arch)
    with pytest.raises(ValueError,match="exact resolved Phi"):
        trace_fanout_cpp(model,emission,architecture=phi_arch(model))
