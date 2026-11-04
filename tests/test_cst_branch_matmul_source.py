"""CST L5e/L5f: both mapped MatMul readers, exact per-reduction last-use.

Full Functional graph and source identities remain constant across independent
Phi changes. These are bounded E0/E1 and original GCC tests, not Vitis QoR.
"""
from dataclasses import replace
from hashlib import sha256
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.branch_matmul import (
    emit_branch_matmul_hls, disjoint_branch_offsets,
)
from fpgai_research.model_ir.branch_matmul_contract import (
    BranchEvent, branch_contract, branch_matmul_source_events,
    validate_branch_events,
)
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.validation.cpp_branch_matmul_trace import trace_branch_cpp
from fpgai_research.validation.host import validate_host


def mapped_arch(model,*,p=None,r0=None,r1=None):
    base=ArchitectureRequest()
    overrides={}
    for op,config in zip(model.operators,(p,r0,r1,None,None)):
        if config is not None:
            overrides[op.name]=replace(base,**config)
    return resolve(model,base,overrides)


@pytest.mark.skipif(shutil.which("g++") is None,reason="actual GCC required")
@pytest.mark.parametrize("dims,mappings",[
    ((2,2,3,2),({"pes":2,"simd":2},{"pes":2,"simd":2},{"pes":1,"simd":1})),
    ((3,3,5,3),({"pes":2,"serialization":2},{"simd":2,"tile":(2,2,2)},{"pes":3,"simd":2})),
    ((2,4,4,3),({"pes":2,"simd":2,"tile":(2,2,2)},{"pes":2,"simd":4},{"serialization":2,"tile":(2,2,2)})),
])
def test_two_independently_mapped_matmul_readers_are_exact_source_refined(dims,mappings):
    model=sample_branch(*dims)
    arch=mapped_arch(model,p=mappings[0],r0=mappings[1],r1=mappings[2])
    offsets=disjoint_branch_offsets(arch,reuse_after_last_read=True)
    e0=branch_contract(arch,offsets)
    m,k,n,q=dims
    assert e0["status"]=="VERIFIED_E0"
    assert e0["mapped_multi_reader_values"]==m*n
    assert e0["actual_required_h_reads"]==2*m*n*q
    events,seqs=branch_matmul_source_events(arch)
    assert validate_branch_events(arch,offsets,events)["status"]=="VERIFIED_E0"
    assert e0["phi_iterations"]=={
        "producer":m*n*k,"reader0":m*q*n,"reader1":m*q*n}
    emitted=emit_branch_matmul_hls(arch,offsets=offsets)
    assert emitted.manifest["scratch_elements"] < emitted.manifest["disjoint_source_slots"]
    x=np.asarray([(-1)**i*(i+1)/7 for i in range(m*k)],dtype=np.float32).reshape(m,k)
    numeric=validate_host(model,emitted,x)
    assert numeric["accepted"]
    traced=trace_branch_cpp(arch,emitted)
    assert traced["status"]=="EXECUTED_BRANCH_PHI_LIFETIMES_VERIFIED"
    assert traced["h_mapped_reads_verified"]==2*m*n*q
    assert traced["multi_reader_values"]==m*n
    assert traced["phi_iteration_sha256"]==e0["phi_iteration_sha256"]
    assert traced["original_vitis_calls"]==0


def test_reader_mapping_changes_only_its_local_iteration_projection():
    model=sample_branch(3,3,5,3)
    a=mapped_arch(model,r0={"pes":1,"simd":1},r1={"pes":2})
    b=mapped_arch(model,r0={"pes":2,"simd":2},r1={"pes":2})
    oa=disjoint_branch_offsets(a,reuse_after_last_read=True)
    ob=disjoint_branch_offsets(b,reuse_after_last_read=True)
    sa=branch_contract(a,oa)
    sb=branch_contract(b,ob)
    assert sa["functional_sha256"]==sb["functional_sha256"]
    assert sa["phi_iteration_sha256"]["producer"]==sb["phi_iteration_sha256"]["producer"]
    assert sa["phi_iteration_sha256"]["reader1"]==sb["phi_iteration_sha256"]["reader1"]
    assert sa["phi_iteration_sha256"]["reader0"]!=sb["phi_iteration_sha256"]["reader0"]
    assert sa["actual_required_h_reads"]==sb["actual_required_h_reads"]
    assert emit_branch_matmul_hls(a,offsets=oa).manifest["cpp_sha256"]!=(
        emit_branch_matmul_hls(b,offsets=ob).manifest["cpp_sha256"])


def test_overwrite_before_second_mapped_reader_is_concrete_rejection():
    arch=mapped_arch(sample_branch(2,2,3,2))
    good=disjoint_branch_offsets(arch,reuse_after_last_read=True)
    events,_=branch_matmul_source_events(arch)
    bad={**good,"a":0}
    result=validate_branch_events(arch,bad,events)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="OVERWRITE_BEFORE_ALL_MAPPED_READS"
    assert result["counterexample"]["address"]>=0


def test_duplicate_missing_and_unknown_mapped_read_fail_closed():
    arch=mapped_arch(sample_branch(2,2,3,2))
    offsets=disjoint_branch_offsets(arch)
    events,_=branch_matmul_source_events(arch)
    i=next(i for i,e in enumerate(events) if e.kind=="READ")
    duplicated=events[:i+1]+(events[i],)+events[i+1:]
    assert validate_branch_events(arch,offsets,duplicated)["reason"]=="DUPLICATE_MAPPED_READ"
    missing=events[:i]+events[i+1:]
    assert validate_branch_events(arch,offsets,missing)["status"]=="REJECTED"
    changed=replace(events[i],output_element=9999)
    invalid=events[:i]+(changed,)+events[i+1:]
    assert validate_branch_events(arch,offsets,invalid)["reason"]=="WRONG_PHI_READ_COORDINATE"
    with pytest.raises(ValueError,match="bound"):
        branch_matmul_source_events(arch,max_events=1)


def test_matmul_source_phi_owner_is_first_class_not_just_a_hash():
    arch=mapped_arch(sample_branch(2,2,3,2),r0={"pes":2,"simd":2})
    offsets=disjoint_branch_offsets(arch)
    events,_=branch_matmul_source_events(arch)
    i=next(i for i,e in enumerate(events)
           if e.kind=="READ" and e.consumer=="reader0")
    valid=events[i]
    assert valid.phi_owner is not None
    corrupted=events[:i]+(replace(valid,phi_owner=(99,0,0,0)),)+events[i+1:]
    result=validate_branch_events(arch,offsets,corrupted)
    assert result["reason"]=="WRONG_PHI_READ_OWNER"
    assert result["counterexample"]["required_owner"]!=[99,0,0,0]
    absent=events[:i]+(replace(valid,phi_owner=None),)+events[i+1:]
    assert validate_branch_events(arch,offsets,absent)["reason"]=="MISSING_OR_INVALID_PHI_OWNER"


def test_mixed_formats_and_unlowered_join_mapping_reject():
    model=sample_branch()
    mixed=mapped_arch(model,r0={"precision_bits":8,"integer_bits":4})
    with pytest.raises(ValueError,match="Mixed"):
        branch_contract(mixed,disjoint_branch_offsets(mixed))
    unsupported=mapped_arch(model)
    points=list(unsupported.points)
    points[3]=replace(points[3],effective=replace(points[3].effective,pipeline_ii=2),
                      requested=replace(points[3].requested,pipeline_ii=2))
    bad=replace(unsupported,points=tuple(points))
    with pytest.raises(ValueError,match="scalar"):
        branch_contract(bad,disjoint_branch_offsets(bad))


@pytest.mark.skipif(shutil.which("g++") is None,reason="actual GCC required")
def test_actual_original_cpp_trace_mutant_does_not_pass():
    model=sample_branch()
    arch=mapped_arch(model,p={"pes":2},r0={"simd":2},r1={"pes":2})
    emission=emit_branch_matmul_hls(arch,offsets=disjoint_branch_offsets(arch))
    altered=emission.cpp.replace("FPGAI_ITER 1","FPGAI_ITER 9")
    assert altered!=emission.cpp
    forged=Emission(altered,{**emission.manifest,
                              "cpp_sha256":sha256(altered.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="ordinal"):
        trace_branch_cpp(arch,forged)
    with pytest.raises(ValueError,match="identity"):
        trace_branch_cpp(mapped_arch(model),emission)
