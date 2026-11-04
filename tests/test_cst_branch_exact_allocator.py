"""Independent oracle, budget, mutations and E1 original GCC for L5f.

Brute force enumerates every small contiguous-base offset assignment and
replays the production verifier independently of the branch-and-bound
solver's lifetime interval/conflict graph implementation.
"""
from dataclasses import replace
from itertools import product
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import emit_branch_matmul_hls
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.model_ir.branch_matmul_contract import (
    branch_matmul_source_events,validate_branch_events,
)
from fpgai_research.model_ir.branch_exact_allocator import exact_mapped_branch_slots
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_branch_matmul_trace import trace_branch_cpp
from fpgai_research.validation.host import validate_host


def architecture(m=1,k=1,n=2,q=1,*,custom=False):
    graph=sample_branch(m,k,n,q)
    base=ArchitectureRequest()
    overrides={} if not custom else {
        graph.operators[0].name:replace(base,pes=2,simd=2),
        graph.operators[1].name:replace(base,pes=2,simd=2),
        graph.operators[2].name:replace(base,pes=2,serialization=2),
    }
    return resolve(graph,base,overrides)


def brute_force_reference(arch,*,max_slots=15):
    model=arch.functional
    h,a,b,g=(model.operators[i].outputs[0] for i in range(4))
    sizes={name:model.tensor(name).shape[0]*model.tensor(name).shape[1]
           for name in (h,a,b,g)}
    events,_=branch_matmul_source_events(arch)
    count=0
    for cap in range(max(sizes.values()),max_slots+1):
        for values in product(*(range(cap-sizes[name]+1)
                                for name in (h,a,b,g))):
            count+=1
            offsets=dict(zip((h,a,b,g),values))
            result=validate_branch_events(arch,offsets,events)
            if result["status"]=="VERIFIED_E0":
                return cap,offsets,count
    return None,None,count


@pytest.mark.parametrize("dims",[(1,1,2,1),(2,1,2,1),(1,2,3,1)])
def test_optimizer_matches_independent_exhaustive_reference(dims):
    arch=architecture(*dims)
    expected,offsets,attempts=brute_force_reference(arch,max_slots=16)
    proof=exact_mapped_branch_slots(arch,max_slots=16,max_candidates=100_000)
    assert proof["status"]=="OPTIMAL_E0"
    assert proof["proven_optimal"] is True
    assert proof["physical_slots"]==expected
    assert validate_branch_events(
        arch,proof["offsets"],branch_matmul_source_events(arch)[0])["status"]=="VERIFIED_E0"


def test_tiny_budget_is_never_promoted_to_a_fake_optimum():
    arch=architecture(2,2,3,2,custom=True)
    result=exact_mapped_branch_slots(arch,max_slots=40,max_candidates=1)
    assert result["status"]=="UNKNOWN_BUDGET"
    assert result["offsets"] is None
    assert result["proven_optimal"] is False
    assert result["examined_candidates"]==1


def test_too_small_capacity_is_marked_infeasible_not_unknown():
    arch=architecture(2,2,3,2)
    result=exact_mapped_branch_slots(arch,max_slots=3)
    assert result["status"]=="INFEASIBLE_WITHIN_CAPACITY"
    assert result["offsets"] is None


def test_optimum_reacts_to_independent_reader_phi_without_changing_graph_identity():
    graph=sample_branch(2,2,3,2)
    base=ArchitectureRequest()
    a=resolve(graph,base)
    b=resolve(graph,base,{
        graph.operators[1].name:replace(base,pes=2,simd=2)})
    one=exact_mapped_branch_slots(a,max_slots=40,max_candidates=200_000)
    two=exact_mapped_branch_slots(b,max_slots=40,max_candidates=200_000)
    assert one["status"]=="OPTIMAL_E0" and two["status"]=="OPTIMAL_E0"
    assert one["functional_sha256"]==two["functional_sha256"]
    assert one["architecture_sha256"]!=two["architecture_sha256"]
    assert one["mapped_h_reads"]==two["mapped_h_reads"]
    assert one["source_event_sha256"]!=two["source_event_sha256"]


@pytest.mark.skipif(shutil.which("g++") is None,reason="original host GCC required")
def test_true_mapped_optimum_is_emitted_and_gcc_executed():
    arch=architecture(2,2,3,1,custom=True)
    result=exact_mapped_branch_slots(arch,max_slots=30,max_candidates=250_000)
    assert result["status"]=="OPTIMAL_E0"
    assert result["physical_slots"]<result["total_disjoint_slots"]
    source=emit_branch_matmul_hls(arch,offsets=result["offsets"])
    assert source.manifest["scratch_elements"]==result["physical_slots"]
    x=np.asarray([.25,-.5,.75,-1.0],dtype=np.float32).reshape(2,2)
    assert validate_host(arch.functional,source,x)["accepted"]
    traced=trace_branch_cpp(arch,source)
    assert traced["status"]=="EXECUTED_BRANCH_PHI_LIFETIMES_VERIFIED"
    assert traced["h_mapped_reads_verified"]==result["mapped_h_reads"]
