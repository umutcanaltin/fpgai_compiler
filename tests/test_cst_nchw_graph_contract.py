"""General NCHW Conv/Concat/Pool/broadcast DAG E0/E1 correctness gates."""
from dataclasses import replace
from hashlib import sha256
import re,shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls,Emission
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.nchw_graph_contract import (
    admit_nchw_graph,nchw_graph_events,disjoint_nchw_offsets,
    verify_nchw_graph,nchw_graph_contract,nchw_edge_witnesses,
    nchw_witness_receipt,
)
from fpgai_research.validation.nchw_access_oracle import independent_nchw_access
from fpgai_research.validation.cpp_nchw_conv_graph_trace import trace_nchw_conv_sources
from fpgai_research.validation.host import validate_host


def mapped(model,parallel=True):
    base=ArchitectureRequest()
    override={}
    if parallel:
        for i,op in enumerate(model.operators):
            if op.kind=="Conv":
                override[op.name]=replace(base,pes=2,simd=2,
                                           serialization=2 if i%2==0 else 1)
    return resolve(model,base,override)


@pytest.mark.parametrize("kind,height,width",[
    ("branch_concat",4,4),("branch_concat",5,5),
    ("pool_fork",4,4),("broadcast_fork",4,4),
])
@pytest.mark.skipif(shutil.which("g++") is None,reason="Actual GCC required")
def test_original_nchw_dag_source_numeric_and_all_conv_load_owners(kind,height,width):
    model=nchw_case(kind,height=height,width=width)
    arch=mapped(model)
    offsets=disjoint_nchw_offsets(arch)
    _,produced=admit_nchw_graph(arch)
    assert len(produced)==len(model.operators)
    events=nchw_graph_events(arch)
    cert=nchw_graph_contract(arch,offsets)
    independent=independent_nchw_access(arch,events)
    assert cert["status"]=="VERIFIED_E0"
    assert independent["status"]=="VERIFIED_INDEPENDENT_NCHW_ACCESS_E0"
    assert cert["reads"]==independent["reads"]
    assert len(nchw_edge_witnesses(arch,offsets))>=len(model.operators)
    safe={**offsets,"out":offsets["h"]}
    assert verify_nchw_graph(arch,safe,events)["status"]=="VERIFIED_E0"
    assert nchw_witness_receipt(arch,safe)["status"]=="VERIFIED_E0"
    stage1=model.operators[1].outputs[0]
    assert verify_nchw_graph(
        arch,{**offsets,stage1:offsets["h"]},events)["reason"]==(
            "OVERWRITE_BEFORE_GLOBAL_LAST_READ")
    emitted=emit_hls(lower(arch),trace_conv_access=True)
    data=np.asarray([(-1)**i*(i+1)/13
                     for i in range(height*width)],dtype=np.float32)
    x=data.reshape(1,1,height,width)
    assert validate_host(model,emitted,x)["accepted"]
    trace=trace_nchw_conv_sources(arch,emitted)
    assert trace["status"]=="ACTUAL_CONV_SOURCE_READS_VERIFIED_E1"
    assert trace["actual_conv_reader_taps"]>0
    assert trace["source_covered_conv_operators"]>=1
    assert trace["native_vitis_calls"]==0


def test_forged_phi_missing_reduction_and_wrong_access_all_reject():
    arch=mapped(nchw_case("pool_fork"))
    offsets=disjoint_nchw_offsets(arch)
    events=nchw_graph_events(arch)
    assert verify_nchw_graph(arch,offsets,events)["status"]=="VERIFIED_E0"
    pos=next(i for i,e in enumerate(events)
             if e.kind=="READ" and e.consumer=="conv_branch")
    e=events[pos]
    mutated=events[:pos]+(replace(e,phi_owner=(99,0,0,0)),)+events[pos+1:]
    assert verify_nchw_graph(arch,offsets,mutated)["reason"]=="INCORRECT_NCHW_SOURCE_OR_PHI"
    bad_index=events[:pos]+(replace(e,element=(e.element+1)%32),)+events[pos+1:]
    assert independent_nchw_access(
        arch,bad_index)["status"]=="REJECTED_INDEPENDENT_NCHW_ACCESS"
    missing=events[:pos]+events[pos+1:]
    assert verify_nchw_graph(arch,offsets,missing)["reason"]=="INCOMPLETE_GRAPH_CONSUMPTION"
    dup=events[:pos+1]+(e,)+events[pos+1:]
    assert verify_nchw_graph(arch,offsets,dup)["reason"]=="DUPLICATE_READ_TOKEN"
    with pytest.raises(ValueError,match="invalid NCHW"):
        nchw_edge_witnesses(arch,offsets,events=missing)


def test_broadcast_operand_multiplicity_and_concat_channel_port_identity():
    broadcast=mapped(nchw_case("broadcast_fork"),parallel=False)
    events=nchw_graph_events(broadcast)
    narrow=[e for e in events if e.kind=="READ" and
            e.tensor=="narrow" and e.consumer=="broadcast_add"]
    assert len(narrow)==2*4*4
    assert len({e.element for e in narrow})==2
    assert sorted(set(e.element for e in narrow))==[0,1]
    concat=mapped(nchw_case("branch_concat"),parallel=False)
    events=nchw_graph_events(concat)
    cats=[e for e in events if e.kind=="READ" and e.consumer=="concat"]
    assert len(cats)==2*2*4*4
    assert {e.input_port for e in cats}=={0,1}
    assert {e.tensor for e in cats}=={"a","b"}


def test_reject_unsupported_nchw_graph_settings_and_scratch_budget():
    model=nchw_case("pool_fork")
    arch=mapped(model,parallel=False)
    with pytest.raises(ValueError,match="budget"):
        nchw_graph_events(arch,max_events=1)
    with pytest.raises(ValueError,match="single-input/output"):
        admit_nchw_graph(arch,max_operators=2)
    bad=replace(model,operators=tuple(
        replace(x,attributes=(("dilations",(2,2)),)) if x.name=="conv_branch" else x
        for x in model.operators))
    with pytest.raises(ValueError):
        admit_nchw_graph(mapped(bad,parallel=False))
    invalid=replace(model,operators=tuple(
        replace(x,kind="Sigmoid") if x.name=="last" else x
        for x in model.operators))
    with pytest.raises(ValueError,match="Unsupported NCHW"):
        admit_nchw_graph(mapped(invalid,parallel=False))
    with pytest.raises(ValueError,match="scratch"):
        verify_nchw_graph(arch,{"h":0},nchw_graph_events(arch))


@pytest.mark.skipif(shutil.which("g++") is None,reason="Real GCC required")
def test_mutating_actual_nchw_source_index_is_detected():
    arch=mapped(nchw_case("branch_concat"),parallel=False)
    source=emit_hls(lower(arch),trace_conv_access=True)
    m=re.search(r"const int __cst_source_index=[^;]+;",source.cpp)
    assert m is not None
    changed=m.group(0).replace("+ix;","+((ix+1)%4);")
    assert changed!=m.group(0)
    cpp=source.cpp.replace(m.group(0),changed,1)
    edited=Emission(cpp,{**source.manifest,
                        "cpp_sha256":sha256(cpp.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="ACTUAL_NCHW_CONV_LOAD_MAP_MISMATCH"):
        trace_nchw_conv_sources(arch,edited)
    with pytest.raises(ValueError,match="opt-in"):
        trace_nchw_conv_sources(arch,emit_hls(lower(arch)))
