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
    assert independent["phi_owner_validation"]=="VERIFIED_INDEPENDENT_CONV_OWNER_E0"
    assert independent["phi_owner_checked_conv_reads"]>0
    assert len(independent["phi_owner_multiset_sha256"])==64
    assert cert["independent_conv_phi_owner_status"]=="VERIFIED_INDEPENDENT_CONV_OWNER_E0"
    assert cert["independent_conv_phi_owner_multiset_sha256"]==independent["phi_owner_multiset_sha256"]
    assert cert["independent_conv_phi_owner_checked_reads"]==independent["phi_owner_checked_conv_reads"]
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
    assert trace["independent_conv_phi_owner_status"]=="VERIFIED_INDEPENDENT_CONV_OWNER_E0"
    assert trace["independent_conv_phi_owner_checked_reads"]>0
    assert trace["independent_conv_phi_owner_multiset_sha256"]==independent["phi_owner_multiset_sha256"]
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
    # An intermediate READ deletion is now rejected at the *consumer's*
    # COMMIT, rather than at the later end-of-trace completeness check.
    assert verify_nchw_graph(arch,offsets,missing)["reason"]=="COMMIT_BEFORE_MODELED_INPUT_READS"
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


@pytest.mark.parametrize("kind,channels", [
    ("branch_concat",2), ("branch_concat",4),
    ("pool_fork",2), ("pool_fork",3), ("broadcast_fork",4),
])
def test_independent_conv_phi_ownership_rejects_value_safe_mutations(
        kind,channels):
    arch=mapped(nchw_case(kind,channels=channels))
    events=nchw_graph_events(arch)
    baseline=independent_nchw_access(arch,events)
    assert baseline["status"]=="VERIFIED_INDEPENDENT_NCHW_ACCESS_E0"
    assert baseline["phi_owner_checked_conv_reads"]>0
    assert baseline["oracle_independent_of_resolver_coordinate"] is True
    index=next(i for i,event in enumerate(events)
               if event.kind=="READ" and event.phi_owner is not None)
    event=events[index]
    bad=replace(event,phi_owner=(
        event.phi_owner[0],event.phi_owner[1]+1,
        event.phi_owner[2],event.phi_owner[3]))
    mutated=events[:index]+(bad,)+events[index+1:]
    # Input tensor, source element, reduction and consumer stay unchanged:
    # the independent NumPy index multiset itself is still satisfied.
    observed_access=[(e.tensor,e.element,e.consumer,e.output_element,
                      e.input_port,e.reduction_k)
                     for e in mutated if e.kind=="READ"]
    original_access=[(e.tensor,e.element,e.consumer,e.output_element,
                      e.input_port,e.reduction_k)
                     for e in events if e.kind=="READ"]
    assert observed_access==original_access
    rejection=independent_nchw_access(arch,mutated)
    assert rejection["status"]=="REJECTED_INDEPENDENT_NCHW_ACCESS"
    assert rejection["reason"]=="INDEPENDENT_CONV_PHI_OWNER_MISMATCH"
    assert rejection["first_unexpected_owner"] is not None


def test_independent_conv_owner_detects_common_mode_resolver_error(monkeypatch):
    from fpgai_research.model_ir import nchw_graph_contract as graph_module
    from fpgai_research.architecture.resolver import coordinate as original_coordinate
    arch=mapped(nchw_case("branch_concat",channels=4))
    offsets=disjoint_nchw_offsets(arch)
    source_events=nchw_graph_events(arch)
    assert independent_nchw_access(
        arch,source_events)["status"]=="VERIFIED_INDEPENDENT_NCHW_ACCESS_E0"

    def corrupted_coordinate(point,index):
        result=original_coordinate(point,index)
        if point.kind!="Conv":
            return result
        # Simulate one incorrect shared mapper used by BOTH the generator
        # and its non-independent E0 replay verifier. The mutated owner is
        # still within the declared PE set, but is the WRONG owner.
        return {**result,"pe":(result["pe"]+1)%point.effective.pes}

    monkeypatch.setattr(graph_module,"coordinate",corrupted_coordinate)
    common_mode_events=graph_module.nchw_graph_events(arch)
    assert common_mode_events!=source_events
    assert graph_module.verify_nchw_graph(
        arch,offsets,common_mode_events)["status"]=="VERIFIED_E0"
    independent=independent_nchw_access(arch,common_mode_events)
    assert independent["reason"]=="INDEPENDENT_CONV_PHI_OWNER_MISMATCH"
    with pytest.raises(ValueError,match="NCHW_INDEPENDENT_ACCESS_OR_PHI_INVALID"):
        graph_module.nchw_graph_contract(arch,offsets)


def test_independent_conv_owner_has_no_misleading_hardware_claim():
    arch=mapped(nchw_case("pool_fork"))
    report=independent_nchw_access(arch,nchw_graph_events(arch))
    assert report["original_vitis_calls"]==0
    assert report["phi_owner_checked_conv_reads"]>0
    assert any("No physical PE" in item for item in report["limitations"])


@pytest.mark.parametrize("kind",["branch_concat","pool_fork","broadcast_fork"])
def test_nchw_output_commit_requires_all_modeled_produced_input_reads(kind):
    arch=mapped(nchw_case(kind),parallel=False)
    events=nchw_graph_events(arch)
    # The last Relu reads the completed previous tensor element. Moving its
    # output COMMIT before that READ keeps the multiset and readiness intact,
    # but makes the source operation causally impossible.
    first=next(i for i,e in enumerate(events)
               if e.kind=="READ" and e.consumer=="last"
               and e.output_element==0)
    commiti=next(i for i,e in enumerate(events)
                 if e.kind=="COMMIT" and e.tensor=="out" and e.element==0)
    assert first<commiti
    shuffled=list(events)
    earlier=shuffled.pop(commiti)
    shuffled.insert(first,earlier)
    moved=tuple(shuffled)
    expected_access=independent_nchw_access(arch,events)
    actual_access=independent_nchw_access(arch,moved)
    assert expected_access["status"]=="VERIFIED_INDEPENDENT_NCHW_ACCESS_E0"
    assert actual_access["status"]=="VERIFIED_INDEPENDENT_NCHW_ACCESS_E0"
    assert actual_access["access_multiset_sha256"]==expected_access["access_multiset_sha256"]
    result=verify_nchw_graph(arch,disjoint_nchw_offsets(arch),moved)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="COMMIT_BEFORE_MODELED_INPUT_READS"
    assert result["counterexample"]["observed_reads"]<result["counterexample"]["required_reads"]
    original=verify_nchw_graph(arch,disjoint_nchw_offsets(arch),events)
    assert original["status"]=="VERIFIED_E0"
    assert original["modeled_produced_input_read_obligations"]>0


def test_independent_nchw_owner_normalizes_numpy_coordinates_before_digest():
    import json
    from fpgai_research.validation.nchw_access_oracle import (
        _independent_conv_owner,
    )
    arch=mapped(nchw_case("branch_concat",channels=4))
    conv=arch.points[0]
    owner_np=_independent_conv_owner(conv,np.int64(1),np.int64(0))
    owner_python=_independent_conv_owner(conv,1,0)
    assert owner_np==owner_python
    assert all(type(component) is int for component in owner_np)
    assert json.loads(json.dumps({"owner":owner_np}))["owner"]==list(owner_np)
    # This is the path that previously raised TypeError(np.int64) in the
    # owner evidence SHA-256 used by both logical and actual-GCC tests.
    first=independent_nchw_access(arch,nchw_graph_events(arch))
    second=independent_nchw_access(arch,nchw_graph_events(arch))
    assert first["phi_owner_multiset_sha256"]==second["phi_owner_multiset_sha256"]
    assert len(first["phi_owner_multiset_sha256"])==64


def test_independent_nchw_owner_rejects_float_and_boolean_channels():
    from fpgai_research.validation.nchw_access_oracle import (
        _independent_conv_owner,
    )
    conv=mapped(nchw_case("branch_concat",channels=4)).points[0]
    for out_channel in (1.0,True,np.float64(1),np.bool_(True)):
        with pytest.raises(ValueError,match="Integral Conv"):
            _independent_conv_owner(conv,out_channel,0)


def test_missing_external_sink_read_remains_incomplete_consumption():
    arch=mapped(nchw_case("pool_fork"),parallel=False)
    events=nchw_graph_events(arch)
    sink=next(i for i,e in enumerate(events)
              if e.kind=="READ" and e.consumer=="EXTERNAL")
    missing_sink=events[:sink]+events[sink+1:]
    # The new source-operand-before-COMMIT rule does not replace the
    # independent requirement that all final-output sink reads occur.
    result=verify_nchw_graph(
        arch,disjoint_nchw_offsets(arch),missing_sink)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="INCOMPLETE_GRAPH_CONSUMPTION"
