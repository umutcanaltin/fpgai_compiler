"""L6a graph-wide bounded nested fan-out: source, Phi, numerical and errors.

Not a generic ONNX, Vitis or physical-memory certification.
"""
from dataclasses import replace
from hashlib import sha256
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.graph_phi import emit_graph_hls
from fpgai_research.experiments.cst_graph_phi_fixtures import sample_graph
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.graph_phi_contract import (
    graph_contract, graph_disjoint_offsets, graph_source_events,
    verify_graph_events,
)
from fpgai_research.validation.cpp_graph_phi_trace import trace_graph_cpp
from fpgai_research.validation.host import validate_host


def mapped(model):
    base=ArchitectureRequest()
    overrides={}
    for index,op in enumerate(model.operators):
        if op.kind=="MatMul":
            overrides[op.name]=replace(base,
                pes=2 if index%2==0 else 3,
                simd=2 if index%2==0 else 1,
                serialization=2 if index%2 else 1)
    return resolve(model,base,overrides)


@pytest.mark.skipif(shutil.which("g++") is None,reason="Original GCC required")
@pytest.mark.parametrize("kind,rows,columns",[
    ("mixed",2,3),("triple_reader",2,4),("nested",3,4),
])
def test_original_source_executes_generalized_fanout_with_independent_phi(kind,rows,columns):
    model=sample_graph(kind,m=rows,n=columns)
    arch=mapped(model)
    disjoint=graph_disjoint_offsets(arch)
    copied={**disjoint,"out":disjoint["h"]}
    for label,offsets in (("disjoint",disjoint),("released",copied)):
        cert=graph_contract(arch,offsets)
        assert cert["status"]=="VERIFIED_E0"
        assert cert["operator_count"]>=6
        assert cert["edge_count"]>=5
        assert cert["multi_reader_values"]>=rows*columns
        src=emit_graph_hls(arch,offsets=offsets)
        x=np.asarray([(-1)**i*(i+1)/7 for i in range(rows*columns)],
                     dtype=np.float32).reshape(rows,columns)
        host=validate_host(model,src,x)
        assert host["accepted"]
        receipt=trace_graph_cpp(arch,src)
        assert receipt["status"]=="EXECUTED_GRAPH_BOUNDARIES_VERIFIED"
        assert receipt["independent_phi_issue_status"]=="VERIFIED_E0"
        assert receipt["independent_phi_issue_checked_reads"]==cert["independent_phi_issue_checked_reads"]
        assert receipt["independent_phi_issue_checked_operators"]==cert["independent_phi_issue_checked_operators"]
        assert receipt["independent_phi_issue_signature_sha256"]==cert["independent_phi_issue_signature_sha256"]
        assert src.manifest["independent_phi_issue_signature_sha256"]==cert["independent_phi_issue_signature_sha256"]
        assert receipt["source_event_sha256"]==cert["source_event_sha256"]
        assert receipt["last_use_sha256"]==cert["last_use_sha256"]
        assert receipt["native_vitis_calls"]==0
        assert receipt["operator_count"]==len(model.operators)
        if label=="released":
            assert receipt["source_array_elements"]<sum(
                model.tensor(op.outputs[0]).shape[0]*model.tensor(op.outputs[0]).shape[1]
                for op in model.operators)


def test_graph_requires_every_consumer_read_not_only_consumer_ack():
    arch=mapped(sample_graph("nested",m=2,n=3))
    offsets=graph_disjoint_offsets(arch)
    events,_=graph_source_events(arch)
    assert verify_graph_events(arch,offsets,events)["status"]=="VERIFIED_E0"
    target=next(i for i,x in enumerate(events)
                if x.kind=="READ" and x.consumer=="reader0")
    event=events[target]
    missing=events[:target]+events[target+1:]
    assert verify_graph_events(arch,offsets,missing)["reason"]=="INCOMPLETE_GRAPH_RELEASE"
    doubled=events[:target+1]+(event,)+events[target+1:]
    assert verify_graph_events(arch,offsets,doubled)["reason"]=="DUPLICATE_CONTRIBUTION_READ"
    forged=events[:target]+(replace(event,phi_owner=(99,0,0,0)),)+events[target+1:]
    out=verify_graph_events(arch,offsets,forged)
    assert out["reason"]=="WRONG_PHI_CONSUMER_OWNER"


def test_nested_graph_rejects_alias_before_last_branch_and_stale_read():
    arch=mapped(sample_graph("nested"))
    offsets=graph_disjoint_offsets(arch)
    events,_=graph_source_events(arch)
    bad={**offsets,"a":offsets["h"]}
    cert=verify_graph_events(arch,bad,events)
    assert cert["status"]=="REJECTED"
    assert cert["reason"]=="OVERWRITE_BEFORE_LAST_USE"
    assert cert["counterexample"]["address"]>=0
    i=next(i for i,e in enumerate(events)
           if e.kind=="READ" and e.tensor=="h")
    premature=(events[i],)+events
    assert verify_graph_events(arch,offsets,premature)["reason"]=="READ_BEFORE_COMMIT_OR_STALE"


def test_unknown_operator_and_unimplemented_scalar_phi_fail_closed():
    model=sample_graph("mixed")
    base=ArchitectureRequest()
    override={model.operators[-2].name:replace(base,pes=2)}
    with pytest.raises(ValueError,match="scalar"):
        graph_contract(resolve(model,base,override),graph_disjoint_offsets(resolve(model,base,override)))
    broken=replace(model,operators=tuple(
        replace(op,kind="Identity") if op.name=="final" else op
        for op in model.operators))
    arch=resolve(broken,base)
    with pytest.raises(ValueError,match="Unsupported graph operator"):
        emit_graph_hls(arch)
    with pytest.raises(ValueError,match="bound"):
        graph_source_events(mapped(model),max_events=1)


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_real_cpp_mutant_is_not_a_qualified_source_refinement():
    arch=mapped(sample_graph("mixed"))
    source=emit_graph_hls(arch)
    needle="FPGAI_BRANCH READ h"
    assert needle in source.cpp
    changed=source.cpp.replace(needle,"FPGAI_BRANCH READ out")
    forged=Emission(changed,{**source.manifest,
                             "cpp_sha256":sha256(changed.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="Invalid graph source read input binding"):
        trace_graph_cpp(arch,forged)
    wrong=mapped(sample_graph("mixed",m=3,n=3))
    with pytest.raises(ValueError,match="hash mismatch"):
        trace_graph_cpp(wrong,source)
    # With the optional source-linked mode, the *same* expression drives
    # the actual scratch load and the observed read index. A valid-range
    # one-column source-address corruption cannot be hidden behind an
    # independent instrumentation formula.
    assert "fpgai_source_element=mi*3+kk" in source.cpp
    wrong_address=source.cpp.replace("fpgai_source_element=mi*3+kk",
                                      "fpgai_source_element=mi*3+((kk+1)%3)")
    assert wrong_address!=source.cpp
    mutated=Emission(wrong_address,{**source.manifest,
                                   "cpp_sha256":sha256(wrong_address.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="SOURCE_GRAPH_REDUCTION_READ_MISMATCH"):
        trace_graph_cpp(arch,mutated)

@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_executed_cpp_rejects_forged_independent_phi_issue_manifest():
    arch=mapped(sample_graph("nested",m=2,n=3))
    source=emit_graph_hls(arch)
    assert source.manifest["independent_phi_issue_status"]=="VERIFIED_E0"
    altered=Emission(source.cpp,{**source.manifest,
        "independent_phi_issue_signature_sha256":"0"*64})
    with pytest.raises(AssertionError,match="SOURCE_GRAPH_INDEPENDENT_PHI_ISSUE_RECEIPT_MISMATCH"):
        trace_graph_cpp(arch,altered)
