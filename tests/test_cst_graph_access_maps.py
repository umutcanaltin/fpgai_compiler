"""Graph-wide broadcast/Concat source access maps: E0/E1 original GCC."""
from dataclasses import replace
from hashlib import sha256
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.graph_phi import emit_graph_hls
from fpgai_research.experiments.cst_graph_access_fixtures import access_graph
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.graph_access_maps import (
    scalar_output_accesses,validate_scalar_access_shape,
)
from fpgai_research.model_ir.graph_phi_contract import (
    graph_contract,graph_disjoint_offsets,graph_source_events,
    verify_graph_events,
)
from fpgai_research.model_ir.graph_edge_witness import edge_witnesses
from fpgai_research.validation.cpp_graph_phi_trace import trace_graph_cpp
from fpgai_research.validation.host import validate_host


def resolved(model,independent=True):
    base=ArchitectureRequest()
    overrides={}
    if independent:
        for i,op in enumerate(model.operators):
            if op.kind=="MatMul":
                overrides[op.name]=replace(
                    base,pes=2+(i%2),simd=2,serialization=1+i%2)
    return resolve(model,base,overrides)


@pytest.mark.parametrize("kind,m,n,q",[
    ("broadcast",2,3,1),("broadcast",3,4,1),
    ("concat",2,3,2),("concat",3,4,3),
])
@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC is required")
def test_rank_two_nonidentity_access_map_refines_original_cpp(kind,m,n,q):
    model=access_graph(kind,m=m,n=n,q=q)
    arch=resolved(model)
    disjoint=graph_disjoint_offsets(arch)
    alias={**disjoint,"out":disjoint["h"]}
    for offsets in (disjoint,alias):
        contract=graph_contract(arch,offsets)
        assert contract["status"]=="VERIFIED_E0"
        source=emit_graph_hls(arch,offsets=offsets)
        x=np.asarray([(-1)**i*(i+1)/9 for i in range(m*n)],
                     dtype=np.float32).reshape(m,n)
        assert validate_host(model,source,x)["accepted"]
        executed=trace_graph_cpp(arch,source)
        assert executed["status"]=="EXECUTED_GRAPH_BOUNDARIES_VERIFIED"
        assert executed["source_event_sha256"]==contract["source_event_sha256"]
        assert executed["last_use_sha256"]==contract["last_use_sha256"]
        assert executed["edge_witness_sha256"]==source.manifest["edge_witness_sha256"]
        assert executed["native_vitis_calls"]==0
        assert executed["independent_access_reads"]==executed["reads"]
        assert len(edge_witnesses(arch,offsets))==executed["edge_witnesses"]
    assert emit_graph_hls(arch,offsets=alias).manifest["source_array_elements"]<(
        emit_graph_hls(arch,offsets=disjoint).manifest["source_array_elements"])


def test_broadcast_row_column_and_concat_access_independent_expected_indexes():
    model=access_graph("broadcast",m=2,n=3)
    add=next(x for x in model.operators if x.kind=="Add")
    # Narrow [m,1] input is broadcast three times per row.
    assert scalar_output_accesses(model,add,0)==(
        (0,"h",0),(1,"narrow",0))
    assert scalar_output_accesses(model,add,2)==(
        (0,"h",2),(1,"narrow",0))
    assert scalar_output_accesses(model,add,5)==(
        (0,"h",5),(1,"narrow",1))
    arch=resolved(model,False)
    events,_=graph_source_events(arch)
    narrow=[e for e in events if e.kind=="READ" and e.tensor=="narrow"]
    assert len(narrow)==6
    assert [e.element for e in narrow]==[0,0,0,1,1,1]
    missing=tuple(e for i,e in enumerate(events)
                  if i!=next(j for j,x in enumerate(events)
                             if x.kind=="READ" and x.tensor=="narrow"))
    assert verify_graph_events(
        arch,graph_disjoint_offsets(arch),missing)["reason"]=="INCOMPLETE_GRAPH_RELEASE"
    concat=access_graph("concat",m=2,n=3,q=2)
    c=next(x for x in concat.operators if x.kind=="Concat")
    assert scalar_output_accesses(concat,c,0)==((0,"a",0),)
    assert scalar_output_accesses(concat,c,3)==((1,"b",0),)
    assert scalar_output_accesses(concat,c,4)==((1,"b",1),)
    assert scalar_output_accesses(concat,c,5)==((0,"a",3),)
    assert scalar_output_accesses(concat,c,9)==((1,"b",3),)


def test_invalid_axis_and_output_shape_do_not_fall_back_to_scalar():
    model=access_graph("concat")
    base=ArchitectureRequest()
    ops=tuple(replace(op,attributes=(("axis",0),))
              if op.kind=="Concat" else op for op in model.operators)
    wrong=replace(model,operators=ops)
    with pytest.raises(ValueError,match="axis|CONCAT"):
        graph_contract(resolve(wrong,base),graph_disjoint_offsets(resolve(wrong,base)))
    # A shape that violates ONNX broadcast is rejected by the architecture resolver.
    broadcast=access_graph("broadcast")
    wrong_tensors=tuple(replace(t,shape=(2,2)) if t.name=="narrow" else t
                        for t in broadcast.tensors)
    with pytest.raises(ValueError):
        resolve(replace(broadcast,tensors=wrong_tensors),base)
    # A Concat input width that no longer matches output is fail-closed.
    wrong_tensors=tuple(replace(t,shape=(2,4)) if t.name=="cat" else t
                        for t in model.tensors)
    with pytest.raises(ValueError):
        resolve(replace(model,tensors=wrong_tensors),base)


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_actual_concat_load_address_mutant_is_detected():
    arch=resolved(access_graph("concat",m=2,n=3,q=2))
    source=emit_graph_hls(arch)
    assert "source_index=idx/5*3+(col-0)" in source.cpp
    modified=source.cpp.replace("source_index=idx/5*3+(col-0)",
                                "source_index=idx/5*3+((col+1)%3)")
    assert source.cpp!=modified
    forged=Emission(modified,{**source.manifest,
                              "cpp_sha256":sha256(modified.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="SOURCE_GRAPH_EVENT_MISMATCH"):
        trace_graph_cpp(arch,forged)
