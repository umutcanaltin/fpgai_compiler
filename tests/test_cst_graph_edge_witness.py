"""First-class per-tensor-edge Phi/readiness/address witnesses and falsifiers."""
from dataclasses import replace
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.graph_phi import emit_graph_hls
from fpgai_research.experiments.cst_graph_phi_fixtures import sample_graph
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.graph_phi_contract import (
    graph_disjoint_offsets,graph_source_events,
)
from fpgai_research.model_ir.graph_edge_witness import (
    edge_witnesses,edge_witness_receipt,
)


def test_independent_branch_edge_value_witnesses_bind_full_last_reader():
    graph=sample_graph("nested",m=2,n=3)
    base=ArchitectureRequest()
    arch=resolve(graph,base,{"reader0":replace(base,pes=2,simd=2)})
    offsets=graph_disjoint_offsets(arch)
    receipt=edge_witness_receipt(arch,offsets)
    assert receipt["status"]=="VERIFIED_E0"
    assert receipt["edges"]>=9
    witnesses=edge_witnesses(arch,offsets)
    h0=next(x for x in witnesses if x.tensor=="h" and x.consumer=="reader0")
    h1=next(x for x in witnesses if x.tensor=="h" and x.consumer=="reader1")
    assert h0.producer=="producer"
    assert h0.producer_phi_projection_sha256==h1.producer_phi_projection_sha256
    assert h0.consumer_phi_projection_sha256!=h1.consumer_phi_projection_sha256
    for x,y in zip(h0.values,h1.values):
        assert x.element==y.element
        assert x.source_address==offsets["h"]+x.element
        assert x.ready_at_event==y.ready_at_event
        assert x.release_after_event==y.release_after_event
        assert x.release_after_event>max(x.consumer_reads)
        assert x.release_after_event==max(y.consumer_reads)
        assert len(x.consumer_reads)==3
        assert len(y.consumer_reads)==3
        assert all(owner is not None for owner in x.phi_read_owners)
    assert emit_graph_hls(arch,offsets=offsets).manifest["edge_witness_sha256"]==receipt["source_witness_sha256"]


def test_witness_rejects_stale_storage_ownership_before_fabrication():
    arch=resolve(sample_graph("mixed"),ArchitectureRequest())
    good=graph_disjoint_offsets(arch)
    events,_=graph_source_events(arch)
    bad={**good,"a":good["h"]}
    with pytest.raises(ValueError,match="NO_GRAPH_EDGE_CERTIFICATE_FOR_INVALID_EVENTS"):
        edge_witness_receipt(arch,bad,events=events)
    unknown=events[:-1]
    with pytest.raises(ValueError,match="INCOMPLETE_GRAPH_RELEASE"):
        edge_witnesses(arch,good,events=unknown)


def test_operator_local_phi_changes_only_corresponding_edge_projection():
    graph=sample_graph("nested",m=2,n=3)
    base=ArchitectureRequest()
    first=resolve(graph,base)
    second=resolve(graph,base,{"reader0":replace(base,pes=2,simd=2)})
    p=graph_disjoint_offsets(first)
    a=edge_witnesses(first,p)
    b=edge_witnesses(second,p)
    left=next(x for x in a if x.tensor=="h" and x.consumer=="reader0")
    right=next(x for x in b if x.tensor=="h" and x.consumer=="reader0")
    assert left.consumer_phi_projection_sha256!=right.consumer_phi_projection_sha256
    assert left.producer_phi_projection_sha256==right.producer_phi_projection_sha256
