"""Independent access oracle must detect shared-helper semantic bugs."""
from dataclasses import replace
import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_graph_access_fixtures import access_graph
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir import graph_phi_contract as graph
from fpgai_research.model_ir.graph_access_maps import scalar_output_accesses
from fpgai_research.validation.graph_access_oracle import independent_numpy_access_oracle


@pytest.mark.parametrize("kind,m,n,q",[
    ("broadcast",2,3,1),("broadcast",3,4,1),
    ("concat",2,3,2),("concat",3,4,3),
])
def test_numpy_expected_read_multiset_matches_graph_certificate(kind,m,n,q):
    arch=resolve(access_graph(kind,m=m,n=n,q=q),ArchitectureRequest())
    events,_=graph.graph_source_events(arch)
    result=independent_numpy_access_oracle(arch,events)
    assert result["status"]=="VERIFIED_INDEPENDENT_ACCESS_E0"
    assert result["read_obligations"]==sum(e.kind=="READ" for e in events)
    assert result["original_vitis_calls"]==0


def test_numpy_oracle_rejects_a_wrong_in_bounds_concat_map_even_if_e0_shares_bug(monkeypatch):
    # Deliberately make production event construction AND production verifier
    # agree on the *same wrong* per-source index mapping. All tensor values
    # remain read and released, so logical lifetimes do not spot the error.
    arch=resolve(access_graph("concat",m=2,n=3,q=2),ArchitectureRequest())
    expected,_=graph.graph_source_events(arch)
    oracle_ok=independent_numpy_access_oracle(arch,expected)
    assert oracle_ok["status"]=="VERIFIED_INDEPENDENT_ACCESS_E0"
    def wrong_concat(model,op,index):
        result=scalar_output_accesses(model,op,index)
        if op.kind=="Concat":
            port,tensor,flat=result[0]
            cols=model.tensor(tensor).shape[1]
            row,col=divmod(flat,cols)
            return ((port,tensor,row*cols+(col+1)%cols),)
        return result
    monkeypatch.setattr(graph,"scalar_output_accesses",wrong_concat)
    faulty,_=graph.graph_source_events(arch)
    logical=graph.verify_graph_events(arch,graph.graph_disjoint_offsets(arch),faulty)
    assert logical["status"]=="VERIFIED_E0"
    result=independent_numpy_access_oracle(arch,faulty)
    assert result["status"]=="REJECTED_INDEPENDENT_ACCESS_MAP"
    assert result["first_missing"] is not None
    assert result["first_unexpected"] is not None


def test_unknown_or_missing_read_rejected_by_independent_oracle():
    arch=resolve(access_graph("broadcast",m=2,n=3),ArchitectureRequest())
    original,_=graph.graph_source_events(arch)
    pos=next(i for i,e in enumerate(original)
             if e.kind=="READ" and e.consumer=="broadcast_add")
    event=original[pos]
    forged=original[:pos]+(replace(event,element=(event.element+1)%6),)+original[pos+1:]
    assert independent_numpy_access_oracle(
        arch,forged)["status"]=="REJECTED_INDEPENDENT_ACCESS_MAP"
    assert independent_numpy_access_oracle(
        arch,original[:pos]+original[pos+1:])["status"]=="REJECTED_INDEPENDENT_ACCESS_MAP"
    with pytest.raises(ValueError,match="budget"):
        independent_numpy_access_oracle(arch,original,max_events=1)
