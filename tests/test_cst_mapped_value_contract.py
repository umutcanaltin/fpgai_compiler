"""Typed per-logical-value Phi ownership and read/release contract tests.

The E0 certificate counts every reduction and consumer read spatially, while
the separately compiled actual-C++ validator checks source ordering. It does
not model Vitis cycles or arbitrary fan-out.
"""
from __future__ import annotations
from dataclasses import replace

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.experiments.cst_nonlinear_campaign import synthetic_nonlinear_model
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.decision_interface import (
    DecisionAction,Accepted,create_state,transition)
from fpgai_research.model_ir.mapped_value_contract import (
    resolve_mapped_values,mapped_value_receipt)


def test_explicit_phi_value_contracts_complete_matmul_chain():
    f=make_chain(3,(3,5,4,2))
    arch=resolve(f,ArchitectureRequest())
    ir=resolve_mapped_values(arch,row_block=2)
    proof=mapped_value_receipt(ir)
    assert proof["value_elements"]==27
    assert proof["logical_producer_contributions"]==3*5*3+3*4*5
    assert proof["logical_consumer_reads"]==3*5*4+3*4*2
    assert proof["row_block"]==2
    assert len(proof["phi_conditioned_projection_sha256"])==64
    by_element={(v.tensor,v.row,v.column):v for v in ir.values}
    assert by_element[("h0",0,0)].source_slot_address==0
    assert by_element[("h0",2,0)].source_slot_address==0
    assert by_element[("h0",1,4)].source_slot_address==9
    assert all(v.ready_rule=="all_logical_producer_contributions_completed"
               for v in ir.values)
    assert all(v.release_rule=="all_logical_consumer_reads_completed"
               for v in ir.values)
    assert proof["native_vitis_calls"]==0


def test_changing_only_phi_changes_explicit_ownership_not_logical_cardinality():
    graph=make_chain(3,(3,5,2))
    base=ArchitectureRequest()
    first=mapped_value_receipt(resolve_mapped_values(resolve(graph,base)))
    mapped=mapped_value_receipt(resolve_mapped_values(resolve(graph,base,{
        "mm0":replace(base,pes=2,simd=2,serialization=2)
    })))
    assert first["value_elements"]==mapped["value_elements"]
    assert first["logical_producer_contributions"]==mapped[
        "logical_producer_contributions"]
    assert first["logical_consumer_reads"]==mapped["logical_consumer_reads"]
    assert first["phi_conditioned_projection_sha256"]!=mapped[
        "phi_conditioned_projection_sha256"]
    # The present row-atomic mask need not change; do not claim it does.


def test_nonlinear_value_read_contracts_keep_all_operator_stages():
    graph=synthetic_nonlinear_model(3,trans_b=1)
    ir=resolve_mapped_values(resolve(graph,ArchitectureRequest()),row_block=2)
    stats=mapped_value_receipt(ir)
    assert stats["value_elements"]==3*(6+4+4)
    assert stats["logical_consumer_reads"]==3*(6*4+4+4*2)
    flat=next(v for v in ir.values if v.tensor=="flat")
    assert flat.producer_required_contributions==1
    assert flat.consumer_required_reads==4
    activated=next(v for v in ir.values if v.tensor=="activation")
    assert activated.producer_required_contributions==1
    assert activated.consumer_required_reads==2


def test_source_admitted_transition_attaches_mapped_phi_value_receipt():
    graph=make_chain(3,(3,5,2))
    state=create_state(graph,exchange_policy="row_chain")
    result=transition(state,DecisionAction("operator","mm0","pes",2))
    assert isinstance(result,Accepted)
    projected=result.certificate["mapped_value_contract"]
    assert projected["value_elements"]==15
    assert len(projected["verified_row_stage_schedule_sha256"])==64
    assert projected["evidence"]=="E0-spatial-Phi-ownership-and-logical-read-cardinality"
    assert "cycle" in " ".join(projected["limits"])


@pytest.mark.parametrize("row_block,limit",[(0,100),(2,1)])
def test_unsupported_or_over_budget_mapped_value_certificates_fail_closed(row_block,limit):
    graph=make_chain(3,(3,5,2))
    arch=resolve(graph,ArchitectureRequest())
    with pytest.raises(ValueError):
        resolve_mapped_values(arch,row_block=row_block,max_values=limit)
