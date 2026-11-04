"""C1/C2 independent Phi source-order and owner reference, adversarial edits."""
from __future__ import annotations

import pytest

from fpgai_research.architecture.resolver import coordinate as original_coordinate
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.model_ir import branch_matmul_contract as contract
from fpgai_research.model_ir import interleaved_branch_schedule as emitter
from fpgai_research.model_ir.independent_phi_issue_reference import (
    reference_phi_reader_order,validate_phi_issue_conformance,
)
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.architecture.resolver import resolve
from dataclasses import replace


def _arch(q=4,reader0=None,reader1=None):
    model=sample_branch(2,2,3,q)
    base=ArchitectureRequest()
    overrides={}
    if reader0:
        overrides["reader0"]=replace(base,**reader0)
    if reader1:
        overrides["reader1"]=replace(base,**reader1)
    return resolve(model,base,overrides)


@pytest.mark.parametrize("q,r0,r1",[
    (4,None,None),(4,{"pes":2},None),
    (4,None,{"pes":2}),(4,{"pes":2},{"pes":2}),
    (3,None,{"pes":2}),(4,None,{"pes":2,"simd":2}),
])
def test_independent_arithmetic_spec_accepts_real_proposed_issue_order(q,r0,r1):
    arch=_arch(q,reader0=r0,reader1=r1)
    events=emitter.interleaved_branch_events(arch).events
    proof=validate_phi_issue_conformance(arch,events)
    assert proof["status"]=="VERIFIED_E0"
    assert proof["read_population"]==2*2*3*q
    assert len(reference_phi_reader_order(arch))==2*2*3*q


def test_a_legal_permutation_is_not_necessarily_Phi_issue_conforming():
    arch=_arch()
    events=list(emitter.interleaved_branch_events(arch).events)
    first=next(i for i,ev in enumerate(events)
               if ev.kind=="READ" and ev.tensor=="h")
    assert events[first+1].kind=="READ"
    events[first],events[first+1]=events[first+1],events[first]
    changed=tuple(events)
    # The general safety checker must allow a different legal order.
    assert contract.validate_branch_events(
        arch,disjoint_branch_offsets(arch),changed)["status"]=="VERIFIED_E0"
    # A stronger, policy-specific checker disallows it.
    proof=validate_phi_issue_conformance(arch,changed)
    assert proof["status"]=="REJECTED"
    assert proof["reason"]=="PHI_SOURCE_ISSUE_ORDER_OR_OWNER_MISMATCH"
    assert proof["first_read_index"]==0


def test_independent_owner_oracle_detects_shared_generator_verifier_bug(monkeypatch):
    arch=_arch(reader1={"pes":2})
    def wrong_coordinate(point,index):
        result=dict(original_coordinate(point,index))
        result["pe"]=result["pe"]+1
        return result
    # Deliberately infect both generator and its original structural
    # verifier with the SAME bug (a common-mode failure).
    monkeypatch.setattr(emitter,"coordinate",wrong_coordinate)
    monkeypatch.setattr(contract,"coordinate",wrong_coordinate)
    events=emitter.interleaved_branch_events(arch).events
    assert contract.validate_branch_events(
        arch,disjoint_branch_offsets(arch),events)["status"]=="VERIFIED_E0"
    proof=validate_phi_issue_conformance(arch,events)
    assert proof["status"]=="REJECTED"
    assert proof["reason"]=="PHI_SOURCE_ISSUE_ORDER_OR_OWNER_MISMATCH"


def test_independent_reference_rejects_unimplemented_tiling_without_approximation():
    arch=_arch(reader1={"tile":(2,2,3)})
    with pytest.raises(ValueError,match="unsupported policy"):
        reference_phi_reader_order(arch)
