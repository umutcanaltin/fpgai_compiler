"""R2f E0: independently verified fanout-2/3/4 source certificates and faults."""
from __future__ import annotations

from dataclasses import replace

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.model_ir.contracts import ArchitectureRequest,FunctionalModel,Operator
from fpgai_research.model_ir.fanout_contract import (
    FanoutEvent, admit_fanout, disjoint_fanout_offsets,
    generate_fanout_events,verify_fanout_events,verify_fanout_issue,
)


def _arch(readers=2,*,m=2,k=2,n=3,q=4,policies=None):
    model=sample_fanout(readers,m=m,k=k,n=n,q=q)
    base=ArchitectureRequest()
    overrides={
        f"reader{i}":replace(base,pes=pes)
        for i,pes in enumerate(policies or (1,)*readers) if pes!=1
    }
    return resolve(model,base,overrides)


@pytest.mark.parametrize("readers", [2,3,4])
@pytest.mark.parametrize("shape",[(1,2,2,3),(2,2,3,4),(2,3,4,4)])
def test_true_compositional_fanout_reduction_and_last_use(readers,shape):
    m,k,n,q=shape
    policies=tuple(2 if i%2 else 1 for i in range(readers))
    arch=_arch(readers,m=m,k=k,n=n,q=q,policies=policies)
    c=admit_fanout(arch)
    assert len(c.readers)==readers
    assert len(c.add_ops)==readers-1
    assert len(c.intermediates)==2*readers
    offsets=disjoint_fanout_offsets(c)
    events=generate_fanout_events(arch)
    evidence=verify_fanout_events(arch,offsets,events)
    phi=verify_fanout_issue(arch,events)
    assert evidence["status"]=="VERIFIED_E0"
    assert phi["status"]=="VERIFIED_E0"
    assert evidence["fanout_readers"]==readers
    assert evidence["producer_contributions"]==m*k*n
    assert evidence["actual_H_reads"]==readers*m*n*q
    assert evidence["all_intermediate_value_count"]==m*n+(2*readers-1)*m*q
    assert phi["H_read_population"]==readers*m*n*q
    assert evidence["original_GCC_calls"]==0
    assert evidence["original_vendor_calls"]==0
    assert generate_fanout_events(arch)==events
    assert evidence["source_scratch_slots"]>=m*n


@pytest.mark.parametrize("readers", [2,3,4])
def test_fanout_H_producer_commit_fails_on_missing_or_forged_term(readers):
    arch=_arch(readers,k=3)
    c=admit_fanout(arch)
    offsets=disjoint_fanout_offsets(c)
    original=list(generate_fanout_events(arch))
    # Commit H[0] after only one of three contributions.
    commit=next(x for x in original if x.kind=="COMMIT"
                and x.tensor==c.h and x.element==0)
    original.remove(commit)
    original.insert(1,commit)
    rejected=verify_fanout_events(arch,offsets,tuple(original))
    assert rejected["status"]=="REJECTED"
    assert rejected["reason"]=="PREMATURE_H_PRODUCER_COMMIT"
    assert rejected["counterexample"]["missing_terms"]==[1,2]
    original=list(generate_fanout_events(arch))
    wrong=original[0]
    original[0]=replace(wrong,source_element=999)
    assert verify_fanout_events(
        arch,offsets,tuple(original))["reason"]=="WRONG_PRODUCER_OPERAND_INDEX"


@pytest.mark.parametrize("readers", [2,3,4])
def test_fanout_individual_reader_reduction_completeness(readers):
    arch=_arch(readers,n=3)
    c=admit_fanout(arch)
    offsets=disjoint_fanout_offsets(c)
    for stage in range(readers):
        seq=list(generate_fanout_events(arch))
        out=c.reader_values[stage]
        commit=next(e for e in seq if e.kind=="COMMIT"
                    and e.tensor==out and e.element==0)
        seq.remove(commit)
        first_read=next(i for i,e in enumerate(seq)
                        if e.kind=="READ" and e.tensor==c.h
                        and e.operation==c.readers[stage]
                        and e.output_element==0)
        seq.insert(first_read+1,commit)
        result=verify_fanout_events(arch,offsets,tuple(seq))
        assert result["status"]=="REJECTED"
        assert result["reason"]=="PREMATURE_FANOUT_READER_COMMIT"
        assert len(result["counterexample"]["missing_reductions"])==2


@pytest.mark.parametrize("readers", [2,3,4])
def test_compositional_add_chain_rejects_early_merge_commit(readers):
    arch=_arch(readers)
    c=admit_fanout(arch)
    offsets=disjoint_fanout_offsets(c)
    for step,merge_tensor in enumerate(c.add_values):
        seq=list(generate_fanout_events(arch))
        commit=next(e for e in seq if e.kind=="COMMIT"
                    and e.tensor==merge_tensor and e.element==0)
        seq.remove(commit)
        left=c.reader_values[0] if step==0 else c.add_values[step-1]
        at=next(i for i,e in enumerate(seq) if
                e.kind=="READ" and e.tensor==left and e.element==0
                and e.operation==c.add_ops[step])
        seq.insert(at+1,commit)
        rejected=verify_fanout_events(arch,offsets,tuple(seq))
        assert rejected["reason"]=="PREMATURE_FANOUT_ADD_COMMIT"
        assert rejected["counterexample"]["missing_inputs"]==[
            c.reader_values[step+1]]


@pytest.mark.parametrize("readers", [2,3,4])
def test_unsafe_alias_and_wrong_Phi_owner_are_rejected(readers):
    arch=_arch(readers,policies=(1,)*(readers-1)+(2,))
    c=admit_fanout(arch)
    offsets=disjoint_fanout_offsets(c)
    events=generate_fanout_events(arch)
    overlapping={**offsets,c.reader_values[0]:offsets[c.h]}
    assert verify_fanout_events(
        arch,overlapping,events)["reason"]=="FANOUT_OVERWRITE_LIVE_VALUE"
    seq=list(events)
    read=next(i for i,e in enumerate(seq)
              if e.kind=="READ" and e.tensor==c.h)
    seq[read]=replace(seq[read],phi_owner=(0,777,0,0))
    assert verify_fanout_events(
        arch,offsets,tuple(seq))["reason"]==(
            "WRONG_FANOUT_PHI_COORDINATE_OR_OWNER")


@pytest.mark.parametrize("readers", [2,3,4])
def test_safe_but_nonconforming_fanout_Phi_order_and_producer_reader_overlap(readers):
    arch=_arch(readers)
    c=admit_fanout(arch)
    offsets=disjoint_fanout_offsets(c)
    seq=list(generate_fanout_events(arch))
    first=next(i for i,e in enumerate(seq)
               if e.kind=="READ" and e.tensor==c.h)
    assert seq[first+1].kind=="READ"
    seq[first],seq[first+1]=seq[first+1],seq[first]
    assert verify_fanout_events(arch,offsets,tuple(seq))["status"]=="VERIFIED_E0"
    assert verify_fanout_issue(arch,tuple(seq))["reason"]==(
        "FANOUT_PHI_ISSUE_OR_OWNER_ORDER_MISMATCH")
    # Producer/consumer STREAMING at the logical-event level:
    # move an already legal first H read immediately after its H COMMIT,
    # before remaining producer contributions. This preserves the
    # H READ *sequence* while changing the source event interleaving.
    seq=list(generate_fanout_events(arch))
    first_read=next(i for i,e in enumerate(seq)
                    if e.kind=="READ" and e.tensor==c.h)
    read=seq.pop(first_read)
    first_commit=next(i for i,e in enumerate(seq)
                      if e.kind=="COMMIT" and e.tensor==c.h)
    seq.insert(first_commit+1,read)
    assert verify_fanout_events(arch,offsets,tuple(seq))["status"]=="VERIFIED_E0"
    assert verify_fanout_issue(arch,tuple(seq))["status"]=="VERIFIED_E0"


def test_no_silent_acceptance_of_unsupported_fanout_or_invalid_event_budget():
    with pytest.raises(ValueError,match="2..4"):
        sample_fanout(5)
    arch=_arch(4)
    with pytest.raises(ValueError,match="event bound"):
        generate_fanout_events(arch,max_events=1)
    c=admit_fanout(arch)
    events=generate_fanout_events(arch)
    with pytest.raises(ValueError,match="offset"):
        verify_fanout_events(arch,{"h":0},events)
    # If the source has a wrong tensor identity, reject rather than
    # inventing an implicit scratch or source-zero value.
    mutation=(replace(events[0],tensor="UNKNOWN"),)+events[1:]
    assert verify_fanout_events(
        arch,disjoint_fanout_offsets(c),mutation)["reason"]==(
            "INVALID_FANOUT_ELEMENT")
