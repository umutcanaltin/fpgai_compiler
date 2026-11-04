"""R2b E0 only: fixed PE-group source event issue, no generated-C++ claim."""
from __future__ import annotations

import csv
from dataclasses import replace

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_interleaved_schedule import run
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.branch_matmul_contract import (
    BranchEvent,validate_branch_events,
)
from fpgai_research.model_ir.interleaved_branch_schedule import (
    interleaved_branch_events, release_positions, prove_interleaved_source_order,
)


def _arch(*,r0=None,r1=None,q=4):
    model=sample_branch(2,2,3,q)
    base=ArchitectureRequest()
    overrides={}
    if r0:
        overrides["reader0"]=replace(base,**r0)
    if r1:
        overrides["reader1"]=replace(base,**r1)
    return resolve(model,base,overrides)


def test_interleaved_E0_contract_verifies_exact_both_reader_accesses():
    arch=_arch(r0={"pes":2},r1={"pes":2})
    events=interleaved_branch_events(arch)
    proof=prove_interleaved_source_order(
        arch,disjoint_branch_offsets(arch))
    assert proof["status"]=="VERIFIED_E0"
    assert proof["mapped_multi_reader_values"]==2*3
    assert proof["actual_required_h_reads"]==2*2*3*4
    assert proof["h_last_read_ordinals"]==list(release_positions(events,"h"))
    assert proof["mapping_event_effect_source_executed"] is False
    assert proof["original_gcc_calls"]==0
    assert proof["evidence_level"]=="E0-LOGICAL-NOT-GCC-OR-HARDWARE"


def test_legal_phi_mutation_alters_actual_proposed_logical_element_release():
    scalar=_arch()
    changed=_arch(r1={"pes":2})
    before=release_positions(interleaved_branch_events(scalar),"h")
    after=release_positions(interleaved_branch_events(changed),"h")
    assert [i for i,_ in before]==[i for i,_ in after]
    assert any(a[1]!=b[1] for a,b in zip(before,after))
    # A changed candidate EVENT schedule is evidence at E0 only:
    assert prove_interleaved_source_order(
        changed,disjoint_branch_offsets(changed)
    )["mapping_event_effect_source_executed"] is False


def test_illegal_read_and_premature_alias_refused_with_counterexample():
    arch=_arch(r0={"pes":2})
    plan=interleaved_branch_events(arch)
    offsets=disjoint_branch_offsets(arch)
    first=next(e for e in plan.events if e.kind=="READ" and e.tensor=="h")
    # Moving a real consumer read before the first H COMMIT is unsafe.
    premature=(first,)+tuple(e for e in plan.events if e is not first)
    bad=validate_branch_events(arch,offsets,premature)
    assert bad["status"]=="REJECTED"
    assert bad["reason"]=="READ_BEFORE_READY_OR_STALE"
    # All H cannot be overwritten when reader1 still needs some values.
    alias={**offsets,"a":0}
    conflict=validate_branch_events(arch,alias,plan.events)
    assert conflict["status"]=="REJECTED"
    assert conflict["reason"]=="OVERWRITE_BEFORE_ALL_MAPPED_READS"
    assert "address" in conflict["counterexample"]


def test_partial_reduction_or_unsupported_source_issue_policy_fails_closed():
    arch=_arch()
    with pytest.raises(ValueError,match="bound"):
        interleaved_branch_events(arch,max_events=1)
    unsupported=_arch(r0={"tile":(2,2,3)})
    with pytest.raises(ValueError):
        interleaved_branch_events(unsupported)


def test_complete_original_prospective_E0_campaign_preserves_negative_outcomes(tmp_path):
    dest=tmp_path/"r2b"
    report=run(output=dest)
    assert report["prespecified_cases"]==12
    assert report["verified_e0_cases"]==12
    assert report["nonbaseline_phi_interventions"]==9
    assert report["source_executed_lifetime_change_established"] is False
    assert report["actual_source_memory_savings_established"] is False
    assert report["original_gcc_calls"]==0
    assert report["original_vendor_calls"]==0
    with (dest/"cases.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==12
    assert all(row["certificate_verified_e0"]=="True" for row in rows)
    assert all(row["source_executed_e1"]=="False" for row in rows)
    assert all(int(row["mapped_h_last_read_elements_changed_vs_scalar"])==0
               for row in rows if row["policy"]=="scalar")
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=dest)


def test_C2_rejects_premature_A_or_B_reduction_commit_independently():
    # Moving an A/B COMMIT near the first corresponding H READ previously
    # passed the structural verifier: it only checked whether the slot
    # was vacant. The new obligation requires every one of n reductions.
    arch=_arch()
    original=interleaved_branch_events(arch).events
    offsets=disjoint_branch_offsets(arch)
    for tensor,reader_name in (("a","reader0"),("b","reader1")):
        commit=next(ev for ev in original
                    if ev.kind=="COMMIT" and ev.tensor==tensor
                    and ev.element==0)
        reduced=[ev for ev in original if ev is not commit]
        insertion=next(i+1 for i,ev in enumerate(reduced)
                       if ev.kind=="READ" and ev.tensor=="h"
                       and ev.consumer==reader_name
                       and ev.output_element==0)
        corrupted=tuple(reduced[:insertion]+[commit]+reduced[insertion:])
        result=validate_branch_events(arch,offsets,corrupted)
        assert result["status"]=="REJECTED"
        assert result["reason"]=="PREMATURE_REDUCTION_COMMIT"
        assert result["counterexample"]["missing_reduction_indices"]
        assert result["counterexample"]["output"]==0


def test_C2_rejects_add_COMMIT_before_both_operand_reads():
    arch=_arch()
    original=interleaved_branch_events(arch).events
    offsets=disjoint_branch_offsets(arch)
    commit=next(ev for ev in original
                if ev.kind=="COMMIT" and ev.tensor=="g"
                and ev.element==0)
    remaining=[ev for ev in original if ev is not commit]
    insert=next(i+1 for i,ev in enumerate(remaining)
                if ev.kind=="READ" and ev.tensor=="a"
                and ev.element==0)
    corrupted=tuple(remaining[:insert]+[commit]+remaining[insert:])
    result=validate_branch_events(arch,offsets,corrupted)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="PREMATURE_ADD_COMMIT"
    assert result["counterexample"]["missing_inputs"]==["b"]


def test_C2_original_complete_diamond_certificate_still_verified():
    for r0,r1 in ((None,None),({"pes":2},None),
                  (None,{"pes":2}),({"pes":2},{"pes":2})):
        arch=_arch(r0=r0,r1=r1)
        events=interleaved_branch_events(arch).events
        result=validate_branch_events(
            arch,disjoint_branch_offsets(arch),events)
        assert result["status"]=="VERIFIED_E0"
        assert result["actual_required_h_reads"]==2*2*3*4
