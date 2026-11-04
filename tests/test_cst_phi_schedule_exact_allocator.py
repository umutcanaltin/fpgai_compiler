"""C1 F3: contiguous-base exact allocator can compare schedules fairly."""
from __future__ import annotations

from dataclasses import replace

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.branch_exact_allocator import exact_mapped_branch_slots
from fpgai_research.model_ir.interleaved_branch_schedule import interleaved_branch_events


def _arch(pes=1):
    model=sample_branch(1,1,2,2)
    base=ArchitectureRequest()
    overrides={} if pes==1 else {"reader1":replace(base,pes=pes)}
    return resolve(model,base,overrides)


def test_exact_contiguous_allocator_accepts_fixed_independent_Phi_schedule():
    for pes in (1,2):
        arch=_arch(pes)
        events=interleaved_branch_events(arch).events
        comparison=exact_mapped_branch_slots(
            arch,source_events=events,max_slots=20,max_candidates=300_000)
        assert comparison["status"]=="OPTIMAL_E0"
        assert comparison["proven_optimal"] is True
        assert comparison["schedule_source"]=="SUPPLIED_INDEPENDENTLY_PHI_CONFORMING"
        assert comparison["source_event_sha256"]
        assert comparison["physical_slots"]<=20


def test_original_serial_source_api_remains_supported():
    arch=_arch()
    result=exact_mapped_branch_slots(
        arch,max_slots=20,max_candidates=300_000)
    assert result["status"]=="OPTIMAL_E0"
    assert result["schedule_source"]=="GENERATED_SERIAL"


def test_nonconforming_but_type_correct_Phi_schedule_cannot_enter_exact_solver():
    arch=_arch()
    events=list(interleaved_branch_events(arch).events)
    first=next(i for i,e in enumerate(events)
               if e.kind=="READ" and e.tensor=="h")
    events[first],events[first+1]=events[first+1],events[first]
    with pytest.raises(ValueError,match="not Phi-issue conforming"):
        exact_mapped_branch_slots(
            arch,source_events=tuple(events),max_slots=20,max_candidates=10000)
