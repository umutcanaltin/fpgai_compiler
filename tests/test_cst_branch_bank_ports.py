"""L6b UNKNOWN is never a hardware legal-action mask."""
from dataclasses import replace
import pytest
from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.model_ir.branch_bank_ports import (
    DeclaredPortModel,CycleAssignment,check_declared_ports,
)
from fpgai_research.model_ir.branch_matmul_contract import branch_matmul_source_events
from fpgai_research.model_ir.contracts import ArchitectureRequest


def fixture():
    model=sample_branch(2,2,3,2)
    base=ArchitectureRequest()
    arch=resolve(model,base,{
        model.operators[1].name:replace(base,pes=2,simd=2),
        model.operators[2].name:replace(base,pes=2,serialization=2)})
    events,_=branch_matmul_source_events(arch)
    return arch,disjoint_branch_offsets(arch),events


def test_unknown_without_declared_schedule_or_memory_model():
    arch,offsets,events=fixture()
    port=DeclaredPortModel(2,1,1)
    no_model=check_declared_ports(arch,offsets)
    assert no_model["status"]=="UNKNOWN_UNDECLARED_MEMORY_MODEL"
    no_schedule=check_declared_ports(arch,offsets,port_model=port)
    assert no_schedule["status"]=="UNKNOWN_UNSCHEDULED"
    assert not no_schedule["certified_abstract_only"]
    assert not no_schedule["hardware_qualified"]


def test_sequential_assumed_source_timing_only_qualifies_abstract_ports():
    arch,offsets,events=fixture()
    timing=tuple(CycleAssignment(i,i) for i in range(len(events)))
    result=check_declared_ports(
        arch,offsets,port_model=DeclaredPortModel(2,1,1),schedule=timing)
    assert result["status"]=="ABSTRACT_PORT_FEASIBLE_E0"
    assert result["source_events_checked"]==len(events)
    assert result["certified_abstract_only"] is True
    assert result["hardware_qualified"] is False
    assert result["original_vitis_calls"]==0


def test_declared_oversubscription_is_rejected_with_concrete_bank():
    arch,offsets,events=fixture()
    timing=tuple(CycleAssignment(i,0) for i in range(len(events)))
    result=check_declared_ports(
        arch,offsets,port_model=DeclaredPortModel(1,1,1),schedule=timing)
    assert result["status"]=="REJECTED_ABSTRACT_PORT_OVERSUBSCRIPTION"
    assert result["counterexample"]["declared_ports"]==1
    assert result["hardware_qualified"] is False


def test_incomplete_wrong_order_and_bad_types_fail_closed():
    arch,offsets,events=fixture()
    port=DeclaredPortModel(1,1,1)
    timing=tuple(CycleAssignment(i,i) for i in range(len(events)))
    assert check_declared_ports(
        arch,offsets,port_model=port,
        schedule=timing[:-1])["status"]=="REJECTED_INCOMPLETE_OR_INVALID_SCHEDULE"
    scrambled=(timing[1],timing[0])+timing[2:]
    assert check_declared_ports(
        arch,offsets,port_model=port,
        schedule=scrambled)["status"]=="REJECTED_INCOMPLETE_OR_INVALID_SCHEDULE"
    reversed_tick=(timing[0],CycleAssignment(1,0),CycleAssignment(2,0),
                   CycleAssignment(3,2),CycleAssignment(4,1))+timing[5:]
    assert check_declared_ports(
        arch,offsets,port_model=port,
        schedule=reversed_tick)["status"]=="REJECTED_NONMONOTONE_SOURCE_ORDER"
    for k in (0,True,-1):
        with pytest.raises(ValueError):
            DeclaredPortModel(k,1,1)
    with pytest.raises(TypeError):
        check_declared_ports(arch,offsets,port_model=port,schedule=[*timing])


def test_invalid_lifetime_is_not_rescued_by_extra_ports():
    arch,offsets,events=fixture()
    h,a,b,g=[arch.functional.operators[i].outputs[0] for i in range(4)]
    bad={**offsets,b:0}
    result=check_declared_ports(arch,bad,
        port_model=DeclaredPortModel(64,8,8),
        schedule=tuple(CycleAssignment(i,i) for i in range(len(events))))
    assert result["status"]=="REJECTED_LIFETIME"
    assert result["hardware_qualified"] is False
