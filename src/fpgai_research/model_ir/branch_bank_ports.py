"""L6b: fail-closed *abstract* bank-port feasibility for mapped tensor edges.

The physical-clock schedule is NOT recoverable from sequential GCC source.
Without an explicitly supplied candidate cycle assignment and declared bank
model, return UNKNOWN, NEVER hardware-certified. With both, this checks
*only* an assumed striped logical-memory access model; original Vitis
CSynth/RTL validation is still mandatory before any hardware legality claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from fpgai_research.model_ir.branch_matmul_contract import (
    branch_matmul_source_events,validate_branch_events,admit_branch,
)
from fpgai_research.model_ir.contracts import ArchitecturalModel

SCHEMA="fpgai-research/cst-declared-abstract-bank-port-check-e0/v1"


@dataclass(frozen=True)
class DeclaredPortModel:
    banks: int
    read_ports_per_bank: int
    write_ports_per_bank: int

    def __post_init__(self):
        if (type(self.banks) is not int or type(self.read_ports_per_bank) is not int
                or type(self.write_ports_per_bank) is not int or
                min(self.banks,self.read_ports_per_bank,self.write_ports_per_bank)<1):
            raise ValueError("Declared abstract bank and port counts must be finite positive integers")


@dataclass(frozen=True)
class CycleAssignment:
    event_index: int
    cycle: int


def check_declared_ports(
    arch:ArchitecturalModel,offsets:dict[str,int],*,
    schedule:tuple[CycleAssignment,...]|None=None,
    port_model:DeclaredPortModel|None=None,
    max_events:int=100_000,
)->dict:
    """Fail closed, without predicting HLS cycles or silently choosing ports."""
    events,_=branch_matmul_source_events(arch,max_events=max_events)
    temporal=validate_branch_events(arch,offsets,events,max_events=max_events)
    base={
        "schema":SCHEMA,"functional_sha256":temporal.get("functional_sha256"),
        "architecture_sha256":temporal.get("architecture_sha256"),
        "source_event_sha256":temporal.get("source_order_event_sha256"),
        "assumed_schedule_origin":"CALLER_DECLARED_NOT_HLS_OBSERVED",
        "hardware_qualified":False,"original_vitis_calls":0,
        "evidence":"E0-explicitly-declared-abstract-source-access-bank-check",
    }
    if temporal["status"]!="VERIFIED_E0":
        return {**base,"status":"REJECTED_LIFETIME","certified_abstract_only":False,
                "reason":temporal["reason"]}
    if port_model is None:
        return {**base,"status":"UNKNOWN_UNDECLARED_MEMORY_MODEL",
                "certified_abstract_only":False}
    if not isinstance(port_model,DeclaredPortModel):
        raise TypeError("Typed immutable port model required")
    if schedule is None:
        return {**base,"status":"UNKNOWN_UNSCHEDULED",
                "certified_abstract_only":False}
    if (type(schedule) is not tuple or
            any(not isinstance(x,CycleAssignment) for x in schedule)):
        raise TypeError("Exact typed immutable source-event cycle assignment required")
    if (len(schedule)!=len(events) or any(
            type(x.event_index) is not int or x.event_index!=i or
            type(x.cycle) is not int or x.cycle<0 or x.cycle>max_events*2
            for i,x in enumerate(schedule))):
        return {**base,"status":"REJECTED_INCOMPLETE_OR_INVALID_SCHEDULE",
                "certified_abstract_only":False}
    if any(schedule[i].cycle>schedule[i+1].cycle
           for i in range(len(schedule)-1)):
        return {**base,"status":"REJECTED_NONMONOTONE_SOURCE_ORDER",
                "certified_abstract_only":False}
    # Only the explicitly declared striped layout is checked. The emitted
    # C++ still has one monolithic scratch[]; this is NOT an HLS partition.
    histogram={}
    accesses={}
    for index,(e,timing) in enumerate(zip(events,schedule)):
        address=offsets[e.tensor]+e.element
        bank=address%port_model.banks
        access_kind="read" if e.kind=="READ" else "write"
        key=(timing.cycle,bank,access_kind)
        histogram[key]=histogram.get(key,0)+1
        cap=(port_model.read_ports_per_bank if access_kind=="read"
             else port_model.write_ports_per_bank)
        if histogram[key]>cap:
            return {**base,"status":"REJECTED_ABSTRACT_PORT_OVERSUBSCRIPTION",
                    "certified_abstract_only":False,
                    "counterexample":{
                        "event_index":index,"cycle":timing.cycle,"bank":bank,
                        "port_type":access_kind,"required":histogram[key],
                        "declared_ports":cap,"logical_source_address":address,
                    }}
        alias_key=(timing.cycle,address)
        previous=accesses.get(alias_key)
        if previous is not None and (previous=="write" or access_kind=="write"):
            return {**base,"status":"REJECTED_SAME_CYCLE_ADDRESS_HAZARD",
                    "certified_abstract_only":False,
                    "counterexample":{
                        "event_index":index,"cycle":timing.cycle,
                        "logical_source_address":address,
                        "earlier_access":previous,"current_access":access_kind,
                    }}
        accesses[alias_key]=access_kind
    sig=sha256(json.dumps(
        {"ports":port_model.__dict__,
         "ticks":[x.cycle for x in schedule],
         "source":temporal["source_order_event_sha256"]},
        sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {**base,"status":"ABSTRACT_PORT_FEASIBLE_E0",
            "certified_abstract_only":True,
            "hardware_qualified":False,
            "declared_bank_model_sha256":sig,
            "source_events_checked":len(events),
            "assumed_cycles":schedule[-1].cycle+1 if schedule else 0,
            "limitations":[
                "Caller-declared cycle labels, NOT achieved HLS scheduling",
                "Modulo-striped abstract bank model, NOT actual synthesized HLS memory partition",
                "No original Vitis CSim/CSynth, RTL memory ports, place-route or clock proof",
            ]}
