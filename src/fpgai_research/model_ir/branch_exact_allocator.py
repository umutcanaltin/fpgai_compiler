"""Exact bounded tensor-base scratch allocation under *all* Phi-mapped reads.

The event sequence includes individual MatMul consumer reduction reads,
not one ACK per consumer. This solver certifies the *minimum contiguous
tensor-base source array extent* under a fixed, complete sequential C++ loop
order. This is a finite combinatorial source model: no hardware banks,
RAM port limits, throughput, timing, or physical resource optimality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from fpgai_research.model_ir.branch_matmul_contract import (
    BranchEvent, admit_branch, branch_matmul_source_events, validate_branch_events,
)
from fpgai_research.model_ir.contracts import ArchitecturalModel

SCHEMA = "fpgai-research/cst-mapped-multi-reader-exact-base-allocation-e0/v1"


@dataclass(frozen=True)
class Interval:
    tensor: str
    element: int
    begin: int
    end: int


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",",":")).encode()).hexdigest()


def _intervals(arch: ArchitecturalModel, *, max_events: int,
               source_events: tuple[BranchEvent,...] | None = None):
    (m,k,n,q),(h,a,b,g),_weights,_ops=admit_branch(arch.functional)
    sizes = {h:m*n, a:m*q, b:m*q, g:m*q}
    disjoint={}
    start=0
    for name in (h,a,b,g):
        disjoint[name]=start
        start+=sizes[name]
    if source_events is None:
        events,_iterations=branch_matmul_source_events(arch,max_events=max_events)
    else:
        if type(source_events) is not tuple or any(
                not isinstance(ev,BranchEvent) for ev in source_events):
            raise ValueError("Explicit source schedule requires immutable BranchEvents")
        # This alternative arm must satisfy a separately defined
        # Phi-issue order, not just generic address/readiness legality.
        from fpgai_research.model_ir.independent_phi_issue_reference import (
            validate_phi_issue_conformance,
        )
        conformance=validate_phi_issue_conformance(arch,source_events)
        if conformance["status"]!="VERIFIED_E0":
            raise ValueError("Exact alternative schedule is not Phi-issue conforming: "+
                             str(conformance))
        events=source_events
    proof=validate_branch_events(arch,disjoint,events,max_events=max_events)
    if proof["status"]!="VERIFIED_E0":
        raise ValueError("Exact allocation requires independently legal source events: "+str(proof))
    commits={}
    last_reads={}
    for i,e in enumerate(events):
        key=(e.tensor,e.element)
        if e.kind=="COMMIT":
            if key in commits:
                raise AssertionError("Double commit in source event stream")
            commits[key]=i
        elif e.kind=="READ":
            last_reads[key]=i
        else:
            raise AssertionError("Unknown event in source lifetime proof")
    if commits.keys()!=last_reads.keys():
        raise AssertionError("Missing source value release")
    intervals=tuple(Interval(t,j,commits[(t,j)],last_reads[(t,j)])
                    for t in (h,a,b,g) for j in range(sizes[t]))
    return (h,a,b,g),sizes,events,proof,intervals


def exact_mapped_branch_slots(
    arch:ArchitecturalModel,*,max_slots:int=64,
    max_candidates:int=300_000,max_events:int=100_000,
    source_events:tuple[BranchEvent,...]|None=None,
) -> dict:
    """Exhaustive bounded base assignment with proof-replayed first optimum.

    UNKNOWN_* is never a valid mask/action or a source-emit permit.
    If the budget expires at any smaller capacity, a larger found capacity
    must NOT be called optimal. Inputs must use the typed admitted diamond.
    """
    if any(type(x) is not int or x<1 for x in
           (max_slots,max_candidates,max_events)):
        raise ValueError("Positive finite solver bounds required")
    tensors,sizes,events,proof,intervals=_intervals(
        arch,max_events=max_events,source_events=source_events)
    total=sum(sizes.values())
    result={
        "schema":SCHEMA,
        "status":"UNKNOWN",
        "source_event_sha256":proof["source_order_event_sha256"],
        "functional_sha256":proof["functional_sha256"],
        "architecture_sha256":proof["architecture_sha256"],
        "logical_values":proof["logical_values"],
        "mapped_h_reads":proof["actual_required_h_reads"],
        "max_slots":max_slots,"max_candidates":max_candidates,
        "max_events":max_events,
        "schedule_source":"GENERATED_SERIAL" if source_events is None else "SUPPLIED_INDEPENDENTLY_PHI_CONFORMING",
        "total_disjoint_slots":total,
        "proven_optimal":False,
        "physical_slots":None,"offsets":None,
        "examined_candidates":0,
        "original_vitis_calls":0,
        "evidence":"E0-exact-finite-Phi-multi-read-source-order-base-allocation",
        "limitations":[
            "Minimum over contiguous tensor-base addresses and one FIXED supplied or generated source event order only",
            "No hardware bank/port limits, clock overlap, RTL, BRAM or timing inference",
            "No generalized tensor layout or graph DAG lowerer",
        ],
    }
    if max_slots < max(sizes.values()):
        return {**result,"status":"INFEASIBLE_WITHIN_CAPACITY"}
    by_name={name:[x for x in intervals if x.tensor==name] for name in tensors}
    # Avoid recomputing overlaps for every base candidate:
    # forbidden differences of physical offsets are induced by exactly
    # those logical elements whose [COMMIT, last READ] intervals overlap.
    forbidden={}
    for i,name in enumerate(tensors):
        for other in tensors[:i]:
            differences=set()
            for left in by_name[name]:
                for right in by_name[other]:
                    if left.begin<=right.end and right.begin<=left.end:
                        differences.add(right.element-left.element)
            forbidden[(name,other)]=frozenset(differences)
    explored=0
    budget_hit=False
    def solve(capacity):
        nonlocal explored,budget_hit
        offsets={}
        def visit(index):
            nonlocal explored,budget_hit
            if index==len(tensors):
                return dict(offsets)
            tensor=tensors[index]
            for base in range(capacity-sizes[tensor]+1):
                if explored>=max_candidates:
                    budget_hit=True
                    return None
                explored+=1
                if any(base-offsets[other] in forbidden[(tensor,other)]
                       for other in tensors[:index]):
                    continue
                offsets[tensor]=base
                answer=visit(index+1)
                if answer is not None:
                    return answer
                del offsets[tensor]
                if budget_hit:
                    return None
            return None
        return visit(0)
    for slots in range(max(sizes.values()),max_slots+1):
        answer=solve(slots)
        if answer is not None:
            independently_checked=validate_branch_events(
                arch,answer,events,max_events=max_events)
            if independently_checked["status"]!="VERIFIED_E0":
                raise AssertionError("Minimum address layout failed independent production verifier")
            if independently_checked["source_scratch_slots"]>slots:
                raise AssertionError("Finite solver returned an overflowing address layout")
            return {
                **result,"status":"OPTIMAL_E0","proven_optimal":True,
                "physical_slots":slots,"offsets":answer,
                "examined_candidates":explored,
                "verification_last_use_sha256":independently_checked["last_use_sha256"],
                "allocation_sha256":_digest({
                    "event_sha256":proof["source_order_event_sha256"],
                    "offsets":answer,"slots":slots,
                }),
            }
        if budget_hit:
            return {**result,"status":"UNKNOWN_BUDGET",
                    "examined_candidates":explored}
    return {**result,"status":"INFEASIBLE_WITHIN_CAPACITY",
            "examined_candidates":explored}
