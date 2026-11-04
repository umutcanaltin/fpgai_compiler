"""Independent bounded R2b/R2c Phi source-issue reference (E0 only).

Unlike the generator, this module NEVER calls mapped_matmul_order,
interleaved_branch_events, resolver.coordinate, or the source codegen.
It ranks each independent reader reduction coordinate by a closed-form
group/rank formula and checks the actual proposed source READ sequence.

This characterizes the fixed sequential PE-GROUP issue rule only. It
is not an FPGA hardware cycle, port, II, placement or HLS schedule
specification. The reference verifies policy conformance; it does
not replace the separate functional, reduction or address verifier.
"""
from __future__ import annotations

from hashlib import sha256
import json

from fpgai_research.model_ir.branch_matmul_contract import (
    admit_branch, BranchEvent,
)

SCHEMA="fpgai-research/cst-independent-Phi-issue-rule-E0/v1"


def reference_phi_reader_order(arch,*,max_reads:int=100_000):
    (m,k,n,q),(h,a,b,g),_weights,ops=admit_branch(arch.functional)
    if type(max_reads) is not int or max_reads<1 or 2*m*n*q>max_reads:
        raise ValueError("Independent Phi issue reference bounded by exact H READ population")
    sortable=[]
    for stage in (1,2):
        effective=arch.points[stage].effective
        width=effective.pes
        lane_count=effective.simd
        if (effective.tile or effective.serialization!=1
                or effective.parallelism!=1
                or effective.pipeline_ii is not None
                or not 1<=width<=q or not 1<=lane_count<=n):
            raise ValueError("Independent R2b Phi issue reference: unsupported policy")
        for row in range(m):
            for out_col in range(q):
                group_start=(out_col//width)*width
                group_width=min(width,q-group_start)
                for reduction in range(n):
                    # Closed-form rank in per-reader source issue order;
                    # handles a final incomplete PE group without
                    # sharing the generator's for-loop nesting.
                    rank=(row*n*q + group_start*n
                          + reduction*group_width + out_col-group_start)
                    # R2b v1 is untiled, unreplicated, un-serialized.
                    # The ownership reference is arithmetic independent
                    # of resolver.coordinate, including SIMD fold.
                    # phi_owner[3] is the OUTPUT serialization fold
                    # (reuse_step[2]), not the reduction SIMD group.
                    # Serialization is explicitly fixed to one here.
                    owner=(0,out_col%width,reduction%lane_count,0)
                    record=BranchEvent(
                        "READ",h,row*n+reduction,ops[stage].name,
                        row*q+out_col,owner)
                    sortable.append((rank,stage,record))
    result=tuple(v for _,_,v in sorted(sortable,key=lambda x:(x[0],x[1])))
    if len(result)!=2*m*n*q:
        raise AssertionError("Independent Phi issue reference lost read tokens")
    return result


def validate_phi_issue_conformance(arch,events,*,max_reads:int=100_000)->dict:
    if not isinstance(events,tuple) or any(
            not isinstance(e,BranchEvent) for e in events):
        raise ValueError("Typed immutable event tuple required")
    expected=reference_phi_reader_order(arch,max_reads=max_reads)
    h=arch.functional.operators[0].outputs[0]
    observed=tuple(e for e in events if e.kind=="READ" and e.tensor==h)
    for index in range(max(len(observed),len(expected))):
        got=observed[index] if index<len(observed) else None
        want=expected[index] if index<len(expected) else None
        if got!=want:
            return {
                "schema":SCHEMA,"status":"REJECTED",
                "reason":"PHI_SOURCE_ISSUE_ORDER_OR_OWNER_MISMATCH",
                "first_read_index":index,
                "observed":None if got is None else (
                    got.consumer,got.output_element,got.element,got.phi_owner),
                "expected":None if want is None else (
                    want.consumer,want.output_element,want.element,want.phi_owner),
                "oracle_independent_of_source_generator":True,
                "original_gcc_calls":0,"original_vendor_calls":0,
            }
    digest=sha256(json.dumps(
        [(e.consumer,e.output_element,e.element,e.phi_owner)
         for e in observed],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {
        "schema":SCHEMA,"status":"VERIFIED_E0",
        "read_population":len(observed),
        "observed_issue_signature_sha256":digest,
        "oracle_independent_of_source_generator":True,
        "original_gcc_calls":0,"original_vendor_calls":0,
        "limitations":[
            "Conformance only for R2b fixed two-reader PE-group source issue policy",
            "No hardware timing, SIMD concurrency, bank ports or physical scheduling",
            "Does not certify independent H producer reduction completeness",
        ],
    }
