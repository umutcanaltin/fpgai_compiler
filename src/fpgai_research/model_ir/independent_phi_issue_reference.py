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


def reference_graph_phi_reader_order(arch, *, max_reads: int = 100_000):
    """Independently derive exact source READ order for graph MatMul consumers.

    This enumeration uses closed-form source-loop ranks. It does not call
    mapped_matmul_order, resolver.coordinate, graph_source_events or the HLS
    emitter. Only MatMul operators reading a produced intermediate have
    observable per-reduction READ tokens in the bounded graph source trace.
    """
    from fpgai_research.model_ir.graph_phi_contract import GraphEvent, admit_graph

    if type(max_reads) is not int or max_reads < 1:
        raise ValueError("Positive finite graph Phi READ budget required")
    model, produced = admit_graph(arch)
    reference = {}
    population = 0
    for op, point in zip(model.operators, arch.points):
        if op.kind != "MatMul" or op.inputs[0] not in produced:
            continue
        if point.kind != "MatMul" or point.operator != op.name:
            raise ValueError("Graph Phi point/operator identity mismatch")
        m, n, k = point.iteration_shape
        a = point.effective
        _tm, tn, tk = a.tile or point.iteration_shape
        if min(tn, tk, a.parallelism, a.pes, a.simd, a.serialization) < 1:
            raise ValueError("Nonpositive graph Phi source-loop factor")
        count = m * n * k
        if count > max_reads - population:
            raise ValueError("Graph Phi READ population exceeds finite budget")
        population += count
        width = a.parallelism * a.pes
        span = width * a.serialization
        sortable = []
        for mi in range(m):
            for ni in range(n):
                local_n = ni % tn
                group = (local_n // span) * span
                after_group = local_n - group
                fold, spatial = divmod(after_group, width)
                replica, pe = divmod(spatial, a.pes)
                for kk in range(k):
                    local_k = kk % tk
                    group_k, lane = divmod(local_k, a.simd)
                    rank = (mi, ni // tn, group, fold, replica, pe,
                            kk // tk, group_k, lane)
                    event = GraphEvent(
                        "READ", op.inputs[0], mi * k + kk, op.name,
                        mi * n + ni, 0, kk, (replica, pe, lane, fold))
                    sortable.append((rank, event))
        reference[op.name] = tuple(event for _, event in sorted(sortable,
                                                               key=lambda x: x[0]))
    return reference


def validate_graph_phi_issue_conformance(arch, events, *, max_reads: int = 100_000) -> dict:
    """Check operator-local mapped MatMul issue order separately from value safety.

    The static graph and admission rules are shared with the compiler, while
    the iteration ranking and Phi owners are independently calculated. No
    claim is made for an externally sourced MatMul (no producer READ tokens),
    scalar operations, inter-operator scheduling or achieved hardware cycles.
    """
    from fpgai_research.model_ir.graph_phi_contract import GraphEvent

    if type(events) is not tuple or any(not isinstance(e, GraphEvent) for e in events):
        raise ValueError("Graph Phi issue check requires immutable typed events")
    expected = reference_graph_phi_reader_order(arch, max_reads=max_reads)
    checked = 0
    observed_signatures = []
    for operator, wanted in expected.items():
        observed = tuple(e for e in events if e.kind == "READ"
                         and e.consumer == operator and e.input_port == 0)
        checked += len(wanted)
        for index in range(max(len(observed), len(wanted))):
            got = observed[index] if index < len(observed) else None
            want = wanted[index] if index < len(wanted) else None
            if got != want:
                def describe(e):
                    return None if e is None else (
                        e.tensor, e.element, e.consumer, e.output_element,
                        e.input_port, e.reduction_k, e.phi_owner)
                return {
                    "schema": "fpgai-research/independent-graph-Phi-source-issue/v1",
                    "status": "REJECTED",
                    "reason": "GRAPH_PHI_SOURCE_ISSUE_ORDER_OR_OWNER_MISMATCH",
                    "operator": operator,
                    "first_read_index": index,
                    "observed": describe(got),
                    "expected": describe(want),
                    "checked_reads": checked,
                    "original_gcc_calls": 0,
                    "original_vendor_calls": 0,
                }
        observed_signatures.extend(
            (e.tensor, e.element, e.consumer, e.output_element,
             e.input_port, e.reduction_k, e.phi_owner) for e in observed)
    digest = sha256(json.dumps(observed_signatures, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "schema": "fpgai-research/independent-graph-Phi-source-issue/v1",
        "status": "VERIFIED_E0" if checked else "NOT_APPLICABLE",
        "checked_operators": len(expected),
        "checked_reads": checked,
        "observed_issue_signature_sha256": digest if checked else None,
        "original_gcc_calls": 0,
        "original_vendor_calls": 0,
        "limitations": [
            "Shared static graph admission; independently calculated loop ranks",
            "Only MatMul READ tokens with a produced left-hand input",
            "No guarantee for scalar source order or inter-operator scheduling",
            "Source order is not hardware cycle, initiation interval, or RTL timing",
        ],
    }
