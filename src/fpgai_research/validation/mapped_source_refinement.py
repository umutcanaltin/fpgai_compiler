"""Bind an E0 Phi-aware logical value plan to independent executed GCC source.

This is a *bounded sequential-chain* source-refinement receipt: first verify
a complete typed logical event proposal derived from operator Phi; then run
the actual emitted HLS C++ with independent per-reduction value/read tracing.
The two checkers use different representations and cross-check cardinality,
source identity, row policy and release completion. Does not prove arbitrary
C++ equivalence, RTL timing or DAG fan-out physical reuse.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod

from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.mapped_event_schedule import (
    canonical_value_events, verify_value_events,
)
from fpgai_research.model_ir.mapped_value_contract import resolve_mapped_values
from fpgai_research.validation.cpp_value_trace import trace_cpp_values
from fpgai_research.model_ir.semantics import functional_fingerprint

def qualify_mapped_source(arch: ArchitecturalModel, *, row_block:int=1,
                          max_events:int=50_000, cxx:str="g++") -> dict:
    """Execute real generated C++ to qualify the E0 event relation (E1)."""
    if not isinstance(arch,ArchitecturalModel):
        raise TypeError("Expected resolved ArchitecturalModel")
    if type(max_events) is not int or max_events<1:
        raise ValueError("Finite source trace event budget required")
    compiled=compile_row_chain(arch,row_block=row_block)
    proposed=canonical_value_events(arch,row_block=row_block)
    e0=verify_value_events(arch,proposed,row_block=row_block,max_events=max_events)
    if e0["status"]!="VERIFIED":
        raise ValueError("Proposed Phi-conditioned logical event contract rejected")
    trace=trace_cpp_values(compiled.ir,emission=compiled.emission,
                           max_events=max_events,cxx=cxx)
    mapped=resolve_mapped_values(arch,row_block=row_block)
    n=len(mapped.values)
    if trace["status"]!="SOURCE_VALUE_LIFETIMES_VERIFIED":
        raise AssertionError("Actual generated source did not meet its value trace")
    if trace["last_reader_releases_verified"]!=n:
        raise AssertionError("Actual C++ last-reader release count differs from IR")
    if trace["row_block"]!=row_block:
        raise AssertionError("Emitted row slot layout differs from certified plan")
    if trace["cpp_sha256"]!=compiled.emission.manifest["cpp_sha256"]:
        raise AssertionError("Independently executed source differs from compiler emission")
    if trace["functional_sha256"]!=functional_fingerprint(arch.functional):
        raise AssertionError("Independent source trace changed Functional graph")
    evidence={
        "schema":"fpgai-research/cst-mapped-e0-to-e1-refinement/v1",
        "status":"EXECUTED_SOURCE_REFINEMENT_VERIFIED",
        "functional_sha256":trace["functional_sha256"],
        "source_cpp_sha256":trace["cpp_sha256"],
        "row_block":row_block,
        "logical_phi_event_sha256":e0["event_sha256"],
        "logical_phi_projection_sha256":e0["mapping_projection_sha256"],
        "logical_value_count":n,
        "logical_event_count":e0["events_checked"],
        "actual_cpp_trace_sha256":trace["source_trace_sha256"],
        "actual_cpp_iteration_count":trace["mapped_reduction_iterations_verified"],
        "actual_cpp_last_reader_releases":trace["last_reader_releases_verified"],
        "original_vitis_calls":0,
        "evidence":"E0-Phi-logical-plus-E1-independent-executed-C++-value-trace",
        "limits":[
            "E0 logical event order is not asserted identical to C++ loop interleaving",
            "Source tracer independently verifies actual reduction/read/last-use",
            "Single-reader sequential row chain, no arbitrary fan-out",
            "Host C++ execution does not prove original Vitis/RTL or physical timing",
        ],
    }
    evidence["receipt_sha256"]=sha256(json.dumps(evidence,sort_keys=True,
                             separators=(",",":")).encode()).hexdigest()
    return evidence
