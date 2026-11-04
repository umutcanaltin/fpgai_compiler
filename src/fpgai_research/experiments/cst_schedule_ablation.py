"""Prespecified counterexample/ablation study for compositional schedules.

Scientific design: paired positive schedules and deliberately constructed
negative schedule interventions over the *same* fixed Functional IR/Phi.
The operator-only refinement has no event-order input and is a scoped
ablation, not a competing schedule checker. Report every attempted case,
negative-control class, bounded status, and witness; no Vitis/DRL calls.

No claim that these intervention classes represent real-world bug rates.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter_ns

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.row_chain import resolve_row_chain
from fpgai_research.model_ir.temporal_schedule import (
    ComputeEvent, blocked_schedule, check_schedule,
)
from fpgai_research.model_ir.semantics import verify_refinement
from fpgai_research.validation.cpp_chain_trace import trace_chain_epochs
from fpgai_research.validation.host import validate_host


SCHEMA = "fpgai-research/cst-independent-schedule-ablation/v1"


def _architecture(graph, heterogeneous: bool):
    default = ArchitectureRequest()
    if not heterogeneous:
        return resolve(graph, default)
    return resolve(graph, default, {
        op.name: replace(
            default, pes=(i%3)+1, simd=(i%3)+1,
            parallelism=(i%2)+1, serialization=(i%2)+1)
        for i, op in enumerate(graph.operators)
    })


def _interventions(events, slots, rows):
    """Fixed mutation operators with known expected illegality classes."""
    return [
        ("early_consumer", (ComputeEvent(0, 1),) + events, slots,
         "READ_BEFORE_PRODUCE_OR_STALE_ROW"),
        ("duplicate_stage", (events[0],) + events, slots,
         "DUPLICATE_STAGE_EXECUTION"),
        ("missing_terminal_stage", events[:-1], slots,
         "MISSING_STAGE_EXECUTION"),
        ("invalid_operator_id", (ComputeEvent(0, 999),) + events, slots,
         "INVALID_STAGE_OR_ROW"),
    ] + ([
        ("insufficient_scratch_slots", events, 1,
         "OVERWRITE_UNCONSUMED_VALUE")
    ] if slots > 1 else [])


def run(*, output: str | Path, max_graphs: int = 12,
        host_cases: int = 6, row_trace_cases: int = 3) -> dict:
    if type(max_graphs) is not int or not 1 <= max_graphs <= 18:
        raise ValueError("max_graphs must be in 1..18")
    if any(type(v) is not int or v < 0 for v in (host_cases, row_trace_cases)):
        raise ValueError("Host budgets must be nonnegative integers")
    if host_cases > 100 or row_trace_cases > host_cases:
        raise ValueError("Invalid bounded host/trace budgets")
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Create-only scientific schedule study already exists")
    catalog = [
        (rows, widths, hetero)
        for rows in (2, 3, 5)
        for widths in ((3, 5, 4, 2), (3, 5, 4, 3, 2), (4, 3, 7, 2))
        for hetero in (False, True)
    ][:max_graphs]
    cases = []
    host_used = 0
    trace_used = 0
    for graph_id, (rows, widths, heterogeneous) in enumerate(catalog):
        graph = make_chain(rows, widths)
        arch = _architecture(graph, heterogeneous)
        ir = resolve_row_chain(arch)
        per_operator = verify_refinement(ir.composed.lowered)
        if not per_operator["stage_invariants_verified"]:
            raise AssertionError("Unsupported base operator architecture")
        blocks = sorted({1, min(2, rows), min(3, rows), rows})
        for block in blocks:
            events = blocked_schedule(rows, len(ir.chain), block)
            candidates = [("valid", events, block, None)]
            candidates.extend(_interventions(events, block, rows))
            for mutant, candidate, slots, expected_fault in candidates:
                start = perf_counter_ns()
                verdict = check_schedule(ir, candidate, row_slots=slots)
                elapsed_ns = perf_counter_ns() - start
                expected_valid = expected_fault is None
                if verdict["status"] != ("VERIFIED" if expected_valid else "REJECTED"):
                    raise AssertionError(
                        f"Unexpected bounded schedule result: {mutant} {verdict}")
                observed = (verdict["counterexample"] or {}).get("reason")
                if observed != expected_fault:
                    raise AssertionError(
                        f"Wrong counterexample for {mutant}: {observed}; "
                        f"expected {expected_fault}")
                item = {
                    "graph_id": graph_id, "rows": rows,
                    "operator_count": len(ir.chain),
                    "model_widths": "x".join(map(str, widths)),
                    "mapping_mode": ("heterogeneous" if heterogeneous else "uniform"),
                    "row_block": block, "row_slots": slots,
                    "intervention": mutant,
                    "a_priori_expected_legal": expected_valid,
                    "source_operator_refinement_passed": True,
                    "compositional_schedule_status": verdict["status"],
                    "expected_reason": expected_fault,
                    "observed_reason": observed,
                    "first_bad_event": (verdict["counterexample"] or {}).get("event_index"),
                    "diagnostic_tensor": (verdict["counterexample"] or {}).get("tensor"),
                    "checked_element_events": verdict["checked_element_events"],
                    "check_wall_ns": elapsed_ns,
                    "host_numerical_pass": None,
                    "executed_row_timeline_pass": None,
                    "native_vitis_calls": 0,
                }
                if expected_valid and host_used < host_cases:
                    compiled = compile_row_chain(arch, row_block=block)
                    x = np.linspace(-0.125, 0.125, rows*widths[0],
                                    dtype=np.float32).reshape(rows, widths[0])
                    result = validate_host(graph, compiled.emission, x,
                                           atol=1e-5, rtol=1e-4)
                    if not result["accepted"]:
                        raise AssertionError("Actual generated HLS C++ violates functional reference")
                    item["host_numerical_pass"] = True
                    host_used += 1
                    if trace_used < row_trace_cases:
                        trace = trace_chain_epochs(compiled.ir, compiled.emission)
                        if not trace["producer_ready_before_consumer_finished"]:
                            raise AssertionError("Actual host row events disagree with checked schedule")
                        item["executed_row_timeline_pass"] = True
                        trace_used += 1
                cases.append(item)
    target.mkdir(parents=True)
    with (target/"cases.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    counts = {}
    for r in cases:
        k = r["intervention"]
        d = counts.setdefault(k, {"attempts": 0, "recognized": 0})
        d["attempts"] += 1
        d["recognized"] += int(
            (r["compositional_schedule_status"] == "VERIFIED") if k == "valid"
            else (r["compositional_schedule_status"] == "REJECTED"
                  and r["expected_reason"] == r["observed_reason"]))
    cmd = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=5)
    result = {
        "schema": SCHEMA,
        "source_revision": cmd.stdout.strip() if cmd.returncode == 0 else None,
        "prespecified_graphs": len(catalog),
        "prespecified_variant_count": len(cases),
        "positive_controls": sum(r["a_priori_expected_legal"] for r in cases),
        "negative_interventions": sum(not r["a_priori_expected_legal"] for r in cases),
        "positive_control_accepts": counts["valid"]["recognized"],
        "negative_interventions_correctly_rejected": sum(
            d["recognized"] for name, d in counts.items() if name != "valid"),
        "per_intervention": counts,
        "source_operator_refinement_passed_for_all_interventions": True,
        "host_executed_positives": host_used,
        "host_numerical_passes": sum(r["host_numerical_pass"] is True for r in cases),
        "host_executed_temporal_traces": trace_used,
        "median_schedule_check_ns": statistics.median(r["check_wall_ns"] for r in cases),
        "vitis_calls": 0,
        "evidence": "E0-explicit-schedule-counterexample-and-E1-selected-host-cxx",
        "limitations": [
            "Targeted designed schedule mutations, not real-world fault distribution",
            "Operator-only refinement is a deliberately scoped ablation, not a schedule checker",
            "Measurements are CPU process wall times, not HLS compile or physical timing",
            "No statistical generalization beyond the prespecified finite small graphs",
        ],
    }
    (target/"summary.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--max-graphs", type=int, default=12)
    p.add_argument("--host-cases", type=int, default=6)
    p.add_argument("--row-trace-cases", type=int, default=3)
    a = p.parse_args(argv)
    print(json.dumps(run(output=a.output, max_graphs=a.max_graphs,
                         host_cases=a.host_cases,
                         row_trace_cases=a.row_trace_cases), indent=2))


if __name__ == "__main__":
    main()
