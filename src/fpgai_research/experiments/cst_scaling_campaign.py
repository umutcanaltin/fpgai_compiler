"""Deterministic scale sweep for CST finite legality and certificate overhead.

Unlike QoR studies, this measures ONLY CPU checker cost for synthetic graphs,
including the time to construct IR and a finite lifetime certificate. Source-
pinned, create-only, all prespecified sample measurements retained. Do not
claim asymptotic complexity from a handful of empirical points.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter_ns

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.row_chain import resolve_row_chain
from fpgai_research.model_ir.row_chain_liveness import verify_row_chain_liveness
from fpgai_research.model_ir.temporal_schedule import blocked_schedule, check_schedule


# Prespecified dimensions; no learned workloads and no synthesis tool calls.
SIZES = (2, 8, 32, 128, 512)
TOPOLOGIES = (
    (3, 5, 2),           # two MatMul operators / one boundary
    (3, 5, 4, 2),        # three operators / two boundaries
    (3, 5, 4, 3, 2),     # four operators / three boundaries
)


def run(*, output: str | Path, max_graphs: int = 15,
        repetitions: int = 3) -> dict:
    if type(max_graphs) is not int or not 1 <= max_graphs <= 15:
        raise ValueError("max_graphs must be 1..15")
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be 1..10")
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Create-only scaling study exists")
    setups = [(rows, widths) for rows in SIZES for widths in TOPOLOGIES][:max_graphs]
    observations = []
    for graph_id, (rows, widths) in enumerate(setups):
        graph = make_chain(rows, widths)
        ir_start = perf_counter_ns()
        ir = resolve_row_chain(resolve(graph, ArchitectureRequest()))
        ir_elapsed = perf_counter_ns()-ir_start
        lifetime_start = perf_counter_ns()
        lifetime = verify_row_chain_liveness(ir)
        lifetime_elapsed = perf_counter_ns()-lifetime_start
        if lifetime["status"] != "EXHAUSTIVE_BOUNDED_LIVENESS_VERIFIED":
            raise AssertionError("Prespecified scaling case exceeded finite proof bound")
        slots = sorted({1, min(4, rows), rows})
        for block in slots:
            events = blocked_schedule(rows, len(ir.chain), block)
            timings = []
            outcome = None
            for sample in range(repetitions):
                start = perf_counter_ns()
                current = check_schedule(ir, events, row_slots=block)
                timings.append(perf_counter_ns()-start)
                if current["status"] != "VERIFIED":
                    raise AssertionError("Legal scale control not verified")
                if outcome is None:
                    outcome = current
                elif outcome["schedule_sha256"] != current["schedule_sha256"]:
                    raise AssertionError("Different outcomes for identical schedule")
                observations.append({
                    "graph": graph_id,
                    "rows": rows,
                    "operators": len(ir.chain),
                    "topology": "x".join(map(str, widths)),
                    "scratch_row_slots": block,
                    "logical_intermediate_elements": ir.full_intermediate_elements,
                    "physical_scratch_elements": outcome["source_scratch_elements"],
                    "logical_compute_stage_events": len(events),
                    "producer_consumer_element_events": outcome["checked_element_events"],
                    "check_sample": sample,
                    "ir_resolve_ns_one_observation": ir_elapsed,
                    "liveness_check_ns_one_observation": lifetime_elapsed,
                    "schedule_check_ns": timings[-1],
                    "schedule_valid": True,
                    "original_vitis_calls": 0,
                })
    summary = {
        "schema": "fpgai-research/cst-finite-check-scaling/v1",
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=5).stdout.strip(),
        "distinct_graphs": len(setups),
        "schedule_samples": len(observations),
        "all_schedule_samples_accepted": all(x["schedule_valid"] for x in observations),
        "row_sizes": sorted({x["rows"] for x in observations}),
        "operator_counts": sorted({x["operators"] for x in observations}),
        "max_logical_intermediate_elements": max(
            x["logical_intermediate_elements"] for x in observations),
        "max_checked_element_events": max(
            x["producer_consumer_element_events"] for x in observations),
        "median_checker_runtime_ns": statistics.median(
            x["schedule_check_ns"] for x in observations),
        "raw_timing_samples_preserved": True,
        "vitis_calls": 0,
        "evidence": "E0-observed-CPU-checker-scaling-only",
        "limitations": [
            "Host CPU scheduling/thermal noise; no controlled hardware performance claim",
            "Static synthetic MatMul chains only; no non-chain dataflow scalability",
            "CPU checker observations do not equal HLS compilation runtime",
            "Does not establish asymptotic complexity or novel priority",
        ],
    }
    target.mkdir(parents=True)
    with (target/"cases.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    (target/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-graphs", type=int, default=15)
    parser.add_argument("--repetitions", type=int, default=3)
    options = parser.parse_args(argv)
    print(json.dumps(run(output=options.output,
                         max_graphs=options.max_graphs,
                         repetitions=options.repetitions), indent=2))


if __name__ == "__main__":
    main()
