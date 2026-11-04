"""Create-only E0 exact fan-out scratch allocation + independent controls.

Original source and all candidate denominators are preserved. Source-level
HLS resources and native Vitis results are NOT measured by this study.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

from fpgai_research.experiments.cst_fanout_campaign import branch_graph
from fpgai_research.model_ir.fanout_allocator import exact_fanout_slot_plan
from fpgai_research.model_ir.fanout_lifetime import (
    canonical_fanout_events, verify_fanout_events,
)

CASES = (
    ("sequential_release", ("h", "g"), 8, 100_000),
    ("parallel_readers", ("h", "a", "b", "g"), 8, 100_000),
    ("insufficient_capacity", ("h", "a"), 2, 100_000),
    ("bounded_unknown", ("h", "a", "b", "g"), 8, 1),
)


def run(*, output: str | Path) -> dict:
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Exact fan-out E0 campaign is create-only")
    model = branch_graph()
    cases = []
    for name, names, capacity, budget in CASES:
        result = exact_fanout_slot_plan(
            model, names, max_slots=capacity, max_candidates=budget)
        if name in ("sequential_release", "parallel_readers"):
            if result["status"] != "OPTIMAL_E0":
                raise AssertionError("Prespecified small exact E0 case was not solved")
            confirmed = verify_fanout_events(
                model, result["offsets"],
                canonical_fanout_events(model, result["offsets"]))
            if confirmed["status"] != "VERIFIED":
                raise AssertionError("Independent E0 replay disagrees")
        elif result["status"] != (
                "INFEASIBLE_WITHIN_CAPACITY" if name == "insufficient_capacity"
                else "UNKNOWN_BUDGET"):
            raise AssertionError("Bounded E0 negative result was misclassified")
        cases.append({
            "case": name,
            "tensor_names": "|".join(names),
            "status": result["status"],
            "physical_slots": result["physical_slots"],
            "dense_disjoint_slots": result.get("dense_disjoint_slots"),
            "examined_candidates": result["examined_candidates"],
            "logical_event_sha256": result["logical_event_sha256"],
            "allocation_sha256": result.get("allocation_sha256"),
            "offsets_json": json.dumps(result["offsets"], sort_keys=True),
            "generated_cpp_cases": 0,
            "native_vitis_calls": result["original_vitis_calls"],
        })
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True,
        text=True, timeout=5, check=True).stdout.strip()
    summary = {
        "schema": "fpgai-research/cst-fanout-exact-allocation-campaign/v1",
        "source_revision": sha,
        "cases": len(cases),
        "optimal_e0": sum(x["status"] == "OPTIMAL_E0" for x in cases),
        "infeasible_cap": sum(x["status"] == "INFEASIBLE_WITHIN_CAPACITY" for x in cases),
        "unknown_budget": sum(x["status"] == "UNKNOWN_BUDGET" for x in cases),
        "executed_cpp_cases": 0,
        "native_vitis_calls": 0,
        "evidence": "E0-exact-bounded-abstract-consumer-ACK-allocation-only",
        "limitations": [
            "No source-lowered branching C++",
            "No concurrent access or bank port proof",
            "No native HLS or measured FPGA resources",
        ],
    }
    root.mkdir(parents=True)
    with (root / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(output=args.output), indent=2))


if __name__ == "__main__":
    main()
