"""CPU-only, reproducible paper-comparison manifest planner.

This planner never imports vendor runners, creates result directories, or
asserts that prospective agents have been implemented.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .agents import available_agents
from .curriculum import get_stage

SCHEMA = "fpgai-paper-agent-comparison-plan/v1"
VALID_STAGES = tuple(f"L{i}" for i in range(7))
FULL_TRAINING_EPISODES = 100
FULL_TRAINING_CALLS_PER_EPISODE = 10


def build_comparison_plan(*, stages=("L0",), agents=("graph_actor_critic",),
                          seeds=(111, 222, 333), training=100,
                          calls_per_episode=10):
    """Validate actual registry support and freeze the prospective comparison."""
    stages, agents, seeds = tuple(stages), tuple(agents), tuple(seeds)
    for name, values in (("stages", stages), ("agents", agents), ("seeds", seeds)):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and contain no duplicates")
    if any(s not in VALID_STAGES for s in stages):
        raise ValueError("Unknown curriculum stage")
    missing = set(agents) - set(available_agents())
    if missing:
        raise ValueError(f"Unimplemented/unregistered agent(s): {sorted(missing)}")
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if type(training) is not int or training < FULL_TRAINING_EPISODES:
        raise ValueError("Full paper plan requires at least 100 training episodes")
    if type(calls_per_episode) is not int or calls_per_episode != 10:
        raise ValueError("Full paper curriculum requires exactly 10 original HLS calls per training episode")
    matrix = []
    for stage in stages:
        contract = get_stage(stage)
        for agent in agents:
            for seed in seeds:
                matrix.append({
                    "study_id": f"{stage.lower()}__{agent}__seed{seed}",
                    "stage": stage,
                    "algorithm": agent,
                    "seed": seed,
                    "training_episodes": training,
                    "required_training_native_calls": training * calls_per_episode,
                    "training_calls_per_episode": calls_per_episode,
                    "fixed_clock_ns": contract.fixed_clock_ns,
                    "freeze_numeric_format": contract.freeze_numeric_format,
                    "results_status": "NOT_EXECUTED",
                })
    payload = {
        "schema": SCHEMA,
        "status": "PROSPECTIVE_SOURCE_ONLY",
        "vendor_calls_executed": 0,
        "training_native_calls_required_total": sum(
            x["required_training_native_calls"] for x in matrix),
        "evaluation_native_calls_not_included": True,
        "independent_native_audit_required": True,
        "studies": matrix,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=["L0"], choices=VALID_STAGES)
    parser.add_argument("--agents", nargs="+", default=["graph_actor_critic"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[111, 222, 333])
    parser.add_argument("--training", type=int, default=100)
    parser.add_argument("--output", type=Path, help="Optional create-only JSON path")
    args = parser.parse_args(argv)
    result = build_comparison_plan(
        stages=args.stages, agents=args.agents, seeds=args.seeds,
        training=args.training)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            f.write(rendered)
        print(f"Created source-only paper plan: {args.output}")


if __name__ == "__main__":
    main()
