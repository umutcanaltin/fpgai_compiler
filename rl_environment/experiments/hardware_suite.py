"""Create-only, source-only FPGA hardware-RL benchmark scenario suite.

Ingest one existing vetted hardware-RL tasks/v1 manifest with train,
validation and test tasks. Reject structural heldout leakage when requested;
build compatible same/family/general/cross_family x fixed/task_fixed/search
manifests and explicit unavailable decisions. NEVER fabricate ONNX models,
native HLS evidence, rewards, or a surrogate. No vendor calls are made.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

from ..infrastructure.hardware_native import _regular_file
from .hardware_campaign import (
    TASK_SCHEMA, SPLITS, campaign_plan, read_tasks,
)
from .hardware_regimes import (
    onnx_structure_sha256, onnx_operator_family,
)

SUITE_SCHEMA = "fpgai-research/hardware-rl-scenario-suite/v1"
SCENARIOS = ("same", "family", "general", "cross_family")
CLOCKS = ("fixed", "task_fixed", "search")


def _pick_cohort(candidates, *, network_regime, clock_regime):
    """Choose deterministic study membership using source facts only.

    A suite plan never silently changes a timing target. The task manifest
    remains the source of each FPGA part, caps, initial clock and sample.
    """
    by_model = defaultdict(list)
    by_family = defaultdict(list)
    for t in sorted(candidates, key=lambda x: x.id):
        by_model[t.onnx_sha256].append(t)
        by_family[onnx_operator_family(t.onnx)].append(t)
    if network_regime == "same":
        cohorts = list(by_model.values())
    elif network_regime == "family":
        cohorts = list(by_family.values())
    else:
        cohorts = [list(candidates)]
    # Deterministic preference: most distinct task IDs, then ID order.
    cohorts.sort(key=lambda cohort: (-len(cohort), tuple(t.id for t in cohort)))
    for cohort in cohorts:
        if clock_regime == "fixed":
            per_clock = defaultdict(list)
            for t in cohort:
                per_clock[t.initial_clock_ns].append(t)
            slices = sorted(per_clock.values(), key=lambda seq: (
                -len(seq), tuple(t.id for t in seq),
            ))
        else:
            slices = [cohort]
        for choice in slices:
            # same must always mean identical ONNX bytes. Multi-network
            # regimes must retain >=2 different structural fingerprints.
            structures = {onnx_structure_sha256(t.onnx) for t in choice}
            families = {onnx_operator_family(t.onnx) for t in choice}
            if network_regime == "same" and len({t.onnx_sha256 for t in choice}) != 1:
                continue
            if network_regime in ("family", "general", "cross_family") and len(structures) < 2:
                continue
            if network_regime == "family" and len(families) != 1:
                continue
            if network_regime == "cross_family" and len(families) < 2:
                continue
            if clock_regime == "task_fixed" and len({
                t.initial_clock_ns for t in choice
            }) < 2:
                continue
            return tuple(choice)
    raise ValueError(
        "No task cohort satisfies network and clock regime concurrently; "
        "provide distinct ONNX graphs and/or task-fixed clock targets"
    )


def plan_scenario_suite(tasks, *, holdout="structure", seed=101,
                        calls_per_episode=2, method="ppo",
                        reward_mode="best_improvement",
                        reward_reference_ns=10000.0):
    if holdout not in ("structure", "bytes"):
        raise ValueError("Holdout protocol must be structure or bytes")
    if any(not any(t.split == split for t in tasks) for split in SPLITS):
        raise ValueError("A research suite requires train, validation and test tasks")
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Suite task IDs must be unique")
    # A file hash cannot leak across splits; read_tasks also checks this.
    split_by_bytes = {}
    split_by_structure = {}
    graph_families = {}
    for t in tasks:
        previous = split_by_bytes.setdefault(t.onnx_sha256, t.split)
        if previous != t.split:
            raise ValueError("Identical ONNX files leak across suite splits")
        structure = onnx_structure_sha256(t.onnx)
        previous = split_by_structure.setdefault(structure, t.split)
        if holdout == "structure" and previous != t.split:
            raise ValueError("Identical ONNX graph structure leaks across heldout splits")
        graph_families[t.id] = onnx_operator_family(t.onnx)
    runs = []
    for split in SPLITS:
        available = tuple(t for t in tasks if t.split == split)
        for network in SCENARIOS:
            for clock in CLOCKS:
                name = f"{split}_{network}_{clock}"
                try:
                    cohort = _pick_cohort(
                        available, network_regime=network, clock_regime=clock,
                    )
                    plan = campaign_plan(
                        tasks=cohort, split=split, method=method,
                        episodes=len(cohort), calls_per_episode=calls_per_episode,
                        total_authorized_calls=None, seed=seed,
                        evaluation=False, checkpoint=None,
                        hidden=64, ppo_epochs=3, ppo_clip=.2,
                        network_regime=network, clock_regime=clock,
                        reward_mode=reward_mode,
                        reward_reference_ns=reward_reference_ns,
                    )
                    runs.append({
                        "id": name, "status": "SOURCE_PLAN_AVAILABLE",
                        "network_regime": network, "clock_regime": clock,
                        "split": split,
                        "task_ids": [t.id for t in cohort],
                        "task_definitions": [t.metadata() for t in cohort],
                        "plan": plan,
                    })
                except ValueError as error:
                    runs.append({
                        "id": name, "status": "UNAVAILABLE_SOURCE_COHORT",
                        "network_regime": network, "clock_regime": clock,
                        "split": split, "reason": str(error),
                    })
    return {
        "schema": SUITE_SCHEMA, "holdout": holdout,
        "seed": seed, "method": method,
        "calls_per_episode": calls_per_episode,
        "reward_mode": reward_mode,
        "reward_reference_ns": float(reward_reference_ns),
        "task_count": len(tasks),
        "task_family_by_id": dict(sorted(graph_families.items())),
        "source_only": True,
        "native_vitis_calls": 0,
        "scenarios": runs,
    }


def write_scenario_suite(root: str | Path, suite: dict):
    """Write all model *references* atomically to a new root only.

    Each generated scenario manifest references original model/sample files
    by absolute path, and all hashes are rechecked by read_tasks. The suite
    does not copy/modify models or allocate original Vitis cases.
    """
    root = Path(root).expanduser()
    if root.exists() or root.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing scenario suite")
    if suite.get("schema") != SUITE_SCHEMA or suite.get("source_only") is not True:
        raise ValueError("Unknown or native-authorized scenario suite")
    root.mkdir(parents=True, exist_ok=False)
    folder = root / "scenarios"
    folder.mkdir(exist_ok=False)
    for case in suite["scenarios"]:
        if case["status"] != "SOURCE_PLAN_AVAILABLE":
            continue
        path = folder / (case["id"] + ".json")
        records = [{
            key: value for key, value in task.items()
            if key not in ("onnx_sha256", "samples_sha256")
        } for task in case["task_definitions"]]
        with path.open("x", encoding="utf-8") as f:
            json.dump({"schema": TASK_SCHEMA, "tasks": records},
                      f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
        # Validate new path resolves to original immutable model and samples.
        check = read_tasks(path)
        if [(t.onnx_sha256, t.samples_sha256) for t in check] != [
            (t["onnx_sha256"], t["samples_sha256"])
            for t in case["task_definitions"]
        ]:
            raise ValueError("Scenario references changed during suite allocation")
    # Also write full source train/validation/test manifests.
    for split in SPLITS:
        original = [
            t for t in suite["scenarios"]
            if t["split"] == split and t["status"] == "SOURCE_PLAN_AVAILABLE"
        ]
        identities = {}
        for entry in original:
            for definition in entry["task_definitions"]:
                identities[definition["id"]] = definition
        # A split's complete source cohort can include tasks not selected by
        # any compatible scenario. Retain that distinction in suite.json.
        with (root / f"selected_{split}_tasks.json").open("x", encoding="utf-8") as f:
            json.dump({
                "schema": TASK_SCHEMA,
                "tasks": [{
                    k: v for k, v in task.items()
                    if k not in ("onnx_sha256", "samples_sha256")
                } for task in sorted(identities.values(), key=lambda x: x["id"])
                ],
            }, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
    with (root / "suite.json").open("x", encoding="utf-8") as f:
        json.dump(suite, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True,
                        help="Existing tasks/v1 with train, validation and test")
    parser.add_argument("--output-root",
                        help="Fresh source-only suite folder; omit for read-only preview")
    parser.add_argument("--holdout", choices=("structure", "bytes"),
                        default="structure")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--calls-per-episode", type=int, default=2)
    parser.add_argument("--method", choices=("ppo", "reinforce",
                        "actor_critic", "random"), default="ppo")
    parser.add_argument("--reward-mode",
                        choices=("best_improvement", "dense_latency"),
                        default="best_improvement")
    parser.add_argument("--reward-reference-ns", type=float, default=10000.0)
    args = parser.parse_args(argv)
    try:
        source = _regular_file(Path(args.tasks))
        tasks = read_tasks(source)
        suite = plan_scenario_suite(
            tasks, holdout=args.holdout, seed=args.seed,
            calls_per_episode=args.calls_per_episode,
            method=args.method, reward_mode=args.reward_mode,
            reward_reference_ns=args.reward_reference_ns,
        )
        suite["source_manifest_sha256"] = sha256(source.read_bytes()).hexdigest()
        if args.output_root:
            written = write_scenario_suite(args.output_root, suite)
            summary = {
                "schema": SUITE_SCHEMA, "source_only": True,
                "output_root": str(written),
                "available_scenarios": sum(
                    x["status"] == "SOURCE_PLAN_AVAILABLE"
                    for x in suite["scenarios"]
                ),
                "unavailable_scenarios": sum(
                    x["status"] != "SOURCE_PLAN_AVAILABLE"
                    for x in suite["scenarios"]
                ),
                "native_vitis_calls": 0,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(json.dumps(suite, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, TypeError, ValueError, KeyError, RuntimeError) as exc:
        print(f"HW-RL scenario suite stopped: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
