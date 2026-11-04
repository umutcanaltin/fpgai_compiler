"""Independent original-HLS audited scorecards for completed Gym stages.

No cross-task absolute latency averaging, no unqualified speedups, no
synthetic reward, and no native tool calls. Each stage must be fully source-
pinned and independently replay-auditable at its original code revision.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import isfinite
from pathlib import Path

from .gym_audit import audit
from .curriculum import get_stage


def _read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def scorecard(study):
    root = Path(study).resolve(strict=True)
    campaign = json.loads((root / "campaign.json").read_text())
    stage = get_stage(campaign.get("curriculum_stage"))
    if stage is None:
        raise ValueError("Standard legacy Gym campaigns do not have staged results")
    if campaign.get("curriculum_mode") != "full" or (
            campaign["training_networks_per_seed"] < 100 or
            campaign["calls_per_episode_max"] != 10):
        raise ValueError("Only independently audited >=100-episode x10 full curriculum studies can support learning scorecards")
    receipt = audit(root)  # Source case replay and reward verification, zero Vitis
    episodes = _read_jsonl(root / "completed_episodes.jsonl")
    calls = _read_jsonl(root / "native_steps.jsonl")
    train = [r for r in calls if r["method"] == "online_train"]
    if not train or len(train) != campaign["training_networks_per_seed"] * campaign[
            "calls_per_episode_max"] * len(campaign["training_seeds"]):
        raise ValueError("Incomplete original-HLS curriculum training")
    test = [r for r in episodes if r["task_partition"] == "test"]
    test_tasks = {}
    for row in test:
        test_tasks.setdefault(row["task_sha"], {})[row["method"]] = row
    rows = []
    for task_sha, arms in sorted(test_tasks.items()):
        trained = arms.get("frozen_stochastic")
        never = arms.get("untrained_stochastic")
        ref = arms.get("random")
        if not all([trained, never, ref]):
            raise ValueError("Missing same-task, independent original-HLS test controls")
        values = [trained["best_feasible_latency_ns"],
                  never["best_feasible_latency_ns"]]
        rows.append({
            "stage": stage.name,
            "test_task_sha": task_sha,
            "trained_feasible": values[0] is not None,
            "untrained_feasible": values[1] is not None,
            "random_feasible": ref["best_feasible_latency_ns"] is not None,
            "trained_best_hls_ns": values[0],
            "untrained_best_hls_ns": values[1],
            "random_best_hls_ns": ref["best_feasible_latency_ns"],
            "trained_vs_untrained_relative_improvement": (
                (values[1]-values[0])/values[1]
                if all(v is not None and v > 0 for v in values)
                else None),
            "paired_feasible": all(v is not None for v in values),
            "equal_10_step_budget": (
                trained["original_vitis_calls"] ==
                never["original_vitis_calls"] ==
                campaign["calls_per_episode_max"]),
        })
    if not rows:
        raise ValueError("No source-attested heldout trained/untrained comparisons")
    if any(not r["equal_10_step_budget"] for r in rows):
        raise ValueError("Heldout trained/untrained policies did not have equal 10-call budgets")
    first = train[:len(train)//2]
    last = train[len(train)//2:]
    def rate(rows):
        return sum(int(x["qualified"]) for x in rows)/len(rows) if rows else None
    paired = [r for r in rows if r["paired_feasible"]]
    if not paired:
        learning_status = "INCONCLUSIVE_NO_PAIRED_FEASIBLE_TEST_LATENCIES"
    elif all(r["trained_vs_untrained_relative_improvement"] > 0
             for r in paired):
        learning_status = "OBSERVED_PILOT_IMPROVEMENT_NOT_STATISTICAL_PROOF"
    else:
        learning_status = "NO_CONSISTENT_PILOT_IMPROVEMENT_NOT_PROOF_OF_NO_LEARNING"
    return {
        "stage": stage.name,
        "minimum_100_original_training_episodes_required": True,
        "training_episodes_completed": sum(
            r["method"] == "online_train" for r in episodes),
        "learning_evidence_status": learning_status,
        "source_revision": campaign["source_revision"],
        "original_hls_calls": len(calls),
        "train_hls_calls": len(train),
        "first_half_qualified_rate": rate(first),
        "last_half_qualified_rate": rate(last),
        "all_training_qualified_rate": rate(train),
        "test_tasks": len(rows),
        "paired_feasible_trained_vs_untrained": sum(
            x["paired_feasible"] for x in rows),
        "trained_better_paired_tasks": sum(
            x["paired_feasible"] and
            x["trained_vs_untrained_relative_improvement"] > 0 for x in rows),
        "heldout": rows,
        "scientific_reward": campaign["paper_evidence"]["reward_definition"],
        "independent_case_audit": receipt,
        "source_evidence_only": True,
        "not_physical_latency": True,
        "one_seed_not_generalization_proof": True,
    }


def export(studies, output, *, require_all_seven=False):
    if require_all_seven and {stage for stage, _ in studies} != {
            f"L{i}" for i in range(7)}:
        raise ValueError("Full seven-environment comparison requires exactly L0..L6")
    if not studies:
        raise ValueError("Supply at least one completed stage")
    names = [x[0] for x in studies]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate stage labels are not permitted")
    cards = []
    for expected, root in studies:
        card = scorecard(root)
        if card["stage"] != expected:
            raise ValueError("User stage label does not match actual source-pinned campaign")
        cards.append(card)
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Curriculum comparison output must be fresh/create-only")
    target.mkdir(parents=True)
    stages = [{
        "stage": r["stage"],
        "training_episodes_completed": r["training_episodes_completed"],
        "learning_evidence_status": r["learning_evidence_status"],
        "train_hls_calls": r["train_hls_calls"],
        "training_feasibility_first_half": r["first_half_qualified_rate"],
        "training_feasibility_last_half": r["last_half_qualified_rate"],
        "heldout_test_tasks": r["test_tasks"],
        "paired_feasible_test_tasks": r["paired_feasible_trained_vs_untrained"],
        "trained_better_paired_tasks": r["trained_better_paired_tasks"],
    } for r in cards]
    # Every recorded training episode appears in a separate, source-attested
    # learning curve. Never mix rewards from different environments or replace
    # source measurement with a smoothed value.
    curves = []
    for (expected, original_root), card in zip(studies, cards):
        original = Path(original_root).resolve(strict=True)
        events = _read_jsonl(original / "native_steps.jsonl")
        episodes_by_seed = {}
        for event in events:
            if event["method"] != "online_train":
                continue
            raw_id = event["task_id"]
            if not raw_id.startswith("train_"):
                raise ValueError("Malformed original curriculum training task id")
            number = int(raw_id.split("_")[1]) + 1
            orig = Path(event["private_episode_path"]).resolve(strict=True)
            if not orig.is_relative_to(original):
                raise ValueError("Training evidence escaped original curriculum study")
            seed_id = orig.relative_to(original).parts[0]
            key = (seed_id, number)
            episodes_by_seed.setdefault(key, []).append(event)
        seeds = sorted({seed for seed, _ in episodes_by_seed})
        for seed in seeds:
            n = card["training_episodes_completed"] // len(seeds)
            sums = []
            proportions = []
            for episode in range(1, n+1):
                events_one = episodes_by_seed.get((seed, episode), [])
                if len(events_one) != 10:
                    raise ValueError("Curriculum learning curve missing one of ten actual original HLS steps")
                sums.append(sum(float(r["reward"]) for r in events_one))
                proportions.append(sum(int(r["qualified"]) for r in events_one)/10.)
                window = min(10, episode)
                curves.append({
                    "stage": expected, "seed": seed,
                    "training_episode": episode,
                    "original_hls_calls": 10,
                    "episode_actual_reward": sums[-1],
                    "episode_qualified_fraction": proportions[-1],
                    "trailing_10_episode_reward_mean": (
                        sum(sums[-window:])/window),
                    "trailing_10_episode_feasibility_mean": (
                        sum(proportions[-window:])/window),
                })
    with (target / "training_learning_curve.csv").open("x", newline="") as handle:
        fields = list(curves[0])
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(curves)
    plots = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        for stage in names:
            for seed in sorted({r["seed"] for r in curves if r["stage"] == stage}):
                sub = sorted(
                    (r for r in curves if r["stage"] == stage and r["seed"] == seed),
                    key=lambda r: r["training_episode"])
                for y_raw, y_roll, ylabel, label in (
                        ("episode_actual_reward",
                         "trailing_10_episode_reward_mean",
                         "Original HLS episode reward", "reward"),
                        ("episode_qualified_fraction",
                         "trailing_10_episode_feasibility_mean",
                         "Fraction qualified by original HLS", "feasibility")):
                    fig, ax = plt.subplots(figsize=(9, 4))
                    x = [r["training_episode"] for r in sub]
                    ax.plot(x, [r[y_raw] for r in sub], alpha=.35,
                            label="Actual episode result")
                    ax.plot(x, [r[y_roll] for r in sub],
                            label="Descriptive trailing-10 average")
                    ax.set(xlabel="Training episode (different network/task)",
                           ylabel=ylabel,
                           title=f"{stage} {seed}: source-measured {label}, no proxy QoR")
                    ax.legend()
                    fig.tight_layout()
                    filename = f"{stage}_{seed}_training_{label}.png"
                    fig.savefig(target / filename, dpi=160)
                    plt.close(fig)
                    plots.append(filename)
    with (target / "stage_scorecard.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)
    with (target / "heldout_same_task_comparison.csv").open("x",newline="") as handle:
        fields = list(cards[0]["heldout"][0]) if cards[0]["heldout"] else []
        if fields:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(row for card in cards for row in card["heldout"])
    (target / "stage_report.json").write_text(
        json.dumps({
            "schema": "fpgai-gym-multi-stage-audited-scorecard/v1",
            "stages": cards,
            "training_curve_rows": len(curves),
            "training_curve_files": ["training_learning_curve.csv", *plots],
            "native_tool_calls_during_report": 0,
            "cross_stage_reward_comparisons_prohibited": True,
            "cross_task_absolute_ns_averages_prohibited": True,
            "no_feasible_pair_means_no_speedup_claim": True,
        }, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {
        "status": "AUDITED_REAL_HLS_CURRICULUM_SCORECARDS",
        "stages_reported": names,
        "native_tool_calls_during_report": 0,
        "output": str(target),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", required=True,
                        help="StageName=/absolute/or/relative/study (repeat)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-all-seven", action="store_true",
                        help="Reject incomplete L0..L6 aggregate experiment")
    args=parser.parse_args(argv)
    studies=[]
    for item in args.study:
        stage, sep, path = item.partition("=")
        if not sep or not path:
            parser.error("--study requires L0=path form")
        get_stage(stage)
        studies.append((stage,path))
    print(json.dumps(export(
        studies,args.output,require_all_seven=args.require_all_seven),
        sort_keys=True,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
