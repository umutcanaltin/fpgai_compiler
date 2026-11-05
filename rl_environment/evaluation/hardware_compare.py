"""HW-RL-05 read-only, audited matched heldout HLS comparison.

No vendor invocation, no optimizer update and no reinterpretation of
unqualified raw HLS cycles as qualified latency. A paired comparison
requires *two* complete independently audited frozen evaluation campaigns
on identical original task order, hardware constraints and native budgets.

Writes only to a fresh distinct output directory; original study folders
remain immutable. Negative and censored outcomes are retained explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import isfinite
from pathlib import Path
import statistics
import sys

import numpy as np

from ..experiments.hardware_campaign import (
    CAMPAIGN_SCHEMA, audit_campaign, read_tasks, _read_jsonl,
)
from ..infrastructure.hardware_native import _regular_file
from ..study import _write_json_new


def _read_audited_run(root: str | Path, tasks) -> dict:
    directory = Path(root).expanduser()
    # The only source of native validity is an independent replay of
    # EVERY original immutable case, not the saved headline summary.
    verdict = audit_campaign(root=directory, tasks=tasks)
    if verdict["status"] != "AUDITED_COMPLETE":
        raise ValueError("A matched scientific comparison requires an audited complete study")
    meta = json.loads(_regular_file(directory / "campaign.json").read_text())
    episodes = _read_jsonl(directory / "episodes.jsonl")
    trials = _read_jsonl(directory / "trials.jsonl")
    if (meta.get("schema") != CAMPAIGN_SCHEMA
            or len(episodes) != meta["episodes"]
            or len(trials) != meta["planned_max_original_vitis_calls"]):
        raise ValueError("Completed campaign summary or case count disagrees")
    by_episode = {}
    for trial in trials:
        by_episode.setdefault(trial["episode"], []).append(trial)
    for i, episode in enumerate(episodes):
        group = by_episode.get(i, ())
        if (episode["episode"] != i
                or episode["task_id"] != meta["schedule"][i]
                or len(group) != meta["original_vitis_calls_per_episode"]):
            raise ValueError("Original episode/task schedule disagrees")
    return {
        "root": str(directory),
        "metadata": meta,
        "audit": verdict,
        "episodes": episodes,
        "trials_by_episode": by_episode,
    }


def _label(meta: dict) -> str:
    name = meta["method"]
    if name != "random" and not meta["message_passing"]:
        name += "_no_message_passing"
    if name != "random" and not meta["trial_history"]:
        name += "_no_trial_history"
    if meta.get("clock_regime") in ("fixed", "task_fixed", "search"):
        name += "_" + meta["clock_regime"] + "_clock"
    if meta.get("checkpoint_training_network_regime") in (
        "same", "family", "general", "cross_family"
    ):
        name += "_trained_" + meta["checkpoint_training_network_regime"]
    if meta.get("checkpoint_training_clock_regime") in ("fixed", "task_fixed", "search"):
        name += "_trainedclock_" + meta["checkpoint_training_clock_regime"]
    return name


def _first_feasible(group):
    for i, trial in enumerate(group, 1):
        if trial["qualified"] is True:
            return i
    return None


def paired_summary(baseline: dict, candidate: dict, *,
                   allow_clock_regime_ablation: bool = False) -> tuple[list[dict], dict]:
    """Provenance-checked per-task censoring categories and latency ratios."""
    a, b = baseline["metadata"], candidate["metadata"]
    if (a["split"] not in ("test", "validation")
            or b["split"] != a["split"]
            or not a["evaluation"] or not b["evaluation"]
            or a["seed"] != b["seed"]
            or a["schedule"] != b["schedule"]
            or a["episodes"] != b["episodes"]
            or a["original_vitis_calls_per_episode"] !=
               b["original_vitis_calls_per_episode"]
            or a["task_definitions"] != b["task_definitions"]
            or a.get("network_regime", "auto") != b.get("network_regime", "auto")
            or a["source_revision_sha"] != b["source_revision_sha"]):
        raise ValueError(
            "Paired comparison requires same seed, disjoint heldout split, "
            "source revision, task cohort and real Vitis call budget"
        )
    clock_ablation = (
        a.get("clock_regime", "search") != b.get("clock_regime", "search")
        or a.get("fixed_clock_ns") != b.get("fixed_clock_ns")
    )
    if clock_ablation and not allow_clock_regime_ablation:
        raise ValueError(
            "Different clock action spaces require explicit "
            "--allow-clock-regime-ablation with identical heldout cases/budgets"
        )
    if _label(a) == _label(b):
        raise ValueError("Paired method/ablation labels must differ")
    rows = []
    categories = {
        "both_feasible": 0, "baseline_only_feasible": 0,
        "candidate_only_feasible": 0, "neither_feasible": 0,
    }
    for i in range(a["episodes"]):
        ar, br = baseline["episodes"][i], candidate["episodes"][i]
        aa = ar["best_qualified_latency_ns"]
        bb = br["best_qualified_latency_ns"]
        if aa is not None and (type(aa) not in (int, float)
                               or not isfinite(aa) or aa <= 0):
            raise ValueError("Invalid qualified baseline latency")
        if bb is not None and (type(bb) not in (int, float)
                               or not isfinite(bb) or bb <= 0):
            raise ValueError("Invalid qualified candidate latency")
        if aa is None and bb is None:
            category = "neither_feasible"
        elif aa is None:
            category = "candidate_only_feasible"
        elif bb is None:
            category = "baseline_only_feasible"
        else:
            category = "both_feasible"
        categories[category] += 1
        if ar["task_id"] != br["task_id"]:
            raise ValueError("Different network task in paired experiment")
        ratio = bb / aa if category == "both_feasible" else None
        rows.append({
            "episode": i,
            "task_id": ar["task_id"],
            "seed": a["seed"],
            "split": a["split"],
            "baseline_method": _label(a),
            "candidate_method": _label(b),
            "baseline_best_qualified_ns": aa,
            "candidate_best_qualified_ns": bb,
            "feasibility_category": category,
            "candidate_over_baseline_latency_ratio": ratio,
            "conditional_candidate_improvement_pct":
                100 * (1. - ratio) if ratio is not None else None,
            "baseline_first_qualified_call":
                _first_feasible(baseline["trials_by_episode"][i]),
            "candidate_first_qualified_call":
                _first_feasible(candidate["trials_by_episode"][i]),
            "original_vitis_calls_per_method":
                a["original_vitis_calls_per_episode"],
        })
    ratios = [r["candidate_over_baseline_latency_ratio"]
              for r in rows if r["candidate_over_baseline_latency_ratio"] is not None]
    all_n = len(rows)
    # No fabricated zero/infinity for a task when no qualified design exists.
    summary = {
        "schema": "fpgai-research/hardware-rl-paired-result/v1",
        "status": "DESCRIPTIVE_PAIR_REQUIRES_ORIGINAL_CASE_AUDIT",
        "source_revision_sha": a["source_revision_sha"],
        "split": a["split"], "seed": a["seed"],
        "baseline": _label(a), "candidate": _label(b),
        "baseline_training_network_regime": a.get("checkpoint_training_network_regime"),
        "candidate_training_network_regime": b.get("checkpoint_training_network_regime"),
        "baseline_training_clock_regime": a.get("checkpoint_training_clock_regime"),
        "candidate_training_clock_regime": b.get("checkpoint_training_clock_regime"),
        "baseline_clock_regime": a.get("clock_regime", "search"),
        "candidate_clock_regime": b.get("clock_regime", "search"),
        "cross_clock_action_space_ablation": clock_ablation,
        "cross_clock_ablation_explicitly_authorized": (
            bool(clock_ablation and allow_clock_regime_ablation)
        ),
        "task_episode_pairs": all_n,
        "distinct_task_ids": len(set(r["task_id"] for r in rows)),
        "original_vitis_calls_per_method":
            all_n * a["original_vitis_calls_per_episode"],
        "categories": categories,
        "baseline_feasible_rate": (
            categories["both_feasible"] +
            categories["baseline_only_feasible"]
        ) / all_n,
        "candidate_feasible_rate": (
            categories["both_feasible"] +
            categories["candidate_only_feasible"]
        ) / all_n,
        "conditional_both_feasible_median_ratio": (
            statistics.median(ratios) if ratios else None
        ),
        "conditional_both_feasible_pair_count": len(ratios),
        "paired_significance_claim": False,
        "observed_latency_is_original_HLS_estimate_only": True,
        "physical_fpga_performance_observed": False,
        "reported_partial_campaigns": False,
    }
    return rows, summary


def compare_audited_campaigns(*, tasks, baseline_root, candidate_root,
                              allow_clock_regime_ablation=False):
    if (Path(baseline_root).resolve() == Path(candidate_root).resolve()):
        raise ValueError("Comparison requires two distinct source-pinned campaign roots")
    a = _read_audited_run(baseline_root, tasks)
    b = _read_audited_run(candidate_root, tasks)
    rows, summary = paired_summary(
        a, b, allow_clock_regime_ablation=allow_clock_regime_ablation,
    )
    # This status can only be granted by the two successful full source/native
    # audits above; ordinary fixture dictionaries never receive it.
    summary.update({
        "status": "INDEPENDENT_NATIVE_CASES_AUDITED_PAIR_DESCRIPTIVE",
        "baseline_campaign": a["root"],
        "candidate_campaign": b["root"],
        "baseline_audit": a["audit"],
        "candidate_audit": b["audit"],
    })
    return rows, summary


def export_comparison(output: str | Path, rows: list[dict],
                      summary: dict) -> Path:
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Create-only paper comparison output already exists")
    if not rows or not summary.get("status", "").startswith("INDEPENDENT_"):
        raise ValueError("Only genuine independently audited complete pairs may be exported")
    destination.mkdir(parents=True, exist_ok=False)
    _write_json_new(destination / "comparison.json", summary)
    with (destination / "paired_native_hls.csv").open("x", newline="",
                                                       encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--baseline", required=True,
                        help="Root of independently auditable frozen control campaign")
    parser.add_argument("--candidate", required=True,
                        help="Root of independently auditable frozen learned campaign")
    parser.add_argument("--output", required=True,
                        help="New paper comparison directory, never an original native root")
    parser.add_argument("--allow-clock-regime-ablation", action="store_true",
                        help="Explicitly compare different clock action spaces on identical heldout tasks/budgets")
    args = parser.parse_args(argv)
    try:
        tasks = read_tasks(args.tasks)
        rows, summary = compare_audited_campaigns(
            tasks=tasks, baseline_root=args.baseline,
            candidate_root=args.candidate,
            allow_clock_regime_ablation=args.allow_clock_regime_ablation,
        )
        saved = export_comparison(args.output, rows, summary)
        print(json.dumps({**summary, "comparison_output": str(saved)},
                         sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        print(f"Hardware RL scientific comparison blocked: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
