"""Independent, read-only replay for original incomplete Gym campaigns.

A source-pinned incomplete campaign must NEVER be presented as completed.
This explicitly audits original cases already executed (including interrupted
validation), and separately verifies 100 fully complete training episodes and
the immutable original policy checkpoint. It invokes no licensed vendor tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

from fpgai_research.experiments.architecture_knobs_v2 import (
    build_catalog, compile_candidate,
)
from .cohort import make_cohort
from .gym_eval_recovery import training_reference, _read_journal, _json
from .post_hls_env import _receipt
from .reward import latency_only_reward
from .research_metadata import summarize_trial
from .paper_evidence import _git_snapshot


def audit(study):
    old = training_reference(study)
    root, campaign = old["root"], old["campaign"]
    # Verify recorded original source files from the actual original Git SHA.
    paper = campaign["paper_evidence"]
    for path, digest in paper["source_file_sha256"].items():
        source = _git_snapshot(campaign["source_revision"], path)
        if sha256(source).hexdigest() != digest:
            raise ValueError("Original paper-method source pin mismatch: " + path)
    # Reuse current parsers only where their implementation has identical
    # content to the source-pinned original; gym_study changes are not replayed.
    for path in ("rl_environment/post_hls_env.py",
                 "rl_environment/reward.py",
                 "rl_environment/clock_actions.py"):
        current = Path(__file__).resolve().parents[1] / path
        if sha256(current.read_bytes()).hexdigest() != (
                paper["source_file_sha256"][path]):
            raise ValueError("Native replay parser/reward differs from original source: "
                             + path)
    seed = old["seed"]
    tasks = make_cohort(
        training=100, validation=campaign["validation_networks_per_seed"],
        test=campaign["test_networks_per_seed"], seed=seed,
        max_vitis_calls=10, clock_ns=None,
        fixed_clock_ablation_ns=campaign.get("fixed_clock_ablation_ns"))
    by_id = {t.task_id: t for cohort in tasks.values() for t in cohort}
    journal = _read_journal(root / "native_steps.jsonl")
    completed = _read_journal(root / "completed_episodes.jsonl")
    complete_paths = {Path(r["private_episode_path"]).resolve(strict=True)
                      for r in completed}
    per_episode_count = Counter()
    per_episode_reward = Counter()
    per_episode_best = {}
    per_episode_choices = {}
    audit_counts = Counter()
    for row in journal:
        ep = Path(row["private_episode_path"]).resolve(strict=True)
        if not ep.is_relative_to(root):
            raise ValueError("Incomplete original study case escaped study root")
        path_parts = ep.relative_to(root).parts
        if (len(path_parts) != 4 or path_parts[0] != f"seed_{seed}" or
                path_parts[2] != row["task_id"] or
                path_parts[-1] != "episode_000000"):
            raise ValueError("Unexpected incomplete native case path")
        task = by_id.get(row["task_id"])
        if task is None or row["task_sha"] != task.task_sha:
            raise ValueError("Task changed or missing from original source cohort")
        index = per_episode_count[ep]
        if row["native_call_index"] != index or index >= 10:
            raise ValueError("Nonsequential native step journal")
        case = ep / f"case_{index:04d}"
        action = _json(case / "action.json")
        chosen = tuple(tuple(v) for v in action["choices"])
        key = (chosen, action["requested_clock_ns"])
        seen = per_episode_choices.setdefault(ep, set())
        if key in seen:
            raise ValueError("Original native study evaluated repeated architecture-clock pair")
        seen.add(key)
        emitted, _, _ = compile_candidate(task.model, build_catalog(task.model), chosen)
        if (sha256((case / "model.cpp").read_bytes()).hexdigest() !=
                emitted.manifest["cpp_sha256"] or
                action["requested_clock_ns"] != row["clock_ns"]):
            raise ValueError("Original native C++ or requested HLS period changed")
        actual = _receipt(case, task)
        original_step = _json(case / "online_step.json")
        if original_step["result"] != actual:
            raise ValueError("Original qualified HLS receipt changed")
        if (actual["qualified"] != row["qualified"]
                or actual["hls_estimated_latency_ns"] !=
                    row["hls_estimated_latency_ns"]
                or actual["vitis_invocation_sha256"] !=
                    row["vitis_invocation_sha256"]
                or actual["hls_report_sha256"] != row["hls_report_sha256"]):
            raise ValueError("Original native receipt or HLS latency changed")
        trial = summarize_trial(original_step, task=task,
                                training_metrics=row.get("training_metrics"))
        if row["trial_evidence"] != trial:
            raise ValueError("Original per-trial scientific evidence drift")
        previous = per_episode_best.get(ep)
        if actual["qualified"]:
            observed = actual["hls_estimated_latency_ns"]
            updated = min(previous, observed) if previous is not None else observed
        else:
            updated = previous
        reward = latency_only_reward(previous, updated, qualified=actual["qualified"])
        if (not isclose(float(row["reward"]), reward, rel_tol=1e-11, abs_tol=1e-11)
                or not isclose(float(original_step["reward"]), reward,
                               rel_tol=1e-11, abs_tol=1e-11)
                or row["best_feasible_latency_ns"] != updated):
            raise ValueError("Original source-pinned native reward mismatch")
        if (row["method"] == "online_train") != (
                row.get("training_metrics") is not None):
            raise ValueError("Original online-vs-frozen policy update evidence mismatch")
        per_episode_count[ep] += 1
        per_episode_reward[ep] += reward
        per_episode_best[ep] = updated
        audit_counts[row["method"]] += 1
        audit_counts["qualified"] += int(actual["qualified"])
        audit_counts["rejected"] += int(not actual["qualified"])
    for row in completed:
        ep = Path(row["private_episode_path"]).resolve(strict=True)
        if (per_episode_count[ep] != row["original_vitis_calls"]
                or row["best_feasible_latency_ns"] != per_episode_best.get(ep)
                or not isclose(float(row["episode_return"]),
                               per_episode_reward[ep],
                               rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("Completed original training episode/accounting drift")
    if (len(complete_paths) != 100 or
            any(per_episode_count[ep] != 10 for ep in complete_paths)):
        raise ValueError("Original campaign completed-episode journal changed")
    if audit_counts["online_train"] != 1000:
        raise ValueError("Original 100-episode training journal not complete")
    return {
        "status": "INDEPENDENT_INCOMPLETE_CAMPAIGN_NATIVE_CASE_REPLAY_PASSED",
        "source_revision": campaign["source_revision"],
        "full_campaign_completed": False,
        "training_episodes_completed": 100,
        "training_original_native_calls_audited": 1000,
        "partial_validation_original_native_calls_audited":
            audit_counts["frozen"],
        "total_original_native_calls_replayed": len(journal),
        "all_methods_qualified_calls": audit_counts["qualified"],
        "all_methods_rejected_calls": audit_counts["rejected"],
        "source_and_original_files_modified": 0,
        "vendor_invocations_during_audit": 0,
        "full_1090_case_study_audit_passed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.study), sort_keys=True,
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
