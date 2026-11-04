"""Finish held-out evaluation after an immutable Gym training study was interrupted.

This never modifies, resumes, truncates, copies or re-runs original native
training cases. Creates a DIFFERENT source-pinned evaluation campaign referencing
the exact original 100-episode checkpoint. An independent read-only receipt
replay confirms every new Vitis case. Separate source commits are explicitly
recorded: results must not be presented as one original unbroken campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

from fpgai_research.experiments.architecture_knobs_v2 import (
    build_catalog, compile_candidate,
)
from .cohort import make_cohort
from .baselines import METHODS
from .gym_agent import OnlineGraphTrainer
from .gym_study import _episode, plan
from .post_hls_env import _receipt
from .reward import latency_only_reward
from .study import _source_revision, _write_json_new, _append_jsonl


METHODS_EVAL = (
    "validation_frozen", "untrained_stochastic", "frozen",
    "frozen_stochastic", *METHODS, "online_adapt",
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_journal(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def training_reference(study):
    """Validate immutable completed training and exact private checkpoint.

    The previous crash means the study has no final summary; do not invent one.
    This read-only preflight checks original journals/checkpoint integrity and
    rejects an incomplete 100-episode training series. Full original native
    case receipt replay remains a separate source-pinned original-code audit.
    """
    root = Path(study).resolve(strict=True)
    campaign = _json(root / "campaign.json")
    if campaign.get("schema") != "fpgai-gym-native-plan/v1":
        raise ValueError("Original study is not the source-pinned Gym campaign")
    seeds = campaign["training_seeds"]
    if len(seeds) != 1:
        raise ValueError("Separate evaluation recovery currently requires one seed")
    seed = seeds[0]
    training = campaign["training_networks_per_seed"]
    if training != 100 or campaign["calls_per_episode_max"] != 10:
        raise ValueError("This recovery is specifically for 100 training episodes x10 calls")
    if (root / "summary.json").exists():
        raise ValueError("Original study is complete; do not create recovery evaluation")
    completed = _read_journal(root / "completed_episodes.jsonl")
    training_rows = [r for r in completed if r["method"] == "online_train"]
    if (len(training_rows) != training
            or {r["task_id"] for r in training_rows} !=
               {f"train_{i:05d}" for i in range(training)}
            or any(r["original_vitis_calls"] != 10 for r in training_rows)):
        raise ValueError("Training journal does not contain 100 complete 10-step episodes")
    records = _read_journal(root / "native_steps.jsonl")
    training_steps = [r for r in records if r["method"] == "online_train"]
    if len(training_steps) != 1000:
        raise ValueError("Training native case journal does not contain 1,000 original steps")
    by_task = Counter(r["task_id"] for r in training_steps)
    if any(by_task[f"train_{i:05d}"] != 10 for i in range(training)):
        raise ValueError("Original training steps are not 10 per distinct task")
    for episode in training_rows:
        ep = Path(episode["private_episode_path"]).resolve(strict=True)
        if not ep.is_relative_to(root) or not ep.is_dir():
            raise ValueError("Original training native episode is unavailable")
    cp = root / f"seed_{seed}" / "checkpoint"
    meta = _json(cp / "manifest.json")
    if (meta.get("source_revision") != campaign["source_revision"]
            or meta.get("online_updates") != 1000
            or meta.get("training_episodes_completed", 100) != 100
            or len(meta.get("train_task_shas", [])) != 100
            or meta.get("policy_sha256") != _sha(cp / "shared_gnn_policy.npz")
            or meta.get("optimizer_sha256") != _sha(cp / "optimizer_private.pt")):
        raise ValueError("Original training checkpoint provenance/hash mismatch")
    return {
        "root": root, "campaign": campaign, "seed": seed,
        "checkpoint": cp, "checkpoint_manifest": meta,
        "completed_training_episodes": 100,
        "original_training_calls": 1000,
        "original_journal_rows": len(records),
    }


def run(*, training_study, output, execute_native=False,
        max_total_vitis_calls=None, vitis_executable="vitis_hls",
        timeout_seconds=7200):
    original = training_reference(training_study)
    source = _source_revision()
    seed = original["seed"]
    old = original["campaign"]
    nval, ntest = (old["validation_networks_per_seed"],
                   old["test_networks_per_seed"])
    if nval != 1 or ntest != 1:
        raise ValueError("Current matched continuation requires one validation/test task")
    calls_per = old["calls_per_episode_max"]
    count = (nval + 8 * ntest) * calls_per
    if not execute_native:
        return {
            "status": "EVALUATION_RECOVERY_DRY_RUN_ZERO_VENDOR_CALLS",
            "training_source_revision": old["source_revision"],
            "evaluation_source_revision": source,
            "old_training_original_vitis_calls": 1000,
            "new_evaluation_max_vitis_calls": count,
            "new_evaluation_episodes": nval + 8 * ntest,
            "source_revisions_explicitly_distinct": source != old["source_revision"],
        }
    if max_total_vitis_calls is None or max_total_vitis_calls < count:
        raise ValueError("Explicit full evaluation-only native budget is required")
    if type(timeout_seconds) is not int or not 60 <= timeout_seconds <= 14400:
        raise ValueError("Invalid original Vitis timeout")
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Recovery output must be entirely new, never overwrite")
    cohort = make_cohort(
        training=100, validation=nval, test=ntest,
        max_vitis_calls=calls_per, seed=seed, clock_ns=None,
        fixed_clock_ablation_ns=old.get("fixed_clock_ablation_ns"))
    if cohort["train"][-1].task_sha != original["checkpoint_manifest"]["train_task_shas"][-1]:
        raise ValueError("Training cohort fingerprint differs from checkpoint")
    dest.mkdir(parents=True)
    dest = dest.resolve(strict=True)
    checkpoint = original["checkpoint"]
    initial_checkpoint = {
        "manifest_sha256": _sha(checkpoint / "manifest.json"),
        "policy_sha256": original["checkpoint_manifest"]["policy_sha256"],
        "optimizer_sha256": original["checkpoint_manifest"]["optimizer_sha256"],
    }
    _write_json_new(dest / "campaign.json", {
        "schema": "fpgai-gym-separate-evaluation-continuation/v1",
        "status": "IN_PROGRESS",
        "evaluation_source_revision": source,
        "training_source_revision": old["source_revision"],
        "original_training_study_path": str(original["root"]),
        "training_checkpoint_path": str(checkpoint),
        "original_training_calls": 1000,
        "original_training_episode_count": 100,
        "checkpoint_files_sha256": initial_checkpoint,
        "seed": seed, "validation_tasks": nval, "test_tasks": ntest,
        "calls_per_episode": calls_per,
        "max_total_vitis_calls_authorized": max_total_vitis_calls,
        "expected_evaluation_calls_max": count,
        "original_vitis_executable": vitis_executable,
        "explicitly_separate_source_pinned_studies": True,
        "source_changed_for_duplicate_action_rejection_handling": True,
        "scientific_reward_unchanged": True,
        "not_same_revision_as_original_training": source != old["source_revision"],
    })
    agent = OnlineGraphTrainer(
        seed=seed, use_message_passing=old["message_passing_enabled"])
    agent.load_private(checkpoint, load_optimizer=False)
    adapted = OnlineGraphTrainer(
        seed=seed, use_message_passing=old["message_passing_enabled"])
    adapted.load_private(checkpoint, load_optimizer=True)
    untrained = OnlineGraphTrainer(
        seed=seed, use_message_passing=old["message_passing_enabled"])
    outrows = []
    native_calls = 0

    def log(row):
        nonlocal native_calls
        if _source_revision() != source:
            raise RuntimeError("Recovery source changed mid-experiment")
        native_calls += 1
        if native_calls > max_total_vitis_calls:
            raise RuntimeError("Recovery native-call budget exceeded")
        _append_jsonl(dest / "native_steps.jsonl", row)

    def episode(task, subdir, policy, *, method, action_seed):
        result = _episode(
            task, dest / f"seed_{seed}" / subdir / task.task_id,
            policy, method=method, seed=action_seed, native=True,
            vitis_executable=vitis_executable,
            timeout_seconds=timeout_seconds, log=log)
        outrows.append(result)
        _append_jsonl(dest / "completed_episodes.jsonl", result)

    for i, task in enumerate(cohort["validation"]):
        episode(task, "validation_frozen", agent,
                method="frozen", action_seed=seed+1000+i)
    for i, task in enumerate(cohort["test"]):
        same_seed = seed+2000+i
        episode(task, "test_untrained_stochastic", untrained,
                method="untrained_stochastic", action_seed=same_seed)
        episode(task, "test_frozen", agent,
                method="frozen", action_seed=same_seed)
        episode(task, "test_frozen_stochastic", agent,
                method="frozen_stochastic", action_seed=same_seed)
        for method in METHODS:
            episode(task, "test_"+method, agent,
                    method=method, action_seed=same_seed)
        episode(task, "test_adaptive", adapted,
                method="online_adapt", action_seed=same_seed)
    _write_json_new(dest / "summary.json", {
        "schema": "fpgai-gym-separate-evaluation-results/v1",
        "status": "RECOVERED_SEPARATE_NATIVE_EVALUATION_AWAIT_AUDIT",
        "evaluation_source_revision": source,
        "training_source_revision": old["source_revision"],
        "original_vitis_calls_recorded": native_calls,
        "results": outrows,
        "full_original_training_replay_claim": False,
        "contains_training_source_data": False,
        "no_physical_fmax_claim": True,
    })
    return {
        "status": "SEPARATE_EVALUATION_NATIVE_COMPLETE_PENDING_AUDIT",
        "new_evaluation_calls": native_calls,
        "original_training_calls_not_repeated": 1000,
        "evaluation_root": str(dest),
    }


def audit(study):
    """Replay every NEW original native evaluation receipt without Vitis calls.

    Training evidence is checkpoint-verified against original journals;
    the original unfinished study is NOT relabeled a fully audited campaign.
    """
    root = Path(study).resolve(strict=True)
    campaign = _json(root / "campaign.json")
    summary = _json(root / "summary.json")
    if (campaign.get("schema") != "fpgai-gym-separate-evaluation-continuation/v1"
            or summary.get("schema") != "fpgai-gym-separate-evaluation-results/v1"
            or summary.get("evaluation_source_revision") != _source_revision()
            or campaign.get("evaluation_source_revision") != _source_revision()):
        raise ValueError("Recovery audit source revision or campaign contract drift")
    original = training_reference(campaign["original_training_study_path"])
    cp = original["checkpoint"]
    expected = campaign["checkpoint_files_sha256"]
    if (expected["manifest_sha256"] != _sha(cp / "manifest.json")
            or expected["policy_sha256"] != _sha(cp / "shared_gnn_policy.npz")
            or expected["optimizer_sha256"] != _sha(cp / "optimizer_private.pt")
            or original["campaign"]["source_revision"] !=
                campaign["training_source_revision"]):
        raise ValueError("Recovery training-source checkpoint provenance changed")
    records = _read_journal(root / "native_steps.jsonl")
    completed = _read_journal(root / "completed_episodes.jsonl")
    if (completed != summary["results"] or len(completed) != 9
            or sum(r["original_vitis_calls"] for r in completed) != len(records)
            or len(records) > campaign["expected_evaluation_calls_max"]):
        raise ValueError("Recovery episode/journal accounting failure")
    seed = campaign["seed"]
    tasks = make_cohort(
        training=100, validation=1, test=1,
        seed=seed, max_vitis_calls=campaign["calls_per_episode"],
        clock_ns=None, fixed_clock_ablation_ns=original["campaign"].get(
            "fixed_clock_ablation_ns"))
    taskmap = {task.task_id: task for split in tasks.values() for task in split}
    seen = set()
    best = {}
    observed_counts = Counter()
    observed_reward = Counter()
    for row in records:
        ep = Path(row["private_episode_path"]).resolve(strict=True)
        if not ep.is_relative_to(root):
            raise ValueError("Recovery case escaped new evaluation root")
        rel = ep.relative_to(root).parts
        if (len(rel) != 4 or rel[0] != f"seed_{seed}" or
                rel[2] != row["task_id"] or rel[3] != "episode_000000"):
            raise ValueError("Recovery episode layout drift")
        task = taskmap.get(row["task_id"])
        if task is None or task.task_sha != row["task_sha"]:
            raise ValueError("Recovery task fingerprint drift")
        idx = row["native_call_index"]
        if (ep, idx) in seen:
            raise ValueError("Duplicate original HLS evaluation journal index")
        seen.add((ep, idx))
        case = ep / f"case_{idx:04d}"
        action = _json(case / "action.json")
        chosen = tuple(tuple(v) for v in action["choices"])
        emitted, _, _ = compile_candidate(task.model, build_catalog(task.model), chosen)
        if (_sha(case / "model.cpp") != emitted.manifest["cpp_sha256"]
                or row["clock_ns"] != action["requested_clock_ns"]):
            raise ValueError("Recovery original C++ or clock different from source")
        actual = _receipt(case, task)
        original_step = _json(case / "online_step.json")
        if original_step["result"] != actual:
            raise ValueError("Recovery original Vitis receipt or qualification drift")
        if (row["qualified"] != actual["qualified"]
                or row["hls_estimated_latency_ns"] != actual["hls_estimated_latency_ns"]
                or row["vitis_invocation_sha256"] !=
                    actual["vitis_invocation_sha256"]):
            raise ValueError("Recovery qualification/native HLS hashes disagree")
        previous = best.get(ep)
        candidate_latency = actual["hls_estimated_latency_ns"]
        new = (min(previous, candidate_latency) if previous is not None
               else candidate_latency) if actual["qualified"] else previous
        reward = latency_only_reward(previous, new, qualified=actual["qualified"])
        if (not isclose(float(row["reward"]), reward, rel_tol=1e-10, abs_tol=1e-10)
                or not isclose(float(original_step["reward"]), reward,
                               rel_tol=1e-10, abs_tol=1e-10)
                or row["best_feasible_latency_ns"] != new):
            raise ValueError("Recovery real HLS latency or scientific reward drift")
        if (row["method"] == "online_adapt") != (row.get("training_metrics") is not None):
            raise ValueError("Only online adaptation may update during heldout testing")
        best[ep] = new
        observed_counts[ep] += 1
        observed_reward[ep] += reward
    for row in completed:
        ep = Path(row["private_episode_path"]).resolve(strict=True)
        if (observed_counts[ep] != row["original_vitis_calls"]
                or row["best_feasible_latency_ns"] != best.get(ep)
                or not isclose(float(row["episode_return"]),
                               observed_reward[ep], rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("Recovery completed episode disagrees with HLS journal")
    if any(count != campaign["calls_per_episode"] for count in observed_counts.values()):
        raise ValueError("Recovery evaluation did not consume all pre-registered calls")
    return {
        "status": "INDEPENDENT_SEPARATE_EVALUATION_HLS_AUDIT_PASSED",
        "original_training_source_revision": campaign["training_source_revision"],
        "evaluation_source_revision": campaign["evaluation_source_revision"],
        "original_training_episodes_complete": 100,
        "original_training_cases_not_rerun": 1000,
        "new_evaluation_cases_replayed": len(records),
        "evaluation_episodes": len(completed),
        "original_cases_modified": 0,
        "vendor_invocations_during_audit": 0,
        "note": "This is a separately source-pinned heldout continuation, not a completed original single-revision 1090-call campaign",
    }


def report(study, output):
    receipt = audit(study)
    root = Path(study).resolve(strict=True)
    out = Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Create-only evaluation report folder required")
    out.mkdir(parents=True)
    summary = _json(root / "summary.json")
    fields = ["method", "task_id", "task_sha", "native_calls", "feasible",
              "best_feasible_hls_ns", "episode_return", "online_updates_after_episode"]
    rows = [{
        "method": r["method"], "task_id": r["task_id"],
        "task_sha": r["task_sha"], "native_calls": r["original_vitis_calls"],
        "feasible": r["feasible"],
        "best_feasible_hls_ns": r["best_feasible_latency_ns"],
        "episode_return": r["episode_return"],
        "online_updates_after_episode": r["online_updates_after_episode"],
    } for r in summary["results"]]
    with (out / "heldout_comparison.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "audit.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    return {"status": "AUDITED_SEPARATE_EVALUATION_REPORT", "rows": rows,
            "audit": receipt, "output": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-study")
    parser.add_argument("--output")
    parser.add_argument("--study", help="Completed separate recovery evaluation for audit/report")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--max-total-vitis-calls", type=int)
    parser.add_argument("--vitis-executable", default="vitis_hls")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    if args.audit:
        if not args.study:
            parser.error("--audit needs --study")
        result = audit(args.study)
    elif args.report:
        if not args.study or not args.output:
            parser.error("--report needs --study and --output")
        result = report(args.study, args.output)
    else:
        if not args.training_study:
            parser.error("New recovery needs --training-study")
        if args.execute_native and not args.output:
            parser.error("--execute-native requires a new --output")
        result = run(
            training_study=args.training_study,
            output=args.output, execute_native=args.execute_native,
            max_total_vitis_calls=args.max_total_vitis_calls,
            vitis_executable=args.vitis_executable,
            timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
