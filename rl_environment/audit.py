"""Independent read-only audit of an original native online Post-HLS campaign.

Reconstructs all frozen task/network/numerical input cohorts from their
deterministic generator and checks every original case against compiler HLS
emission, CSim testbench/Tcl, source-attested Vitis receipts and measured
post-HLS eligibility. It NEVER starts Vitis and NEVER trusts stored
"qualified" flags, best latency or study rewards as ground truth.

Run on a *fresh review checkout* of this branch, not inside the original
source-pinned native worktree:
  python -m rl_environment.audit --study /path/to/study --seed 20261030

The original campaign SHA may be an older source revision: keep that pin
immutable and report both study and independent audit revisions.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import subprocess

from fpgai_research.experiments.architecture_knobs_v2 import (
    build_catalog, compile_candidate, legal_candidate,
)
from fpgai_research.toolchains.vitis_hls.package import make_vitis_tcl
from .cohort import make_cohort
from .post_hls_env import SCHEMA as CASE_SCHEMA, _receipt
from .study import plan

SCHEMA = "fpgai-research/independent-online-post-hls-audit/v1"


def _load(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing/symlinked original evidence: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _check(test, description):
    if not test:
        raise ValueError("Original native audit failed: " + description)


def _current_sha():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        check=True, text=True, timeout=10)
    return result.stdout.strip()


def _expected_episodes(cohort, validation_interval):
    """Reconstruct exactly the historical interleaved training/eval order."""
    for i, task in enumerate(cohort["train"]):
        yield ("train", task, "online_drl", True)
        if (i + 1) % validation_interval == 0 or i + 1 == len(cohort["train"]):
            tag = f"after_{i+1:05d}"
            for item in cohort["validation"]:
                yield ("validation_" + tag, item, "online_drl", False)
                yield ("validation_random_" + tag, item, "random", False)
    for item in cohort["test"]:
        yield ("test_rl", item, "online_drl", False)
        yield ("test_random", item, "random", False)


def audit(study, *, seed=20261030):
    root = Path(study)
    _check(root.is_dir() and not root.is_symlink(), "study directory unavailable")
    campaign = _load(root / "campaign.json")
    budget = campaign.get("budget", {})
    _check(campaign.get("schema") ==
           "fpgai-research/real-online-post-hls-campaign/v1", "study schema")
    _check(campaign.get("source_proxy_reward_used") is False, "proxy/native confusion")
    _check(campaign.get("no_post_vivado_execution") is True, "unexpected Vivado scope")
    if campaign.get("cohort_seed") is not None:
        _check(campaign["cohort_seed"] == seed, "cohort seed mismatch")
    study_sha = campaign.get("source_revision")
    _check(isinstance(study_sha, str) and len(study_sha) == 40 and
           all(c in "0123456789abcdef" for c in study_sha), "source pin")
    counts = (
        budget.get("training_networks"),
        budget.get("heldout_validation_networks"),
        budget.get("heldout_test_networks"),
        budget.get("vitis_calls_per_episode_including_baseline"),
        budget.get("frozen_validation_rounds"),
    )
    _check(all(type(x) is int and x > 0 for x in counts), "invalid campaign budget")
    training, validation, testing, cap, rounds = counts
    interval = (training + rounds - 1) // rounds
    # Historical study manifest omits validate-every. Determine the actual
    # checkpoint numbers from the original learning_curve.jsonl instead of
    # inventing a training interval.
    points = [_load_json_line(line) for line in
              (root / "learning_curve.jsonl").read_text().splitlines() if line.strip()]
    ends = [p["training_networks_completed"] for p in points]
    _check(ends and ends[-1] == training and ends == sorted(set(ends)),
           "invalid validation checkpoint order")
    # The historical runner validates at fixed intervals, with a possible
    # final partial interval. The first checkpoint is the real interval.
    interval = ends[0]
    expected_ends = [min(x, training) for x in
                     range(interval, training + interval, interval)]
    _check(ends == expected_ends, "validation checkpoints mismatch runner")
    declared = plan(training=training, validation=validation, test=testing,
                    calls_per_network=cap, validate_every=interval)
    for field in ("maximum_original_vitis_invocations",
                  "planned_training_calls", "planned_validation_calls",
                  "planned_final_test_and_random_calls",
                  "frozen_validation_rounds"):
        _check(declared[field] == budget.get(field), "budget mismatch: " + field)

    tasks = make_cohort(
        training=training, validation=validation, test=testing, seed=seed,
        max_vitis_calls=cap, part=campaign.get("part"))
    original_cohorts = _load(root / "cohort_hashes.json")
    _check(original_cohorts == {
        split: [t.public_descriptor() for t in tasks[split]]
        for split in ("train", "validation", "test")
    }, "model/sample/constraint cohort hash mismatch (check --seed)")

    old_steps_path = root / "native_steps.jsonl"
    _check(old_steps_path.is_file() and not old_steps_path.is_symlink(),
           "native transition log missing")
    saved_steps = [_load_json_line(line) for line in
                   old_steps_path.read_text().splitlines() if line.strip()]
    summary = _load(root / "summary.json")
    _check(summary.get("status") == "COMPLETED_WITH_ORIGINAL_VITIS_ATTEMPTS",
           "incomplete native campaign")

    checked = []
    counts_by_partition = {"train": 0, "validation": 0, "test": 0}
    paired = {}
    step_index = 0
    total_seconds = 0.0
    qualified_steps = 0
    numeric_failures = 0
    missing_metrics = 0
    qualified_baselines = 0
    for subdir, task, method, training_episode in _expected_episodes(tasks, interval):
        directory = root / "episodes" / subdir / task.task_id
        _check(directory.is_dir() and not directory.is_symlink(),
               "missing episode " + str(directory))
        meta = _load(directory / "task.json")
        _check(meta.get("schema") == CASE_SCHEMA and
               meta.get("task") == task.public_descriptor(),
               "episode/task immutable identity")
        catalog = build_catalog(task.model)
        used = set()
        previous_best = None
        first_qualified = False
        row = {"task_id": task.task_id, "subdir": subdir, "method": method,
               "partition": task.partition, "qualified_cases": 0,
               "best_hls_worst_case_cycles": None, "native_seconds": 0.0}
        observed_cases = [p.name for p in directory.iterdir()
                          if p.name.startswith("case_")]
        _check(sorted(observed_cases) == [
            f"case_{i:04d}" for i in range(cap)], "native case count/sequence")
        for i in range(cap):
            case = directory / f"case_{i:04d}"
            _check(case.is_dir() and not case.is_symlink(), "unsafe original case")
            action = _load(case / "action.json")
            choices = tuple(tuple(items) for items in action.get("choices", []))
            _check(len(choices) == len(task.model.operators) and
                   choices not in used and
                   legal_candidate(task.model, catalog, choices),
                   "duplicate/illegal original candidate")
            used.add(choices)
            emitted, _, _ = compile_candidate(task.model, catalog, choices)
            _check(_sha(case / "model.cpp") == emitted.manifest["cpp_sha256"],
                   "original HLS source mismatch vs real compiler")
            _check((case / "run_hls.tcl").read_text() ==
                   make_vitis_tcl(task.part, task.clock_ns),
                   "original HLS clock/tool script differs")
            result = _receipt(case, task)
            _check(result["vitis_invocation_sha256"] is not None,
                   "missing original Vitis invocation")
            recorded = _load(case / "online_step.json")
            _check(recorded.get("task_sha") == task.task_sha and
                   recorded.get("candidate") == action.get("choices") and
                   recorded.get("native_calls_used") == i+1 and
                   recorded.get("status") == "AUDITED_POST_HLS_STEP",
                   "original native step identity")
            for key in ("qualified", "hls_worst_case_cycles",
                        "vitis_invocation_sha256", "hls_report_sha256",
                        "hls_estimated_resources", "csim_max_absolute_error"):
                _check(recorded.get("result", {}).get(key) == result.get(key),
                       "native result re-audit mismatch: " + key)
            _check(result["vendor_seconds"] is not None,
                   "missing real process execution cost")
            total_seconds += result["vendor_seconds"]
            row["native_seconds"] += result["vendor_seconds"]
            if result["qualified"]:
                qualified_steps += 1
                row["qualified_cases"] += 1
                first_qualified = first_qualified or i == 0
                new_best = (result["hls_worst_case_cycles"] if previous_best is None
                            else min(previous_best, result["hls_worst_case_cycles"]))
                improvement = (max(0.0, (previous_best - new_best)/previous_best)
                               if previous_best is not None else 0.0)
                expected_reward = improvement - 0.005
                previous_best = new_best
            else:
                numeric = result["csim_max_absolute_error"]
                if numeric is not None and numeric > task.max_absolute_error:
                    numeric_failures += 1
                if result["hls_worst_case_cycles"] is None:
                    missing_metrics += 1
                expected_reward = -0.05
            _check(isclose(recorded.get("reward"), expected_reward,
                           abs_tol=1e-10, rel_tol=1e-10),
                   "stored reward does not match independently reaudited native")
            _check(recorded.get("best_cycles") == previous_best,
                   "stored best native latency differs")
            _check(step_index < len(saved_steps), "native_steps.jsonl truncated")
            log = saved_steps[step_index]
            _check(
                log.get("task_id") == task.task_id and
                log.get("task_sha256") == task.task_sha and
                log.get("method") == method and
                log.get("native_call") == i+1 and
                log.get("best_hls_worst_case_cycles") == previous_best and
                log.get("measured_hls_cycles") == result["hls_worst_case_cycles"] and
                log.get("native_receipt_sha256") ==
                    result["vitis_invocation_sha256"] and
                log.get("source_proxy_reward") is False,
                "transitions do not match original cases")
            _check(isclose(float(log.get("reward")), expected_reward,
                           abs_tol=1e-10, rel_tol=1e-10),
                   "online learning log reward mismatch")
            step_index += 1
        if first_qualified:
            qualified_baselines += 1
        row["baseline_qualified"] = first_qualified
        row["best_hls_worst_case_cycles"] = previous_best
        counts_by_partition[task.partition] += 1
        checked.append(row)
        if task.partition == "test":
            paired.setdefault(task.task_id, {})[method] = previous_best

    _check(step_index == len(saved_steps) ==
           budget["maximum_original_vitis_invocations"],
           "expected original tool-call denominator mismatch")
    _check(summary.get("original_vitis_calls_audited") == step_index,
           "summary original tool calls mismatch")
    _check(summary.get("training_networks") == training and
           summary.get("validation_networks") == validation and
           summary.get("test_networks") == testing,
           "summary network counts mismatch")
    _check(all(set(value) == {"online_drl", "random"} for value in paired.values()) and
           len(paired) == testing, "missing matched heldout random/DRL pairs")
    _check(all(p["no_validation_gradient_updates"] is True for p in points),
           "validation must not train")
    _check(sum(
        1 for row in saved_steps
        if row["partition"] == "train" and row["online_gradient_loss"] is not None
    ) == summary.get("online_gradient_updates"), "online update count mismatched")
    return {
        "schema": SCHEMA,
        "original_study_source_sha256": study_sha,
        "independent_audit_source_sha256": _current_sha(),
        "original_study_summary_sha256": _sha(root / "summary.json"),
        "cohort_seed": seed,
        "tool_invocation_count": step_index,
        "source_reemitted_and_native_cases_reaudited": step_index,
        "qualified_post_hls_cases": qualified_steps,
        "nonqualified_post_hls_cases": step_index-qualified_steps,
        "qualified_baseline_episodes": qualified_baselines,
        "episodes_by_partition": counts_by_partition,
        "actual_recorded_vendor_seconds": round(total_seconds, 6),
        "numeric_error_exceedance_cases": numeric_failures,
        "unqualified_or_missing_hls_cycles": missing_metrics,
        "online_policy_gradient_updates": summary["online_gradient_updates"],
        "heldout_network_pairs": paired,
        "hardware_fmax_verified": False,
        "vendor_programs_invoked_by_audit": 0,
        "original_evidence_files_modified": False,
    }


def _load_json_line(text):
    return json.loads(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.study, seed=args.seed), sort_keys=True,
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
