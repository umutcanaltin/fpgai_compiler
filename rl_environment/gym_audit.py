"""Independent source-reemitting, read-only audit of the NEW clock-aware Gym study.

No Vitis/Vivado invocations, no writes to original native cases.
Synthetic fixtures do not constitute native evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path

from fpgai_research.experiments.architecture_knobs_v2 import (
    build_catalog, compile_candidate,
)
from .cohort import make_cohort
from .clock_actions import (
    CLOCK_ACTION_MODE, CLOCK_MIN_NS, CLOCK_MAX_NS, CLOCK_RESOLUTION_NS,
    validate_clock_action,
)
from .post_hls_env import _receipt
from .reward import (latency_only_reward, beginner_real_hls_latency_reward,
                     curriculum_real_hls_cycle_reward)
from .research_metadata import experiment_metadata, summarize_trial
from .study import _source_revision


def _require(condition, message):
    if not condition:
        raise ValueError("Independent Gym audit: " + message)


def audit(directory):
    root = Path(directory).resolve(strict=True)
    campaign = json.loads((root / "campaign.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    _require(campaign["schema"] == "fpgai-gym-native-plan/v1", "unexpected contract")
    _require(campaign.get("requested_clock_is_policy_output") is True
             and campaign.get("fixed_clock_task_input") is False
             and campaign.get("clock_action_is_discrete") is False
             and campaign.get("clock_action_mode") == CLOCK_ACTION_MODE
             and campaign.get("clock_action_bounds_ns")
                 == [CLOCK_MIN_NS, CLOCK_MAX_NS]
             and campaign.get("clock_action_resolution_ns") ==
                 (.01 if campaign.get("curriculum_stage") is not None
                  else CLOCK_RESOLUTION_NS),
             "invalid source-defined floating clock action contract")
    fixed = campaign.get("fixed_clock_ablation_ns")
    frozen_numeric = campaign.get("freeze_numeric_format", False)
    beginner = campaign.get("beginner_pilot", False)
    from .agents import available_agents
    from .experiments import available_experiments
    algorithm = campaign.get("agent_algorithm", "graph_actor_critic")
    experiment_id = campaign.get("experiment_id")
    _require(algorithm in available_agents(), "registered algorithm unavailable at source")
    _require(experiment_id is None or experiment_id in available_experiments(),
             "registered research experiment unavailable at source")
    from .curriculum import get_stage, stage_contract, stage_validation
    stage = get_stage(campaign.get("curriculum_stage"))
    _require(campaign.get("curriculum_contract") ==
             (stage_contract(stage) if stage else None),
             "curriculum stage source contract drift")
    if stage:
        stage_validation(stage, fixed_clock=fixed, frozen_numeric=frozen_numeric,
                         beginner=beginner)
        mode = campaign.get("curriculum_mode")
        _require(mode in ("smoke", "full"), "missing curriculum experiment mode")
        if mode == "full":
            _require(campaign["training_networks_per_seed"] >= 100
                     and campaign["calls_per_episode_max"] == 10
                     and campaign.get("training_episode_minimum") == 100
                     and campaign.get("required_original_train_calls_per_episode") == 10,
                     "full curriculum requires >=100 real training episodes x10 HLS")
        else:
            _require(campaign["training_networks_per_seed"] == 2
                     and campaign["calls_per_episode_max"] == 2
                     and campaign["validation_networks_per_seed"] == 1
                     and campaign["test_networks_per_seed"] == 1,
                     "smoke cannot be counted as full 100-episode learning experiment")
        _require(campaign.get("clock_report_resolution_ns") == .01,
                 "curriculum original-Vitis report clock resolution changed")
    _require(type(beginner) is bool, "invalid beginner cohort flag")
    if beginner:
        _require(fixed == 32.0 and frozen_numeric and
                 campaign.get("beginner_pilot_schema") ==
                 "single_matmul_four_knobs_v1" and
                 campaign.get("searchable_architecture_fields") ==
                 ["parallelism", "pes", "simd", "serialization"],
                 "beginner study scientific action/clock contract changed")
    _require(type(frozen_numeric) is bool, "invalid frozen numeric configuration")
    _require(campaign.get("fixed_numeric_precision_bits") ==
             (16 if frozen_numeric else None) and
             campaign.get("fixed_numeric_integer_bits") ==
             (6 if frozen_numeric else None),
             "unbound fixed numeric format metadata")
    if fixed is None:
        _require(campaign.get("clock_action_domain_origin")
                 == "learned_bounded_float",
                 "floating-clock experiment wrongly treated as fixed-clock input")
    else:
        _require(campaign.get("clock_action_domain_origin")
                 == "explicit_fixed_clock_ablation",
                 "fixed-clock arm not explicitly registered")
    _require(campaign["source_revision"] == _source_revision(), "source SHA drift")
    paper = campaign.get("paper_evidence")
    _require(isinstance(paper, dict)
             and isinstance(paper.get("software_environment"), dict),
             "original experiment software environment is missing")
    expected_paper = experiment_metadata(
        source_revision=campaign["source_revision"],
        use_message_passing=campaign["message_passing_enabled"],
        beginner_pilot=beginner,
        curriculum_stage=stage.name if stage else None,
        agent_algorithm=algorithm, experiment_id=experiment_id)
    # The recorded Python/torch/host environment is ORIGINAL provenance,
    # not something to replace with the auditor's possibly newer machine.
    expected_paper["software_environment"] = paper["software_environment"]
    _require(paper == expected_paper,
             "reward formula, GNN hyperparameters or source file digest drift")
    _require(summary["source_revision"] == campaign["source_revision"], "summary source drift")
    _require(summary["status"] == "SOURCE_ATTRIBUTED_CASES_RECORDED_AWAIT_INDEPENDENT_REPLAY_AUDIT",
             "native study incomplete")
    records = [json.loads(line) for line in
               (root / "native_steps.jsonl").read_text().splitlines()]
    _require(records, "no original Vitis cases")
    _require(len(records) == summary["original_vitis_calls_recorded"],
             "missing or extra step rows")
    _require(len(records) <= campaign["maximum_original_vitis_invocations"],
             "exceeded authorized vendor budget")
    _require(campaign["calls_per_episode_max"] <= 30
             and campaign["hard_cap_vitis_compiles_per_episode"] == 30,
             "Gym episode exceeds hard 30-Vitis-attempt limit")
    _require(campaign.get("wall_clock_limit_seconds") is None
             and campaign.get("estimates_are_not_stopping_conditions") is True,
             "unexpected campaign wall-clock stop or time-estimate contract")
    _require(campaign["max_total_vitis_calls_authorized"] >=
             campaign["maximum_original_vitis_invocations"],
             "unauthorized aggregate Vitis-call budget")
    completed = [json.loads(line) for line in
                 (root / "completed_episodes.jsonl").read_text().splitlines()]
    _require(completed == summary["results"],
             "append-only completed-episode progress does not match final summary")
    _require(len(completed) == campaign["total_episodes_planned"],
             "missing completed train/validation/test/control episodes")
    if stage and campaign["curriculum_mode"] == "full":
        train_rows = [row for row in completed if row["method"] == "online_train"]
        _require(
            len(train_rows) >= 100 * len(campaign["training_seeds"]) and
            all(row["original_vitis_calls"] == 10 for row in train_rows),
            "incomplete original 100x10 full-stage training; not proof of learning/non-learning")
    _require(sum(row["original_vitis_calls"] for row in completed) == len(records),
             "completed-episode call accounting differs from original HLS steps")
    tasks = {}
    for seed in campaign["training_seeds"]:
        cohort = make_cohort(
            training=campaign["training_networks_per_seed"],
            validation=campaign["validation_networks_per_seed"],
            test=campaign["test_networks_per_seed"],
            max_vitis_calls=campaign["calls_per_episode_max"], seed=seed,
            clock_ns=None, fixed_clock_ablation_ns=fixed,
            freeze_numeric_format=frozen_numeric,
            beginner_pilot=beginner,
            curriculum_stage=stage.name if stage else None)
        for split in cohort:
            for task in cohort[split]:
                tasks[(seed, task.task_id)] = task
    seen = set()
    best_by_episode = {}
    count_by_episode = {}
    reward_sum_by_episode = {}
    last_update_by_arm = {}
    for row in records:
        ep = Path(row["private_episode_path"]).resolve(strict=True)
        _require(ep.is_relative_to(root), "native case escaped study root")
        match = ep.relative_to(root).parts
        _require(len(match) == 4 and match[0].startswith("seed_")
                 and match[-1] == "episode_000000",
                 "unexpected native episode layout")
        seed = int(match[0][5:])
        task = tasks.get((seed, row["task_id"]))
        _require(task is not None and match[2] == task.task_id,
                 "unknown experiment task")
        _require(row["task_sha"] == task.task_sha, "task fingerprint drift")
        idx = count_by_episode.get(ep, 0)
        _require(idx < task.max_vitis_calls and idx == row["native_call_index"],
                 "out-of-order or extra native call")
        case = ep / f"case_{idx:04d}"
        _require(case.is_dir() and not case.is_symlink(), "missing immutable case")
        action = json.loads((case / "action.json").read_text())
        candidate = tuple(tuple(v) for v in action["choices"])
        _require(validate_clock_action(task, action["requested_clock_ns"])
                 == row["clock_ns"], "noncanonical or unbound floating clock")
        catalog = build_catalog(task.model)
        if frozen_numeric:
            _require(all(row[:2] == catalog.baseline[i][:2]
                         for i, row in enumerate(candidate)),
                     "source native action changed fixed precision/integer format")
        if stage:
            from fpgai_research.experiments.architecture_knobs_v2 import FIELDS
            _require(all(all(row[j] == catalog.baseline[i][j]
                             for j, field in enumerate(FIELDS)
                             if field not in stage.knob_fields)
                         for i, row in enumerate(candidate)),
                     "source native case mutated curriculum-locked knob")
        if beginner:
            _require(len(task.model.operators) == 1 and
                     task.model.operators[0].kind == "MatMul",
                     "beginner native model is not one MatMul")
            _require(all(all(row[j] == catalog.baseline[i][j]
                             for j in (0, 1, 6, 7))
                         for i, row in enumerate(candidate)),
                     "beginner source-native case mutated locked knob")
        emitted, _, _ = compile_candidate(task.model, catalog, candidate)
        _require(sha256((case / "model.cpp").read_bytes()).hexdigest()
                 == emitted.manifest["cpp_sha256"],
                 "independent source regeneration mismatch")
        actual = _receipt(case, task)
        expected = json.loads((case / "online_step.json").read_text())
        _require(expected["result"] == actual, "native case result changed")
        _require(row["vitis_invocation_sha256"] == actual["vitis_invocation_sha256"]
                 and row["hls_report_sha256"] == actual["hls_report_sha256"]
                 and row["clock_ns"] == actual["requested_clock_ns"],
                 "native invocation/clock/report mismatch")
        _require(row["hls_estimated_latency_ns"] == actual["hls_estimated_latency_ns"]
                 and row["hls_cycles"] == actual["hls_worst_case_cycles"]
                 and row["qualified"] == actual["qualified"],
                 "HLS numeric or feasibility drift")
        _require(
            row["hls_clock_uncertainty_ns"] == actual["hls_clock_uncertainty_ns"]
            and row["hls_effective_scheduling_budget_ns"]
                == actual["hls_effective_scheduling_budget_ns"]
            and row["hls_scheduling_margin_ns"] == actual["hls_scheduling_margin_ns"]
            and row["hls_timing_feasible"] == actual["hls_timing_feasible"],
            "HLS uncertainty-adjusted effective timing budget or feasibility drift")
        old = best_by_episode.get(ep)
        new = old
        if actual["qualified"]:
            latency = actual["hls_estimated_latency_ns"]
            new = latency if old is None else min(old, latency)
        if stage is not None:
            reward = curriculum_real_hls_cycle_reward(
                qualified=actual["qualified"],
                hls_cycles=actual["hls_worst_case_cycles"],
                model=task.model,
                requested_clock_ns=row["clock_ns"])
            _require(expected.get("reward_scope") ==
                     "CURRICULUM_REAL_HLS_NS_EFFICIENCY_PER_FEASIBLE_CANDIDATE",
                     "curriculum reward scope changed")
        elif beginner:
            reward = beginner_real_hls_latency_reward(
                qualified=actual["qualified"],
                hls_cycles=actual["hls_worst_case_cycles"],
                input_width=task.model.tensor(task.model.inputs[0]).shape[-1],
                output_width=task.model.tensor(task.model.outputs[0]).shape[-1])
            _require(expected.get("reward_scope") ==
                     "BEGINNER_REAL_HLS_CYCLE_EFFICIENCY_PER_FEASIBLE_CANDIDATE",
                     "incorrect original beginner reward scope")
        else:
            reward = latency_only_reward(old, new, qualified=actual["qualified"])
        _require(isclose(float(row["reward"]), reward, rel_tol=1e-11, abs_tol=1e-11)
                 and isclose(float(expected["reward"]), reward, rel_tol=1e-11, abs_tol=1e-11),
                 "HLS-ns reward independently disagrees")
        _require(row["best_feasible_latency_ns"] == new
                 and expected["best_hls_estimated_latency_ns"] == new,
                 "best feasible HLS ns drift")
        _require(row.get("qualification_gates_failed") ==
                 actual["qualification_gates_failed"] and
                 row.get("raw_hls_worst_case_cycles") ==
                 actual["raw_hls_worst_case_cycles"] and
                 row.get("csim_passed") == actual["csim_passed"] and
                 row.get("original_hls_timing_parse_issue") ==
                 actual["original_hls_timing_parse_issue"],
                 "per-stage original HLS rejection diagnosis drift")
        metrics = row.get("training_metrics")
        is_training = row["method"] in ("online_train", "online_adapt")
        _require((metrics is not None) == is_training,
                 "missing or unexpected online policy diagnostics")
        if is_training:
            _require(metrics.get("schema") == "fpgai-gym-policy-update-diagnostics/v1",
                     "unexpected policy diagnostic schema")
            names = ("loss_total", "loss_actor", "loss_critic",
                     "loss_entropy_term", "policy_entropy", "log_probability",
                     "value_prediction", "discounted_td_target", "advantage",
                     "gradient_l2_before_clip")
            _require(all(type(metrics.get(k)) in (float, int)
                         and isfinite(metrics[k]) for k in names),
                     "invalid online update scalar")
            _require(metrics.get("native_feasibility_label") is actual["qualified"],
                     "supervised feasibility label changed from original HLS")
            prob = metrics.get("feasibility_prediction_before_hls")
            brier = metrics.get("feasibility_brier_score")
            _require(type(prob) in (float, int) and isfinite(prob) and
                     0. <= prob <= 1. and type(brier) in (float, int) and
                     isfinite(brier) and isclose(
                         brier, (prob - float(actual["qualified"])) ** 2,
                         rel_tol=1e-9, abs_tol=1e-9),
                     "original-HLS online feasibility prediction/Brier drift")
            _require(type(metrics.get("loss_auxiliary_feasibility")) in
                     (float, int) and isfinite(metrics["loss_auxiliary_feasibility"])
                     and metrics["loss_auxiliary_feasibility"] >= 0,
                     "invalid real-HLS feasibility auxiliary loss")
            _require(metrics.get("auxiliary_feasibility_weight") == .2 and
                     metrics.get("auxiliary_positive_label_weight") == 8.,
                     "auxiliary supervision setup drift")
            _require(isclose(metrics["loss_total"],
                             metrics["loss_actor"] + metrics["loss_critic"]
                             + metrics["loss_entropy_term"] +
                             .2 * metrics["loss_auxiliary_feasibility"],
                             rel_tol=1e-5, abs_tol=1e-6),
                     "online loss components disagree")
            _require(isclose(metrics["advantage"],
                             metrics["discounted_td_target"] -
                             metrics["value_prediction"], rel_tol=1e-5,
                             abs_tol=1e-6),
                     "policy advantage does not match target/value")
            arm = (seed, "online_train" if row["method"] == "online_train"
                   else "online_adapt")
            # Adaptive test starts from the *trained* checkpoint and its
            # optimizer/update counter, not from update zero.
            initial = (last_update_by_arm.get((seed, "online_train"), 0)
                       if row["method"] == "online_adapt" else 0)
            last = last_update_by_arm.get(arm, initial)
            _require(metrics["training_updates"] == last + 1,
                     "nonmonotonic policy update index within training arm")
            last_update_by_arm[arm] = last + 1
        _require(row.get("trial_evidence") ==
                 summarize_trial(expected, task=task,
                                 training_metrics=metrics),
                 "published HLS source/clock/numeric/resource trial evidence drift")
        reward_sum_by_episode[ep] = reward_sum_by_episode.get(ep, 0.) + reward
        _require(isclose(row.get("cumulative_episode_reward"),
                         reward_sum_by_episode[ep], rel_tol=1e-10, abs_tol=1e-10),
                 "cumulative episode reward drift")
        best_by_episode[ep] = new
        count_by_episode[ep] = idx + 1
        _require((ep, idx) not in seen, "duplicated original case")
        seen.add((ep, idx))
    for ep, count in count_by_episode.items():
        _require(len(list(ep.glob("case_*"))) == count,
                 "unreported native attempt or case")
    completed_by_path = {}
    for row in completed:
        ep = Path(row["private_episode_path"]).resolve(strict=True)
        _require(ep.is_relative_to(root) and ep not in completed_by_path,
                 "duplicate or escaped completed-episode receipt")
        completed_by_path[ep] = row
        _require(row["original_vitis_calls"] == count_by_episode.get(ep, 0),
                 "episode journal differs from original case count")
        _require(isclose(row["episode_return"],
                         reward_sum_by_episode.get(ep, 0.),
                         rel_tol=1e-10, abs_tol=1e-10),
                 "episode journal reward differs from original HLS steps")
        _require(row["best_feasible_latency_ns"] ==
                 best_by_episode.get(ep), "episode journal best latency drift")
    return {
        "status": "INDEPENDENT_GYM_HLS_AUDIT_PASSED",
        "original_cases_recomputed": len(seen),
        "episodes_checked": len(count_by_episode),
        "HLS_estimated_only": True,
        "vendor_invocations_during_audit": 0,
        "original_files_modified": 0,
        "source_revision": campaign["source_revision"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    a = p.parse_args(argv)
    print(json.dumps(audit(a.study), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
