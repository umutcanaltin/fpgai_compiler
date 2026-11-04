"""Source-pinned, foreground online Gym/GNN HLS study.

No native program is launched without --execute-native and a complete explicit
budget. Every HLS case is unique, immutable and independently re-auditable.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .cohort import make_cohort
from .tasks import MAX_GYM_VITIS_CALLS_PER_EPISODE
from .clock_actions import (CLOCK_MIN_NS, CLOCK_MAX_NS, CLOCK_RESOLUTION_NS,
                            CLOCK_ACTION_MODE, normalize_clock_period_ns)
from .baselines import BaselineChooser, METHODS
from .agents import create_agent, available_agents
from .environments import create_environment
from .study import _source_revision, _write_json_new, _append_jsonl
from .research_metadata import experiment_metadata, summarize_trial


def plan(*, training, validation, test, calls_per_network, clocks_ns=None,
         seeds=(101,), use_message_passing=True,
         estimated_seconds_per_compile=None,
         freeze_numeric_format=False, beginner_pilot=False,
         curriculum_stage=None, curriculum_mode="full",
         agent_algorithm="graph_actor_critic", experiment_id=None):
    if experiment_id is not None:
        from .experiments import available_experiments
        if experiment_id not in available_experiments():
            raise ValueError("Experiment identifier is not registered in source")
    if agent_algorithm not in available_agents():
        raise ValueError("Agent algorithm is not registered in this source revision")
    if type(freeze_numeric_format) is not bool:
        raise ValueError("freeze_numeric_format must be explicit boolean")
    if type(beginner_pilot) is not bool:
        raise ValueError("beginner_pilot must be explicit boolean")
    from .curriculum import get_stage, stage_validation, stage_contract
    stage = get_stage(curriculum_stage)
    if stage is not None:
        if curriculum_mode not in ("full", "smoke"):
            raise ValueError("Curriculum mode must be full or explicitly smoke")
        if curriculum_mode == "full" and (training < 100 or calls_per_network != 10):
            raise ValueError("Full curriculum requires >=100 training episodes and exactly 10 original HLS calls per episode")
        if curriculum_mode == "smoke" and (training != 2 or validation != 1
                                            or test != 1 or calls_per_network != 2):
            raise ValueError("Curriculum smoke is exactly 2 training, 1 validation, 1 test task, 2 calls each")
        stage_validation(stage, fixed_clock=clocks_ns[0] if clocks_ns else None,
                         frozen_numeric=freeze_numeric_format,
                         beginner=beginner_pilot)
    if beginner_pilot and (clocks_ns != (32.,) or not freeze_numeric_format):
        raise ValueError("Beginner pilot requires fixed 32.000 ns and --freeze-numeric-format")
    counts = (training, validation, test, calls_per_network)
    if any(type(v) is not int or v <= 0 for v in counts):
        raise ValueError("Strictly positive cohort and native-call budget required")
    if calls_per_network > MAX_GYM_VITIS_CALLS_PER_EPISODE:
        raise ValueError("Maximum 30 Vitis HLS compile attempts per Gym episode")
    if estimated_seconds_per_compile is not None and (
            type(estimated_seconds_per_compile) not in (int, float)
            or not isfinite(estimated_seconds_per_compile)
            or estimated_seconds_per_compile <= 0):
        raise ValueError("Estimated seconds per compile must be finite and positive")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Independent unique training seeds required")
    # Standard study learns a real-valued requested clock; the only clock
    # override allowed is an intentionally fixed one-period ablation.
    if clocks_ns is not None and (
            not isinstance(clocks_ns, tuple) or len(clocks_ns) != 1):
        raise ValueError("Clock grids are prohibited: RL learns floating clocks")
    fixed_clock = (normalize_clock_period_ns(clocks_ns[0])
                   if clocks_ns is not None else None)
    # Each policy/control is a separate episode on the same frozen task.
    per_seed = {
        "train": training,
        "validation_frozen": validation,
        "test_frozen": test,
        "test_frozen_stochastic": test,
        "test_untrained_stochastic": test,
        "test_adaptive": test,
        **{f"test_{method}": test for method in METHODS},
    }
    episode_breakdown = {name: count * len(seeds)
                         for name, count in per_seed.items()}
    total_episodes = sum(episode_breakdown.values())
    # This is an *upper bound* on calls; real episodes may terminate early.
    total = total_episodes * calls_per_network
    estimate_hours = (round(
        total * float(estimated_seconds_per_compile) / 3600., 3)
        if estimated_seconds_per_compile is not None else None)
    return {
        "schema": "fpgai-gym-native-plan/v1",
        "training_networks_per_seed": training,
        "validation_networks_per_seed": validation,
        "test_networks_per_seed": test,
        "training_seeds": list(seeds),
        "agent_algorithm": agent_algorithm,
        "experiment_id": experiment_id,
        "calls_per_episode_max": calls_per_network,
        "hard_cap_vitis_compiles_per_episode": MAX_GYM_VITIS_CALLS_PER_EPISODE,
        "episode_counts_by_method": episode_breakdown,
        "total_episodes_planned": total_episodes,
        "estimated_seconds_per_compile_assumed": (
            float(estimated_seconds_per_compile)
            if estimated_seconds_per_compile is not None else None),
        "estimated_vitis_hours_at_max_calls": estimate_hours,
        "wall_clock_limit_seconds": None,
        "estimates_are_not_stopping_conditions": True,
        "clock_action_mode": CLOCK_ACTION_MODE,
        "clock_action_bounds_ns": [CLOCK_MIN_NS, CLOCK_MAX_NS],
        "clock_action_resolution_ns": .01 if stage is not None else CLOCK_RESOLUTION_NS,
        "clock_action_domain_origin": (
            "learned_bounded_float" if fixed_clock is None
            else "explicit_fixed_clock_ablation"),
        "fixed_clock_ablation_ns": fixed_clock,
        "freeze_numeric_format": freeze_numeric_format,
        "beginner_pilot": beginner_pilot,
        "curriculum_stage": curriculum_stage,
        "curriculum_mode": curriculum_mode if stage is not None else None,
        "training_episode_minimum": 100 if stage is not None and curriculum_mode == "full" else None,
        "required_original_train_calls_per_episode": 10 if stage is not None and curriculum_mode == "full" else None,
        "clock_report_resolution_ns": 0.01 if stage is not None else 0.001,
        "curriculum_contract": stage_contract(stage) if stage else None,
        "beginner_pilot_schema": (
            "single_matmul_four_knobs_v1" if beginner_pilot else None),
        "searchable_architecture_fields": (
            ["parallelism", "pes", "simd", "serialization"]
            if beginner_pilot else None),
        "fixed_numeric_precision_bits": 16 if freeze_numeric_format else None,
        "fixed_numeric_integer_bits": 6 if freeze_numeric_format else None,
        "clock_action_is_discrete": False,
        "requested_clock_is_policy_output": True,
        "fixed_clock_task_input": False,
        "clock_ablation_mode": "fixed_clock" if fixed_clock is not None else "joint_clock",
        "message_passing_enabled": bool(use_message_passing),
        "maximum_original_vitis_invocations": total,
        "baseline_methods": list(METHODS),
        "frozen_and_adaptive_test_from_same_checkpoint": True,
        "proxy_hls_rewards": False,
        "post_route_evidence": False,
    }


def _choose_unused_hls_action(*, agent, obs, env, method, action, tape):
    """Exclude source-recorded architecture/clock pairs BEFORE a vendor call.

    A deterministic policy's exact Beta mean can repeat indefinitely. For
    frozen deterministic evaluation, the nearest untried 1-ps clock is a
    deterministic duplicate-resolution convention, NOT an extra reward,
    policy gradient, HLS estimate or new architecture optimization decision.
    Stochastic policies resample the real action distribution, retaining only
    the final action's gradient tape. Never silently re-run an original case.
    """
    from .clock_actions import normalize_clock_period_ns
    slot, raw_period = action
    decimals = (2 if getattr(getattr(env, "task", None), "curriculum_stage", None)
                in ("L5", "L6") else 3)
    ticks_per_ns = 10 ** decimals
    # Canonicalize for the ORIGINAL experiment clock encoding before comparing
    # prior original HLS cases, not just before invoking the native backend:
    # e.g. Python's 7.2010000000000005 equals canonical 7.201 at 1 ps.
    period = normalize_clock_period_ns(raw_period, decimals=decimals)
    candidate = env.offers[slot]
    tried = env.backend.tried_configs
    if (candidate, period) not in tried:
        return (slot, period), tape
    low, high = (float(x) for x in obs["clock_bounds_ns"])
    if method == "frozen":
        # Preserve the selected architecture and choose the closest
        # distinct legal requested clock in canonical millips resolution.
        center = int(round(period * ticks_per_ns))
        min_tick, max_tick = int(round(low * ticks_per_ns)), int(round(high * ticks_per_ns))
        for delta in range(1, max_tick - min_tick + 1):
            for tick in (center + delta, center - delta):
                if not min_tick <= tick <= max_tick:
                    continue
                alternate = normalize_clock_period_ns(tick / ticks_per_ns, decimals=decimals)
                if (candidate, alternate) not in tried:
                    return (slot, alternate), None
        # Only fixed-clock exhaustive exhaustion can reach this with 10 calls.
        raise ValueError("No unused clock for deterministic frozen candidate")
    if method in METHODS:
        # The reference policy is not a gradient-bearing distribution.
        # Distinctive nearest-period resolution prevents duplicate HLS cases.
        center = int(round(period * ticks_per_ns))
        min_tick, max_tick = int(round(low * ticks_per_ns)), int(round(high * ticks_per_ns))
        for delta in range(1, max_tick - min_tick + 1):
            for tick in (center + delta, center - delta):
                if not min_tick <= tick <= max_tick:
                    continue
                alternate = normalize_clock_period_ns(tick / ticks_per_ns, decimals=decimals)
                if (candidate, alternate) not in tried:
                    return (slot, alternate), None
        raise ValueError("No unused reference-clock candidate")
    for _ in range(64):
        next_action, next_tape = agent.choose(
            obs, training=method in ("online_train", "online_adapt"),
            sample=True)
        next_slot, raw_next_clock = next_action
        next_clock = normalize_clock_period_ns(raw_next_clock, decimals=decimals)
        if (env.offers[next_slot], next_clock) not in tried:
            return (next_slot, next_clock), next_tape
    raise ValueError("No fresh stochastic HLS action after bounded resampling")


def _episode(task, path, agent, *, method, seed, native,
             vitis_executable, timeout_seconds, log):
    env = create_environment(
        "graph_native_hls", task, path, execute_native=native,
        vitis_executable=vitis_executable,
        timeout_seconds=timeout_seconds)
    obs, _ = env.reset(seed=seed)
    # Separate deterministic baseline streams; the four reference algorithms
    # must not be four copies of one identical random fallback trajectory.
    baseline_seed = seed + {"random": 1009, "greedy": 2003,
                           "evolutionary": 3001, "parzen": 4003}.get(method, 0)
    chooser = BaselineChooser(method, seed=baseline_seed) if method in METHODS else None
    if method in ("frozen_stochastic", "untrained_stochastic", "online_adapt"):
        import torch
        torch.manual_seed(seed)
    steps = 0
    best = None
    episode_return = 0.0
    while np.any(obs["action_mask"]):
        if chooser is not None:
            action = (chooser.choose(obs), chooser.choose_clock(obs))
            tape = None
        else:
            action, tape = agent.choose(
                obs, training=method in ("online_train", "online_adapt"),
                sample=method in ("online_train", "online_adapt",
                                  "frozen_stochastic", "untrained_stochastic"))
        action, tape = _choose_unused_hls_action(
            agent=agent, obs=obs, env=env, method=method,
            action=action, tape=tape)
        slot, requested_clock_ns = action
        selected_features = obs["action_features"][slot].copy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        training_metrics = None
        result = info["source_attributed_result"]["result"]
        if method in ("online_train", "online_adapt"):
            training_metrics = agent.update(
                tape, reward, next_obs=next_obs,
                terminated=terminated or truncated, return_metrics=True,
                qualified=bool(result["qualified"]))
        episode_return += float(reward)
        if chooser is not None:
            chooser.observe(selected_features, result["hls_estimated_latency_ns"],
                            clock_ns=result["requested_clock_ns"])
        best = env.backend.best_latency_ns
        trial = summarize_trial(
            info["source_attributed_result"], task=task,
            training_metrics=training_metrics)
        log({
            "task_id": task.task_id, "task_sha": task.task_sha,
            "source": "original-vitis-hls", "method": method,
            "native_call_index": steps, "clock_ns": result["requested_clock_ns"],
            "qualified": result["qualified"],
            "hls_estimated_latency_ns": result["hls_estimated_latency_ns"],
            "hls_cycles": result["hls_worst_case_cycles"],
            "hls_estimated_clock_ns": result["hls_estimated_clock_ns"],
            "hls_clock_uncertainty_ns": result["hls_clock_uncertainty_ns"],
            "hls_effective_scheduling_budget_ns": result["hls_effective_scheduling_budget_ns"],
            "hls_scheduling_margin_ns": result["hls_scheduling_margin_ns"],
            "hls_timing_feasible": result["hls_timing_feasible"],
            "max_observed_hls_qualified_requested_mhz":
                env.backend.max_observed_hls_qualified_requested_mhz,
            "hls_report_sha256": result["hls_report_sha256"],
            "vitis_invocation_sha256": result["vitis_invocation_sha256"],
            "wall_seconds": result["vendor_seconds"], "reward": reward,
            "qualification_gates_failed": result["qualification_gates_failed"],
            "raw_hls_worst_case_cycles": result["raw_hls_worst_case_cycles"],
            "csim_passed": result["csim_passed"],
            "original_hls_timing_parse_issue": result[
                "original_hls_timing_parse_issue"],
            "best_feasible_latency_ns": best,
            "cumulative_episode_reward": episode_return,
            "training_metrics": training_metrics,
            "trial_evidence": trial,
            "private_episode_path": str(path / "episode_000000"),
        })
        steps += 1
        obs = next_obs
        if terminated or truncated:
            break
    return {
        "task_id": task.task_id, "task_sha": task.task_sha,
        "method": method,
        "original_vitis_calls": steps,
        "best_feasible_latency_ns": best,
        "feasible": best is not None,
        "episode_return": episode_return,
        "online_updates_after_episode": agent.updates,
        "private_episode_path": str(path / "episode_000000"),
        "task_partition": task.partition,
    }


def run(*, output, training=2, validation=1, test=1, calls_per_network=3,
        clocks_ns=None, estimated_seconds_per_compile=None,
        seeds=(101,), execute_native=False,
        use_message_passing=True,
        max_total_vitis_calls=None, vitis_executable="vitis_hls",
        timeout_seconds=7200, checkpoint_every_train_episodes=10,
        freeze_numeric_format=False, beginner_pilot=False,
        curriculum_stage=None, curriculum_mode="full",
        agent_algorithm="graph_actor_critic", experiment_id=None):
    budget = plan(training=training, validation=validation, test=test,
                  calls_per_network=calls_per_network, clocks_ns=clocks_ns,
                  seeds=seeds, use_message_passing=use_message_passing,
                  estimated_seconds_per_compile=estimated_seconds_per_compile,
                  freeze_numeric_format=freeze_numeric_format,
                  beginner_pilot=beginner_pilot,
                  curriculum_stage=curriculum_stage,
                  curriculum_mode=curriculum_mode,
                  agent_algorithm=agent_algorithm,
                  experiment_id=experiment_id)
    if (type(checkpoint_every_train_episodes) is not int
            or not 1 <= checkpoint_every_train_episodes <= 10000):
        raise ValueError("Checkpoint interval must be 1..10000 training episodes")
    if not execute_native:
        return {"status": "DRY_RUN_ZERO_VENDOR_CALLS",
                "checkpoint_every_train_episodes": checkpoint_every_train_episodes,
                **budget}
    total = budget["maximum_original_vitis_invocations"]
    if type(max_total_vitis_calls) is not int or max_total_vitis_calls < total:
        raise ValueError("Explicit full --max-total-vitis-calls budget is mandatory")
    if type(timeout_seconds) is not int or not 60 <= timeout_seconds <= 14400:
        raise ValueError("Invalid explicit native call timeout")
    # CPU test runs prepend src/ (pytest.ini), but an ordinary `python -m`
    # native invocation may resolve an outdated site-packages compiler.
    # This must fail BEFORE the create-only study directory is allocated.
    from .runtime_contract import verify_local_compiler_package
    verify_local_compiler_package()
    revision = _source_revision()
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Native study output must be create-only")
    dest.mkdir(parents=True)
    dest = dest.resolve(strict=True)
    _write_json_new(dest / "campaign.json", {
        **budget, "status": "IN_PROGRESS", "source_revision": revision,
        "original_vitis_executable": vitis_executable,
        "max_total_vitis_calls_authorized": max_total_vitis_calls,
        "native_tool_only_explicit_foreground": True,
        "checkpoint_every_train_episodes": checkpoint_every_train_episodes,
        "paper_evidence": experiment_metadata(
            source_revision=revision, use_message_passing=use_message_passing,
            beginner_pilot=budget["beginner_pilot"],
            curriculum_stage=budget["curriculum_stage"],
            agent_algorithm=agent_algorithm,
            experiment_id=experiment_id),
    })
    all_results = []
    executed = 0

    def record_episode(row):
        # Durable, append-only progress even if an hours-long experiment later
        # gets interrupted. Final summary is only written on full completion.
        all_results.append(row)
        _append_jsonl(dest / "completed_episodes.jsonl", row)

    def log(row):
        nonlocal executed
        if _source_revision() != revision:
            raise RuntimeError("Source changed during immutable native experiment")
        executed += 1
        if executed > max_total_vitis_calls:
            raise RuntimeError("Authorized Vitis budget exceeded")
        _append_jsonl(dest / "native_steps.jsonl", row)

    for seed in seeds:
        # The task contains no requested clock or list. The shared actor
        # selects a floating period every real native step.
        cohort = make_cohort(
            training=training, validation=validation, test=test,
            max_vitis_calls=calls_per_network, seed=seed,
            clock_ns=None,
            fixed_clock_ablation_ns=budget["fixed_clock_ablation_ns"],
            freeze_numeric_format=budget["freeze_numeric_format"],
            beginner_pilot=budget["beginner_pilot"],
            curriculum_stage=budget["curriculum_stage"])
        root = dest / f"seed_{seed}"
        _write_json_new(root / "tasks.json" if root.exists() else
                        _make_tasks_root(root), {
            split: [t.public_descriptor() for t in items]
            for split, items in cohort.items()})
        agent = create_agent(agent_algorithm, seed=seed,
                             use_message_passing=use_message_passing)
        for i, task in enumerate(cohort["train"]):
            training_episode = _episode(
                task, root / "train" / task.task_id, agent,
                method="online_train", seed=seed+i, native=True,
                vitis_executable=vitis_executable, timeout_seconds=timeout_seconds,
                log=log)
            record_episode(training_episode)
            if (budget["curriculum_mode"] == "full" and
                    training_episode["original_vitis_calls"] != 10):
                raise RuntimeError("Full curriculum must record exactly 10 original HLS trials per training episode; study incomplete")
            if (i + 1) % checkpoint_every_train_episodes == 0:
                agent.save_private(
                    root / "training_checkpoints" / f"after_{i+1:06d}_episodes",
                    {"source_revision": revision,
                     "training_episodes_completed": i+1,
                     "train_task_shas": [t.task_sha for t in cohort["train"][:i+1]],
                     "agent_algorithm": agent_algorithm})
        cp = root / "checkpoint"
        agent.save_private(cp, {
            "source_revision": revision,
            "train_task_shas": [t.task_sha for t in cohort["train"]],
            "agent_algorithm": agent_algorithm,
        })
        for i, task in enumerate(cohort["validation"]):
            record_episode(_episode(
                task, root / "validation_frozen" / task.task_id, agent,
                method="frozen", seed=seed+1000+i, native=True,
                vitis_executable=vitis_executable, timeout_seconds=timeout_seconds,
                log=log))
        adapted = create_agent(agent_algorithm, seed=seed,
                               use_message_passing=use_message_passing)
        adapted.load_private(cp)
        # A same-initialization, never-trained GNN is a critical causal
        # control: frozen trained vs frozen untrained isolates learning.
        never_trained = create_agent(
            agent_algorithm, seed=seed,
            use_message_passing=use_message_passing)
        for i, task in enumerate(cohort["test"]):
            record_episode(_episode(
                task, root / "test_untrained_stochastic" / task.task_id,
                never_trained, method="untrained_stochastic",
                seed=seed+2000+i, native=True,
                vitis_executable=vitis_executable,
                timeout_seconds=timeout_seconds, log=log))
            record_episode(_episode(
                task, root / "test_frozen" / task.task_id, agent,
                method="frozen", seed=seed+2000+i, native=True,
                vitis_executable=vitis_executable, timeout_seconds=timeout_seconds,
                log=log))
            record_episode(_episode(
                task, root / "test_frozen_stochastic" / task.task_id, agent,
                method="frozen_stochastic", seed=seed+2000+i, native=True,
                vitis_executable=vitis_executable, timeout_seconds=timeout_seconds,
                log=log))
            for baseline in METHODS:
                record_episode(_episode(
                    task, root / ("test_" + baseline) / task.task_id, agent,
                    method=baseline, seed=seed+2000+i, native=True,
                    vitis_executable=vitis_executable,
                    timeout_seconds=timeout_seconds, log=log))
            record_episode(_episode(
                task, root / "test_adaptive" / task.task_id, adapted,
                method="online_adapt", seed=seed+2000+i, native=True,
                vitis_executable=vitis_executable, timeout_seconds=timeout_seconds,
                log=log))
    _write_json_new(dest / "summary.json", {
        "schema": "fpgai-gym-original-hls-results/v1",
        "status": "SOURCE_ATTRIBUTED_CASES_RECORDED_AWAIT_INDEPENDENT_REPLAY_AUDIT",
        "source_revision": revision,
        "original_vitis_calls_recorded": executed,
        "planned_max_calls": total,
        "results": all_results,
        "adaptation_protocol": "sequential update across test tasks; later adaptive tasks are contaminated by prior test-task updates",
        "bayesian_tpe_claim": False,
        "no_physical_fmax_claim": True,
    })
    return {"status": "SOURCE_ATTRIBUTED_AWAIT_REPLAY_AUDIT",
            "original_vitis_calls": executed, "output": str(dest)}


def _make_tasks_root(root):
    root.mkdir(parents=True, exist_ok=False)
    return root / "tasks.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="build/gym_native_smoke_v1")
    p.add_argument("--train-episodes", "--train-networks",
                   dest="train_networks", type=int, default=2,
                   help="Number of online-training episodes per seed")
    p.add_argument("--validation-episodes", "--validation-networks",
                   dest="validation_networks", type=int, default=1)
    p.add_argument("--test-tasks", "--test-networks",
                   dest="test_networks", type=int, default=1,
                   help="Frozen tasks per seed; eight separate test/control episodes per task")
    p.add_argument("--max-vitis-compiles-per-episode", "--calls-per-network",
                   dest="calls_per_network", type=int, default=3,
                   help="Upper limit of original Vitis attempts per episode (1..30)")
    p.add_argument("--estimated-seconds-per-compile", type=float,
                   help="Forecast-only assumed average original Vitis seconds per call; "
                        "never a timeout or stopping criterion")
    p.add_argument("--fixed-clock-ablation-ns", type=float,
                   help="Only for an intentionally fixed-clock control experiment; "
                        "normal DRL runs choose clocks automatically")
    p.add_argument("--freeze-numeric-format", action="store_true",
                   help="Opt-in architecture-only pilot: hold model-global precision at 16 bits "
                        "and integer width at 6 bits; no precision or integer-width actions")
    p.add_argument("--beginner-pilot", action="store_true",
                   help="Opt-in easiest real HLS Gym: one MatMul per network, unseen heldout "
                        "dimensions, fixed 32 ns and 16/6 numeric format, four hardware knobs")
    p.add_argument("--curriculum-smoke", action="store_true",
                   help="Explicit 2-train/2-call source-audited smoke; never valid as a 100-episode learning result")
    p.add_argument("--curriculum-stage", choices=[f"L{i}" for i in range(7)],
                   help="Separate source-pinned L0-L6 real HLS benchmark stage; use "
                        "python -m rl_environment.gym_curriculum to apply its exact contract")
    p.add_argument("--agent-algorithm", choices=available_agents(),
                   default="graph_actor_critic",
                   help="Source-registered from-scratch DRL algorithm for all train/eval arms")
    p.add_argument("--seed", type=int, action="append")
    p.add_argument("--execute-native", action="store_true")
    p.add_argument("--no-message-passing", action="store_true")
    p.add_argument("--max-total-vitis-calls", type=int)
    p.add_argument("--vitis-executable", default="vitis_hls")
    p.add_argument("--timeout-seconds", type=int, default=7200,
                   help="Timeout of each individual Vitis subprocess; not a campaign wall-clock limit")
    p.add_argument("--checkpoint-every-train-episodes", type=int, default=10,
                   help="Create an immutable private GNN checkpoint every N completed training episodes")
    a = p.parse_args(argv)
    result = run(
        output=a.output, training=a.train_networks,
        validation=a.validation_networks, test=a.test_networks,
        calls_per_network=a.calls_per_network,
        estimated_seconds_per_compile=a.estimated_seconds_per_compile,
        clocks_ns=((a.fixed_clock_ablation_ns,)
                   if a.fixed_clock_ablation_ns is not None else None),
        seeds=tuple(a.seed or [101]), execute_native=a.execute_native,
        use_message_passing=not a.no_message_passing,
        max_total_vitis_calls=a.max_total_vitis_calls,
        vitis_executable=a.vitis_executable, timeout_seconds=a.timeout_seconds,
        checkpoint_every_train_episodes=a.checkpoint_every_train_episodes,
        freeze_numeric_format=a.freeze_numeric_format,
        beginner_pilot=a.beginner_pilot,
        curriculum_stage=a.curriculum_stage,
        curriculum_mode="smoke" if a.curriculum_smoke else "full",
        agent_algorithm=a.agent_algorithm)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
