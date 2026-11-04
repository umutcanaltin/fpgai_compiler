"""CLI for seven separate, source-pinned, REAL-Vitis Gym environments.

No synthetic latency, no pretrained policy and no hidden stage promotion.
Every stage is a fresh training cohort with separate output, independent
heldout arms, immutable original HLS cases and per-stage audit/report.
Run --list or --plan-all without any native tool invocation.
"""
from __future__ import annotations

import argparse
import json

from .curriculum import STAGES, get_stage, stage_contract
from .gym_study import plan, run
from .agents import available_agents
from .experiments import available_experiments, experiment_config


def config(name, *, train=100, validation=1, test=1, calls=10, seed=111):
    get_stage(name)
    return experiment_config(
        "curriculum_" + name.lower(), train=train, validation=validation,
        test=test, calls=calls, seed=seed)


def plans(*, train=100, validation=1, test=1, calls=10, seed=111):
    result = []
    for stage in STAGES:
        kw = config(stage.name, train=train, validation=validation,
                    test=test, calls=calls, seed=seed)
        p = plan(**kw)
        result.append({
            "stage": stage_contract(stage),
            "research_plan": p,
            "native_vendor_calls_executed": 0,
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-stages", action="store_true")
    group.add_argument("--list-experiments", action="store_true")
    group.add_argument("--plan-all", action="store_true")
    group.add_argument("--stage", choices=[x.name for x in STAGES])
    group.add_argument("--experiment", choices=available_experiments(),
                       help="Registered source-defined alternative protocol; "
                            "all source/audit and minimum training gates apply")
    parser.add_argument("--smoke", action="store_true",
                        help="Explicit 22-call 2x2 smoke, not a learning experiment")
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=1)
    parser.add_argument("--test-tasks", type=int, default=1)
    parser.add_argument("--max-vitis-compiles-per-episode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--agent-algorithm", choices=available_agents(),
                        default="graph_actor_critic",
                        help="Explicit source-registered from-scratch DRL algorithm")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--output", help="New, create-only stage output directory")
    parser.add_argument("--max-total-vitis-calls", type=int)
    parser.add_argument("--vitis-executable", default="vitis_hls")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--checkpoint-every-train-episodes", type=int, default=10)
    args = parser.parse_args(argv)
    if args.smoke and (args.list_stages or args.plan_all or args.list_experiments):
        parser.error("--smoke requires a specific --stage L0..L6")
    if args.list_stages:
        if args.execute_native:
            parser.error("--list-stages never invokes native tools")
        answer = [stage_contract(x) for x in STAGES]
    elif args.list_experiments:
        if args.execute_native:
            parser.error("--list-experiments never invokes native tools")
        answer = list(available_experiments())
    elif args.plan_all:
        if args.execute_native:
            parser.error("--plan-all never invokes native tools")
        answer = plans(
            train=args.train_episodes, validation=args.validation_episodes,
            test=args.test_tasks, calls=args.max_vitis_compiles_per_episode,
            seed=args.seed)
    else:
        if args.smoke and (args.train_episodes not in (2, 100)
                           or args.max_vitis_compiles_per_episode not in (2, 10)):
            parser.error("--smoke uses exactly 2 training episodes x 2 HLS calls; omit conflicting flags")
        train = 2 if args.smoke else args.train_episodes
        calls = 2 if args.smoke else args.max_vitis_compiles_per_episode
        kw = (experiment_config(
            args.experiment, train=train, validation=args.validation_episodes,
            test=args.test_tasks, calls=calls, seed=args.seed)
              if args.experiment else config(
                args.stage, train=train,
                validation=args.validation_episodes, test=args.test_tasks,
                calls=calls, seed=args.seed))
        stage_name = kw["curriculum_stage"]
        get_stage(stage_name)
        if args.execute_native and not args.output:
            parser.error("Original Vitis execution requires explicit fresh --output")
        answer = run(
            output=args.output or f"build/stage_{stage_name}_dry_plan",
            training=kw["training"], validation=kw["validation"],
            test=kw["test"], calls_per_network=kw["calls_per_network"],
            seeds=kw["seeds"], clocks_ns=kw["clocks_ns"],
            freeze_numeric_format=kw["freeze_numeric_format"],
            beginner_pilot=kw["beginner_pilot"],
            curriculum_stage=kw["curriculum_stage"],
            curriculum_mode="smoke" if args.smoke else "full",
            execute_native=args.execute_native,
            max_total_vitis_calls=args.max_total_vitis_calls,
            vitis_executable=args.vitis_executable,
            timeout_seconds=args.timeout_seconds,
            checkpoint_every_train_episodes=args.checkpoint_every_train_episodes,
            agent_algorithm=args.agent_algorithm,
            experiment_id=args.experiment)
    print(json.dumps(answer, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
