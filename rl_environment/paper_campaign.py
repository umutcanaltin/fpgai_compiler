"""Prepare, explicitly launch, and independently audit one paper HLS study.

Default is a source-only plan. Native execution is permitted only with an
explicit --execute-native flag and complete study-specific Vitis call budget.
Full training additionally requires an independently audited qualifying smoke
from the same clean source revision. This CLI runs only on the invoking host.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import available_agents
from .gym_curriculum import config
from .gym_study import plan, run
from .study import _source_revision
from .runtime_contract import verify_local_compiler_package


def study_arguments(*, stage="L0", agent="graph_actor_critic",
                    mode="smoke", seed=111) -> dict:
    """Use source-defined immutable stage settings; do not rewrite HLS rules."""
    if mode not in ("smoke", "full"):
        raise ValueError("Mode must be smoke or full")
    if agent not in available_agents():
        raise ValueError(f"Algorithm has no tested source registration: {agent}")
    if type(seed) is not int or seed < 0:
        raise ValueError("A nonnegative integer seed is required")
    kw = config(stage, train=2 if mode == "smoke" else 100,
                calls=2 if mode == "smoke" else 10, seed=seed)
    return {
        "training": kw["training"],
        "validation": kw["validation"],
        "test": kw["test"],
        "calls_per_network": kw["calls_per_network"],
        "clocks_ns": kw["clocks_ns"],
        "seeds": kw["seeds"],
        "freeze_numeric_format": kw["freeze_numeric_format"],
        "beginner_pilot": kw["beginner_pilot"],
        "curriculum_stage": kw["curriculum_stage"],
        "curriculum_mode": mode,
        "agent_algorithm": agent,
    }


def study_plan(*, stage="L0", agent="graph_actor_critic",
               mode="smoke", seed=111) -> dict:
    kw = study_arguments(stage=stage, agent=agent, mode=mode, seed=seed)
    p = plan(**kw)
    return {
        "status": "SOURCE_ONLY_NOT_EXECUTED",
        "mode": mode,
        "stage": stage,
        "agent": agent,
        "seed": seed,
        "original_vitis_calls_executed": 0,
        "requires_independent_audit": True,
        "full_run_requires_qualifying_smoke": mode == "full",
        "native_budget": p["maximum_original_vitis_invocations"],
        "source_plan": p,
    }


def check_smoke_receipts(*, audit_report: dict, funnel: dict) -> dict:
    """Fixed engineering acceptance gate, not proof of RL learning."""
    counts = funnel.get("counts", {})
    checks = {
        "independent_audit_passed":
            audit_report.get("status") == "INDEPENDENT_GYM_HLS_AUDIT_PASSED",
        "exactly_22_original_calls":
            audit_report.get("original_cases_recomputed") == 22
            and counts.get("original_cases", 0) == 22,
        "at_least_14_qualified": counts.get("qualified", 0) >= 14,
        "no_invalid_timing_parse": counts.get("timing_parse_exception", 0) == 0,
    }
    return {
        "status": "SMOKE_ACCEPTED_FOR_FULL_RESEARCH" if all(checks.values())
                  else "SMOKE_REJECTED_DO_NOT_START_FULL",
        "checks": checks,
        "qualified_original_cases": counts.get("qualified", 0),
        "original_cases": counts.get("original_cases", 0),
        "source_revision": audit_report.get("source_revision"),
        "vendor_calls_during_gate": 0,
    }


def verify_smoke(study: str | Path, *, stage: str,
                 agent: str, seed: int, revision: str) -> dict:
    """Read-only check that an exact-source smoke authorizes a full study."""
    root = Path(study).resolve(strict=True)
    campaign = json.loads((root / "campaign.json").read_text())
    if (campaign.get("curriculum_mode") != "smoke"
            or campaign.get("curriculum_stage") != stage
            or campaign.get("agent_algorithm") != agent
            or campaign.get("training_seeds") != [seed]
            or campaign.get("source_revision") != revision):
        raise ValueError("Smoke is not for this stage/agent/seed/source revision")
    from .gym_audit import audit
    from .qualification_funnel import diagnose
    result = check_smoke_receipts(audit_report=audit(root),
                                  funnel=diagnose(root, examples=0))
    if result["status"] != "SMOKE_ACCEPTED_FOR_FULL_RESEARCH":
        raise RuntimeError("Strict original-Vitis smoke gate rejected: "
                           + json.dumps(result, sort_keys=True))
    return result


def execute(*, stage, agent, mode, seed, output, max_total_vitis_calls,
            vitis_executable="vitis_hls", smoke_study=None,
            timeout_seconds=7200) -> dict:
    """Run original HLS on THIS machine only after explicit CLI opt-in."""
    kw = study_arguments(stage=stage, agent=agent, mode=mode, seed=seed)
    expected = plan(**kw)["maximum_original_vitis_invocations"]
    if type(max_total_vitis_calls) is not int or max_total_vitis_calls != expected:
        raise ValueError(f"Explicit native budget must be exactly {expected}")
    if not output:
        raise ValueError("Explicit create-only output directory required")
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"Refusing to overwrite original study: {dest}")
    verify_local_compiler_package()
    revision = _source_revision()
    if mode == "full":
        if not smoke_study:
            raise ValueError("Full study requires --smoke-study independently audited")
        verify_smoke(smoke_study, stage=stage, agent=agent,
                     seed=seed, revision=revision)
    run_result = run(
        output=output, **kw, execute_native=True,
        max_total_vitis_calls=max_total_vitis_calls,
        vitis_executable=vitis_executable,
        timeout_seconds=timeout_seconds)
    # These source-only checks do not modify native cases or run vendor tools.
    from .gym_audit import audit
    from .qualification_funnel import diagnose
    audit_report = audit(dest)
    funnel = diagnose(dest, examples=0)
    answer = {
        "status": "ORIGINAL_HLS_RECORDED_AND_INDEPENDENTLY_AUDITED",
        "run": run_result,
        "audit": audit_report,
        "qualification": funnel,
        "full_paper_learning_claim": False,
    }
    if mode == "smoke":
        answer["smoke_gate"] = check_smoke_receipts(
            audit_report=audit_report, funnel=funnel)
    return answer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=[f"L{i}" for i in range(7)], default="L0")
    p.add_argument("--agent", choices=available_agents(), default="graph_actor_critic")
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--execute-native", action="store_true",
                   help="Actually run Vitis on this machine; default is plan only")
    p.add_argument("--max-total-vitis-calls", type=int)
    p.add_argument("--output", help="Fresh create-only path for original HLS receipts")
    p.add_argument("--smoke-study",
                   help="Required original audited same-revision smoke for full mode")
    p.add_argument("--vitis-executable", default="vitis_hls")
    p.add_argument("--timeout-seconds", type=int, default=7200)
    args = p.parse_args(argv)
    if not args.execute_native:
        if args.max_total_vitis_calls is not None or args.smoke_study:
            p.error("Native authorization arguments require --execute-native")
        result = study_plan(stage=args.stage, agent=args.agent,
                            mode=args.mode, seed=args.seed)
    else:
        if args.max_total_vitis_calls is None or not args.output:
            p.error("Native HLS requires --output and --max-total-vitis-calls")
        result = execute(
            stage=args.stage, agent=args.agent, mode=args.mode,
            seed=args.seed, output=args.output, smoke_study=args.smoke_study,
            max_total_vitis_calls=args.max_total_vitis_calls,
            vitis_executable=args.vitis_executable,
            timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
