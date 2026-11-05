"""HW-RL-05: bounded, foreground native-HLS multi-network RL campaigns.

Plan is read-only and invokes ZERO vendor programs. Native campaigns need
explicit --execute-native, the exact full call budget, a clean Git source
revision, and a fresh output root. Old source-pinned studies are immutable.

Trainable methods: shared GNN one-step actor-critic, episodic REINFORCE with
a learned baseline, and real clipped PPO. Random and no-message-passing
controls share the same compiler legal masks, clock bounds, HLS gates and
task schedule. No surrogate latency, cached native reward or fake hardware.

A completed campaign is NOT scientific evidence until independently audited.
Private checkpoints are evaluation-only NPZ; optimizer state is intentionally
not advertised as resumable after a crash.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import sys

import numpy as np

from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.architecture.resolver import TargetLimits

from ..agents.hardware_algorithms import (
    METHODS, HardwareEpisodicTrainer, RandomHardwarePolicy,
    create_hardware_method,
)
from ..infrastructure.hardware_checkpoint import save_training_state
from ..agents.hardware_agent import original_hls_diagnostics
from ..environments.hardware_gym import GymHardwareNativeEnv, MAX_ACTIONS
from ..infrastructure.hardware_native import _regular_file, audit_native_case
from ..environments.hardware_space import (
    HardwareEdit, candidate_actions, initial_state, transition,
)
from ..reward import latency_only_reward
from ..study import _append_jsonl, _source_revision, _write_json_new
from .hardware_regimes import (
    CLOCK_REGIMES, NETWORK_REGIMES, validate_experiment_regimes,
)

TASK_SCHEMA = "fpgai-research/hardware-rl-tasks/v1"
CAMPAIGN_SCHEMA = "fpgai-research/hardware-rl-campaign/v1"
SPLITS = ("train", "validation", "test")
_TASK_KEYS = frozenset((
    "id", "split", "onnx", "samples", "part", "resource_caps",
    "max_absolute_error", "initial_clock_ns", "max_parallel_units",
))


@dataclass(frozen=True)
class HardwareTask:
    id: str
    split: str
    onnx: Path
    samples: Path
    part: str
    resource_caps: tuple[int, int, int, int, int]
    max_absolute_error: float
    initial_clock_ns: float
    max_parallel_units: int
    onnx_sha256: str
    samples_sha256: str

    def metadata(self) -> dict:
        """JSON-native provenance; persisted and replayed representations match."""
        return {
            key: (str(value) if isinstance(value, Path)
                  else list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(self).items()
        }


def _file_hash(path: Path) -> str:
    return sha256(_regular_file(path).read_bytes()).hexdigest()


def read_tasks(path: str | Path) -> tuple[HardwareTask, ...]:
    """Validate immutable task inputs and prohibit train/heldout overlap."""
    source = _regular_file(Path(path).expanduser(), max_bytes=1_000_000)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema", "tasks"}:
        raise ValueError("Expected exact hardware RL task schema and tasks")
    if document["schema"] != TASK_SCHEMA:
        raise ValueError("Unknown hardware RL task schema")
    records = document["tasks"]
    if not isinstance(records, list) or not 1 <= len(records) <= 512:
        raise ValueError("Expected 1..512 bounded task records")
    result = []
    ids = set()
    fingerprint_splits = {}
    for record in records:
        if (not isinstance(record, dict)
                or not {"id", "split", "onnx", "samples", "part",
                        "resource_caps", "max_absolute_error"} <= record.keys()
                or set(record) - _TASK_KEYS):
            raise ValueError("Task record has missing or unsupported fields")
        name = record["id"]
        split = record["split"]
        if (not isinstance(name, str)
                or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name) is None
                or name in ids or split not in SPLITS):
            raise ValueError("Task ID must be unique/safe with an explicit split")
        ids.add(name)
        if not all(isinstance(record[k], str) and record[k]
                   for k in ("onnx", "samples", "part")):
            raise ValueError("Task source paths/FPGA part must be nonempty strings")
        onnx = _regular_file(source.parent / record["onnx"])
        samples = _regular_file(source.parent / record["samples"])
        caps = record["resource_caps"]
        if (not isinstance(caps, list) or len(caps) != 5
                or any(type(v) is not int or v < 0 for v in caps)):
            raise ValueError("Exactly five explicit nonnegative native resource caps required")
        eps = record["max_absolute_error"]
        clock = record.get("initial_clock_ns", 14.)
        max_pe = record.get("max_parallel_units", 8)
        if (type(eps) not in (int, float) or not isfinite(eps) or eps <= 0
                or type(clock) not in (int, float) or not isfinite(clock)
                or clock != round(float(clock), 2)
                or not 2. <= clock <= 32.
                or type(max_pe) is not int or not 1 <= max_pe <= 128):
            raise ValueError("Invalid numerical error, initial clock, or parallelism bound")
        model_hash = _file_hash(onnx)
        data_hash = _file_hash(samples)
        if model_hash in fingerprint_splits and fingerprint_splits[model_hash] != split:
            raise ValueError("Identical ONNX bytes cannot occur across train/heldout splits")
        fingerprint_splits[model_hash] = split
        result.append(HardwareTask(
            name, split, onnx, samples, record["part"], tuple(caps),
            float(eps), float(clock), max_pe, model_hash, data_hash
        ))
    return tuple(result)


def episode_schedule(tasks: tuple[HardwareTask, ...], *,
                     split: str, episodes: int, seed: int):
    if split not in SPLITS or type(episodes) is not int or not 1 <= episodes <= 1000:
        raise ValueError("Split and bounded episode count required")
    if type(seed) is not int or not 0 <= seed <= 2**31:
        raise ValueError("Explicit bounded schedule seed required")
    selected = sorted((task for task in tasks if task.split == split),
                      key=lambda task: task.id)
    if not selected:
        raise ValueError("No tasks in the requested split")
    # The *same* task order for every method given the same schedule seed.
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < episodes:
        permutation = rng.permutation(len(selected))
        order.extend(selected[int(i)] for i in permutation)
    return tuple(order[:episodes])


def campaign_plan(*, tasks, split, method, episodes, calls_per_episode,
                  total_authorized_calls, seed, evaluation,
                  checkpoint, hidden, ppo_epochs, ppo_clip,
                  message_passing=True, use_trial_history=True,
                  network_regime="auto", clock_regime="search"):
    if method not in METHODS:
        raise ValueError("Unsupported hardware optimization method")
    if (type(calls_per_episode) is not int
            or not 1 <= calls_per_episode <= 30):
        raise ValueError("Native calls per episode must be in [1,30]")
    total = episodes * calls_per_episode
    if total > 5000:
        raise ValueError("Campaign exceeds prospective 5000-call safety cap")
    if total_authorized_calls is not None and (
            type(total_authorized_calls) is not int
            or total_authorized_calls != total):
        raise ValueError("Authorized original Vitis budget must match planned calls exactly")
    if method == "random" and checkpoint is not None:
        raise ValueError("Random baseline cannot load a trained checkpoint")
    if evaluation and method != "random" and checkpoint is None:
        raise ValueError("Frozen neural-policy evaluation requires an original private checkpoint")
    if not evaluation and checkpoint is not None:
        raise ValueError("Training from scratch cannot load a prior checkpoint")
    if type(hidden) is not int or not 8 <= hidden <= 512:
        raise ValueError("GNN hidden width must be in [8,512]")
    if type(ppo_epochs) is not int or not 1 <= ppo_epochs <= 8:
        raise ValueError("PPO epochs must be in [1,8]")
    if type(ppo_clip) not in (float, int) or not .05 <= ppo_clip <= .4:
        raise ValueError("PPO clip must be in [0.05,0.4]")
    schedule = episode_schedule(tasks, split=split, episodes=episodes, seed=seed)
    unique_tasks = len({task.id for task in schedule})
    regime = validate_experiment_regimes(
        schedule, network_regime=network_regime, clock_regime=clock_regime,
    )
    return {
        **regime,
        "schema": CAMPAIGN_SCHEMA,
        "method": method,
        "split": split,
        "evaluation": bool(evaluation),
        "seed": seed,
        "episodes": episodes,
        "original_vitis_calls_per_episode": calls_per_episode,
        "planned_max_original_vitis_calls": total,
        "unique_network_task_ids": unique_tasks,
        "task_repetitions": episodes - unique_tasks,
        "schedule": [task.id for task in schedule],
        "hidden": hidden,
        "ppo_epochs": ppo_epochs if method == "ppo" else None,
        "ppo_clip": ppo_clip if method == "ppo" else None,
        "message_passing": bool(message_passing),
        "trial_history": bool(use_trial_history),
        "checkpoint_loaded_for_frozen_evaluation": str(checkpoint) if checkpoint else None,
        "native_execution_authorized": total_authorized_calls == total,
        "source_only_plan": True,
        "physical_fpga_measurements": False,
        "cached_or_surrogate_hls_reward": False,
    }


def _load_task(task: HardwareTask):
    model = load_onnx(str(task.onnx))
    samples = np.load(_regular_file(task.samples), allow_pickle=False)
    return model, samples


def _environment(task: HardwareTask, *, root: Path,
                 calls_per_episode: int, native: bool,
                 vitis: str, timeout: int,
                 clock_regime: str = "search"):
    model, samples = _load_task(task)
    return GymHardwareNativeEnv(
        model, samples, root,
        part=task.part, resource_caps=task.resource_caps,
        max_absolute_error=task.max_absolute_error,
        initial_clock_ns=task.initial_clock_ns,
        fixed_clock_ns=(task.initial_clock_ns if clock_regime in ("fixed", "task_fixed") else None),
        max_parallel_units=task.max_parallel_units,
        max_vitis_calls=calls_per_episode,
        execute_native=native,
        vitis_executable=vitis,
        timeout_seconds=timeout,
    )


def _trial_record(episode: int, step: int, task: HardwareTask, *,
                  source_action: HardwareEdit | None,
                  action, reward, info, gradient, method: str,
                  rejected_source_proposals: int = 0):
    """Store the pre-step compiler action, never the next state's offer."""
    audit = info["original_vitis_audit"]
    chosen = source_action
    return {
        "schema": "fpgai-research/hardware-rl-trial/v1",
        "episode": episode, "step": step, "task_id": task.id,
        "method": method, "source_action_index": action[0],
        "source_action": asdict(chosen) if chosen else None,
        "rejected_source_proposals_before_native": rejected_source_proposals,
        "requested_clock_ns": action[1],
        "native_state_sha256": info["state_sha256"],
        "native_case": info["native_case"],
        "qualified": info["qualified"], "reward": reward,
        "best_qualified_latency_ns": info["best_hls_estimated_latency_ns"],
        "qualified_latency_ns": audit["hls_estimated_latency_ns"],
        "original_audit": original_hls_diagnostics(audit),
        "gradient": gradient,
    }


def _choose_admitted_action(agent, environment, observation, *,
                           train: bool, stochastic_evaluation: bool,
                           max_source_proposals: int = 64):
    """Reject no-op/visited complete states before an original Vitis call.

    Only an accepted original policy sample enters the learning trajectory.
    These are acceptance-conditioned samples; proposal rejections are logged
    and must be disclosed in any scientific policy comparison. Frozen
    deterministic evaluation never silently changes its policy.
    """
    if (type(max_source_proposals) is not int
            or not 1 <= max_source_proposals <= 256):
        raise ValueError("Bounded source-only proposal budget required")
    rejected = 0
    rejected_tapes = []
    deterministic = (not train and not stochastic_evaluation
                     and not isinstance(agent, RandomHardwarePolicy))
    ranked = iter(agent.ranked_frozen_actions(observation)) if deterministic else None
    for _ in range(max_source_proposals):
        if deterministic:
            try:
                action, tape = next(ranked), None
            except StopIteration:
                break
        elif isinstance(agent, RandomHardwarePolicy):
            action, tape = agent.choose(observation), None
        else:
            action, tape = agent.choose(
                observation, training=train,
                sample=(train or stochastic_evaluation),
            )
        try:
            admitted = environment.preview_action(action)
        except ValueError as exc:
            reason = str(exc)
            if not (reason.startswith("No-op architecture-clock")
                    or reason.startswith("Repeated complete hardware state")):
                raise
            rejected += 1
            if train:
                if tape is None:
                    raise RuntimeError("Trainable rejected proposal lacks original policy tape")
                rejected_tapes.append(tape)
            continue
        if train and rejected_tapes:
            if tape is None:
                raise RuntimeError("Accepted proposal lacks original policy tape")
            # Rejection sampling is a random-length trajectory. Its exact
            # likelihood is the PRODUCT of rejected and accepted original
            # draws. The sum of their log probabilities is the unbiased
            # score-function estimator for the accepted HLS reward.
            # Rejected proposals have no fabricated HLS rewards/labels.
            tape = replace(
                tape,
                log_probability=(tape.log_probability + sum(
                    (t.log_probability for t in rejected_tapes), 0
                )),
                entropy=(tape.entropy + sum(
                    (t.entropy for t in rejected_tapes), 0
                )),
                rejected_proposal_tapes=tuple(rejected_tapes),
            )
        return action, tape, environment.offers[action[0]], admitted, rejected
    raise RuntimeError(
        f"No novel source-backed proposal after {max_source_proposals} "
        "source-only attempts; no Vitis call made"
    )


def run_campaign(*, tasks, plan: dict, output_root: str | Path,
                 authorized_calls: int, checkpoint: str | None,
                 vitis: str, timeout: int, stochastic_evaluation: bool):
    if authorized_calls != plan["planned_max_original_vitis_calls"]:
        raise ValueError("The exact native Vitis authorization is required")
    if type(timeout) is not int or not 1 <= timeout <= 172800:
        raise ValueError("Native Vitis timeout outside allowed range")
    # Explicit provenance; no dirty *tracked* source is allowed. Untracked
    # private build products can remain, but are never silently committed.
    source_rev = _source_revision()
    root = Path(output_root).expanduser()
    if root.exists() or root.is_symlink():
        raise FileExistsError("Create-only native campaign root already exists")
    schedule = episode_schedule(
        tasks, split=plan["split"], episodes=plan["episodes"], seed=plan["seed"]
    )
    # Preflight every distinct network *before* allocating original cases.
    # All setup is source-only and cannot invoke the Vitis runner.
    for task in {v.id: v for v in schedule}.values():
        _environment(
            task, root=root / "preflight" / task.id,
            calls_per_episode=plan["original_vitis_calls_per_episode"],
            native=False, vitis=vitis, timeout=timeout,
            clock_regime=plan.get("clock_regime", "search"),
        )
    agent = create_hardware_method(
        plan["method"], seed=plan["seed"], hidden=plan["hidden"],
        ppo_epochs=plan["ppo_epochs"] or 3, ppo_clip=plan["ppo_clip"] or .2,
        message_passing=plan["message_passing"],
        use_trial_history=plan["trial_history"],
    )
    if checkpoint is not None:
        agent.load_private(checkpoint, load_optimizer=False)
    root.mkdir(parents=True, exist_ok=False)
    metadata = {
        **plan, "source_only_plan": False, "native_execution_authorized": True,
        "source_revision_sha": source_rev, "original_cases_private": True,
        "task_definitions": [task.metadata() for task in tasks],
        "training_checkpoint_resume_supported": False,
        "independent_campaign_audit_completed": False,
    }
    _write_json_new(root / "campaign.json", metadata)
    journal = root / "trials.jsonl"
    episode_journal = root / "episodes.jsonl"
    total_calls = 0
    qualified_calls = 0
    for i, task in enumerate(schedule):
        environment = _environment(
            task, root=root / "episodes" / f"episode_{i:06d}",
            calls_per_episode=plan["original_vitis_calls_per_episode"],
            native=True, vitis=vitis, timeout=timeout,
            clock_regime=plan.get("clock_regime", "search"),
        )
        observation, info = environment.reset(seed=plan["seed"] + i)
        rewards = []
        best = None
        gradient_receipt = None
        for step in range(plan["original_vitis_calls_per_episode"]):
            train = not plan["evaluation"] and plan["method"] != "random"
            (action, tape, source_action, admitted, rejected_proposals) = (
                _choose_admitted_action(
                    agent, environment, observation, train=train,
                    stochastic_evaluation=stochastic_evaluation,
                )
            )
            if total_calls >= authorized_calls:
                raise RuntimeError("Native Vitis call authorization exceeded")
            # Gym.step() rebuilds the action catalog for the NEXT state,
            # so the exact pre-step offer was captured above.
            observation, reward, terminated, truncated, info = environment.step(action)
            total_calls += 1
            if admitted.state_sha256 != info["state_sha256"]:
                raise RuntimeError("Preflight source state and original HLS receipt disagree")
            if train:
                if isinstance(agent, HardwareEpisodicTrainer):
                    gradient_receipt = agent.observe(
                        tape, reward, info=info, env=environment
                    )
                else:
                    gradient_receipt = agent.update(
                        tape, reward, info=info, env=environment,
                        next_obs=observation if not (terminated or truncated) else None,
                        terminated=bool(terminated or truncated),
                    )
            else:
                gradient_receipt = None
            _append_jsonl(journal, _trial_record(
                i, step, task, source_action=source_action, action=action,
                reward=reward, info=info, gradient=gradient_receipt,
                method=plan["method"],
                rejected_source_proposals=rejected_proposals,
            ))
            rewards.append(reward)
            qualified_calls += int(info["qualified"])
            best = info["best_hls_estimated_latency_ns"]
            if terminated or truncated:
                if step != plan["original_vitis_calls_per_episode"] - 1:
                    raise RuntimeError(
                        "Incomplete native episode: retain cases, do not publish as completed"
                    )
                break
        if len(rewards) != plan["original_vitis_calls_per_episode"]:
            raise RuntimeError("Native episode did not finish full budget")
        if (not plan["evaluation"]
                and isinstance(agent, HardwareEpisodicTrainer)):
            gradient_receipt = agent.finish_episode(complete=True)
        export = None
        optimizer_snapshot = None
        if (not plan["evaluation"]
                and not isinstance(agent, RandomHardwarePolicy)):
            provenance = {
                "campaign_source_sha": source_rev,
                "task_id": task.id,
                "episode": i,
                "algorithm": plan["method"],
                "task_manifest_provenance": [x.metadata() for x in tasks],
                "learning_mode": "online_original_hls",
                "network_regime": plan.get("network_regime", "auto"),
                "clock_regime": plan.get("clock_regime", "search"),
                "fixed_clock_ns": plan.get("fixed_clock_ns"),
            }
            export = (root / "checkpoints" / f"episode_{i:06d}")
            agent.save_private(export, provenance)
            optimizer_snapshot = (
                root / "training_state" / f"episode_{i:06d}"
            )
            save_training_state(
                agent, optimizer_snapshot,
                method=plan["method"],
                source_revision_sha=source_rev,
                metadata=provenance,
            )
        _append_jsonl(episode_journal, {
            "schema": "fpgai-research/hardware-rl-episode/v1",
            "episode": i, "task_id": task.id,
            "native_calls": len(rewards),
            "qualified_calls": sum(
                bool(t["audit"]["qualified"]) for t in environment.history
            ),
            "best_qualified_latency_ns": best,
            "no_feasible_design_censored": best is None,
            "reward_sum": float(sum(rewards)),
            "gradient_receipt": (
                gradient_receipt if isinstance(agent, HardwareEpisodicTrainer)
                else None
            ),
            "evaluation_only_checkpoint": str(export) if export else None,
            "full_optimizer_rng_snapshot": (
                str(optimizer_snapshot) if optimizer_snapshot else None
            ),
            "automatic_resume_authorized": False,
        })
        print(json.dumps({
            "completed_episode": i + 1, "of": len(schedule),
            "task_id": task.id, "native_calls": total_calls,
            "best_qualified_latency_ns": best,
            "qualified_calls_so_far": qualified_calls,
        }, sort_keys=True), flush=True)
    result = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "COMPLETE_PENDING_INDEPENDENT_CAMPAIGN_AUDIT",
        "native_calls": total_calls,
        "qualified_calls": qualified_calls,
        "episodes": len(schedule),
        "method": plan["method"],
        "split": plan["split"],
        "original_case_root": str(root.absolute()),
        "physical_fpga_measurements": False,
        "native_replay_completed": False,
    }
    _write_json_new(root / "summary.json", result)
    return result


def _read_jsonl(path: Path):
    with _regular_file(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(item, dict) for item in records):
        raise ValueError("Invalid campaign JSONL entries")
    return records


def audit_campaign(*, root: str | Path, tasks, allow_incomplete: bool = False):
    """Independent source+native replay of every recorded case, ZERO Vitis calls.

    This does not replay policy optimizer dynamics (eval-only checkpoints).
    Incomplete/corrupt campaigns are never mislabeled complete.
    """
    directory = Path(root).expanduser()
    metadata = json.loads(_regular_file(directory / "campaign.json").read_text())
    if metadata.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("Unknown native hardware campaign schema")
    if metadata.get("source_revision_sha") != _source_revision():
        raise ValueError("Original source revision mismatch: checkout pinned commit to audit")
    if [t.metadata() for t in tasks] != metadata["task_definitions"]:
        raise ValueError("Original task models/samples/constraints changed")
    if (metadata["planned_max_original_vitis_calls"] !=
            metadata["episodes"] * metadata["original_vitis_calls_per_episode"]):
        raise ValueError("Original native-call budget was changed")
    task_index = {t.id: t for t in tasks}
    validated_regime = validate_experiment_regimes(
        tuple(task_index[k] for k in metadata["schedule"]),
        network_regime=metadata.get("network_regime", "auto"),
        clock_regime=metadata.get("clock_regime", "search"),
    )
    if any(metadata.get(k) != v for k, v in validated_regime.items()
           if k in metadata):
        raise ValueError("Campaign network/clock regime metadata disagrees")
    trials = _read_jsonl(directory / "trials.jsonl")
    episode_records = _read_jsonl(directory / "episodes.jsonl")
    expected = metadata["planned_max_original_vitis_calls"]
    if not allow_incomplete and (
        len(trials) != expected
        or len(episode_records) != metadata["episodes"]
        or not (directory / "summary.json").is_file()
    ):
        raise ValueError("Incomplete native campaign; no completed paper claim")
    count = 0
    qualified = 0
    groups = {}
    for t in trials:
        groups.setdefault(t["episode"], []).append(t)
    for episode_index, group in sorted(groups.items()):
        if not 0 <= episode_index < metadata["episodes"]:
            raise ValueError("Invalid episode index in original trial journal")
        task = task_index[metadata["schedule"][episode_index]]
        if any(x["task_id"] != task.id for x in group):
            raise ValueError("Campaign task schedule mismatch")
        model, samples = _load_task(task)
        state = initial_state(
            model, part=task.part, clock_ns=task.initial_clock_ns,
            resource_caps=task.resource_caps,
            max_absolute_error=task.max_absolute_error,
            limits=TargetLimits(max_parallel_units=task.max_parallel_units),
        )
        best = None
        for j, trial in enumerate(group):
            if (trial["step"] != j
                    or trial["episode"] != episode_index
                    or trial.get("method") != metadata["method"]
                    or type(trial.get("rejected_source_proposals_before_native", 0))
                       is not int
                    or not 0 <= trial.get(
                        "rejected_source_proposals_before_native", 0) < 64):
                raise ValueError("Out-of-order, duplicated, or wrong-method native trial")
            # Recreate the exact source-defined action index from the
            # *pre-step* architecture state, not merely a plausible edit.
            original_offers = (
                None, *candidate_actions(state, max_actions=MAX_ACTIONS - 1)
            )
            index = trial["source_action_index"]
            if type(index) is not int or not 0 <= index < len(original_offers):
                raise ValueError("Original GNN action index is not in the source catalog")
            offered = original_offers[index]
            serialized_offer = (
                json.loads(json.dumps(asdict(offered), allow_nan=False))
                if offered is not None else None
            )
            if trial["source_action"] != serialized_offer:
                raise ValueError("Archived action index/edit disagrees with exact compiler menu")
            if metadata.get("clock_regime", "search") in ("fixed", "task_fixed") and (
                index == 0
                or trial["requested_clock_ns"] != task.initial_clock_ns
            ):
                raise ValueError("Fixed-clock campaign contains a clock edit")
            edit = trial["source_action"]
            if edit is not None:
                decision = transition(state, HardwareEdit(**edit))
                if decision.status != "SOURCE_ACCEPTED":
                    raise ValueError("Archived source action is not compiler-legal")
                state = decision.next
            clock = trial["requested_clock_ns"]
            if clock != state.clock_ns:
                decision = transition(
                    state, HardwareEdit("clock", "clock_ns", clock)
                )
                if decision.status != "SOURCE_ACCEPTED":
                    raise ValueError("Archived native clock is not compiler-legal")
                state = decision.next
            if trial["native_state_sha256"] != state.state_sha256:
                raise ValueError("Archived source state digest mismatch")
            case = (directory / "episodes" / f"episode_{episode_index:06d}"
                    / "episode_000000" / f"case_{j:04d}")
            receipt = json.loads(_regular_file(
                case.parent / f"step_{j:04d}.json"
            ).read_text())
            independent = audit_native_case(case, state, samples)
            if (receipt["original_case_audit"] != independent
                    or receipt["reward"] != trial["reward"]
                    or receipt["action_index"] != trial["source_action_index"]
                    or receipt["requested_clock_ns"] != trial["requested_clock_ns"]
                    or receipt["state_sha256"] != trial["native_state_sha256"]
                    or receipt["native_case"] != str(case)
                    or independent["state_sha256"] != trial["native_state_sha256"]
                    or original_hls_diagnostics(independent) != trial["original_audit"]
                    or trial["qualified"] is not independent["qualified"]
                    or trial["qualified_latency_ns"] != independent["hls_estimated_latency_ns"]
                    or trial["native_case"] != independent["case"]):
                raise ValueError("Independent original HLS campaign replay disagrees")
            prior_best = best
            if independent["qualified"]:
                latency = independent["hls_estimated_latency_ns"]
                best = latency if best is None else min(best, latency)
            expected_reward = latency_only_reward(
                prior_best, best, qualified=independent["qualified"]
            )
            if (abs(float(trial["reward"]) - float(expected_reward)) > 1e-10
                    or trial["best_qualified_latency_ns"] != best):
                raise ValueError("Original HLS latency-only reward was replaced")
            count += 1
            qualified += int(independent["qualified"])
    for i, record in enumerate(episode_records):
        if (record.get("schema") != "fpgai-research/hardware-rl-episode/v1"
                or record.get("episode") != i
                or record.get("task_id") != metadata["schedule"][i]
                or record.get("native_calls") != metadata["original_vitis_calls_per_episode"]
                or record.get("native_calls") != len(groups.get(i, ()))
                or record.get("qualified_calls") != sum(
                    bool(x["qualified"]) for x in groups.get(i, ())
                )):
            raise ValueError("Original episode journal or task schedule disagrees")
        group = groups[i]
        last = group[-1] if group else None
        if (last is None
                or record.get("best_qualified_latency_ns") != last["best_qualified_latency_ns"]
                or record.get("no_feasible_design_censored") is not (
                    last["best_qualified_latency_ns"] is None
                )
                or abs(record.get("reward_sum") - sum(
                    float(x["reward"]) for x in group
                )) > 1e-10):
            raise ValueError("Original episode reward/censoring summary was altered")
    complete = (
        count == expected and len(episode_records) == metadata["episodes"]
        and len(groups) == metadata["episodes"]
        and (directory / "summary.json").is_file()
    )
    if not complete and not allow_incomplete:
        raise ValueError("Native case count does not match preregistered budget")
    return {
        "schema": "fpgai-research/hardware-rl-campaign-audit/v1",
        "status": "AUDITED_COMPLETE" if complete else "AUDITED_PARTIAL_PROVISIONAL",
        "method": metadata["method"], "split": metadata["split"],
        "original_native_cases_replayed": count,
        "original_native_cases_expected": expected,
        "qualified_cases": qualified,
        "source_revision_sha": metadata["source_revision_sha"],
        "network_regime": metadata.get("network_regime", "auto"),
        "clock_regime": metadata.get("clock_regime", "search"),
        "native_vitis_calls_by_audit": 0,
        "physical_routed_timing": False,
        "no_learning_superiority_inferred": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True,
                        help="Original JSON manifest of disjoint train/validation/test tasks")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--method", choices=METHODS, default="actor_critic")
    parser.add_argument(
        "--network-regime", choices=NETWORK_REGIMES, default="auto",
        help="auto=legacy scheduling; same=one immutable model; general=2+ genuinely different ONNX graphs",
    )
    parser.add_argument(
        "--clock-regime", choices=CLOCK_REGIMES, default="search",
        help="search=learned 2..32ns clock; fixed=one global clock; task_fixed=distinct per-task fixed clocks, no clock actor",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--calls-per-episode", type=int, default=10)
    parser.add_argument("--max-total-vitis-calls", type=int)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-clip", type=float, default=.2)
    parser.add_argument("--no-message-passing", action="store_true")
    parser.add_argument("--no-trial-history", action="store_true")
    parser.add_argument("--evaluation", action="store_true",
                        help="Frozen evaluation; trainable methods require --checkpoint")
    parser.add_argument("--stochastic-evaluation", action="store_true")
    parser.add_argument("--checkpoint",
                        help="Private evaluation-only NPZ checkpoint; no optimizer resumption")
    parser.add_argument("--output-root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-native", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Only for read-only provisional audit of interrupted studies")
    parser.add_argument("--vitis", default="vitis_hls")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    try:
        if args.allow_incomplete and not args.audit:
            raise ValueError("Incomplete provisional audit flag requires --audit")
        tasks = read_tasks(args.tasks)
        if args.audit:
            if not args.output_root:
                raise ValueError("--audit requires the original campaign --output-root")
            output = audit_campaign(
                root=args.output_root, tasks=tasks,
                allow_incomplete=args.allow_incomplete,
            )
        else:
            plan = campaign_plan(
                tasks=tasks, split=args.split, method=args.method,
                episodes=args.episodes, calls_per_episode=args.calls_per_episode,
                total_authorized_calls=args.max_total_vitis_calls,
                seed=args.seed, evaluation=args.evaluation,
                checkpoint=args.checkpoint, hidden=args.hidden,
                ppo_epochs=args.ppo_epochs, ppo_clip=args.ppo_clip,
                message_passing=not args.no_message_passing,
                use_trial_history=not args.no_trial_history,
                network_regime=args.network_regime,
                clock_regime=args.clock_regime,
            )
            if args.execute_native:
                if not args.output_root or args.max_total_vitis_calls is None:
                    raise ValueError(
                        "--execute-native requires fresh --output-root and "
                        "exact --max-total-vitis-calls"
                    )
                output = run_campaign(
                    tasks=tasks, plan=plan,
                    output_root=args.output_root,
                    authorized_calls=args.max_total_vitis_calls,
                    checkpoint=args.checkpoint,
                    vitis=args.vitis, timeout=args.timeout,
                    stochastic_evaluation=args.stochastic_evaluation,
                )
            else:
                output = plan
        print(json.dumps(output, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        print(f"Hardware RL campaign stopped: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
