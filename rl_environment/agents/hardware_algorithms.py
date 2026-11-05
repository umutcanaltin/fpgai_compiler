"""Hardware GNN research algorithms over one existing audited native Gym.

A genuine on-policy episodic REINFORCE-with-value-baseline, PPO-Clip and the
existing one-step actor-critic share the exact same GNN/clock actor. No
surrogate reward, cached HLS measurement, or second compiler/backend.
All observed rewards are independently re-audited against a live Gym case
*before* they may enter any policy optimization buffer.

All method-specific changes occur here, outside compiler/Environment code.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import numpy as np

from ..clock_actions import normalize_clock_period_ns
from .hardware_agent import (
    HardwareActionTape, HardwareGraphTrainer, verify_original_hardware_step,
    _require_torch, torch, Categorical, Beta,
)

METHODS = ("actor_critic", "reinforce", "ppo", "random")


@dataclass(frozen=True)
class AuditedExperience:
    tape: HardwareActionTape
    reward: float
    qualified: bool
    state_sha256: str


def _discounted_returns(experiences: list[AuditedExperience], gamma: float):
    if not experiences or not 0 <= gamma < 1 or not isfinite(gamma):
        raise ValueError("A nonempty audited episode and finite gamma in [0,1) required")
    results = []
    total = 0.
    for item in reversed(experiences):
        total = float(item.reward) + gamma * total
        results.append(total)
    return torch.tensor(list(reversed(results)), dtype=torch.float32)


class HardwareEpisodicTrainer(HardwareGraphTrainer):
    """Episode-buffered REINFORCE or clipped PPO; no updates on unverified data.

    The same shared model is carried across different network episodes.
    Online feedback is observed and audited each step. Parameters update
    only at the end of an explicitly bounded complete episode.
    """
    def __init__(self, *, algorithm: str, gamma: float = .95,
                 ppo_epochs: int = 3, ppo_clip: float = .2, **kwargs):
        if algorithm not in ("reinforce", "ppo"):
            raise ValueError("Only the real implemented episodic algorithms are allowed")
        if type(gamma) not in (float, int) or not 0 <= gamma < 1:
            raise ValueError("Explicit finite discount must be in [0,1)")
        if type(ppo_epochs) is not int or not 1 <= ppo_epochs <= 8:
            raise ValueError("PPO epochs must be an integer in [1,8]")
        if type(ppo_clip) not in (float, int) or not .05 <= ppo_clip <= .4:
            raise ValueError("PPO clipping must be in [0.05,0.4]")
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.gamma = float(gamma)
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = float(ppo_clip)
        self.pending: list[AuditedExperience] = []

    def observe(self, tape: HardwareActionTape, reward: float, *,
                info: dict, env) -> dict:
        if len(self.pending) >= 100:
            raise ValueError("Native training episode exceeded 100 audited steps")
        if not isinstance(tape, HardwareActionTape):
            raise ValueError("A genuine sampled policy tape is required")
        if self.algorithm == "ppo" and (
            tape.observation is None or tape.sampled_clock_fraction is None
        ):
            raise ValueError("PPO requires the genuine original sampled observation/clock")
        qualified = verify_original_hardware_step(
            tape, reward, info=info, env=env
        )
        item = AuditedExperience(tape, float(reward), qualified, info["state_sha256"])
        self.pending.append(item)
        return {
            "schema": "fpgai-research/hardware-rl-audited-buffer/v1",
            "algorithm": self.algorithm,
            "buffered_steps": len(self.pending),
            "qualified": qualified,
            "native_calls_by_trainer": 0,
            "gradient_updates": self.updates,
            "training_label_from_original_vitis_only": True,
        }

    def _objective(self, experiences, returns):
        logprobs = torch.stack([v.tape.log_probability for v in experiences])
        values = torch.stack([v.tape.value for v in experiences])
        entropy = torch.stack([v.tape.entropy for v in experiences])
        advantage = returns - values.detach()
        actor = -(logprobs * advantage).mean()
        critic = .5 * (values - returns).square().mean()
        predicted = torch.stack([
            self.policy.feasibility_head(
                torch.cat((v.tape.selected_context,
                           v.tape.requested_clock_fraction), dim=-1)
            ).squeeze(-1)
            for v in experiences
        ])
        labels = torch.tensor([float(v.qualified) for v in experiences],
                              dtype=torch.float32)
        feasibility = torch.nn.functional.binary_cross_entropy_with_logits(
            predicted, labels, pos_weight=torch.tensor(8.)
        )
        entropy_loss = -.001 * entropy.mean()
        return actor, critic, entropy_loss, feasibility, 0.

    def _ppo_objective(self, experiences, returns):
        # Fresh policy likelihood for the entire source-admission trace,
        # including rejected source-only draws and the final accepted Vitis
        # candidate. No invented rewards from rejected proposals.
        current_logprobs = []
        values = []
        entropies = []
        feasibility_logits = []
        for item in experiences:
            accepted = item.tape
            trace = (*accepted.rejected_proposal_tapes, accepted)
            trace_logprobs = []
            trace_entropies = []
            for attempt in trace:
                if (attempt.observation is None
                        or attempt.sampled_clock_fraction is None):
                    raise ValueError("PPO admission trace lacks a sampled original policy action")
                logits, value, contexts = self.policy(
                    attempt.observation, with_context=True
                )
                chosen = attempt.source_action_index
                cat = Categorical(logits=logits)
                params = torch.nn.functional.softplus(
                    self.policy.clock_head(contexts[chosen])
                ) + 1.01
                beta = Beta(params[0], params[1])
                trace_logprobs.append(
                    cat.log_prob(torch.tensor(chosen))
                    + beta.log_prob(attempt.sampled_clock_fraction)
                )
                trace_entropies.append(cat.entropy() + beta.entropy())
                if attempt is accepted:
                    values.append(value)
                    feasibility_logits.append(self.policy.feasibility_head(
                        torch.cat((contexts[chosen],
                                   accepted.requested_clock_fraction), dim=-1)
                    ).squeeze(-1))
            current_logprobs.append(torch.stack(trace_logprobs).sum())
            entropies.append(torch.stack(trace_entropies).sum())
        new_lp = torch.stack(current_logprobs)
        old_lp = torch.stack([
            v.tape.log_probability.detach() for v in experiences
        ])
        old_values = torch.stack([
            v.tape.value.detach() for v in experiences
        ])
        advantage = returns - old_values
        ratio = torch.exp(new_lp - old_lp)
        clipped_ratio = ratio.clamp(1.-self.ppo_clip, 1.+self.ppo_clip)
        actor = -torch.min(ratio * advantage,
                           clipped_ratio * advantage).mean()
        critic = .5 * (torch.stack(values) - returns).square().mean()
        entropy_loss = -.001 * torch.stack(entropies).mean()
        labels = torch.tensor([float(v.qualified) for v in experiences])
        feasibility = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.stack(feasibility_logits), labels,
            pos_weight=torch.tensor(8.),
        )
        clip_fraction = float(
            ((ratio.detach()-1.).abs() > self.ppo_clip).float().mean()
        )
        return actor, critic, entropy_loss, feasibility, clip_fraction

    def finish_episode(self, *, complete: bool = True) -> dict:
        """Update on an entire audited trajectory; never optimize partial runs."""
        if not complete:
            raise ValueError("Interrupted native episode cannot enter episodic policy update")
        if not self.pending:
            raise ValueError("Cannot train an episodic policy without audited native steps")
        episodes = self.pending
        returns = _discounted_returns(episodes, self.gamma)
        # PPO uses clipped *on-policy* ratios and fresh graph re-evaluation
        # on each epoch, not multiple backward passes through stale tape graphs.
        repeats = self.ppo_epochs if self.algorithm == "ppo" else 1
        epochs = []
        for epoch in range(repeats):
            parts = (self._ppo_objective(episodes, returns)
                     if self.algorithm == "ppo"
                     else self._objective(episodes, returns))
            actor, critic, entropy, feasibility, clipped = parts
            loss = actor + critic + entropy + .2 * feasibility
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite audited episodic RL loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 5.
            )
            if not bool(torch.isfinite(grad)):
                self.optimizer.zero_grad(set_to_none=True)
                raise ValueError("Nonfinite audited episodic RL gradients")
            self.optimizer.step()
            self.updates += 1
            epochs.append({
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "actor_loss": float(actor.detach()),
                "critic_loss": float(critic.detach()),
                "entropy_loss": float(entropy.detach()),
                "supervised_feasibility_loss": float(feasibility.detach()),
                "gradient_norm_before_clip": float(grad.detach()),
                "ppo_clipped_fraction": clipped if self.algorithm == "ppo" else None,
            })
        self.pending = []
        return {
            "schema": "fpgai-research/hardware-rl-episodic-gradient/v1",
            "algorithm": self.algorithm,
            "audited_steps": len(episodes),
            "qualified_steps": sum(v.qualified for v in episodes),
            "observed_return": sum(v.reward for v in episodes),
            "discounted_first_return": float(returns[0]),
            "gradient_epochs": epochs,
            "total_optimizer_updates": self.updates,
            "original_native_calls_by_optimizer": 0,
            "training_label_from_original_vitis_only": True,
            "source_state_sha256": [v.state_sha256 for v in episodes],
        }


class RandomHardwarePolicy:
    """Policy-neutral real-source baseline, distinct independent NumPy RNG."""
    def __init__(self, *, seed: int = 101):
        if type(seed) is not int or not 0 <= seed <= 2**31:
            raise ValueError("Explicit bounded random baseline seed required")
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def choose(self, observation: dict):
        allowed = np.flatnonzero(observation["action_mask"])
        if not len(allowed):
            raise ValueError("No source-legal architecture proposals")
        chosen = int(self.rng.choice(allowed))
        low, high = (float(v) for v in observation["clock_bounds_ns"])
        clock = normalize_clock_period_ns(
            float(self.rng.uniform(low, high)), decimals=2
        )
        return chosen, clock


def create_hardware_method(method: str, *, seed: int = 101,
                           hidden: int = 64, ppo_epochs: int = 3,
                           ppo_clip: float = .2,
                           message_passing: bool = True,
                           use_trial_history: bool = True):
    """Explicit four-method registry with no plugin loading or fake policies."""
    if method not in METHODS:
        raise ValueError(f"Unsupported hardware RL method: {method!r}")
    if method == "random":
        return RandomHardwarePolicy(seed=seed)
    _require_torch()
    common = dict(
        seed=seed, hidden=hidden,
        message_passing=message_passing, use_trial_history=use_trial_history,
    )
    if method == "actor_critic":
        return HardwareGraphTrainer(**common)
    return HardwareEpisodicTrainer(
        algorithm=method, ppo_epochs=ppo_epochs, ppo_clip=ppo_clip,
        **common,
    )
