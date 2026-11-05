"""HW-RL-04: one shared GNN actor/critic for the verified hardware-native Gym.

The old shared GNN's directed MessagePass primitive is reused, but the
four-operator action/context heads cannot consume the new typed,
mixed-Conv/MatMul/Gemm tensor/operator graph. This is an explicit new
observation adapter, not a new compiler emitter or QoR oracle.

No pretrained weights, cached HLS reward, simulated vendor response,
physical Fmax, or automatic native campaign. Updates are allowed only
after an independently audited original Vitis Gym transition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import numpy as np

_TORCH_IMPORT_ERROR = None
try:
    import torch
    from torch import nn
    from torch.distributions import Beta, Categorical
except (ImportError, OSError) as exc:
    # Distinguish a missing wheel from a broken torch/shared-library import.
    # Neither may silently fall back to an untrained or synthetic policy.
    _TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    torch, nn, Beta, Categorical = None, None, None, None

from ..clock_actions import CLOCK_MAX_NS, CLOCK_MIN_NS, normalize_clock_period_ns
from ..environments.hardware_gym import (
    ACTION_FEATURES, GLOBAL_FEATURES, HISTORY_FEATURES, MAX_ACTIONS,
    MAX_HISTORY, MAX_NODES, NODE_FEATURES,
)

if nn is not None:
    # Reuse the *existing* directed graph message-passing primitive;
    # do not create a competing second graph convolution implementation.
    from ..gym_agent import MessagePass

    class SharedHardwareGraphPolicy(nn.Module):
        """Shared variable-size GNN, attention over actual trials, joint actor.

        Tensor and operator nodes are both represented, with directed
        produces/consumes edges and per-operator candidate target embeddings.
        The policy predicts action preference and a conditional continuous
        Beta distribution for the requested clock. It never predicts the
        HLS latency used as an environment reward.
        """
        def __init__(self, hidden: int = 64, *, message_passing: bool = True,
                     use_trial_history: bool = True):
            super().__init__()
            if type(hidden) is not int or not 8 <= hidden <= 512:
                raise ValueError("Explicit bounded hidden dimension required")
            self.hidden = hidden
            self.message_passing = bool(message_passing)
            self.use_trial_history = bool(use_trial_history)
            self.node_in = nn.Linear(NODE_FEATURES, hidden)
            self.mp1 = MessagePass(hidden)
            self.mp2 = MessagePass(hidden)
            self.task_in = nn.Sequential(nn.Linear(GLOBAL_FEATURES, hidden),
                                         nn.ReLU(), nn.Linear(hidden, hidden))
            self.action_in = nn.Sequential(nn.Linear(ACTION_FEATURES, hidden),
                                           nn.ReLU(), nn.Linear(hidden, hidden))
            self.history_in = nn.Sequential(nn.Linear(HISTORY_FEATURES, hidden),
                                            nn.ReLU(), nn.Linear(hidden, hidden))
            self.history_query = nn.Linear(2 * hidden, hidden)
            self.actor = nn.Sequential(nn.Linear(5 * hidden, hidden),
                                       nn.ReLU(), nn.Linear(hidden, 1))
            self.clock_head = nn.Sequential(nn.Linear(5 * hidden, hidden),
                                            nn.ReLU(), nn.Linear(hidden, 2))
            self.critic = nn.Sequential(nn.Linear(3 * hidden, hidden),
                                        nn.ReLU(), nn.Linear(hidden, 1))
            self.feasibility_head = nn.Sequential(
                nn.Linear(5 * hidden + 1, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def _graph(self, obs):
            mask = torch.as_tensor(obs["node_mask"], dtype=torch.float32)
            nodes = torch.as_tensor(obs["nodes"], dtype=torch.float32)
            if nodes.shape != (MAX_NODES, NODE_FEATURES):
                raise ValueError("Unexpected hardware GNN node tensor shape")
            if not torch.any(mask):
                raise ValueError("Empty hardware graph cannot be encoded")
            edges = torch.as_tensor(obs["edge_index"], dtype=torch.long)
            edge_mask = torch.as_tensor(obs["edge_mask"], dtype=torch.bool)
            selected = edges[edge_mask]
            if selected.numel() and (torch.any(selected < 0)
                                     or torch.any(selected >= MAX_NODES)):
                raise ValueError("Out-of-bounds graph edges")
            if selected.numel() and (torch.any(mask[selected[:, 0]] == 0)
                                     or torch.any(mask[selected[:, 1]] == 0)):
                raise ValueError("Edges reference padded graph nodes")
            # adjacency[p, c] matches the original MessagePass semantics:
            # directed producer/consumer messages remain separate channels.
            adjacency = torch.zeros((MAX_NODES, MAX_NODES), dtype=torch.float32)
            if selected.numel():
                adjacency.index_put_(
                    (selected[:, 0], selected[:, 1]),
                    torch.ones(selected.shape[0], dtype=torch.float32),
                    accumulate=True,
                )
                adjacency.clamp_(max=1.)
            x = torch.relu(self.node_in(nodes)) * mask[:, None]
            if self.message_passing:
                x = self.mp1(x, adjacency, mask)
                x = self.mp2(x, adjacency, mask)
            pooled = x.sum(0) / mask.sum().clamp(min=1)
            return x, pooled

        def _history(self, obs, pooled, task):
            if not self.use_trial_history:
                return torch.zeros_like(pooled)
            rows = torch.as_tensor(obs["history"], dtype=torch.float32)
            mask = torch.as_tensor(obs["history_mask"], dtype=torch.bool)
            if rows.shape != (MAX_HISTORY, HISTORY_FEATURES):
                raise ValueError("Unexpected native trial history features")
            if not bool(mask.any()):
                return torch.zeros_like(pooled)
            encoded = self.history_in(rows[mask])
            query = self.history_query(torch.cat((pooled, task)))
            weights = torch.softmax(
                (encoded @ query) / self.hidden ** .5, dim=0
            )
            return (weights[:, None] * encoded).sum(0)

        def forward(self, obs, *, with_context=False):
            x, graph = self._graph(obs)
            task = self.task_in(torch.as_tensor(
                obs["global"], dtype=torch.float32
            ))
            past = self._history(obs, graph, task)
            features = torch.as_tensor(obs["action_features"],
                                       dtype=torch.float32)
            mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
            target_nodes = torch.as_tensor(obs["action_target_node"],
                                           dtype=torch.long)
            if features.shape != (MAX_ACTIONS, ACTION_FEATURES):
                raise ValueError("Unexpected hardware action feature shape")
            if mask.shape != (MAX_ACTIONS,) or not bool(mask.any()):
                raise ValueError("At least one real source-legal action required")
            if target_nodes.shape != (MAX_ACTIONS,) or bool(
                    torch.any(target_nodes < -1) or
                    torch.any(target_nodes >= MAX_NODES)):
                raise ValueError("Action refers to an invalid operator target node")
            real_target = target_nodes >= 0
            if torch.any(real_target & (torch.as_tensor(
                    obs["node_mask"], dtype=torch.bool
            )[target_nodes.clamp(min=0)] == 0)):
                raise ValueError("Action targets a padded/unavailable graph node")
            targeted = x[target_nodes.clamp(min=0)] * real_target[:, None]
            candidate = self.action_in(features)
            n = candidate.shape[0]
            joint = torch.cat((
                candidate,
                targeted,
                graph.expand(n, -1),
                task.expand(n, -1),
                past.expand(n, -1),
            ), dim=-1)
            bounds = np.asarray(obs["clock_bounds_ns"], dtype=np.float32)
            if bounds.shape != (2,) or not bool(np.isfinite(bounds).all()):
                raise ValueError("Nonfinite or invalid hardware clock bounds")
            if float(bounds[0]) == float(bounds[1]):
                # Fixed-clock ablation must not leak clock-head gradients
                # through the auxiliary feasibility-conditioned actor.
                predicted_clock = torch.full(
                    (joint.shape[0], 1), .5, dtype=joint.dtype,
                    device=joint.device,
                )
            else:
                params = torch.nn.functional.softplus(self.clock_head(joint)) + 1.01
                predicted_clock = (params[:, :1] / params.sum(-1, keepdim=True))
            # Feasibility auxiliary influences *selection only*, and learns
            # exclusively from audited E3 labels after real vendor steps.
            feasibility_logits = self.feasibility_head(
                torch.cat((joint, predicted_clock), dim=-1)
            ).squeeze(-1)
            logits = self.actor(joint).squeeze(-1) + .5 * feasibility_logits
            logits = logits.masked_fill(~mask, float("-inf"))
            value = self.critic(torch.cat((graph, task, past))).squeeze(-1)
            if not bool(torch.isfinite(logits[mask]).all()) or not bool(
                    torch.isfinite(value)):
                raise ValueError("Nonfinite real-source policy logits/value")
            return (logits, value, joint) if with_context else (logits, value)
else:
    SharedHardwareGraphPolicy = None


def _require_torch() -> None:
    """Fail before touching a model or native output when RL extras are absent."""
    if torch is not None:
        return
    reason = f" Import failure: {_TORCH_IMPORT_ERROR}." if _TORCH_IMPORT_ERROR else ""
    raise RuntimeError(
        "PyTorch is required for the hardware GNN agent." + reason
        + " Install RL dependencies in this virtualenv with "
        + "python -m pip install -e '.[rl,onnx]' (without --no-deps), "
        + "or install a CPU-only PyTorch wheel from "
        + "https://download.pytorch.org/whl/cpu."
    )


@dataclass(frozen=True)
class HardwareActionTape:
    log_probability: object
    entropy: object
    value: object
    selected_context: object
    requested_clock_fraction: object
    predicted_feasibility: float
    source_action_index: int
    sampled_clock_fraction: object | None = None
    observation: dict | None = None
    # Previous source-only rejected draws are part of the probability
    # of the accepted native step. They have no vendor reward/label of
    # their own, but must enter the joint score-function likelihood.
    rejected_proposal_tapes: tuple["HardwareActionTape", ...] = ()


def _native_feedback(info: dict, reward: float) -> bool:
    """Strict policy update gate for a previously verified Gym step receipt.

    This does NOT replace the environment's independent re-audit. Mock
    receipts may exercise the trainer in CPU tests, but must never be
    published as real vendor observations.
    """
    if not isinstance(info, dict) or info.get("schema") != (
            "fpgai-research/rl-hardware-gym-native/v1"):
        raise ValueError("Missing original-native Gym transition receipt")
    audit = info.get("original_vitis_audit")
    if not isinstance(audit, dict) or any((
        audit.get("schema") != "fpgai-research/rl-hardware-native-audit/v1",
        audit.get("state_sha256") != info.get("state_sha256"),
        audit.get("qualified") is not info.get("qualified"),
        audit.get("native_invocation_present") is not True,
        audit.get("native_calls_by_audit") != 0,
        audit.get("original_vitis_completed_successfully") is not True,
        audit.get("csim_passed") is not True,
        audit.get("evidence") != "E3-HLS-synthesis-report",
        type(info.get("qualified")) is not bool,
        type(info.get("native_vitis_calls")) is not int,
        info.get("native_vitis_calls") <= 0 if type(
            info.get("native_vitis_calls")) is int else True,
        type(reward) not in (float, int) or not isfinite(reward),
    )):
        raise ValueError("Unattributed or incomplete original HLS feedback")
    if info["qualified"]:
        latency = audit.get("hls_estimated_latency_ns")
        cycles = audit.get("hls_worst_case_cycles")
        if (type(latency) not in (int, float) or not isfinite(latency)
                or latency <= 0 or type(cycles) is not int or cycles <= 0
                or audit.get("gate_failures") != []):
            raise ValueError("Qualified policy label lacks audited HLS latency")
    else:
        if (audit.get("hls_estimated_latency_ns") is not None
                or not audit.get("gate_failures") or reward != -.05):
            raise ValueError("Infeasible policy update is not an audited HLS rejection")
    return info["qualified"]


def verify_original_hardware_step(
        tape: HardwareActionTape, reward: float, *,
        info: dict, env) -> bool:
    """Replay the matching *live* original Vitis case before any algorithm learns.

    This is a shared hard gate for actor-critic, REINFORCE and PPO. Metadata
    alone cannot authorize an update, and this never launches vendor tools.
    """
    qualified = _native_feedback(info, reward)
    # Require the original Gym instance and re-audit the immutable
    # original vendor case. A caller-supplied metadata dict (even if
    # structurally E3-like) is NEVER sufficient to update weights.
    from ..infrastructure import hardware_native
    from ..environments.hardware_gym import GymHardwareNativeEnv
    if (not isinstance(env, GymHardwareNativeEnv)
            or env.current is None or env.calls == 0
            or env.current.state_sha256 != info["state_sha256"]
            or env.last_audit != info["original_vitis_audit"]
            or not isinstance(info.get("native_case"), str)
            or not env.history
            or env.history[-1]["audit"] != info["original_vitis_audit"]
            or env.history[-1]["action_index"] != tape.source_action_index
            or abs(float(env.history[-1]["reward"]) - float(reward)) > 1e-10):
        raise ValueError("Policy update needs the matching live native Gym environment and exact reward")
    reproduced = hardware_native.audit_native_case(
        info["native_case"], env.current, env.samples
    )
    if reproduced != info["original_vitis_audit"]:
        raise ValueError("Independent original Vitis replay differs before gradient update")
    return qualified




class HardwareGraphTrainer:
    """One shared policy across models, source-masked categorical + Beta clock.

    The trainer has no vendor dependency of its own. A caller must first
    obtain a real return from GymHardwareNativeEnv.step() and provide its
    exact reward and audit info; policy weights cannot learn from planning.
    """
    def __init__(self, *, seed: int = 101, hidden: int = 64,
                 lr: float = 5e-4, message_passing: bool = True,
                 use_trial_history: bool = True):
        _require_torch()
        if type(seed) is not int or not 0 <= seed <= 2**31:
            raise ValueError("Explicit bounded shared-policy seed required")
        if type(lr) not in (float, int) or not isfinite(lr) or lr <= 0:
            raise ValueError("Positive finite actor-critic learning rate required")
        torch.manual_seed(seed)
        self.seed = seed
        self.policy = SharedHardwareGraphPolicy(
            hidden, message_passing=message_passing,
            use_trial_history=use_trial_history,
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(lr))
        self.updates = 0

    def choose(self, obs: dict, *, training: bool,
               sample: bool | None = None):
        if sample is None:
            sample = training
        logits, value, contexts = self.policy(obs, with_context=True)
        if sample:
            dist = Categorical(logits=logits)
            chosen = dist.sample()
            base_logprob = dist.log_prob(chosen)
            entropy = dist.entropy()
            low, high = [float(x) for x in obs["clock_bounds_ns"]]
            if high == low:
                # Discrete-only fixed-clock policy; do not sample a dummy Beta
                # latent or include its log-probability in a gradient tape.
                clock = normalize_clock_period_ns(low, decimals=2)
                fraction = torch.tensor([.5], dtype=torch.float32)
                latent = None
                log_probability = base_logprob
            else:
                params = torch.nn.functional.softplus(
                    self.policy.clock_head(contexts[chosen])
                ) + 1.01
                beta = Beta(params[0], params[1])
                # The raw continuous latent is preserved for source-based
                # PPO replay. Native clock periods are canonical to .01ns.
                latent = beta.sample().clamp(1e-6, 1.-1e-6)
                clock = normalize_clock_period_ns(
                    low + (high-low)*float(latent.item()), decimals=2
                )
                fraction = torch.tensor([(clock-low)/(high-low)],
                                        dtype=torch.float32)
                entropy = entropy + beta.entropy()
                log_probability = base_logprob + beta.log_prob(latent)
            with torch.no_grad():
                pred = float(torch.sigmoid(self.policy.feasibility_head(
                    torch.cat((contexts[chosen].detach(), fraction), dim=-1)
                ).squeeze()).item())
            tape = HardwareActionTape(
                log_probability, entropy, value, contexts[chosen],
                fraction, pred, int(chosen.item()),
                sampled_clock_fraction=(
                    latent.detach() if latent is not None else None
                ),
                observation={
                    k: np.array(v, copy=True) for k, v in obs.items()
                },
            ) if training else None
            return (int(chosen.item()), clock), tape
        with torch.no_grad():
            chosen = int(logits.argmax().item())
            low, high = [float(x) for x in obs["clock_bounds_ns"]]
            if high == low:
                clock = normalize_clock_period_ns(low, decimals=2)
            else:
                params = torch.nn.functional.softplus(
                    self.policy.clock_head(contexts[chosen])
                ) + 1.01
                fraction = float((params[0]/params.sum()).item())
                clock = normalize_clock_period_ns(
                    low + (high-low)*fraction, decimals=2
                )
        return (chosen, clock), None

    def ranked_frozen_actions(self, obs: dict) -> tuple[tuple[int, float], ...]:
        """Deterministic greedy candidates in actor-score order, no sampling.

        Each candidate keeps its *own* conditional Beta mean clock. Callers
        may select the highest-ranked source-novel candidate without changing
        weights, sampling a hidden random fallback, or spending a vendor call.
        The sorted fallback rank must be reported in evaluation receipts.
        """
        with torch.no_grad():
            logits, _, contexts = self.policy(obs, with_context=True)
            mask = np.asarray(obs["action_mask"], dtype=bool)
            allowed = np.flatnonzero(mask).tolist()
            ranked = sorted(allowed, key=lambda i: (-float(logits[i]), i))
            low, high = [float(x) for x in obs["clock_bounds_ns"]]
            result = []
            for index in ranked:
                if high == low:
                    clock = normalize_clock_period_ns(low, decimals=2)
                else:
                    params = torch.nn.functional.softplus(
                        self.policy.clock_head(contexts[index])
                    ) + 1.01
                    fraction = float((params[0] / params.sum()).item())
                    clock = normalize_clock_period_ns(
                        low + (high - low) * fraction, decimals=2
                    )
                result.append((index, clock))
            return tuple(result)

    def update(self, tape: HardwareActionTape, reward: float, *,
               info: dict, env, next_obs: dict | None = None,
               terminated: bool = True, gamma: float = .95) -> dict:
        if not isinstance(tape, HardwareActionTape):
            raise ValueError("Policy update requires a prior trainable sampled tape")
        qualified = verify_original_hardware_step(
            tape, reward, info=info, env=env
        )
        if type(gamma) not in (float, int) or not 0 <= gamma < 1:
            raise ValueError("Actor-critic TD discount must be within [0,1)")
        if not terminated and next_obs is None:
            raise ValueError("Nonterminal update requires the new audited observation")
        target = float(reward)
        if not terminated:
            with torch.no_grad():
                _, next_value = self.policy(next_obs)
                target += gamma * float(next_value.item())
        target_tensor = torch.tensor(target, dtype=torch.float32)
        advantage = target_tensor - tape.value.detach()
        actor_loss = -tape.log_probability * advantage
        critic_loss = .5 * (tape.value-target_tensor).square()
        entropy_loss = -.001 * tape.entropy
        supervised_logits = self.policy.feasibility_head(
            torch.cat((tape.selected_context, tape.requested_clock_fraction),
                      dim=-1)
        ).squeeze(-1)
        supervised = torch.nn.functional.binary_cross_entropy_with_logits(
            supervised_logits, torch.tensor(float(qualified)),
            pos_weight=torch.tensor(8.),
        )
        loss = actor_loss + critic_loss + entropy_loss + .2 * supervised
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite policy loss; no gradient update allowed")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before_clip = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), 5.
        )
        if not bool(torch.isfinite(before_clip)):
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("Nonfinite policy gradients; no optimizer update")
        self.optimizer.step()
        self.updates += 1
        return {
            "schema": "fpgai-research/rl-hardware-online-gradient/v1",
            "original_audit_state_sha256": info["state_sha256"],
            "original_native_calls": info["native_vitis_calls"],
            "qualified": qualified,
            "hls_estimated_latency_ns":
                info["original_vitis_audit"]["hls_estimated_latency_ns"],
            "reward": reward,
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "entropy_loss": float(entropy_loss.detach()),
            "supervised_feasibility_loss": float(supervised.detach()),
            "gradient_norm_before_clip": float(before_clip.detach()),
            "shared_policy_updates": self.updates,
            "no_new_vendor_calls_by_trainer": True,
            "independent_read_only_vendor_case_replay": True,
            "weights_pretrained": False,
        }

    def save_private(self, directory: str | Path, metadata: dict):
        """Export tensor-only, no pretrained model; no unsafe pickle loading."""
        dest = Path(directory)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Original private policy checkpoint cannot be overwritten")
        if not isinstance(metadata, dict):
            raise TypeError("Source/task provenance metadata required for checkpoint")
        dest.mkdir(parents=True, exist_ok=False)
        weights = dest / "shared_hardware_gnn.npz"
        np.savez(weights, **{
            k: v.detach().cpu().numpy()
            for k, v in self.policy.state_dict().items()
        })
        with (dest / "manifest.json").open("x") as f:
            json.dump({
                **metadata, "schema": "fpgai-research/rl-hardware-policy/v1",
                "seed": self.seed, "online_updates": self.updates,
                "trained_on_original_vitis_hls": self.updates > 0,
                "private_weights": True,
                "pretrained": False,
                "message_passing": self.policy.message_passing,
                "trial_history": self.policy.use_trial_history,
                "weights_sha256": sha256(weights.read_bytes()).hexdigest(),
                "optimizer_state_included": False,
                "full_training_resume_supported": False,
            }, f, sort_keys=True, indent=2)

    def load_private(self, directory: str | Path, *,
                     load_optimizer: bool = True):
        """Safe tensor-only evaluation restore; cannot claim exact TD resume."""
        if load_optimizer:
            raise ValueError(
                "Exact training resume requires original optimizer state; "
                "this tensor-only checkpoint supports evaluation only"
            )
        dest = Path(directory)
        manifest = json.loads((dest / "manifest.json").read_text())
        path = dest / "shared_hardware_gnn.npz"
        if (manifest.get("schema") != "fpgai-research/rl-hardware-policy/v1"
                or manifest.get("pretrained") is not False
                or manifest.get("message_passing") != self.policy.message_passing
                or manifest.get("trial_history") != self.policy.use_trial_history
                or sha256(path.read_bytes()).hexdigest() != manifest.get("weights_sha256")):
            raise ValueError("Private original-policy checkpoint provenance mismatch")
        with np.load(path, allow_pickle=False) as saved:
            expected = self.policy.state_dict()
            if set(saved.files) != set(expected):
                raise ValueError("Checkpoint tensor names differ from shared GNN")
            values = {
                k: torch.tensor(saved[k], dtype=expected[k].dtype)
                for k in expected
            }
        self.policy.load_state_dict(values, strict=True)
        self.updates = int(manifest["online_updates"])
        return manifest



def original_hls_diagnostics(audit: dict) -> dict:
    """Read-only presentation of already independently audited evidence.

    Never infer latency or timing qualification from raw cycle counts.
    The original audit remains the only authority for the result.
    """
    fields = (
        "gate_failures", "csim_passed", "csim_max_absolute_error",
        "raw_hls_worst_case_cycles", "requested_clock_ns", "hls_timing",
        "original_hls_resource_consumed", "resource_caps",
    )
    return {key: audit[key] for key in fields}


def main(argv=None) -> int:
    """Source-only shared GNN preview or exactly ONE original Vitis online step.

    No multi-episode campaign, hidden pretrained weights, synthetic reward,
    or disk output in default preview. Native execution and the first
    gradient update require --execute-native and a new experiment root.
    """
    import argparse
    import sys
    from ..environments.hardware_gym import GymHardwareNativeEnv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--resource-caps", required=True, type=int, nargs=5,
                        metavar="N", help="LUT FF DSP BRAM18K URAM")
    parser.add_argument("--max-absolute-error", type=float, required=True)
    parser.add_argument("--initial-clock-ns", type=float, default=14.)
    parser.add_argument("--max-parallel-units", type=int, default=8)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--disable-message-passing", action="store_true")
    parser.add_argument("--disable-trial-history", action="store_true")
    parser.add_argument("--execute-native", action="store_true",
                        help="Explicit one original Vitis call and one gradient update")
    parser.add_argument("--checkpoint",
                        help="Optional fresh private tensor-only export after verified step")
    parser.add_argument("--vitis", default="vitis_hls")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    try:
        if args.checkpoint and not args.execute_native:
            raise ValueError("Cannot save a training checkpoint from source-only preview")
        _require_torch()
        from fpgai_research.frontend.onnx import load_onnx
        from ..infrastructure.hardware_native import _regular_file
        model = load_onnx(args.onnx)
        cohort = np.load(_regular_file(Path(args.samples).expanduser()),
                         allow_pickle=False)
        environment = GymHardwareNativeEnv(
            model, cohort, args.output_root, part=args.part,
            resource_caps=tuple(args.resource_caps),
            max_absolute_error=args.max_absolute_error,
            initial_clock_ns=args.initial_clock_ns,
            max_parallel_units=args.max_parallel_units,
            max_vitis_calls=1,
            execute_native=bool(args.execute_native),
            vitis_executable=args.vitis,
            timeout_seconds=args.timeout,
        )
        agent = HardwareGraphTrainer(
            seed=args.seed, hidden=args.hidden,
            message_passing=not args.disable_message_passing,
            use_trial_history=not args.disable_trial_history,
        )
        observation, status = environment.reset(seed=args.seed)
        action, tape = agent.choose(
            observation, training=bool(args.execute_native),
            sample=bool(args.execute_native),
        )
        chosen_edit = environment.offers[action[0]]
        out = {
            "schema": "fpgai-research/rl-hardware-gnn-one-step/v1",
            "status": "SHARED_GNN_SOURCE_ONLY_PREVIEW",
            "seed": args.seed,
            "part": args.part,
            "initial_state_sha256": status["state_sha256"],
            "selected_source_action_index": action[0],
            "selected_source_action": asdict(chosen_edit) if chosen_edit else None,
            "requested_clock_ns": action[1],
            "source_mask_is_not_native_feasibility": True,
            "original_vitis_calls": 0,
            "online_gradient_updates": agent.updates,
            "reward": None,
            "training_started": False,
            "pretrained": False,
        }
        if args.execute_native:
            next_obs, reward, terminated, truncated, info = environment.step(action)
            metrics = agent.update(
                tape, reward, info=info, env=environment,
                next_obs=None, terminated=True,
            )
            out.update({
                "status": "ONE_ORIGINAL_VITIS_STEP_AND_GRADIENT_AUDITED",
                "original_vitis_calls": info["native_vitis_calls"],
                "qualified": info["qualified"],
                "hls_estimated_latency_ns": info["hls_estimated_latency_ns"],
                "native_case": info["native_case"],
                "original_native_state_sha256": info["state_sha256"],
                "reward": reward,
                "online_gradient_updates": agent.updates,
                "gradient_receipt": metrics,
                "original_hls_diagnostics": original_hls_diagnostics(
                    info["original_vitis_audit"]
                ),
                "training_started": True,
                "private_weights_saved": bool(args.checkpoint),
                "terminated": terminated, "truncated": truncated,
            })
            if args.checkpoint:
                agent.save_private(args.checkpoint, {
                    "source_state_sha256": info["state_sha256"],
                    "native_case": info["native_case"],
                    "native_gym_root": str(Path(args.output_root).absolute()),
                    "training_mode": "one_step_original_vitis",
                })
                out["private_checkpoint"] = str(Path(args.checkpoint).absolute())
        print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Hardware GNN stopped: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
