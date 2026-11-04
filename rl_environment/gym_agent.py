"""One shared learnable message-passing GNN + actor/critic for online HLS RL.

No pretrained model, proxy latency, or per-network policy. Trainer, not Env,
performs the gradient update after each independently audited Vitis transition.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical, Beta
except ImportError:
    torch, nn, Categorical, Beta = None, None, None, None

from .gym_env import CONTEXT, KNOBS, MAX_NODES, NODE_FEATURES, HISTORY_FEATURES
from .clock_actions import normalize_clock_period_ns


if nn is not None:
    class MessagePass(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.self_layer = nn.Linear(hidden, hidden)
            self.parent_layer = nn.Linear(hidden, hidden)
            self.child_layer = nn.Linear(hidden, hidden)

        def forward(self, x, edges, node_mask):
            # edges[p,c]=1: two distinct learned directed message channels
            parent_norm = edges.sum(0).clamp(min=1).unsqueeze(1)
            child_norm = edges.sum(1).clamp(min=1).unsqueeze(1)
            parents = edges.T @ x / parent_norm
            children = edges @ x / child_norm
            return torch.relu(self.self_layer(x)
                              + self.parent_layer(parents)
                              + self.child_layer(children)) * node_mask[:, None]


    class SharedGraphPolicy(nn.Module):
        def __init__(self, hidden=64, use_message_passing=True):
            super().__init__()
            self.use_message_passing = bool(use_message_passing)
            self.node_in = nn.Linear(NODE_FEATURES, hidden)
            self.mp1 = MessagePass(hidden)
            self.mp2 = MessagePass(hidden)
            self.constraints = nn.Sequential(
                nn.Linear(CONTEXT, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU())
            self.knob = nn.Sequential(nn.Linear(KNOBS, hidden), nn.ReLU())
            self.trial_encoder = nn.Sequential(
                nn.Linear(HISTORY_FEATURES, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU())
            # Architecture-conditioned Beta density: trainable floating target.
            self.clock_head = nn.Sequential(
                nn.Linear(5 * hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 2))
            self.history_arch = nn.Sequential(nn.Linear(2 * KNOBS, hidden), nn.ReLU())
            self.actor = nn.Sequential(
                nn.Linear(5 * hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))
            self.critic = nn.Sequential(
                nn.Linear(4 * hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))
            # Real-HLS-labelled supervised feasibility auxiliary; never a
            # substituted QoR metric, reward or fake Vitis observation.
            self.feasibility_head = nn.Sequential(
                nn.Linear(5 * hidden + 1, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))

        def encode_graph(self, obs):
            mask = torch.as_tensor(obs["node_mask"], dtype=torch.float32)
            nodes = torch.as_tensor(obs["nodes"], dtype=torch.float32)
            edges = torch.as_tensor(obs["edges"], dtype=torch.float32)
            h = torch.relu(self.node_in(nodes)) * mask[:, None]
            if self.use_message_passing:
                h = self.mp1(h, edges, mask)
                h = self.mp2(h, edges, mask)
            graph = h.sum(0) / mask.sum().clamp(min=1)
            return h, graph

        def forward(self, obs, *, with_arch_context=False):
            h, graph = self.encode_graph(obs)
            context = self.constraints(torch.as_tensor(
                obs["global"], dtype=torch.float32))
            features = torch.as_tensor(obs["action_features"], dtype=torch.float32)
            a = features.shape[0]
            knobs = features[:, :MAX_NODES * KNOBS].reshape(a, MAX_NODES, KNOBS)
            projected = self.knob(knobs)
            mask = torch.as_tensor(obs["node_mask"], dtype=torch.float32)
            per_node = projected * h.unsqueeze(0) * mask[None, :, None]
            choice = per_node.sum(1) / mask.sum().clamp(min=1)
            history = torch.cat([
                torch.as_tensor(obs["current_arch"], dtype=torch.float32),
                torch.as_tensor(obs["best_arch"], dtype=torch.float32),
            ], dim=-1)
            mask = torch.as_tensor(obs["node_mask"], dtype=torch.float32)
            search_state = (self.history_arch(history) * mask[:, None]).sum(0) / mask.sum().clamp(min=1)
            raw_trials = torch.as_tensor(obs["history"], dtype=torch.float32)
            trial_mask = torch.as_tensor(obs["history_mask"], dtype=torch.bool)
            if torch.any(trial_mask):
                encoded = self.trial_encoder(raw_trials[trial_mask])
                scores = (encoded * graph.unsqueeze(0)).sum(-1) / (graph.numel() ** .5)
                trial_state = (
                    torch.softmax(scores, dim=0).unsqueeze(-1) * encoded).sum(0)
            else:
                trial_state = torch.zeros_like(graph)
            joint = torch.cat([
                choice, graph.expand(a, -1), context.expand(a, -1),
                search_state.expand(a, -1), trial_state.expand(a, -1)
            ], dim=-1)
            # Learned feasibility prior affects which *real* HLS proposal
            # is sampled. Labels for this head come only from original
            # verified CSim/CSynth outcomes; it never replaces an evaluation
            # and it does not change the latency-only scientific reward.
            clock_params = torch.nn.functional.softplus(self.clock_head(joint)) + 1.001
            mean_clock_fraction = (
                clock_params[:, :1] / clock_params.sum(dim=-1, keepdim=True))
            feasibility_logits = self.feasibility_head(
                torch.cat([joint, mean_clock_fraction], dim=-1)).squeeze(-1)
            logits = self.actor(joint).squeeze(-1) + .5 * feasibility_logits
            action_mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
            logits = logits.masked_fill(~action_mask, -1e9)
            value = self.critic(
                torch.cat([graph, context, search_state, trial_state])).squeeze()
            return (logits, value, joint) if with_arch_context else (logits, value)

else:
    SharedGraphPolicy = None


class OnlineGraphTrainer:
    def __init__(self, *, seed=101, learning_rate=5e-4, hidden=64,
                 use_message_passing=True):
        if torch is None:
            raise RuntimeError("Install PyTorch for the trainable graph policy")
        torch.manual_seed(seed)
        self.seed = seed
        self.policy = SharedGraphPolicy(hidden=hidden,
                                        use_message_passing=use_message_passing)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.updates = 0

    def choose(self, obs, *, training, sample=None):
        if not np.any(obs["action_mask"]):
            raise ValueError("No legal architecture action")
        if sample is None:
            sample = training
        if training or sample:
            # No backward pass in matched frozen stochastic evaluation.
            logits, value, contexts = self.policy(
                obs, with_arch_context=True)
            architecture_dist = Categorical(logits=logits)
            choice = architecture_dist.sample()
            base_logprob = architecture_dist.log_prob(choice)
            entropy = architecture_dist.entropy()
            raw_params = self.policy.clock_head(contexts[choice])
            alpha, beta = torch.nn.functional.softplus(raw_params) + 1.001
            low, high = (float(x) for x in obs["clock_bounds_ns"])
            if low == high:  # Explicit fixed-clock scientific ablation only.
                period = low
            else:
                clock_dist = Beta(alpha, beta)
                raw = clock_dist.sample().clamp(1e-6, 1. - 1e-6)
                period = low + (high - low) * float(raw.item())
                base_logprob = base_logprob + clock_dist.log_prob(raw)
                entropy = entropy + clock_dist.entropy()
            period = normalize_clock_period_ns(period)
            period_fraction = ((period - low) / (high - low)
                               if high > low else 0.)
            clock_feature = torch.tensor([period_fraction], dtype=torch.float32)
            with torch.no_grad():
                feasibility_before_hls = float(torch.sigmoid(
                    self.policy.feasibility_head(torch.cat([
                        contexts[choice].detach(), clock_feature
                    ])).squeeze()).item())
            tape = (base_logprob, entropy, value, contexts[choice],
                    clock_feature, feasibility_before_hls)
            return (int(choice.item()), period), tape if training else None
        with torch.no_grad():
            logits, _, contexts = self.policy(obs, with_arch_context=True)
            choice = logits.argmax()
            low, high = (float(x) for x in obs["clock_bounds_ns"])
            if low == high:
                period = low
            else:
                raw_params = self.policy.clock_head(contexts[choice])
                params = torch.nn.functional.softplus(raw_params) + 1.001
                fraction = float((params[0] / params.sum()).item())
                period = low + (high - low) * fraction
        return (int(choice.item()), normalize_clock_period_ns(period)), None

    def update(self, tape, reward, *, next_obs=None, terminated=False,
               gamma=0.95, return_metrics=False, qualified=None):
        if tape is None:
            raise ValueError("Frozen evaluation must never update gradients")
        if not np.isfinite(reward) or not 0 <= gamma < 1:
            raise ValueError("Invalid RL reward/discount")
        if len(tape) == 6:
            (logprob, entropy, value, selected_arch, selected_clock,
             feasibility_before_hls) = tape
        elif len(tape) == 5:
            logprob, entropy, value, selected_arch, selected_clock = tape
            feasibility_before_hls = None
        else:
            logprob, entropy, value = tape
            selected_arch = selected_clock = None
            feasibility_before_hls = None
        target = float(reward)
        if not terminated and next_obs is not None and np.any(next_obs["action_mask"]):
            with torch.no_grad():
                _, next_value = self.policy(next_obs)
                target += gamma * float(next_value.item())
        target = torch.tensor(target, dtype=torch.float32)
        advantage = target - value.detach()
        actor_loss = -logprob * advantage
        critic_loss = .5 * (value - target).square()
        entropy_loss = -.001 * entropy
        # This auxiliary classifier learns exclusively from actual original
        # Vitis qualification labels. It helps the shared graph/history
        # representation learn from negative outcomes, but DOES NOT modify
        # the scientific latency reward or use synthetic HLS measurements.
        supervised_loss = torch.zeros((), dtype=torch.float32)
        if qualified is not None:
            if type(qualified) is not bool or selected_arch is None:
                raise ValueError("Feasibility label requires a real native action tape")
            predicted = self.policy.feasibility_head(
                torch.cat([selected_arch, selected_clock])).squeeze()
            supervised_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                predicted, torch.tensor(float(qualified)),
                pos_weight=torch.tensor(8.))
        loss = actor_loss + critic_loss + entropy_loss + .2 * supervised_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), 5.)
        self.optimizer.step()
        self.updates += 1
        if return_metrics:
            # Diagnostic scalars only; no change to policy, reward or updates.
            return {
                "schema": "fpgai-gym-policy-update-diagnostics/v1",
                "loss_total": float(loss.detach().item()),
                "loss_actor": float(actor_loss.detach().item()),
                "loss_critic": float(critic_loss.detach().item()),
                "loss_entropy_term": float(entropy_loss.detach().item()),
                "loss_auxiliary_feasibility": float(
                    supervised_loss.detach().item()),
                "auxiliary_feasibility_weight": 0.2,
                "auxiliary_positive_label_weight": 8.0,
                "native_feasibility_label": qualified,
                "feasibility_prediction_before_hls": feasibility_before_hls,
                "feasibility_brier_score": (
                    (feasibility_before_hls - float(qualified)) ** 2
                    if qualified is not None and
                    feasibility_before_hls is not None else None),
                "policy_entropy": float(entropy.detach().item()),
                "log_probability": float(logprob.detach().item()),
                "value_prediction": float(value.detach().item()),
                "discounted_td_target": float(target.detach().item()),
                "advantage": float(advantage.detach().item()),
                "gradient_l2_before_clip": float(
                    gradient_norm_before_clip.detach().item()),
                "training_updates": self.updates,
            }
        return float(loss.detach().item())

    def save_private(self, directory, metadata):
        """Create-only checkpoint; no external pretrained weights."""
        dest = Path(directory)
        dest.mkdir(parents=True, exist_ok=False)
        tensors = dest / "shared_gnn_policy.npz"
        np.savez(tensors, **{k: v.detach().cpu().numpy()
                            for k, v in self.policy.state_dict().items()})
        # The optimizer is resumable only from trusted private checkpoints.
        state = dest / "optimizer_private.pt"
        torch.save(self.optimizer.state_dict(), state)
        with (dest / "manifest.json").open("x") as stream:
            json.dump({
                **metadata, "schema": "fpgai-gym-online-checkpoint/v1",
                "source": "from-scratch-online-original-vitis-hls",
                "seed": self.seed, "online_updates": self.updates,
                "message_passing_enabled": self.policy.use_message_passing,
                "policy_sha256": sha256(tensors.read_bytes()).hexdigest(),
                "optimizer_sha256": sha256(state.read_bytes()).hexdigest(),
                "private": True,
            }, stream, sort_keys=True, indent=2)

    def load_private(self, directory, *, load_optimizer=True):
        """Load only locally generated, trusted in-project checkpoints."""
        dest = Path(directory)
        meta = json.loads((dest / "manifest.json").read_text())
        weights = dest / "shared_gnn_policy.npz"
        if meta.get("message_passing_enabled") != self.policy.use_message_passing:
            raise ValueError("GNN ablation checkpoint architecture mismatch")
        if meta.get("source") != "from-scratch-online-original-vitis-hls":
            raise ValueError("External pretrained model is prohibited")
        if sha256(weights.read_bytes()).hexdigest() != meta["policy_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        with np.load(weights, allow_pickle=False) as archive:
            state = {k: torch.tensor(archive[k]) for k in self.policy.state_dict()}
        self.policy.load_state_dict(state, strict=True)
        if load_optimizer:
            optim = dest / "optimizer_private.pt"
            if sha256(optim.read_bytes()).hexdigest() != meta["optimizer_sha256"]:
                raise ValueError("Optimizer checkpoint hash mismatch")
            self.optimizer.load_state_dict(torch.load(
                optim, map_location="cpu", weights_only=True))
        self.updates = int(meta["online_updates"])
        return meta
