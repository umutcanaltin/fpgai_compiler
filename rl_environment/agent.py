"""Shared task-conditioned online actor-critic for whole configurations.

One model across every network; training updates happen only AFTER a real
audited Vitis step. No offline dataset, no pretraining, no proxy reward.
"""
from __future__ import annotations
from math import log1p, prod
import numpy as np

from fpgai_research.architecture.resolver import iteration_shape

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError:
    torch=None
    nn=None
    Categorical=None


def _require_torch():
    if torch is None:
        raise RuntimeError("Online graph DRL requires the optional PyTorch dependency")


def encode(task,catalog,choices,best_cycles,calls,last_result=None):
    """True Model IR graph + current compiler-legal proposed complete design."""
    nodes=[]
    for op,indices in zip(task.model.operators,choices):
        shape=iteration_shape(task.model,op)
        nodes.append([
            log1p(prod(shape))/20.0,
            log1p(shape[1])/12.0,
            log1p(shape[2])/12.0,
            float(op.kind=="MatMul"),float(op.kind=="Gemm"),
            *[index/max(1,len(catalog.options[len(nodes)][i])-1)
              for i,index in enumerate(indices)],
        ])
    features=np.asarray(nodes,dtype=np.float32)
    # Graph pooling works on arbitrary supported operator counts. Retain
    # actual producer connectivity through directed mean aggregation.
    producer={t:i for i,op in enumerate(task.model.operators) for t in op.outputs}
    neighbors=[]
    for i,op in enumerate(task.model.operators):
        upstream=[producer[t] for t in op.inputs if t in producer]
        neighbors.append(features[upstream].mean(axis=0) if upstream
                         else np.zeros(features.shape[1],np.float32))
    context=np.asarray([
        log1p(task.model.tensor(task.model.inputs[0]).shape[-1])/12.0,
        log1p(task.max_absolute_error)/12.0,
        log1p(task.clock_ns)/12.0,
        *[log1p(value)/20.0 for value in task.resources],
        log1p(calls)/8.0,
        log1p(best_cycles)/15.0 if best_cycles is not None else 0.0,
        float(best_cycles is not None),
    ],dtype=np.float32)
    prior=last_result or {}
    resources=prior.get("hls_estimated_resources") or {}
    cycles=prior.get("hls_worst_case_cycles")
    error=prior.get("csim_max_absolute_error")
    seconds=prior.get("vendor_seconds")
    feedback=[
        float(last_result is not None),
        float(prior.get("qualified") is True),
        float(cycles is not None),
        log1p(cycles)/15.0 if cycles is not None else 0.0,
        float(error is not None),
        log1p(error)/8.0 if error is not None else 0.0,
        float(seconds is not None),
        log1p(seconds)/15.0 if seconds is not None else 0.0,
    ]
    for name in ("lut","ff","dsp","bram_18k","uram"):
        value=resources.get(name)
        feedback.extend((
            float(value is not None),
            log1p(value)/20.0 if value is not None else 0.0,
        ))
    # 19 feedback features: status (2), latency (2), numeric (2),
    # vendor-time (2), five resource presence/value pairs (10), evidence (1).
    feedback.append(float(prior.get("evidence")=="E3-HLS-synthesis-report"))
    row=np.concatenate((features.mean(axis=0),np.asarray(neighbors).mean(axis=0),
                        context,np.asarray(feedback,dtype=np.float32)))
    if row.shape!=(56,) or not np.isfinite(row).all():
        raise ValueError("Unexpected source-derived graph/context dimension")
    return row


if torch is not None:
    class OnlineGraphActorCritic(nn.Module):
        def __init__(self,hidden=64):
            super().__init__()
            self.feature=nn.Sequential(nn.Linear(56,hidden),nn.ReLU(),
                                       nn.Linear(hidden,hidden),nn.ReLU())
            self.score=nn.Linear(hidden,1)
            self.value=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),
                                     nn.Linear(hidden,1))

        def forward(self,rows):
            embedding=self.feature(rows)
            return self.score(embedding).flatten(),self.value(
                embedding.mean(dim=0)).squeeze()
else:
    OnlineGraphActorCritic=None


class OnlineAgent:
    def __init__(self,*,seed=101,learning_rate=0.0005):
        _require_torch()
        torch.manual_seed(seed)
        self.policy=OnlineGraphActorCritic()
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=learning_rate)
        self.updates=0

    def choose(self,task,catalog,candidates,best_cycles,calls,*,training,last_result=None):
        if not candidates:
            raise ValueError("No legal fresh hardware configurations")
        feature=np.stack([
            encode(task,catalog,choices,best_cycles,calls,last_result)
            for choices in candidates])
        x=torch.tensor(feature,dtype=torch.float32)
        logits,value=self.policy(x)
        if not torch.isfinite(logits).all():
            raise ValueError("Invalid online policy logits")
        if training:
            distribution=Categorical(logits=logits)
            selected=distribution.sample()
            return candidates[int(selected)],(distribution.log_prob(selected),
                                               distribution.entropy(),value)
        # No gradients, exploration, or updates for held-out evaluation.
        with torch.no_grad():
            selected=int(logits.argmax())
        return candidates[selected],None

    def estimate_next_value(self,task,catalog,candidates,best_cycles,calls,
                            *,last_result=None):
        """Source/observed-state bootstrap only; NO future HLS labels."""
        if not candidates:
            return 0.0
        with torch.no_grad():
            features=np.stack([
                encode(task,catalog,c,best_cycles,calls,last_result)
                for c in candidates])
            _,value=self.policy(torch.tensor(features,dtype=torch.float32))
            return float(value.detach())

    def update(self,tape,reward,*,next_value=0.0,terminated=True,gamma=0.90):
        """Online one-step actor-critic TD after a REAL audited Vitis step."""
        if tape is None:
            return None
        logprob,entropy,value=tape
        if not np.isfinite(float(reward)) or not np.isfinite(float(next_value)):
            raise ValueError("Cannot train on nonfinite feedback/bootstrapped value")
        if not 0<=gamma<1:
            raise ValueError("Invalid TD discount")
        target=torch.tensor(
            float(reward) + (0.0 if terminated else gamma*float(next_value)),
            dtype=torch.float32)
        advantage=target-value.detach()
        loss=-(logprob*advantage)+0.5*(value-target).square()-0.001*entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),5.0)
        self.optimizer.step()
        self.updates+=1
        return float(loss.detach())

    def save_private(self,folder,metadata):
        """Tensor-only source-bound checkpoint; never publicly release weights."""
        from hashlib import sha256
        import json
        from pathlib import Path
        root=Path(folder)
        root.mkdir(parents=True,exist_ok=False)
        path=root/"policy.npz"
        np.savez(path,**{name:value.detach().cpu().numpy()
                         for name,value in self.policy.state_dict().items()})
        with (root/"manifest.json").open("x") as stream:
            json.dump({
                **metadata,"tensor_only":True,"private_weights":True,
                "online_gradient_updates":self.updates,
                "policy_sha256":sha256(path.read_bytes()).hexdigest(),
                "native_trained":True,"heldout_updates":0,
            },stream,indent=2,sort_keys=True)
