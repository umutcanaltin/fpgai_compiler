"""Explicit source-controlled policy registry; default actor-critic unchanged.

Research algorithms are registered in source code, not via arbitrary import
strings, so campaign Git SHA pins the algorithm implementation and its
evaluation/checkpoint semantics. Every new algorithm must be independently
tested on CPU and with genuine HLS evidence.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .api import validate_agent


AgentFactory = Callable[..., Any]


def _original_gnn(*, seed: int, use_message_passing: bool):
    from ..gym_agent import OnlineGraphTrainer
    return OnlineGraphTrainer(
        seed=seed, use_message_passing=use_message_passing)


_FACTORIES: dict[str, AgentFactory] = {
    "graph_actor_critic": _original_gnn,
}


def register_agent(name: str, factory: AgentFactory) -> None:
    """Add a DRL algorithm explicitly, without modifying Gym dynamics."""
    if (not isinstance(name, str) or not name or
            any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)):
        raise ValueError("Algorithm name must be lowercase snake_case")
    if name in _FACTORIES:
        raise ValueError(f"Algorithm already registered: {name}")
    if not callable(factory):
        raise TypeError("Agent factory must be callable")
    _FACTORIES[name] = factory


def available_agents() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def create_agent(name: str, *, seed: int,
                 use_message_passing: bool = True):
    if name not in _FACTORIES:
        raise ValueError(
            f"Unknown agent '{name}'. Registered: {', '.join(available_agents())}")
    agent = _FACTORIES[name](
        seed=seed, use_message_passing=use_message_passing)
    validate_agent(agent)
    return agent
