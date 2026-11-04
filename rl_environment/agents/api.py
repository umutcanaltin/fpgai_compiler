"""Stable minimal API for a from-scratch agent in the native-HLS Gym loop."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class AgentProtocol(Protocol):
    """Algorithms must implement these semantics, regardless of architecture.

    choose(): (slot, requested_hls_clock_ns), trainable action tape.
    update(): use only the latest *original* HLS reward/qualification.
    save_private/load_private(): source-bound, create-only checkpoint handling.
    """

    def choose(self, observation: dict, *, training: bool, sample: bool = ...) -> tuple:
        ...

    def update(self, tape: Any, reward: float, *, next_obs: dict,
               terminated: bool, return_metrics: bool, qualified: bool) -> dict:
        ...

    def save_private(self, directory: str | Path, metadata: dict) -> None:
        ...

    def load_private(self, directory: str | Path, *, load_optimizer: bool = True) -> dict:
        ...


REQUIRED_METHODS = ("choose", "update", "save_private", "load_private")


def validate_agent(agent: object) -> None:
    for method in REQUIRED_METHODS:
        if not callable(getattr(agent, method, None)):
            raise TypeError(f"Research agent must implement {method}()")
