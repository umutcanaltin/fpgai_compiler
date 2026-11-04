"""Experimental protocol extension point, without vendor-side effects.

Every registered experiment is a deterministic function of its arguments.
It must bind to an explicit curriculum stage with independent heldout tasks;
additional experiment families can add verified adapters in source.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

ConfigFactory = Callable[..., dict[str, Any]]
_REGISTRY: dict[str, ConfigFactory] = {}


def register_experiment(name: str, factory: ConfigFactory) -> None:
    if (not isinstance(name, str) or not name or
            any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)):
        raise ValueError("Experiment names must be lowercase snake_case")
    if name in _REGISTRY:
        raise ValueError("Experiment already registered")
    if not callable(factory):
        raise TypeError("Experiment factory must be callable")
    _REGISTRY[name] = factory


def available_experiments() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def experiment_config(name: str, **kwargs) -> dict[str, Any]:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown registered experiment: {name}") from exc
    result = factory(**kwargs)
    if not isinstance(result, dict):
        raise TypeError("Experiment configuration factory must return a dictionary")
    return dict(result)


def _curriculum_config(stage_name: str, *, train: int = 100,
                       validation: int = 1, test: int = 1,
                       calls: int = 10, seed: int = 111) -> dict:
    """Built-in explicit protocols; no vendor work or hidden warm starts."""
    from ..curriculum import get_stage
    stage = get_stage(stage_name)
    return {
        "training": train, "validation": validation, "test": test,
        "calls_per_network": calls, "seeds": (seed,),
        "clocks_ns": (
            (stage.fixed_clock_ns,) if stage.fixed_clock_ns is not None
            else None),
        "freeze_numeric_format": stage.freeze_numeric_format,
        "beginner_pilot": stage.name == "L0",
        "curriculum_stage": stage.name,
    }


for _stage_name in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
    def _stage_factory(*, _fixed_stage=_stage_name, **kwargs):
        return _curriculum_config(_fixed_stage, **kwargs)

    register_experiment(f"curriculum_{_stage_name.lower()}", _stage_factory)
