from __future__ import annotations

from dataclasses import dataclass

from .compatibility import CompatibilityRequest
from .implementation_contract import valid_selection_policy


@dataclass(frozen=True)
class ImplementationSelectionRequest:
    operator_id: str
    compatibility: CompatibilityRequest
    preferred_packages: tuple[str, ...] = ()
    allow_fallback: bool = True
    policy: str = "balanced"

    def __post_init__(self) -> None:
        if not valid_selection_policy(self.policy):
            raise ValueError(f"Unsupported implementation selection policy: {self.policy}")
