from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .implementation_contract import ImplementationContract


@dataclass(frozen=True)
class CandidateDecision:
    contract: ImplementationContract
    status: str
    reasons: tuple[str, ...]
    score: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.contract.package_id,
            "version": self.contract.version,
            "status": self.status,
            "reasons": list(self.reasons),
            "score": list(self.score),
        }


@dataclass(frozen=True)
class ImplementationSelectionResult:
    operator_id: str
    policy: str
    selected: ImplementationContract | None
    candidates: tuple[CandidateDecision, ...]
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.selected is not None and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.implementation-selection/v1",
            "operator_id": self.operator_id,
            "policy": self.policy,
            "status": "selected" if self.ok else "failed",
            "selected": None if self.selected is None else {
                "package_id": self.selected.package_id,
                "version": self.selected.version,
            },
            "candidates": [item.to_dict() for item in self.candidates],
            "errors": list(self.errors),
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }
