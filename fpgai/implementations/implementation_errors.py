from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImplementationIssue:
    code: str
    field: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
        }
