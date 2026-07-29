from __future__ import annotations

from dataclasses import dataclass

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


def parse_version(value: str) -> Version:
    """Parse a package version using PEP 440-compatible semantic version syntax."""
    return Version(str(value).strip())


def _release_parts(value: str) -> tuple[int, ...]:
    version = parse_version(value)
    return tuple(version.release)


def _caret_upper_bound(value: str) -> str:
    parts = list(_release_parts(value))
    while len(parts) < 3:
        parts.append(0)
    major, minor, patch = parts[:3]
    if major > 0:
        return f"{major + 1}.0.0"
    if minor > 0:
        return f"0.{minor + 1}.0"
    return f"0.0.{patch + 1}"


def _tilde_upper_bound(value: str) -> str:
    parts = list(_release_parts(value))
    if len(parts) == 1:
        return f"{parts[0] + 1}.0"
    return f"{parts[0]}.{parts[1] + 1}.0"


def normalize_version_range(value: str) -> str:
    """Normalize FPGAI package version syntax into a packaging SpecifierSet string.

    Supported forms include normal PEP 440 specifiers, exact versions, caret
    ranges (``^1.2.3``), and tilde ranges (``~1.2`` or ``~1.2.3``).
    """
    raw = str(value).strip()
    if not raw or raw == "*":
        return ""
    if raw.startswith("^"):
        base = raw[1:].strip()
        parse_version(base)
        return f">={base},<{_caret_upper_bound(base)}"
    if raw.startswith("~") and not raw.startswith("~="):
        base = raw[1:].strip()
        parse_version(base)
        return f">={base},<{_tilde_upper_bound(base)}"
    if not any(token in raw for token in ("<", ">", "=", "!", "~", ",")):
        parse_version(raw)
        return f"=={raw}"
    SpecifierSet(raw)
    return raw


@dataclass(frozen=True)
class VersionRange:
    raw: str
    normalized: str

    @classmethod
    def parse(cls, value: str) -> "VersionRange":
        raw = str(value).strip()
        normalized = normalize_version_range(raw)
        # Construct once here so invalid syntax is rejected at the boundary.
        SpecifierSet(normalized)
        return cls(raw=raw or "*", normalized=normalized)

    def contains(self, version: str | Version) -> bool:
        parsed = version if isinstance(version, Version) else parse_version(str(version))
        return parsed in SpecifierSet(self.normalized)

    def to_dict(self) -> dict[str, str]:
        return {"raw": self.raw, "normalized": self.normalized}


__all__ = [
    "InvalidSpecifier",
    "InvalidVersion",
    "VersionRange",
    "normalize_version_range",
    "parse_version",
]
