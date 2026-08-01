from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fpgai.registries import RegistryCatalogue, RegistrySource

from .discovery_errors import DiscoveryIssue


@dataclass(frozen=True)
class DiscoveryRequest:
    project_root: Path | str | None = None
    configured_directories: tuple[Path | str, ...] = ()
    include_builtin: bool = True
    max_depth: int = 2
    strict: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "configured_directories", tuple(self.configured_directories))


@dataclass(frozen=True)
class SearchRoot:
    path: Path
    source: RegistrySource
    required: bool
    label: str


@dataclass(frozen=True)
class DiscoveredPackage:
    package_root: Path
    manifest_path: Path
    package_id: str
    version: str
    asset_type: str
    source: RegistrySource
    search_root: Path
    manifest_hash: str
    status: str
    message: str = ""
    manifest_errors: tuple[dict[str, Any], ...] = ()

    @property
    def identity(self) -> tuple[str, str]:
        return (self.package_id, self.version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_root": str(self.package_root),
            "manifest_path": str(self.manifest_path),
            "package_id": self.package_id,
            "version": self.version,
            "asset_type": self.asset_type,
            "source": self.source.value,
            "search_root": str(self.search_root),
            "manifest_sha256": self.manifest_hash,
            "status": self.status,
            "message": self.message,
            "manifest_errors": list(self.manifest_errors),
        }


@dataclass
class DiscoveryResult:
    catalogue: RegistryCatalogue = field(default_factory=RegistryCatalogue)
    discovered: tuple[DiscoveredPackage, ...] = ()
    quarantined: tuple[DiscoveredPackage, ...] = ()
    deduplicated: tuple[DiscoveredPackage, ...] = ()
    conflicts: tuple[DiscoveredPackage, ...] = ()
    errors: tuple[DiscoveryIssue, ...] = ()
    warnings: tuple[DiscoveryIssue, ...] = ()
    search_roots: tuple[SearchRoot, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> dict[str, int]:
        return {
            "search_roots": len(self.search_roots),
            "manifests_found": len(self.discovered) + len(self.quarantined) + len(self.deduplicated) + len(self.conflicts),
            "accepted": len(self.discovered),
            "deduplicated": len(self.deduplicated),
            "quarantined": len(self.quarantined),
            "conflicts": len(self.conflicts),
            "warnings": len(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        packages = sorted(
            (*self.discovered, *self.quarantined, *self.deduplicated, *self.conflicts),
            key=lambda item: (item.package_id, item.version, item.source.value, str(item.package_root)),
        )
        return {
            "schema": "fpgai.package-discovery/v1",
            "status": "passed" if self.ok else "failed",
            "summary": self.summary(),
            "search_roots": [
                {
                    "path": str(item.path),
                    "source": item.source.value,
                    "required": item.required,
                    "label": item.label,
                }
                for item in self.search_roots
            ],
            "packages": [item.to_dict() for item in packages],
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
        }
