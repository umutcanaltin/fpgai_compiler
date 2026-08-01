from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from fpgai.contracts.package_manifest import PackageManifest
from fpgai.contracts.package_types import AssetType
from fpgai.contracts.package_version import parse_version

class RegistrySource(str, Enum):
    PROJECT_LOCAL = "project_local"
    ORGANIZATION_PRIVATE = "organization_private"
    INSTALLED_PACKAGE = "installed_package"
    CONFIGURED_DIRECTORY = "configured_directory"
    COMMUNITY_CACHE = "community_cache"
    BUILTIN = "builtin"

    @classmethod
    def normalize(cls, value: str | "RegistrySource") -> "RegistrySource":
        if isinstance(value, cls): return value
        raw = str(value).strip().lower().replace("-", "_")
        return cls(raw)

SOURCE_PRIORITY = {source: rank for rank, source in enumerate(reversed(tuple(RegistrySource)), start=1)}
SOURCE_PRIORITY.update({
    RegistrySource.BUILTIN: 10, RegistrySource.COMMUNITY_CACHE: 20,
    RegistrySource.CONFIGURED_DIRECTORY: 30, RegistrySource.INSTALLED_PACKAGE: 40,
    RegistrySource.ORGANIZATION_PRIVATE: 50, RegistrySource.PROJECT_LOCAL: 60,
})

@dataclass(frozen=True)
class RegistryEntry:
    package_id: str
    version: str
    asset_type: str
    provider: str
    source: RegistrySource
    source_path: Path | None
    manifest_hash: str
    capabilities: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    validation_level: str
    license_category: str
    usage: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        parse_version(self.version)
        AssetType(self.asset_type)
        object.__setattr__(self, "source", RegistrySource.normalize(self.source))
        for name in ("capabilities", "compatibility", "usage", "metadata"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def identity(self) -> tuple[str, str, RegistrySource]:
        return (self.package_id, self.version, self.source)

    @property
    def priority(self) -> int:
        return SOURCE_PRIORITY[self.source]

    @classmethod
    def from_manifest(cls, manifest: PackageManifest, *, source: RegistrySource, manifest_hash: str) -> "RegistryEntry":
        raw=manifest.raw; package=dict(raw.get("package",{})); validation=dict(raw.get("validation",{})); license_cfg=dict(raw.get("license",{}))
        return cls(
            package_id=manifest.package_id, version=manifest.version, asset_type=manifest.asset_type,
            provider=str(package.get("provider", manifest.package_id.split(".",1)[0])), source=source,
            source_path=manifest.package_root, manifest_hash=manifest_hash,
            capabilities=dict(raw.get("capabilities",{})), compatibility=dict(raw.get("compatibility",{})),
            validation_level=str(validation.get("declared_level","unvalidated")),
            license_category=str(license_cfg.get("category","research_only")), usage=dict(raw.get("usage",{})),
            metadata={"name": str(package.get("name", manifest.package_id)), "description": str(package.get("description", ""))},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id, "version": self.version, "asset_type": self.asset_type,
            "provider": self.provider, "source": self.source.value,
            "source_path": str(self.source_path) if self.source_path else None, "manifest_hash": self.manifest_hash,
            "capabilities": dict(self.capabilities), "compatibility": dict(self.compatibility),
            "validation_level": self.validation_level, "license_category": self.license_category,
            "usage": dict(self.usage), "metadata": dict(self.metadata),
        }
