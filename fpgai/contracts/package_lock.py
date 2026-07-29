from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_LOCK_SCHEMA_V1 = "fpgai.package-lock/v1"


@dataclass(frozen=True)
class LockedPackage:
    package_id: str
    version: str
    source: str
    manifest_sha256: str
    dependencies: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.package_id,
            "version": self.version,
            "source": self.source,
            "manifest_sha256": self.manifest_sha256,
            "dependencies": [dict(item) for item in self.dependencies],
        }


@dataclass(frozen=True)
class PackageLock:
    packages: tuple[LockedPackage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PACKAGE_LOCK_SCHEMA_V1,
            "packages": [package.to_dict() for package in self.packages],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def write_package_lock(lock: PackageLock, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(lock.to_json() + "\n", encoding="utf-8")
    else:
        output.write_text(lock.to_yaml(), encoding="utf-8")
    return output


__all__ = ["LockedPackage", "PACKAGE_LOCK_SCHEMA_V1", "PackageLock", "write_package_lock"]
