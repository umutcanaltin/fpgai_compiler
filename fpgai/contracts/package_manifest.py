from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PackageManifest:
    package_root: Path
    manifest_path: Path
    raw: Mapping[str, Any]

    @property
    def schema(self) -> str:
        return str(self.raw.get("schema", ""))

    @property
    def package(self) -> Mapping[str, Any]:
        value = self.raw.get("package", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def package_id(self) -> str:
        return str(self.package.get("id", ""))

    @property
    def version(self) -> str:
        return str(self.package.get("version", ""))

    @property
    def asset_type(self) -> str:
        return str(self.package.get("asset_type", ""))

    @property
    def dependencies(self) -> tuple[Mapping[str, Any], ...]:
        value = self.raw.get("dependencies", [])
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, Mapping))


def load_package_manifest(package_root: str | Path) -> PackageManifest:
    root = Path(package_root).expanduser().resolve()
    path = root / "fpgai.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Package manifest not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("Package manifest root must be a mapping")
    return PackageManifest(package_root=root, manifest_path=path, raw=loaded)


def inspect_package_manifest(package_root: str | Path) -> dict[str, Any]:
    manifest = load_package_manifest(package_root)
    return {
        "schema": manifest.schema,
        "package_id": manifest.package_id,
        "version": manifest.version,
        "asset_type": manifest.asset_type,
        "manifest_path": str(manifest.manifest_path),
    }
