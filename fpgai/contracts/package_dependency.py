from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .package_version import VersionRange


@dataclass(frozen=True)
class PackageDependency:
    package_id: str
    version: VersionRange
    required: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PackageDependency":
        package_id = str(value.get("package", "")).strip()
        version = VersionRange.parse(str(value.get("version", "*")))
        required_raw = value.get("required", True)
        if not isinstance(required_raw, bool):
            raise ValueError("Dependency required must be a Boolean")
        return cls(package_id=package_id, version=version, required=required_raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package_id,
            "version": self.version.raw,
            "required": self.required,
        }


def parse_dependencies(raw: Mapping[str, Any]) -> tuple[PackageDependency, ...]:
    value = raw.get("dependencies", [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Manifest dependencies must be a list")
    dependencies: list[PackageDependency] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Each dependency must be a mapping")
        dependency = PackageDependency.from_mapping(item)
        if not dependency.package_id:
            raise ValueError("Dependency package ID is required")
        dependencies.append(dependency)
    return tuple(dependencies)


__all__ = ["PackageDependency", "parse_dependencies"]
