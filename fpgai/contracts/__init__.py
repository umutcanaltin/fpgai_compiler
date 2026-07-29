"""Public metadata contracts for FPGAI research packages."""

from __future__ import annotations

from typing import Any

__all__ = [
    "PackageDependency",
    "PackageLock",
    "PackageManifest",
    "PackageResolutionResult",
    "PackageValidationResult",
    "VersionRange",
    "inspect_package_manifest",
    "load_package_manifest",
    "resolve_package_set",
    "validate_package_manifest",
    "write_package_lock",
]


def __getattr__(name: str) -> Any:
    if name in {"PackageManifest", "inspect_package_manifest", "load_package_manifest"}:
        from .package_manifest import PackageManifest, inspect_package_manifest, load_package_manifest

        return {
            "PackageManifest": PackageManifest,
            "inspect_package_manifest": inspect_package_manifest,
            "load_package_manifest": load_package_manifest,
        }[name]
    if name in {"PackageValidationResult", "validate_package_manifest"}:
        from .package_validation import PackageValidationResult, validate_package_manifest

        return {
            "PackageValidationResult": PackageValidationResult,
            "validate_package_manifest": validate_package_manifest,
        }[name]
    if name == "PackageDependency":
        from .package_dependency import PackageDependency

        return PackageDependency
    if name == "VersionRange":
        from .package_version import VersionRange

        return VersionRange
    if name in {"PackageResolutionResult", "resolve_package_set"}:
        from .package_resolution import PackageResolutionResult, resolve_package_set

        return {
            "PackageResolutionResult": PackageResolutionResult,
            "resolve_package_set": resolve_package_set,
        }[name]
    if name in {"PackageLock", "write_package_lock"}:
        from .package_lock import PackageLock, write_package_lock

        return {"PackageLock": PackageLock, "write_package_lock": write_package_lock}[name]
    raise AttributeError(name)
