from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from packaging.version import Version

from .package_dependency import PackageDependency, parse_dependencies
from .package_errors import PackageIssue
from .package_lock import LockedPackage, PackageLock
from .package_manifest import PackageManifest, load_package_manifest
from .package_validation import validate_package_manifest
from .package_version import VersionRange, parse_version


@dataclass(frozen=True)
class PackageCandidate:
    manifest: PackageManifest
    source: str
    manifest_sha256: str
    dependencies: tuple[PackageDependency, ...]

    @property
    def package_id(self) -> str:
        return self.manifest.package_id

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def parsed_version(self) -> Version:
        return parse_version(self.version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "version": self.version,
            "source": self.source,
            "manifest_sha256": self.manifest_sha256,
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
        }


@dataclass(frozen=True)
class PackageResolutionResult:
    status: str
    selected: tuple[PackageCandidate, ...]
    resolution_order: tuple[str, ...]
    errors: tuple[PackageIssue, ...]
    warnings: tuple[PackageIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_lock(self) -> PackageLock:
        selected_by_id = {candidate.package_id: candidate for candidate in self.selected}
        ordered_ids = self.resolution_order or tuple(sorted(selected_by_id))
        packages: list[LockedPackage] = []
        for package_id in ordered_ids:
            candidate = selected_by_id[package_id]
            packages.append(
                LockedPackage(
                    package_id=candidate.package_id,
                    version=candidate.version,
                    source=candidate.source,
                    manifest_sha256=candidate.manifest_sha256,
                    dependencies=tuple(dependency.to_dict() for dependency in candidate.dependencies),
                )
            )
        return PackageLock(tuple(packages))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected": [candidate.to_dict() for candidate in self.selected],
            "resolution_order": list(self.resolution_order),
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _issue(code: str, field: str, message: str, *, warning: bool = False) -> PackageIssue:
    return PackageIssue(code, field, message, "warning" if warning else "error")


def _manifest_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_from_root(package_root: str | Path, source: str) -> PackageCandidate:
    manifest = load_package_manifest(package_root)
    return PackageCandidate(
        manifest=manifest,
        source=source,
        manifest_sha256=_manifest_hash(manifest.manifest_path),
        dependencies=parse_dependencies(manifest.raw),
    )


def _topological_order(selected: dict[str, PackageCandidate]) -> tuple[tuple[str, ...], list[PackageIssue]]:
    order: list[str] = []
    errors: list[PackageIssue] = []
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(package_id: str) -> None:
        marker = state.get(package_id, 0)
        if marker == 2:
            return
        if marker == 1:
            start = stack.index(package_id) if package_id in stack else 0
            cycle = stack[start:] + [package_id]
            errors.append(
                _issue(
                    "PKGR005",
                    "dependencies",
                    "Dependency cycle detected: " + " -> ".join(cycle),
                )
            )
            return
        state[package_id] = 1
        stack.append(package_id)
        candidate = selected[package_id]
        for dependency in sorted(candidate.dependencies, key=lambda item: item.package_id):
            if dependency.package_id in selected:
                visit(dependency.package_id)
        stack.pop()
        state[package_id] = 2
        if package_id not in order:
            order.append(package_id)

    for package_id in sorted(selected):
        visit(package_id)
    return tuple(order), errors


def resolve_package_set(
    package_roots: Iterable[str | Path],
    *,
    root_package_ids: Iterable[str] | None = None,
    source: str = "project-local",
) -> PackageResolutionResult:
    """Resolve a closed set of already available local package manifests.

    Resolution is metadata-only. It performs no package imports, subprocess
    execution, network access, or compiler integration.
    """
    errors: list[PackageIssue] = []
    warnings: list[PackageIssue] = []
    candidates: list[PackageCandidate] = []

    for index, package_root in enumerate(package_roots):
        validation = validate_package_manifest(package_root)
        if not validation.ok:
            errors.append(
                _issue(
                    "PKGR001",
                    f"package_roots[{index}]",
                    f"Package manifest is invalid: {validation.package_id or package_root}",
                )
            )
            continue
        try:
            candidates.append(_candidate_from_root(package_root, source))
        except (OSError, ValueError) as exc:
            errors.append(_issue("PKGR001", f"package_roots[{index}]", str(exc)))

    by_identity: dict[tuple[str, str], list[PackageCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_identity[(candidate.package_id, candidate.version)].append(candidate)

    deduplicated: list[PackageCandidate] = []
    for identity, matches in sorted(by_identity.items()):
        hashes = {candidate.manifest_sha256 for candidate in matches}
        if len(hashes) > 1:
            errors.append(
                _issue(
                    "PKGR002",
                    f"packages.{identity[0]}@{identity[1]}",
                    "Duplicate package identity has different manifest content",
                )
            )
            continue
        deduplicated.append(sorted(matches, key=lambda candidate: str(candidate.manifest.package_root))[0])

    available: dict[str, list[PackageCandidate]] = defaultdict(list)
    for candidate in deduplicated:
        available[candidate.package_id].append(candidate)
    for package_id in available:
        available[package_id].sort(
            key=lambda candidate: (candidate.parsed_version, str(candidate.manifest.package_root)),
            reverse=True,
        )

    requested_roots = tuple(sorted(set(root_package_ids or available.keys())))
    requirements: dict[str, list[tuple[VersionRange, str, bool]]] = defaultdict(list)
    for package_id in requested_roots:
        requirements[package_id].append((VersionRange.parse("*"), "root", True))

    selected: dict[str, PackageCandidate] = {}
    pending = list(requested_roots)

    while pending:
        package_id = pending.pop(0)
        constraints = requirements[package_id]
        choices = [
            candidate
            for candidate in available.get(package_id, [])
            if all(version_range.contains(candidate.parsed_version) for version_range, _owner, _required in constraints)
        ]
        if not choices:
            required_constraints = [item for item in constraints if item[2]]
            owners = ", ".join(sorted({owner for _range, owner, _required in required_constraints}))
            ranges = ", ".join(item[0].raw for item in required_constraints)
            errors.append(
                _issue(
                    "PKGR003",
                    f"dependencies.{package_id}",
                    f"No available version satisfies required ranges [{ranges}] requested by {owners}",
                )
            )
            continue

        chosen = choices[0]
        previous = selected.get(package_id)
        if previous and previous.manifest_sha256 == chosen.manifest_sha256:
            continue
        selected[package_id] = chosen

        for dependency in chosen.dependencies:
            if dependency.package_id not in available:
                issue = _issue(
                    "PKGR004" if dependency.required else "PKGRW001",
                    f"packages.{package_id}.dependencies.{dependency.package_id}",
                    f"{'Required' if dependency.required else 'Optional'} dependency is unavailable: {dependency.package_id}",
                    warning=not dependency.required,
                )
                (errors if dependency.required else warnings).append(issue)
                continue
            requirement = (dependency.version, package_id, dependency.required)
            if requirement not in requirements[dependency.package_id]:
                requirements[dependency.package_id].append(requirement)
                if dependency.package_id not in pending:
                    pending.append(dependency.package_id)

    # A later constraint can invalidate a package selected earlier. Recheck the
    # complete selected set and report the conflict deterministically.
    for package_id, candidate in sorted(selected.items()):
        for version_range, owner, required in requirements.get(package_id, []):
            if required and not version_range.contains(candidate.parsed_version):
                errors.append(
                    _issue(
                        "PKGR006",
                        f"packages.{package_id}",
                        f"Selected {candidate.version} does not satisfy {version_range.raw} requested by {owner}",
                    )
                )

    order, cycle_errors = _topological_order(selected)
    errors.extend(cycle_errors)
    selected_tuple = tuple(selected[package_id] for package_id in sorted(selected))
    return PackageResolutionResult(
        status="passed" if not errors else "failed",
        selected=selected_tuple,
        resolution_order=order,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


__all__ = ["PackageCandidate", "PackageResolutionResult", "resolve_package_set"]
