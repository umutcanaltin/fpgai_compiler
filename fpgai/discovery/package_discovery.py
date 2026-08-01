from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from fpgai.contracts.package_manifest import load_package_manifest
from fpgai.contracts.package_validation import validate_package_manifest
from fpgai.registries import RegistryCatalogue, RegistryEntry, RegistrySource, build_builtin_catalogue

from .discovery_errors import DiscoveryIssue
from .discovery_types import DiscoveryRequest, DiscoveryResult, DiscoveredPackage, SearchRoot
from .package_scanner import scan_package_manifests
from .path_policy import validate_package_symlink_boundary


def _issue(code: str, field: str, message: str, *, warning: bool = False) -> DiscoveryIssue:
    return DiscoveryIssue(code, field, message, "warning" if warning else "error")


def _validate_request(request: DiscoveryRequest) -> tuple[DiscoveryIssue, ...]:
    errors: list[DiscoveryIssue] = []
    if not isinstance(request.max_depth, int) or isinstance(request.max_depth, bool) or request.max_depth < 0 or request.max_depth > 5:
        errors.append(_issue("PKGDISC001", "max_depth", "max_depth must be an integer from 0 through 5"))
    return tuple(errors)


def _search_roots(request: DiscoveryRequest) -> tuple[SearchRoot, ...]:
    roots: list[SearchRoot] = []
    for index, configured in enumerate(request.configured_directories):
        roots.append(SearchRoot(Path(configured).expanduser(), RegistrySource.CONFIGURED_DIRECTORY, True, f"configured[{index}]"))
    if request.project_root is not None:
        roots.append(SearchRoot(Path(request.project_root).expanduser() / "packages", RegistrySource.PROJECT_LOCAL, False, "project_local"))
    return tuple(roots)


def _builtin_candidates() -> list[tuple[RegistryEntry, DiscoveredPackage]]:
    candidates: list[tuple[RegistryEntry, DiscoveredPackage]] = []
    for entry in build_builtin_catalogue().inventory():
        candidates.append((entry, DiscoveredPackage(
            package_root=Path("<builtin>"), manifest_path=Path("<builtin>"), package_id=entry.package_id,
            version=entry.version, asset_type=entry.asset_type, source=entry.source,
            search_root=Path("<builtin>"), manifest_hash=entry.manifest_hash, status="candidate",
        )))
    return candidates


def discover_packages(request: DiscoveryRequest) -> DiscoveryResult:
    request_errors = _validate_request(request)
    if request_errors:
        return DiscoveryResult(errors=request_errors)

    errors: list[DiscoveryIssue] = []
    warnings: list[DiscoveryIssue] = []
    quarantined: list[DiscoveredPackage] = []
    all_candidates: list[tuple[RegistryEntry, DiscoveredPackage]] = []
    roots = _search_roots(request)

    if request.include_builtin:
        all_candidates.extend(_builtin_candidates())

    for root_info in roots:
        raw_root = root_info.path
        if raw_root.is_symlink():
            issue = _issue("PKGDISC002", root_info.label, f"Search root may not be a symbolic link: {raw_root}")
            (errors if request.strict else warnings).append(issue if request.strict else _issue(issue.code, issue.field, issue.message, warning=True))
            continue
        if not raw_root.exists():
            if root_info.required:
                issue = _issue("PKGDISCW001", root_info.label, f"Search root does not exist: {raw_root}", warning=True)
                if request.strict:
                    errors.append(_issue("PKGDISC010", root_info.label, issue.message))
                else:
                    warnings.append(issue)
            continue
        if not raw_root.is_dir():
            errors.append(_issue("PKGDISC002", root_info.label, f"Search root is not a directory: {raw_root}"))
            continue

        root = raw_root.resolve()
        scan = scan_package_manifests(root, max_depth=request.max_depth)
        for path in scan.skipped_symlinks:
            warnings.append(_issue("PKGDISC004", str(path), "Symbolic link was not traversed", warning=True))
        for path in scan.depth_limited:
            warnings.append(_issue("PKGDISCW004", str(path), "Scan depth limit reached", warning=True))

        for manifest_path in scan.manifests:
            package_root = manifest_path.parent
            validation = validate_package_manifest(package_root)
            try:
                manifest = load_package_manifest(package_root)
            except (OSError, ValueError) as exc:
                item = DiscoveredPackage(package_root, manifest_path, "", "", "", root_info.source, root, "", "quarantined", str(exc))
                quarantined.append(item)
                continue

            digest = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            symlink_problems = validate_package_symlink_boundary(package_root, manifest.raw)
            validation_errors = tuple(issue.to_dict() for issue in validation.errors)
            if not validation.ok or symlink_problems:
                combined = validation_errors + tuple({"code": "PKGDISC004", "field": "entrypoints", "message": problem, "severity": "error"} for problem in symlink_problems)
                quarantined.append(DiscoveredPackage(
                    package_root, manifest_path, manifest.package_id, manifest.version, manifest.asset_type,
                    root_info.source, root, digest, "quarantined", "Manifest validation failed", combined,
                ))
                warnings.append(_issue("PKGDISCW005", str(package_root), "Package was quarantined", warning=True))
                continue

            try:
                entry = RegistryEntry.from_manifest(manifest, source=root_info.source, manifest_hash=digest)
            except (TypeError, ValueError) as exc:
                quarantined.append(DiscoveredPackage(
                    package_root, manifest_path, manifest.package_id, manifest.version, manifest.asset_type,
                    root_info.source, root, digest, "quarantined", str(exc),
                ))
                continue
            all_candidates.append((entry, DiscoveredPackage(
                package_root, manifest_path, manifest.package_id, manifest.version, manifest.asset_type,
                root_info.source, root, digest, "candidate",
            )))

    grouped: dict[tuple[str, str], list[tuple[RegistryEntry, DiscoveredPackage]]] = defaultdict(list)
    for pair in all_candidates:
        grouped[(pair[0].package_id, pair[0].version)].append(pair)

    catalogue = RegistryCatalogue()
    accepted: list[DiscoveredPackage] = []
    deduplicated: list[DiscoveredPackage] = []
    conflicts: list[DiscoveredPackage] = []

    for identity in sorted(grouped):
        candidates = grouped[identity]
        hashes = {entry.manifest_hash for entry, _ in candidates}
        if len(hashes) > 1:
            for _, item in candidates:
                conflicts.append(DiscoveredPackage(**{**item.__dict__, "status": "conflict", "message": "Same package identity has different manifest content"}))
            errors.append(_issue("PKGDISC009", f"{identity[0]}@{identity[1]}", "Duplicate package identity has conflicting content"))
            continue

        candidates.sort(key=lambda pair: (pair[0].priority, str(pair[1].package_root)), reverse=True)
        selected_entry, selected_item = candidates[0]
        registry_result = catalogue.registry_for(selected_entry.asset_type).register(selected_entry)
        if not registry_result.ok:
            errors.append(_issue("PKGDISC008", selected_entry.package_id, registry_result.errors[0].message))
            continue
        accepted.append(DiscoveredPackage(**{**selected_item.__dict__, "status": "accepted"}))
        for _, duplicate in candidates[1:]:
            deduplicated.append(DiscoveredPackage(**{**duplicate.__dict__, "status": "deduplicated", "message": f"Equivalent package retained from {selected_entry.source.value}"}))
            warnings.append(_issue("PKGDISCW002", duplicate.package_id, "Equivalent package was deduplicated", warning=True))

    if request.strict and (quarantined or conflicts):
        errors.append(_issue("PKGDISC010", "strict", "Strict discovery failed because packages were quarantined or conflicted"))

    return DiscoveryResult(
        catalogue=catalogue,
        discovered=tuple(sorted(accepted, key=lambda item: (item.package_id, item.version, item.source.value))),
        quarantined=tuple(sorted(quarantined, key=lambda item: str(item.package_root))),
        deduplicated=tuple(sorted(deduplicated, key=lambda item: str(item.package_root))),
        conflicts=tuple(sorted(conflicts, key=lambda item: (item.package_id, item.version, str(item.package_root)))),
        errors=tuple(errors), warnings=tuple(warnings), search_roots=roots,
    )
