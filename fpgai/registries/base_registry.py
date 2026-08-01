from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fpgai.contracts.package_manifest import load_package_manifest
from fpgai.contracts.package_validation import validate_package_manifest
from fpgai.contracts.package_version import VersionRange, parse_version
from .registry_entry import RegistryEntry, RegistrySource
from .registry_errors import RegistryIssue

@dataclass(frozen=True)
class RegistryResult:
    selected: RegistryEntry | None = None
    candidates: tuple[RegistryEntry, ...] = ()
    errors: tuple[RegistryIssue, ...] = ()
    warnings: tuple[RegistryIssue, ...] = ()
    @property
    def ok(self) -> bool: return not self.errors
    def to_dict(self) -> dict[str, Any]:
        return {"status":"passed" if self.ok else "failed", "selected":self.selected.to_dict() if self.selected else None,
                "candidates":[e.to_dict() for e in self.candidates], "errors":[e.to_dict() for e in self.errors],
                "warnings":[e.to_dict() for e in self.warnings]}

class BaseRegistry:
    def __init__(self, asset_type: str):
        self.asset_type=asset_type; self._entries: dict[tuple[str,str,RegistrySource],RegistryEntry]={}

    def register(self, entry: RegistryEntry) -> RegistryResult:
        if entry.asset_type != self.asset_type:
            return RegistryResult(errors=(RegistryIssue("PKGREG002","asset_type",f"Expected {self.asset_type}, got {entry.asset_type}"),))
        key=entry.identity; existing=self._entries.get(key)
        if existing and existing.manifest_hash != entry.manifest_hash:
            return RegistryResult(errors=(RegistryIssue("PKGREG003",entry.package_id,"Duplicate identity has different manifest content"),))
        self._entries[key]=entry
        return RegistryResult(selected=entry,candidates=(entry,))

    def register_manifest(self, package_root: str|Path, source: str|RegistrySource) -> RegistryResult:
        validation=validate_package_manifest(package_root)
        if not validation.ok:
            return RegistryResult(errors=(RegistryIssue("PKGREG001",str(package_root),"Package manifest validation failed"),))
        manifest=load_package_manifest(package_root); digest="sha256:"+hashlib.sha256(manifest.manifest_path.read_bytes()).hexdigest()
        try: entry=RegistryEntry.from_manifest(manifest,source=RegistrySource.normalize(source),manifest_hash=digest)
        except (ValueError,TypeError) as exc:
            return RegistryResult(errors=(RegistryIssue("PKGREG001",str(package_root),str(exc)),))
        return self.register(entry)

    def list_entries(self) -> tuple[RegistryEntry,...]:
        return tuple(sorted(self._entries.values(), key=lambda e:(e.package_id,parse_version(e.version),e.priority,e.source.value)))
    def list_versions(self, package_id: str) -> tuple[str,...]:
        return tuple(str(v) for v in sorted({parse_version(e.version) for e in self._entries.values() if e.package_id==package_id}, reverse=True))
    def find(self, *, provider: str|None=None, capability: str|None=None) -> tuple[RegistryEntry,...]:
        entries=self.list_entries()
        if provider is not None: entries=tuple(e for e in entries if e.provider==provider)
        if capability is not None: entries=tuple(e for e in entries if bool(e.capabilities.get(capability)))
        return entries
    def resolve(self, package_id: str, version_range: str="*") -> RegistryResult:
        rng=VersionRange.parse(version_range)
        candidates=[e for e in self._entries.values() if e.package_id==package_id and rng.contains(e.version)]
        if not any(e.package_id==package_id for e in self._entries.values()):
            return RegistryResult(errors=(RegistryIssue("PKGREG005","package_id",f"Package not found: {package_id}"),))
        if not candidates:
            return RegistryResult(errors=(RegistryIssue("PKGREG006","version",f"No version of {package_id} matches {version_range}"),))
        candidates.sort(key=lambda e:(parse_version(e.version),e.priority,e.manifest_hash),reverse=True)
        best=candidates[0]; same=[e for e in candidates if parse_version(e.version)==parse_version(best.version) and e.priority==best.priority]
        if len({e.manifest_hash for e in same})>1:
            return RegistryResult(candidates=tuple(candidates),errors=(RegistryIssue("PKGREG007",package_id,"Resolution is ambiguous"),))
        warnings=tuple(RegistryIssue("PKGREGW001",package_id,f"Lower-priority candidate ignored: {e.source.value}","warning") for e in candidates[1:])
        return RegistryResult(selected=best,candidates=tuple(candidates),warnings=warnings)
    def explain_resolution(self, package_id: str, version_range: str="*") -> dict[str,Any]: return self.resolve(package_id,version_range).to_dict()
    def validate_integrity(self) -> RegistryResult:
        return RegistryResult(candidates=self.list_entries())
