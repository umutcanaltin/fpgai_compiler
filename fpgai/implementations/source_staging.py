from __future__ import annotations

import shutil
from pathlib import Path

from .composition_errors import HLSCompositionError
from .composition_types import HLSCompositionPlan, StagedExternalSources


def _safe_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise HLSCompositionError(f"HLSCOMP007: unsafe package path {relative!r}")
    resolved = (root / path).resolve()
    if root.resolve() not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
        raise HLSCompositionError(f"HLSCOMP008: invalid package file {relative!r}")
    return resolved


def stage_external_sources(plan: HLSCompositionPlan, hls_dir: Path) -> StagedExternalSources:
    sources: list[Path] = []
    headers: list[Path] = []
    include_dirs: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for binding in plan.bindings:
        contract = binding.contract
        if contract.package_root is None:
            raise HLSCompositionError(f"HLSCOMP009: package root missing for {contract.package_id}")
        key = (contract.package_id, contract.version)
        if key in seen:
            continue
        seen.add(key)
        package_root = Path(contract.package_root).resolve()
        namespace = contract.package_id.replace(".", "_").replace("-", "_")
        source_root = hls_dir / "src" / "external" / namespace
        include_root = hls_dir / "include" / "external" / namespace
        source_root.mkdir(parents=True, exist_ok=True)
        include_root.mkdir(parents=True, exist_ok=True)
        include_dirs.append(include_root)
        ordered = contract.source_order or contract.sources
        for index, relative in enumerate(ordered):
            source = _safe_file(package_root, relative)
            target = source_root / f"{index:03d}_{source.name}"
            shutil.copy2(source, target)
            sources.append(target)
        for relative in contract.headers:
            source = _safe_file(package_root, relative)
            target = include_root / source.name
            shutil.copy2(source, target)
            headers.append(target)
    return StagedExternalSources(tuple(sources), tuple(headers), tuple(include_dirs))
