from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .path_policy import should_skip_directory


@dataclass(frozen=True)
class ScanResult:
    manifests: tuple[Path, ...]
    skipped_symlinks: tuple[Path, ...]
    depth_limited: tuple[Path, ...]


def scan_package_manifests(search_root: Path, *, max_depth: int) -> ScanResult:
    root = search_root.resolve()
    manifests: list[Path] = []
    skipped_symlinks: list[Path] = []
    depth_limited: list[Path] = []

    for current_raw, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_raw)
        depth = len(current.relative_to(root).parts)

        retained: list[str] = []
        for name in sorted(directory_names):
            child = current / name
            if child.is_symlink():
                skipped_symlinks.append(child)
                continue
            if should_skip_directory(child):
                continue
            if depth >= max_depth:
                depth_limited.append(child)
                continue
            retained.append(name)
        directory_names[:] = retained

        if "fpgai.yaml" in file_names:
            manifest = current / "fpgai.yaml"
            if manifest.is_symlink():
                skipped_symlinks.append(manifest)
            else:
                manifests.append(manifest)
            directory_names[:] = []

    return ScanResult(
        manifests=tuple(sorted(manifests)),
        skipped_symlinks=tuple(sorted(set(skipped_symlinks))),
        depth_limited=tuple(sorted(set(depth_limited))),
    )
