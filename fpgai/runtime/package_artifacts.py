"""File discovery and copying helpers for runtime packages."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _copy_if_exists(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": src.as_posix(),
        "package_path": dst.as_posix(),
        "bytes": dst.stat().st_size,
    }


def _first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        for hit in hits:
            if hit.is_file():
                return hit
    return None


def _collect_existing(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend([p for p in sorted(root.glob(pattern)) if p.is_file()])
    return out


def _artifact_status(path: Path | None) -> dict[str, Any]:
    return {
        "present": bool(path is not None and path.exists()),
        "path": path.as_posix() if path is not None else None,
    }
