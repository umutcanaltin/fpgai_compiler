"""Shared path rules for repository maintenance tools.

The maintenance tools inspect only repository-owned content. Local virtual
vironments, version-control metadata, generated outputs, and tool caches are
never traversed recursively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
}


def should_skip_directory(path: Path, repository_root: Path) -> bool:
    """Return whether a directory is outside the repository audit scope."""

    if path.name in IGNORED_DIRECTORY_NAMES:
        return True

    try:
        relative_path = path.relative_to(repository_root)
    except ValueError:
        return True

    return any(part in IGNORED_DIRECTORY_NAMES for part in relative_path.parts)


def iter_directories(repository_root: str | Path) -> Iterator[Path]:
    """Yield repository-owned directories without entering ignored trees."""

    root = Path(repository_root).resolve()
    pending = [root]

    while pending:
        current = pending.pop()

        for child in current.iterdir():
            if not child.is_dir():
                continue
            if should_skip_directory(child, root):
                continue

            yield child
            pending.append(child)
