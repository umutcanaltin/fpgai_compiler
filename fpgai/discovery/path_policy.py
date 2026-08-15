from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


SKIPPED_DIRECTORY_NAMES = frozenset({
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", "build", "dist", "generated", "dev_audits",
    "benchmark_results", "node_modules",
})

_PATH_KEYS = frozenset({"path", "sources", "headers", "source_order", "testbench", "reference", "python_module"})


def is_hidden_directory(path: Path) -> bool:
    return path.name.startswith(".")


def should_skip_directory(path: Path, *, is_search_root: bool = False) -> bool:
    if path.name in SKIPPED_DIRECTORY_NAMES:
        return True
    return not is_search_root and is_hidden_directory(path)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def iter_declared_paths(value: Any, field: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_field = f"{field}.{key}" if field else str(key)
            if key in _PATH_KEYS:
                if isinstance(child, str):
                    yield child_field, child
                elif isinstance(child, list):
                    for index, item in enumerate(child):
                        if isinstance(item, str):
                            yield f"{child_field}[{index}]", item
                elif isinstance(child, Mapping):
                    yield from iter_declared_paths(child, child_field)
            else:
                yield from iter_declared_paths(child, child_field)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_declared_paths(child, f"{field}[{index}]")


def validate_package_symlink_boundary(package_root: Path, manifest_raw: Mapping[str, Any]) -> tuple[str, ...]:
    problems: list[str] = []
    root = package_root.resolve()
    manifest = package_root / "fpgai.yaml"
    if manifest.is_symlink():
        problems.append("fpgai.yaml is a symbolic link")

    for field, relative in iter_declared_paths(manifest_raw.get("entrypoints", {}), "entrypoints"):
        candidate = package_root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents if parent != package_root.parent):
            if not is_within(resolved, root):
                problems.append(f"{field} resolves outside package root through a symbolic link")
    return tuple(problems)
