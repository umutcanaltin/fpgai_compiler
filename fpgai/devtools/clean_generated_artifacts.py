"""List or remove generated repository artifacts.

The default mode is a dry run. Use ``--apply`` to remove the listed paths.
Source files, tests, configs, examples, and documentation are never targeted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from fpgai.devtools.path_policy import iter_directories
from fpgai.devtools.repository_audit import CACHE_DIRECTORY_NAMES, GENERATED_ROOTS



def relaunch_without_bytecode() -> None:
    """Restart the command with ``-B`` so it does not create new caches.

    Python imports this module before ``main`` runs, which can create
    ``__pycache__`` directories. The first process removes those small caches
    and immediately replaces itself with a bytecode-disabled process.
    """

    if sys.dont_write_bytecode:
        return

    module_cache = Path(__file__).resolve().parent / "__pycache__"
    package_cache = Path(__file__).resolve().parents[1] / "__pycache__"

    for cache_path in (module_cache, package_cache):
        if cache_path.exists():
            shutil.rmtree(cache_path)

    command = [
        sys.executable,
        "-B",
        "-m",
        "fpgai.devtools.clean_generated_artifacts",
        *sys.argv[1:],
    ]
    os.execv(sys.executable, command)

def find_generated_paths(repository_root: str | Path) -> list[Path]:
    """Return generated roots and cache directories that currently exist."""

    root = Path(repository_root).resolve()
    paths: set[Path] = set()

    for relative_path in GENERATED_ROOTS:
        candidate = root / relative_path
        if candidate.exists():
            paths.add(candidate)

    for candidate in iter_directories(root):
        if candidate.name in CACHE_DIRECTORY_NAMES:
            paths.add(candidate)

    return sorted(paths, key=lambda path: (len(path.parts), path.as_posix()), reverse=True)


def remove_generated_paths(paths: list[Path]) -> None:
    """Remove generated directories and files from the supplied list."""

    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> int:
    relaunch_without_bytecode()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root to clean")
    parser.add_argument("--apply", action="store_true", help="remove listed paths")
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="print every matched path instead of only a compact summary",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = find_generated_paths(root)

    print("Generated artifact cleanup")
    print("--------------------------")
    print(f"mode: {'apply' if args.apply else 'dry_run'}")
    print(f"paths_found: {len(paths)}")

    if args.show_paths:
        for path in paths:
            print(path.relative_to(root).as_posix())
    elif paths:
        category_counts: dict[str, int] = {}
        for path in paths:
            relative_path = path.relative_to(root)
            category = relative_path.parts[0]
            category_counts[category] = category_counts.get(category, 0) + 1

        print("path_groups:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        print("Use --show-paths to print every matched path.")

    if args.apply:
        remove_generated_paths(paths)
        print("status: removed")
    else:
        print("status: no_changes")
        print("Run again with --apply to remove these paths.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
