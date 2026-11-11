"""Read-only, source-pinned map of tracked files and Python import connectivity.

Contributors can locate a file, its owning subsystem, internal imports,
direct importers, top-level definitions and original content SHA without
duplicating implementation blindly. Static Python imports are approximate:
dynamic imports, runtime registration, scripts and non-Python dependencies
are not a complete call graph. Never label files unused based on this map.

No GitHub Actions, Vitis, Vivado, modification or deletion. Optional report
output is create-only and belongs under ignored build/.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess

SCHEMA = "fpgai-research/tracked-source-import-connectivity/v1"
MAX_SOURCE_BYTES = 3_000_000


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def _tracked(root: Path) -> tuple[str, ...]:
    lines = _git(root, "ls-files", "-z").split(b"\0")
    files = sorted(x.decode("utf-8") for x in lines if x)
    if len(files) != len(set(files)):
        raise ValueError("Duplicate Git-tracked path")
    return tuple(files)


def _module(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    rel = path[4:] if path.startswith("src/") else path
    if not (path.startswith("src/") or path.startswith(
            ("tests/", "scripts/", "rl_environment/",
             "experiments/", "examples/"))):
        return None
    bits = PurePosixPath(rel).with_suffix("").parts
    if bits[-1] == "__init__":
        bits = bits[:-1]
    return ".".join(bits) if bits else None


def _owner(path: str) -> str:
    p = PurePosixPath(path).parts
    if path.startswith("src/fpgai_research/"):
        return "compiler/" + (p[2] if len(p) > 2 else "root")
    if path.startswith("paper/compiler_ir/"):
        return "paper/compiler_ir"
    if path.startswith("docs/maintainers/"):
        return "docs/maintainers"
    return p[0] if p else "unknown"


def _target(name: str, modules: dict[str, str]) -> str | None:
    # An import can name a member of a package/module, e.g.
    # "from pkg.mod import Function". Follow the longest tracked module.
    parts = name.split(".")
    for last in range(len(parts), 0, -1):
        module = ".".join(parts[:last])
        if module in modules:
            return modules[module]
    return None


def _internal_imports(tree: ast.AST, module: str,
                      package_init: bool, modules: dict[str, str]) -> list[str]:
    imports: set[str] = set()
    package = module if package_init else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                target = _target(item.name, modules)
                if target:
                    imports.add(target)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                components = package.split(".") if package else []
                up = node.level - 1
                if up > len(components):
                    continue
                prefix = components[:len(components) - up] if up else components
                name = ".".join([*prefix, *([node.module] if node.module else [])])
            else:
                name = node.module or ""
            if not name:
                continue
            target = _target(name, modules)
            if target:
                imports.add(target)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = f"{name}.{alias.name}"
                if child in modules:
                    imports.add(modules[child])
    return sorted(imports)


def inspect(root: str | Path) -> dict:
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise ValueError("Source inventory requires an existing Git checkout")
    tracked = _tracked(root)
    modules = {_module(p): p for p in tracked if _module(p)}
    if len(modules) != sum(1 for p in tracked if _module(p)):
        raise ValueError("Conflicting tracked Python import module identities")
    sha = _git(root, "rev-parse", "HEAD").decode().strip()
    if len(sha) != 40:
        raise ValueError("Git HEAD revision not valid")
    current_dirty = _git(root, "status", "--porcelain")
    # Inventory must still be useful on a contributor's dirty worktree.
    # Record this explicitly, never confuse it with a Git commit snapshot.
    rows: list[dict] = []
    direct_importers: dict[str, set[str]] = defaultdict(set)
    parse_errors = []
    for path in tracked:
        relative = PurePosixPath(path)
        if (relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != path):
            raise ValueError("Unsafe Git-tracked file name")
        target = root / path
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"Missing/symlinked tracked file: {path}")
        size = target.stat().st_size
        if size > MAX_SOURCE_BYTES:
            raise ValueError(f"Tracked source file unexpectedly large: {path}")
        data = target.read_bytes()
        entry: dict = {
            "path": path,
            "owner": _owner(path),
            "size_bytes": len(data),
            "sha256": sha256(data).hexdigest(),
            "module": _module(path),
            "internal_imports": [],
            "direct_importers": [],
            "top_level_definitions": [],
            "parse_error": None,
        }
        if path.endswith(".py"):
            try:
                tree = ast.parse(data, filename=path)
            except (SyntaxError, ValueError, UnicodeError) as exc:
                entry["parse_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                parse_errors.append(path)
            else:
                entry["top_level_definitions"] = sorted(
                    node.name for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef)))
                entry["internal_imports"] = _internal_imports(
                    tree, entry["module"] or "",
                    path.endswith("/__init__.py"), modules)
                for dep in entry["internal_imports"]:
                    direct_importers[dep].add(path)
        rows.append(entry)
    for row in rows:
        row["direct_importers"] = sorted(
            direct_importers.get(row["path"], set()))
    owner_counts = dict(sorted(Counter(r["owner"] for r in rows).items()))
    duplicated_bytes = defaultdict(list)
    for row in rows:
        if row["size_bytes"] >= 80:
            duplicated_bytes[(row["size_bytes"], row["sha256"])].append(row["path"])
    duplicates = sorted(
        [paths for paths in duplicated_bytes.values() if len(paths) > 1],
        key=lambda row: row[0])
    return {
        "schema": SCHEMA,
        "git_head": sha,
        "working_tree_clean": not bool(current_dirty.strip()),
        "working_tree_has_changes_not_identical_to_head": bool(current_dirty.strip()),
        "scope": "tracked worktree files; static imports only; no Git history scan",
        "file_count": len(rows),
        "python_module_count": len(modules),
        "python_parse_error_count": len(parse_errors),
        "python_parse_errors": parse_errors,
        "owner_file_counts": owner_counts,
        "exact_byte_duplicate_groups": duplicates,
        "entries": rows,
        "limitations": [
            "Imports identify static dependencies, not runtime call graph",
            "Zero detected importers does NOT establish a file is unused",
            "Git HEAD is not a content attestation when worktree is dirty",
            "No third-party licensing/credential/history/security audit",
            "No native vendor tools, generated assets or model weights loaded",
        ],
    }


def export_create_only(report: dict, *, output: str | Path) -> dict:
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Tracked source inventory is create-only")
    if report.get("schema") != SCHEMA or not report.get("entries"):
        raise ValueError("Invalid repository inventory report")
    destination.mkdir(parents=True, exist_ok=False)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (destination / "inventory.json").write_text(text, encoding="utf-8")
    digest = sha256(text.encode()).hexdigest()
    (destination / "sha256.json").write_text(
        json.dumps({"inventory.json": digest}, indent=2) + "\n",
        encoding="utf-8")
    return {"output": str(destination), "inventory_sha256": digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None,
                        help="Git source root; defaults to this file's repository")
    parser.add_argument("--output", default=None,
                        help="New create-only local inventory directory")
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    report = inspect(root)
    summary = {
        "schema": report["schema"], "git_head": report["git_head"],
        "working_tree_clean": report["working_tree_clean"],
        "tracked_files": report["file_count"],
        "python_modules": report["python_module_count"],
        "parse_errors": report["python_parse_error_count"],
        "duplicate_byte_groups": len(report["exact_byte_duplicate_groups"]),
        "owners": report["owner_file_counts"],
        "read_only": args.output is None,
        "native_vendor_calls": 0,
    }
    if args.output:
        summary.update(export_create_only(report, output=args.output))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
