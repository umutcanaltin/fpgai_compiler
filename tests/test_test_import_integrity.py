"""Guard retained core regression tests against unsafe source-test deletions.

This test is structural only: no vendor programs, numerical surrogates, or
reliance on archived/removed research entrypoints.
"""
from __future__ import annotations

import ast
from pathlib import Path


def test_all_retained_local_test_imports_exist():
    root = Path(__file__).resolve().parent
    missing = []
    for source in sorted(root.glob("test_*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                name = (node.module or "").split(".")[0]
                if name.startswith("test_") and not (root / (name + ".py")).is_file():
                    missing.append(f"{source.name} -> {name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name.startswith("test_") and not (root / (name + ".py")).is_file():
                        missing.append(f"{source.name} -> {name}")
    assert not missing, "Retained test fixtures missing after cleanup:\n" + "\n".join(missing)
