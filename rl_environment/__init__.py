"""FPGAI research Gym interface, separated from the core compiler.

Importing this package neither launches vendor tools nor reads private results.

Developer source checkouts use a `src/` compiler layout, whereas Python's
editable-install .pth files can still reference an older separately installed
research checkout. Install the *sibling* src/ first so `python -m
rl_environment.gym_curriculum` and pytest resolve the SAME compiler. The
native source binding remains fail-closed in runtime_contract.py.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _prefer_this_checkout_compiler() -> None:
    source_root = (Path(__file__).resolve().parents[1] / "src")
    package_root = source_root / "fpgai_research"
    if not package_root.is_dir():
        return  # A future wheel-only layout may install modules elsewhere.
    canonical = source_root.resolve()
    # Multiple editable installs may have different .pth entries. Prepending
    # this checkout is deterministic, does not import compiler modules, and
    # never mutates any native case or checkpoint.
    sys.path[:] = [
        str(canonical),
        *(entry for entry in sys.path
          if not entry or Path(entry).resolve() != canonical),
    ]


_prefer_this_checkout_compiler()
__all__: list[str] = []
