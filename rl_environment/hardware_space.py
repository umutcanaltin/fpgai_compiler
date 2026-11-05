"""Import-stable compatibility alias for `rl_environment.environments.hardware_space`.

This aliases the same module object, rather than copying symbols, so previous
CLI names and test monkeypatch targets keep their original semantics.
"""
from __future__ import annotations

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    from .environments.hardware_space import main as _main
    raise SystemExit(_main())

_implementation = _importlib.import_module(".environments.hardware_space", package=__package__)
_sys.modules[__name__] = _implementation
