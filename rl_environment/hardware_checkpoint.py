"""Import-stable compatibility alias for `rl_environment.infrastructure.hardware_checkpoint`.

This aliases the same module object, rather than copying symbols, so previous
CLI names and test monkeypatch targets keep their original semantics.
"""
from __future__ import annotations

import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module(".infrastructure.hardware_checkpoint", package=__package__)
_sys.modules[__name__] = _implementation
