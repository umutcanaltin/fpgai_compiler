"""Backward-compatible access to the canonical numeric validation module.

The implementation owner is :mod:`fpgai.validation.numeric`. This module is
kept only for external imports that use the historical ``fpgai.numeric`` path.
"""
from fpgai.validation import numeric as _canonical

for _name in dir(_canonical):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_canonical, _name)

__all__ = [name for name in dir(_canonical) if not name.startswith("_")]
