"""Publish prepared research cases only after every preparation step succeeds.

The CLI never exposes incomplete source, Tcl, private archives or IR snapshots.
Native vendor execution remains explicit and separate from preparation.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

@contextmanager
def staged_experiment(destination: str | Path):
    requested=Path(destination)
    requested.parent.mkdir(parents=True,exist_ok=True)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Experiment destination already exists: {requested}")
    temp=Path(tempfile.mkdtemp(prefix=f".{requested.name}.staging-",dir=requested.parent))
    try:
        yield temp
        if requested.exists() or requested.is_symlink():
            raise FileExistsError(f"Experiment destination created during preparation: {requested}")
        temp.rename(requested)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
