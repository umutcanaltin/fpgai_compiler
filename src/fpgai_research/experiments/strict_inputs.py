"""Strict, provenance-preserving input and output contracts for paper sweeps.

Source-only E1 preparation; this module never executes Vitis or Vivado.
"""
from contextlib import contextmanager
from pathlib import Path
import tempfile

import numpy as np


def read_float32_input(filename: str | Path) -> np.ndarray:
    """Read exact, finite float32 .npy input without silent conversions."""
    filename = Path(filename)
    if filename.is_symlink() or not filename.is_file():
        raise ValueError(f"Experiment input must be a regular, non-linked .npy file: {filename}")
    values = np.load(filename, allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32"):
        raise ValueError(
            f"Input {filename} has dtype {getattr(values, 'dtype', type(values))}; "
            "exact float32 required (no implicit cast or quantization)"
        )
    if values.ndim == 0 or values.size == 0:
        raise ValueError("Input must have at least one axis and one scalar")
    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or infinity")
    return values


@contextmanager
def new_campaign_directory(target: str | Path):
    """Write into private staging and publish only after successful preparation.

    Refuse to overwrite any existing output, including dangling symlinks.
    The final existence check is a fail-closed guard, not a portable
    interprocess no-replace lock; distinct processes must use distinct paths.
    """
    target = Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Campaign already exists; use a fresh path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.stage-", dir=target.parent) as stage:
        stage_dir = Path(stage)
        yield stage_dir
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Campaign created during staging: {target}")
        stage_dir.rename(target)
