from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Optional
import numpy as np


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return int(p)


def infer_feature_dim(shape: Tuple[int, ...]) -> int:
    """
    Given an ONNX-ish input shape, return feature count.
    Common patterns:
      (1, 8) -> 8
      (1, 1, 1, 8) -> 8
      (N, C, H, W) -> C*H*W (excluding batch)
    """
    if not shape:
        return 1
    if len(shape) == 1:
        return int(shape[0])
    # drop batch dim
    return _prod(shape[1:])


def make_deterministic_input_features(
    n_features: int,
    *,
    kind: str = "arange",
    increment: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns float32 vector shape (n_features,).
    """
    n_features = int(n_features)
    if n_features <= 0:
        raise ValueError(f"n_features must be > 0, got {n_features}")

    if kind == "arange":
        x = (np.arange(n_features, dtype=np.float32) + 1.0) * np.float32(increment)
        return x.astype(np.float32, copy=False)

    if kind == "uniform":
        rng = np.random.default_rng(seed)
        x = rng.uniform(low=-1.0, high=1.0, size=(n_features,)).astype(np.float32)
        return x

    raise ValueError(f"Unknown input kind: {kind}")


def write_input_bin(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x.tofile(str(path))
