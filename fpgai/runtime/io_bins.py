from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class InputGenSpec:
    """
    Deterministic input generator for regression tests.

    kind:
      - ramp:   0.1, 0.2, 0.3, ...
      - zeros
      - ones
      - random: seeded normal distribution
    """
    kind: str = "ramp"
    seed: int = 42
    ramp_step: float = 0.1


def infer_1d_feature_count_from_shape(shape: Sequence[int]) -> int:
    """
    Our current hostcpp/hls stubs use a 1D feature vector.
    We treat the last dimension as feature count.
    """
    if not shape:
        raise ValueError("cannot infer feature count from empty shape")
    n = int(shape[-1])
    if n <= 0:
        raise ValueError(f"invalid feature dim: {shape[-1]}")
    return n


def generate_input_vector(n: int, spec: InputGenSpec) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"input size must be > 0, got {n}")

    k = spec.kind.lower().strip()

    if k == "ramp":
        # 0.1, 0.2, ...
        step = np.float32(spec.ramp_step)
        x = (np.arange(n, dtype=np.float32) + 1.0) * step
        return x

    if k == "zeros":
        return np.zeros((n,), dtype=np.float32)

    if k == "ones":
        return np.ones((n,), dtype=np.float32)

    if k == "random":
        rng = np.random.default_rng(int(spec.seed))
        return rng.standard_normal(n, dtype=np.float32)

    raise ValueError(f"unknown input_gen.kind: {spec.kind!r}")


def write_input_bin(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(x, dtype=np.float32).tofile(str(path))


def read_input_bin(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32)
