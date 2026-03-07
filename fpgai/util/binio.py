from __future__ import annotations

from pathlib import Path
import numpy as np


def write_f32_bin(path: Path, vec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    path.write_bytes(vec.tobytes())


def read_f32_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    return np.frombuffer(data, dtype=np.float32)
