from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

DType = str


@dataclass
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: DType = "float32"
