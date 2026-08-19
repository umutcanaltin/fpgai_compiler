from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .contracts import TensorSemantics

DType = str


@dataclass
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: DType = "float32"
    quantization: Optional[Dict[str, Any]] = None
    semantics: TensorSemantics = field(default_factory=TensorSemantics)
