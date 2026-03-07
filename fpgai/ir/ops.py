from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Op:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


def make_name(op_type: str, idx: int, node_name: Optional[str] = None) -> str:
    base = node_name.strip() if (node_name and node_name.strip()) else f"{op_type}_{idx}"
    return base
