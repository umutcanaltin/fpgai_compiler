from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

@dataclass
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str

@dataclass
class Op:
    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)

class Graph:
    def __init__(self, name: str = "main"):
        self.name = name
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.ops: List[Op] = []
        self.tensors: Dict[str, TensorSpec] = {}
        
        # Dictionary to store actual weight values (name -> numpy array)
        self.constants: Dict[str, np.ndarray] = {}

    def add_tensor(self, name: str, shape: Tuple[int, ...], dtype: str = "float32"):
        self.tensors[name] = TensorSpec(name, shape, dtype)

    def get_tensor(self, name: str) -> Optional[TensorSpec]:
        return self.tensors.get(name)

    def add_op(self, op_type: str, inputs: List[str], outputs: List[str], name: str = "", attrs: Dict[str, Any] = None):
        if not name:
            name = f"{op_type}_{len(self.ops)}"
        op = Op(op_type, name, inputs, outputs, attrs or {})
        self.ops.append(op)
        return op

    def summary(self) -> str:
        """Returns a human-readable summary of the graph structure."""
        lines = [f"Graph Name: {self.name}"]
        lines.append(f"Inputs: {self.inputs}")
        lines.append(f"Outputs: {self.outputs}")
        lines.append(f"Constants (Weights): {len(self.constants)} items")
        lines.append("-" * 30)
        lines.append("Operations:")
        for i, op in enumerate(self.ops):
            lines.append(f"  {i:02d}. {op.op_type:<15} | Name: {op.name}")
            lines.append(f"      In: {op.inputs}")
            lines.append(f"      Out: {op.outputs}")
        lines.append("-" * 30)
        return "\n".join(lines)