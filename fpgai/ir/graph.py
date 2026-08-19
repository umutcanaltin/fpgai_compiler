from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .contracts import GraphSemantics, OpSemantics, TensorSemantics
from .ops import Op as CanonicalOp
from .types import TensorSpec


def Op(op_type: str, name: str, inputs, outputs, attrs=None):
    """Backward-compatible constructor for historic fpgai.ir.graph.Op positional order.

    The canonical IR class lives in fpgai.ir.ops.Op with named fields.
    """
    return CanonicalOp(name=name, op_type=op_type, inputs=list(inputs), outputs=list(outputs), attrs=dict(attrs or {}))


class Graph:
    def __init__(self, name: str = "main"):
        self.name = name
        self.schema = "fpgai.ir/v2"
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.ops: List[Op] = []
        self.tensors: Dict[str, TensorSpec] = {}
        self.constants: Dict[str, np.ndarray] = {}
        self.semantics = GraphSemantics()
        self.metadata: Dict[str, Any] = {}

    def add_tensor(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: str = "float32",
        *,
        quantization: Optional[Dict[str, Any]] = None,
        semantics: Optional[TensorSemantics] = None,
    ):
        self.tensors[name] = TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            quantization=quantization,
            semantics=semantics or TensorSemantics(),
        )

    def set_tensor_quantization(self, name: str, quantization: Optional[Dict[str, Any]]) -> None:
        tensor = self.tensors.get(name)
        if tensor is None:
            raise KeyError(f"Unknown tensor {name!r}")
        tensor.quantization = None if quantization is None else dict(quantization)

    def get_tensor(self, name: str) -> Optional[TensorSpec]:
        return self.tensors.get(name)

    def add_op(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        name: str = "",
        attrs: Dict[str, Any] = None,
        *,
        semantics: Optional[OpSemantics] = None,
    ):
        if not name:
            name = f"{op_type}_{len(self.ops)}"
        op = CanonicalOp(
            name=name,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs or {},
            semantics=semantics or OpSemantics(),
        )
        self.ops.append(op)
        return op

    def summary(self) -> str:
        lines = [f"Graph Name: {self.name}", f"IR Schema: {self.schema}"]
        lines.append(f"Inputs: {self.inputs}")
        lines.append(f"Outputs: {self.outputs}")
        lines.append(f"Constants (Weights): {len(self.constants)} items")
        lines.append("-" * 30)
        lines.append("Operations:")
        for i, op in enumerate(self.ops):
            lines.append(f"  {i:02d}. {op.op_type:<20} | Name: {op.name}")
            lines.append(f"      In: {op.inputs}")
            lines.append(f"      Out: {op.outputs}")
            if op.semantics.selected_backend:
                lines.append(f"      Backend: {op.semantics.selected_backend}")
        lines.append("-" * 30)
        return "\n".join(lines)
