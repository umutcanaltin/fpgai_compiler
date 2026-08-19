from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Iterable
import copy
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


    def extract_subgraph(
        self,
        op_names: Iterable[str],
        *,
        name: str | None = None,
    ) -> "Graph":
        """Return an IR-owned standalone subgraph containing the selected ops.

        External runtime dependencies become graph inputs, selected constants stay
        embedded, and values produced by the selection but consumed outside it become
        graph outputs.  The method preserves operator/tensor semantics and metadata so
        backend export uses the same resolved FPGAI IR contracts rather than rebuilding
        an export-only representation.
        """
        requested = tuple(str(item) for item in op_names)
        if not requested:
            raise ValueError("IRSUB001: at least one operator name is required")
        requested_set = set(requested)
        by_name = {op.name: op for op in self.ops}
        missing = [item for item in requested if item not in by_name]
        if missing:
            raise KeyError(f"IRSUB002: unknown operator(s): {sorted(set(missing))}")

        selected_ops = [op for op in self.ops if op.name in requested_set]
        selected_names = {op.name for op in selected_ops}
        produced_by_selected = {str(tensor) for op in selected_ops for tensor in op.outputs}
        consumed_by_selected = {str(tensor) for op in selected_ops for tensor in op.inputs}
        constants = set(self.constants)

        # Preserve dependency ordering based on the selected operations' original order.
        sub_inputs: list[str] = []
        for op in selected_ops:
            for tensor in op.inputs:
                tensor = str(tensor)
                if tensor in constants or tensor in produced_by_selected:
                    continue
                if tensor not in sub_inputs:
                    sub_inputs.append(tensor)

        # Any selected value whose consumer is outside the selection is externally visible.
        outside_consumed: set[str] = set()
        for op in self.ops:
            if op.name in selected_names:
                continue
            outside_consumed.update(str(tensor) for tensor in op.inputs)
        sub_outputs: list[str] = []
        for op in selected_ops:
            for tensor in op.outputs:
                tensor = str(tensor)
                if tensor in outside_consumed or tensor in self.outputs or tensor not in consumed_by_selected:
                    if tensor not in sub_outputs:
                        sub_outputs.append(tensor)
        if not sub_outputs:
            sub_outputs.extend(str(tensor) for tensor in selected_ops[-1].outputs)

        required_tensors = set(sub_inputs) | set(sub_outputs)
        required_constants: set[str] = set()
        for op in selected_ops:
            required_tensors.update(str(tensor) for tensor in op.inputs)
            required_tensors.update(str(tensor) for tensor in op.outputs)
            required_constants.update(str(tensor) for tensor in op.inputs if str(tensor) in constants)

        result = Graph(name=name or f"{self.name}_subgraph")
        result.schema = self.schema
        result.inputs = sub_inputs
        result.outputs = sub_outputs
        result.ops = [copy.deepcopy(op) for op in selected_ops]
        result.tensors = {
            tensor: copy.deepcopy(spec)
            for tensor, spec in self.tensors.items()
            if tensor in required_tensors
        }
        result.constants = {
            tensor: np.array(self.constants[tensor], copy=True)
            for tensor in required_constants
        }
        result.semantics = copy.deepcopy(self.semantics)
        result.metadata = copy.deepcopy(self.metadata)
        result.metadata["subgraph_export"] = {
            "source_graph": self.name,
            "selected_ops": [op.name for op in selected_ops],
            "external_inputs": list(sub_inputs),
            "external_outputs": list(sub_outputs),
        }
        return result

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
