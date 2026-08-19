from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .contracts import ImplementationCandidate
from .graph import Graph


def annotate_default_hardware_semantics(
    graph: Graph,
    *,
    pipeline_mode: str = "inference",
    target_board: str | None = None,
) -> Graph:
    from fpgai.operators import get_builtin_operator_contract
    """Populate conservative hardware-semantic defaults without selecting a backend.

    This pass records candidates and data roles only. It does not silently force HLS,
    VHDL, memory placement, or transport; those remain planner/configuration choices.
    """
    graph.semantics.pipeline_mode = pipeline_mode
    graph.semantics.target_board = target_board

    constants = set(getattr(graph, "constants", {}) or {})
    for name, tensor in graph.tensors.items():
        if name in constants:
            tensor.semantics.training.role = "parameter"
            tensor.semantics.training.requires_gradient = pipeline_mode == "training_on_device"
        elif name in graph.inputs:
            tensor.semantics.training.role = "input"
        elif name in graph.outputs:
            tensor.semantics.training.role = "output"
        else:
            tensor.semantics.training.role = "activation"
            tensor.semantics.training.requires_gradient = pipeline_mode == "training_on_device"

    for op in graph.ops:
        try:
            contract = get_builtin_operator_contract(op.op_type, pipeline_mode=pipeline_mode)
        except KeyError:
            continue
        candidates = []
        if contract.capabilities.inference:
            candidates.append(
                ImplementationCandidate(
                    backend="hls",
                    implementation_id=f"builtin:{contract.canonical_op_type}",
                    status="supported_or_limited",
                )
            )
        # VHDL remains explicit/plugin-driven. Do not claim a builtin implementation.
        op.semantics.implementation_candidates = tuple(candidates)
        op.semantics.training = {
            "forward": bool(contract.capabilities.training_forward),
            "backward_input": bool(contract.capabilities.backward_input),
            "parameter_gradients": bool(contract.capabilities.parameter_gradients),
        }
    return graph


def graph_semantics_report(graph: Graph) -> Dict[str, Any]:
    return {
        "schema": "fpgai.ir-semantics-report/v1",
        "graph": graph.name,
        "ir_schema": getattr(graph, "schema", "fpgai.ir/v1"),
        "graph_semantics": graph.semantics.to_dict(),
        "tensors": {
            name: {
                "shape": list(spec.shape),
                "dtype": spec.dtype,
                "quantization": spec.quantization,
                "semantics": spec.semantics.to_dict(),
            }
            for name, spec in sorted(graph.tensors.items())
        },
        "operators": [
            {
                "name": op.name,
                "op_type": op.op_type,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attributes": dict(op.attrs),
                "semantics": op.semantics.to_dict(),
            }
            for op in graph.ops
        ],
    }


def write_graph_semantics_report(graph: Graph, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph_semantics_report(graph), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
