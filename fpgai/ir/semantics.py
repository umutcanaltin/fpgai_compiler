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
    graph.semantics.ir_level = "functional"
    graph.semantics.provenance.setdefault("source_ir", graph.semantics.source_ir)
    graph.semantics.provenance.setdefault("pipeline_mode", pipeline_mode)
    if target_board:
        graph.semantics.provenance.setdefault("target_board", target_board)

    constants = set(getattr(graph, "constants", {}) or {})
    for name, tensor in graph.tensors.items():
        if name in constants:
            tensor.semantics.training.role = "parameter"
            tensor.semantics.training.requires_gradient = pipeline_mode == "training_on_device"
            tensor.semantics.memory.persistence = "model"
            tensor.semantics.memory.mutable = pipeline_mode == "training_on_device"
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
        op.semantics.execution.setdefault("model", {})
        op.semantics.execution.setdefault("layer", {})
        op.semantics.execution.setdefault("loops", {})
        op.semantics.provenance.setdefault("fpgai_op_name", str(op.name))
        op.semantics.provenance.setdefault("source_op_type", str(op.op_type))
    return graph


def attach_source_provenance(graph: Graph) -> Graph:
    """Mirror frontend/source provenance into stable FPGAI IR semantics.

    Frontends already populate ``graph.metadata["source"]``.  This helper makes
    that provenance part of the authoritative IR contract and assigns stable
    source/lowering breadcrumbs to operators without inventing a second mapping.
    """
    source = dict((getattr(graph, "metadata", {}) or {}).get("source", {}) or {})
    if source:
        graph.semantics.provenance.update(source)
        graph.semantics.source_metadata.update(source)
        graph.semantics.source_ir = str(source.get("format") or graph.semantics.source_ir)
    for index, op in enumerate(graph.ops):
        op.semantics.provenance.setdefault("source_index", index)
        op.semantics.provenance.setdefault("source_framework", source.get("framework"))
        op.semantics.provenance.setdefault("source_format", source.get("format"))
        if not op.semantics.lowering_history:
            op.semantics.lowering_history = ({
                "stage": "functional_import",
                "representation": "fpgai_ir",
                "op_type": str(op.op_type),
            },)
    return graph


def mark_ir_level(graph: Graph, level: str, *, reason: str) -> Graph:
    normalized = str(level).strip().lower()
    if normalized not in {"functional", "architectural", "lowered"}:
        raise ValueError(f"Unsupported FPGAI IR level {level!r}")
    previous = str(getattr(graph.semantics, "ir_level", "functional"))
    graph.semantics.ir_level = normalized
    history = list(graph.semantics.lowering_history)
    entry = {"from": previous, "to": normalized, "reason": str(reason)}
    if not history or history[-1] != entry:
        history.append(entry)
    graph.semantics.lowering_history = tuple(history)
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
