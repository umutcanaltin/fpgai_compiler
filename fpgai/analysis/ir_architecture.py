from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from fpgai.ir.graph import Graph


_ATTENTION_OPS = {"MatMul", "Softmax", "LayerNormalization", "Transpose", "Add", "Mul", "Dense"}


def analyze_ir_architecture(graph: Graph) -> Dict[str, Any]:
    op_types = [op.op_type for op in graph.ops]
    attention_present = {name: op_types.count(name) for name in sorted(_ATTENTION_OPS) if name in op_types}
    semantic_tensor_count = sum(
        1 for spec in graph.tensors.values()
        if spec.quantization is not None
        or spec.semantics.memory.storage != "unspecified"
        or spec.semantics.transport.protocol != "unspecified"
        or spec.semantics.training.role != "activation"
    )
    implementation_annotated = sum(
        1 for op in graph.ops
        if op.semantics.implementation_candidates or op.semantics.selected_backend
    )
    return {
        "schema": "fpgai.ir-architecture-analysis/v1",
        "graph": graph.name,
        "ir_schema": getattr(graph, "schema", "fpgai.ir/v1"),
        "operator_count": len(graph.ops),
        "tensor_count": len(graph.tensors),
        "attention_operator_inventory": attention_present,
        "semantic_tensor_count": semantic_tensor_count,
        "implementation_annotated_operator_count": implementation_annotated,
        "representations": {
            "onnx": {
                "role": "portable model/operator interchange and source graph semantics",
                "owned_by_fpgai": False,
            },
            "mlir": {
                "role": "frontend interoperability/canonicalization layer for StableHLO and supported MLIR dialects before FPGAI architectural semantics",
                "owned_by_fpgai": False,
                "bridge_schema": "fpgai.mlir-bridge/v1",
            },
            "fpgai_ir": {
                "role": "authoritative FPGA architecture IR that preserves functional semantics while adding hierarchical model/layer/loop execution, memory, transport, training state, implementation and validation contracts",
                "owned_by_fpgai": True,
            },
        },
        "scientific_positioning": {
            "claim": "FPGAI IR complements rather than replaces MLIR: framework-native paths lower through MLIR/StableHLO, ONNX remains a parallel interchange frontend, and FPGAI IR is the boundary where functional computation becomes explicit configurable FPGA architecture with model/layer/loop scheduling, memory, transport, persistent training/inference state, implementation selection, validation provenance and runtime contracts.",
            "mlir_replacement_claim": False,
            "contribution_axes": [
                "MLIR/StableHLO interoperability plus ONNX interchange ingress",
                "source-framework-independent computation normalization",
                "explicit user-selectable hierarchical FPGA architecture semantics",
                "model-wise, layer-wise and loop-wise parallelism/pipelining",
                "unified inference/training tensor and operator semantics",
                "memory/transport/persistent-state semantics",
                "backend implementation-selection semantics",
                "requested-to-resolved-to-lowered-to-observed traceability boundary",
            ],
            "physical_validation_boundary": "resolved IR records compiler decisions; HLS, Vivado and runtime artifacts establish physical implementation observations",
        },
    }


def write_ir_architecture_analysis(graph: Graph, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(analyze_ir_architecture(graph), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def resolved_ir_snapshot(
    graph: Graph,
    *,
    compile_plan: Any = None,
    memory_plan: Any = None,
    communication_plan: Any = None,
    runtime_sequence: Any = None,
    training_plan: Any = None,
) -> Dict[str, Any]:
    """Return a deterministic, versioned snapshot of resolved FPGAI IR semantics.

    This is the scientific inspection artifact for the compiler boundary: source
    computation plus FPGAI-owned architecture, memory, transport, training, state,
    runtime and implementation semantics.  It intentionally references the existing
    planner/runtime objects rather than duplicating their resolution logic.
    """
    graph_report = {
        "name": graph.name,
        "schema": getattr(graph, "schema", "fpgai.ir/v2"),
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "semantics": graph.semantics.to_dict(),
        "metadata": _json_safe(getattr(graph, "metadata", {})),
    }
    tensors = {
        name: {
            "shape": list(spec.shape),
            "dtype": spec.dtype,
            "quantization": _json_safe(spec.quantization),
            "semantics": spec.semantics.to_dict(),
            "constant": name in graph.constants,
        }
        for name, spec in sorted(graph.tensors.items())
    }
    operators = [
        {
            "index": index,
            "name": op.name,
            "op_type": op.op_type,
            "inputs": list(op.inputs),
            "outputs": list(op.outputs),
            "attributes": _json_safe(op.attrs),
            "semantics": op.semantics.to_dict(),
        }
        for index, op in enumerate(graph.ops)
    ]
    payload = {
        "schema": "fpgai.resolved-ir/v1",
        "ir_schema": getattr(graph, "schema", "fpgai.ir/v2"),
        "graph": graph_report,
        "tensors": tensors,
        "operators": operators,
        "resolved_plans": {
            "compile": _json_safe(compile_plan) if compile_plan is not None else None,
            "memory": _json_safe(memory_plan) if memory_plan is not None else None,
            "communication": _json_safe(communication_plan) if communication_plan is not None else None,
            "runtime_sequence": _json_safe(runtime_sequence),
            "training": _json_safe(training_plan) if training_plan is not None else None,
        },
        "semantic_ownership": {
            "computation": "FPGAI IR operators/tensors",
            "architecture": "FPGAI IR op.schedule + existing compile plan",
            "memory": "FPGAI IR tensor.memory + existing memory plan",
            "transport": "FPGAI IR tensor.transport + existing communication plan",
            "training": "FPGAI IR tensor/op training semantics + existing training plan",
            "persistent_state": "FPGAI IR tensor.state",
            "hierarchical_execution": "FPGAI graph/op execution semantics + existing compile plan",
            "provenance": "FPGAI graph/op provenance + lowering history",
            "runtime": "FPGAI graph runtime contract + runtime sequence",
            "implementation": "FPGAI IR implementation candidates/selection",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["resolved_ir_fingerprint_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_resolved_ir_snapshot(graph: Graph, path: str | Path, **kwargs: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(resolved_ir_snapshot(graph, **kwargs), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def ir_scientific_capability_matrix(graph: Graph) -> Dict[str, Any]:
    """Summarize which scientific semantic dimensions are materially present in IR."""
    ops = list(graph.ops)
    tensors = list(graph.tensors.values())
    dimensions = {
        "computation": bool(ops),
        "precision_quantization": any(t.quantization is not None for t in tensors) or any("precision" in op.semantics.schedule.get("architecture", {}) for op in ops),
        "pipeline_schedule": any(bool(op.semantics.schedule.get("pipeline")) for op in ops),
        "parallelism": any(bool(op.semantics.schedule.get("parallelism")) for op in ops),
        "partitioning": any(bool(op.semantics.schedule.get("partitioning")) for op in ops),
        "tiling": any(bool(op.semantics.schedule.get("tiling")) for op in ops),
        "buffering": any(bool(op.semantics.buffering) for op in ops),
        "memory": any(t.semantics.memory.storage != "unspecified" for t in tensors) or any("resolved_memory" in op.semantics.resource_constraints for op in ops),
        "transport": any(t.semantics.transport.protocol != "unspecified" for t in tensors),
        "training": any(t.semantics.training.requires_gradient or t.semantics.training.role == "parameter" for t in tensors) or any(bool(op.semantics.training) for op in ops),
        "persistent_state": any(t.semantics.state.persistent_across_invocations for t in tensors),
        "runtime": bool(graph.semantics.runtime_contract),
        "implementation_selection": any(bool(op.semantics.implementation_candidates) or bool(op.semantics.selected_backend) for op in ops),
        "hierarchical_execution": any(bool(getattr(op.semantics, "execution", {})) for op in ops) or bool(getattr(graph.semantics, "execution", {})),
        "source_provenance": bool(getattr(graph.semantics, "provenance", {})) or any(bool(getattr(op.semantics, "provenance", {})) for op in ops),
        "progressive_lowering": bool(getattr(graph.semantics, "lowering_history", ())),
        "memory_initialization": any(t.semantics.memory.initialization_mode != "unspecified" or t.semantics.memory.initialization_source != "unspecified" for t in tensors),
    }
    return {
        "schema": "fpgai.ir-scientific-capability-matrix/v1",
        "ir_schema": getattr(graph, "schema", "fpgai.ir/v2"),
        "graph": graph.name,
        "dimensions": dimensions,
        "represented_dimension_count": sum(1 for value in dimensions.values() if value),
        "dimension_count": len(dimensions),
        "claim_boundary": "presence means represented in authoritative IR semantics; physical validation is reported separately by HLS/Vivado/runtime artifacts",
    }
