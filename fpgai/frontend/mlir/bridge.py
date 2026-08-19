from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np

from fpgai.ir import Graph
from fpgai.ir.contracts import GraphSemantics, OpSemantics, TensorSemantics


class MLIRBridgeError(ValueError):
    pass


_SCHEMA = "fpgai.mlir-bridge/v1"
_PAYLOAD_PREFIX = "// fpgai.bridge.payload = "


def _dtype_to_mlir(dtype: str) -> str:
    value = str(dtype).lower()
    return {
        "float32": "f32",
        "float64": "f64",
        "float16": "f16",
        "int8": "i8",
        "uint8": "ui8",
        "int16": "i16",
        "uint16": "ui16",
        "int32": "i32",
        "uint32": "ui32",
        "int64": "i64",
        "bool": "i1",
    }.get(value, "f32")


def _tensor_type(shape: tuple[int, ...], dtype: str) -> str:
    elem = _dtype_to_mlir(dtype)
    if not shape:
        return f"tensor<{elem}>"
    dims = "x".join(str(int(x)) if int(x) > 0 else "?" for x in shape)
    return f"tensor<{dims}x{elem}>"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def mlir_bridge_manifest(graph: Graph) -> Dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "graph": graph.name,
        "ir_schema": getattr(graph, "schema", "fpgai.ir/v1"),
        "graph_semantics": graph.semantics.to_dict(),
        "metadata": _json_safe(getattr(graph, "metadata", {})),
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "tensors": {
            name: {
                "shape": list(spec.shape),
                "dtype": spec.dtype,
                "quantization": _json_safe(spec.quantization),
                "semantics": spec.semantics.to_dict(),
            }
            for name, spec in graph.tensors.items()
        },
        "constants": {name: _json_safe(value) for name, value in graph.constants.items()},
        "operators": [
            {
                "name": op.name,
                "op_type": op.op_type,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": _json_safe(op.attrs),
                "semantics": op.semantics.to_dict(),
            }
            for op in graph.ops
        ],
    }


def export_fpgai_mlir(graph: Graph) -> str:
    """Export a textual MLIR interoperability form using generic FPGAI dialect ops.

    The first phase intentionally uses generic operation syntax so the artifact can be
    inspected without requiring a compiled FPGAI MLIR dialect. The embedded payload is
    a stable round-trip contract. A native TableGen dialect can consume the same schema
    later without changing FPGAI IR ownership.
    """
    manifest = mlir_bridge_manifest(graph)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    args = []
    for index, name in enumerate(graph.inputs):
        spec = graph.get_tensor(name)
        mlir_ty = _tensor_type(tuple(spec.shape), spec.dtype) if spec else "tensor<*xf32>"
        args.append(f"%arg{index}: {mlir_ty}")

    lines = [
        f'{_PAYLOAD_PREFIX}{payload}',
        'module attributes {fpgai.schema = "fpgai.mlir-bridge/v1"} {',
        f'  func.func @{_sanitize_symbol(graph.name)}({", ".join(args)}) attributes {{fpgai.source_ir = "fpgai.ir"}} {{',
    ]
    value_by_tensor = {name: f"%arg{i}" for i, name in enumerate(graph.inputs)}
    next_value = 0
    for op in graph.ops:
        operands = [value_by_tensor.get(name, f'%unresolved_{_sanitize_symbol(name)}') for name in op.inputs]
        results = []
        result_types = []
        for name in op.outputs:
            value = f"%v{next_value}"
            next_value += 1
            value_by_tensor[name] = value
            results.append(value)
            spec = graph.get_tensor(name)
            result_types.append(_tensor_type(tuple(spec.shape), spec.dtype) if spec else "tensor<*xf32>")
        attrs = {
            "name": op.name,
            "op_type": op.op_type,
            "attrs": _json_safe(op.attrs),
            "semantics": op.semantics.to_dict(),
        }
        attr_json = json.dumps(attrs, sort_keys=True, separators=(",", ":")).replace('\\', '\\\\').replace('"', '\\"')
        lhs = ", ".join(results)
        result_prefix = f"{lhs} = " if lhs else ""
        operand_types = []
        for name in op.inputs:
            spec = graph.get_tensor(name)
            operand_types.append(_tensor_type(tuple(spec.shape), spec.dtype) if spec else "tensor<*xf32>")
        function_type = f"({', '.join(operand_types)}) -> ({', '.join(result_types)})"
        lines.append(
            f'    {result_prefix}"fpgai.op"({", ".join(operands)}) '
            f'{{fpgai.payload = "{attr_json}"}} : {function_type}'
        )
    returns = [value_by_tensor.get(name) for name in graph.outputs]
    returns = [item for item in returns if item]
    if returns:
        types = []
        for name in graph.outputs:
            spec = graph.get_tensor(name)
            types.append(_tensor_type(tuple(spec.shape), spec.dtype) if spec else "tensor<*xf32>")
        lines.append(f"    return {', '.join(returns)} : {', '.join(types)}")
    else:
        lines.append("    return")
    lines.extend(["  }", "}", ""])
    return "\n".join(lines)


def _sanitize_symbol(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.$-]", "_", str(value))
    return text or "main"


def write_fpgai_mlir(graph: Graph, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(export_fpgai_mlir(graph), encoding="utf-8")
    return target


def import_fpgai_mlir(source: str | Path) -> Graph:
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    else:
        candidate = str(source)
        if "\n" not in candidate and len(candidate) < 4096:
            path = Path(candidate)
            text = path.read_text(encoding="utf-8") if path.exists() else candidate
        else:
            text = candidate
    payload_line = next((line for line in text.splitlines() if line.startswith(_PAYLOAD_PREFIX)), None)
    if payload_line is None:
        raise MLIRBridgeError("MLIRBRIDGE001: missing FPGAI bridge payload")
    try:
        data = json.loads(payload_line[len(_PAYLOAD_PREFIX):])
    except json.JSONDecodeError as exc:
        raise MLIRBridgeError("MLIRBRIDGE002: malformed FPGAI bridge payload") from exc
    if data.get("schema") != _SCHEMA:
        raise MLIRBridgeError(f"MLIRBRIDGE003: unsupported bridge schema {data.get('schema')!r}")

    graph = Graph(str(data.get("graph") or "mlir_graph"))
    graph.schema = str(data.get("ir_schema") or "fpgai.ir/v2")
    graph.inputs = [str(x) for x in data.get("inputs", [])]
    graph.outputs = [str(x) for x in data.get("outputs", [])]
    graph.metadata = dict(data.get("metadata", {}) or {})
    graph.semantics = _graph_semantics_from_dict(data.get("graph_semantics", {}) or {})
    # The bridge itself is the immediate source representation, while original
    # framework/StableHLO provenance stays in source_metadata/provenance.
    graph.semantics.source_metadata.setdefault("bridge_schema", _SCHEMA)
    graph.semantics.source_metadata.setdefault("imported_via", "mlir_bridge")
    if not graph.semantics.source_ir or graph.semantics.source_ir == "fpgai":
        graph.semantics.source_ir = "mlir"

    for name, record in (data.get("tensors", {}) or {}).items():
        graph.add_tensor(
            str(name),
            tuple(int(x) for x in record.get("shape", [])),
            str(record.get("dtype", "float32")),
            quantization=record.get("quantization"),
        )
        graph.tensors[str(name)].semantics = _tensor_semantics_from_dict(record.get("semantics", {}) or {})
    for name, value in (data.get("constants", {}) or {}).items():
        dtype = graph.tensors.get(name).dtype if name in graph.tensors else None
        graph.constants[str(name)] = np.asarray(value, dtype=dtype)
    for record in data.get("operators", []) or []:
        op = graph.add_op(
            str(record.get("op_type")),
            [str(x) for x in record.get("inputs", [])],
            [str(x) for x in record.get("outputs", [])],
            name=str(record.get("name") or ""),
            attrs=dict(record.get("attrs", {}) or {}),
        )
        op.semantics = _op_semantics_from_dict(record.get("semantics", {}) or {})
    return graph



def _graph_semantics_from_dict(data: Dict[str, Any]) -> GraphSemantics:
    result = GraphSemantics()
    for key in (
        "pipeline_mode", "target_board", "ir_level", "runtime_contract",
        "resource_constraints", "execution", "source_ir", "source_metadata",
        "provenance", "lowering_history",
    ):
        if key not in data or not hasattr(result, key):
            continue
        value = data[key]
        if key == "lowering_history":
            value = tuple(dict(item) for item in (value or []))
        elif key in {"runtime_contract", "resource_constraints", "execution", "source_metadata", "provenance"}:
            value = dict(value or {})
        setattr(result, key, value)
    return result


def _tensor_semantics_from_dict(data: Dict[str, Any]) -> TensorSemantics:
    result = TensorSemantics()
    memory = data.get("memory", {}) or {}
    transport = data.get("transport", {}) or {}
    training = data.get("training", {}) or {}
    state = data.get("state", {}) or {}
    for key, value in memory.items():
        if hasattr(result.memory, key): setattr(result.memory, key, value)
    for key, value in transport.items():
        if hasattr(result.transport, key): setattr(result.transport, key, value)
    for key, value in training.items():
        if hasattr(result.training, key): setattr(result.training, key, value)
    for key, value in state.items():
        if hasattr(result.state, key): setattr(result.state, key, value)
    result.tags = tuple(str(x) for x in data.get("tags", []) or [])
    return result


def _op_semantics_from_dict(data: Dict[str, Any]) -> OpSemantics:
    result = OpSemantics()
    result.selected_backend = data.get("selected_backend")
    result.selected_implementation_id = data.get("selected_implementation_id")
    result.buffering = dict(data.get("buffering", {}) or {})
    result.schedule = dict(data.get("schedule", {}) or {})
    result.execution = dict(data.get("execution", {}) or {})
    result.training = dict(data.get("training", {}) or {})
    result.resource_constraints = dict(data.get("resource_constraints", {}) or {})
    result.provenance = dict(data.get("provenance", {}) or {})
    result.lowering_history = tuple(dict(item) for item in data.get("lowering_history", []) or [])
    result.tags = tuple(str(x) for x in data.get("tags", []) or [])
    candidates = []
    for item in data.get("implementation_candidates", []) or []:
        from fpgai.ir.contracts import ImplementationCandidate
        candidates.append(ImplementationCandidate(**item))
    result.implementation_candidates = tuple(candidates)
    return result


def canonical_ir_equivalence_manifest(graph: Graph, *, include_parameter_values: bool = False) -> Dict[str, Any]:
    """Build a source-name-independent semantic signature for cross-frontend comparison.

    Tensor/value names and source provenance are deliberately excluded. Constants are
    represented by shape/dtype by default; value hashes can be included when identical
    parameter materialization is part of the experiment.
    """
    token_by_tensor: Dict[str, str] = {}
    for index, name in enumerate(graph.inputs):
        token_by_tensor[name] = f"arg{index}"
    constants = []
    for index, (name, value) in enumerate(sorted(graph.constants.items())):
        arr = np.asarray(value)
        token_by_tensor[name] = f"const{index}"
        rec = {"token": token_by_tensor[name], "shape": list(arr.shape), "dtype": str(arr.dtype)}
        if include_parameter_values:
            rec["sha256"] = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
        constants.append(rec)
    operators = []
    next_value = 0
    ignored_attrs = {"stablehlo_op", "onnx_domain", "onnx_opset", "source_framework"}
    for op in graph.ops:
        inputs = [token_by_tensor.get(name, f"unresolved:{name}") for name in op.inputs]
        outputs = []
        output_types = []
        for name in op.outputs:
            token = f"v{next_value}"
            next_value += 1
            token_by_tensor[name] = token
            outputs.append(token)
            spec = graph.get_tensor(name)
            output_types.append({"shape": list(spec.shape) if spec else None, "dtype": spec.dtype if spec else None})
        attrs = {k: _json_safe(v) for k, v in sorted(op.attrs.items()) if k not in ignored_attrs}
        operators.append({"op_type": op.op_type, "inputs": inputs, "outputs": outputs, "output_types": output_types, "attrs": attrs})
    outputs = [token_by_tensor.get(name, f"unresolved:{name}") for name in graph.outputs]
    payload = {"schema": "fpgai.frontend-equivalence/v1", "inputs": [token_by_tensor[n] for n in graph.inputs], "constants": constants, "operators": operators, "outputs": outputs}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def compare_canonical_ir(graphs: Dict[str, Graph], *, include_parameter_values: bool = False) -> Dict[str, Any]:
    manifests = {name: canonical_ir_equivalence_manifest(graph, include_parameter_values=include_parameter_values) for name, graph in graphs.items()}
    fingerprints = {name: item["fingerprint_sha256"] for name, item in manifests.items()}
    unique = sorted(set(fingerprints.values()))
    return {
        "schema": "fpgai.frontend-equivalence-report/v1",
        "status": "equivalent" if len(unique) <= 1 else "different",
        "include_parameter_values": bool(include_parameter_values),
        "fingerprints": fingerprints,
        "equivalent": len(unique) <= 1,
    }
