from __future__ import annotations
import re
from typing import Any, Mapping
from fpgai.implementations.hls_integration import parse_hls_abi, HLSFlatArrayABI, HLSTensorPortsABI
from .composition_errors import HLSCompositionError
from .composition_types import ExternalNodeBinding, HLSCompositionPlan

def _shape_words(graph: Any, tensor_name: str) -> int:
    spec = graph.get_tensor(tensor_name)
    shape = getattr(spec, "shape", None) if spec is not None else None
    if not shape:
        raise HLSCompositionError(f"HLSCOMP004: missing static shape for tensor {tensor_name!r}")
    total = 1
    for dim in shape:
        value = int(dim)
        if value <= 0:
            raise HLSCompositionError(f"HLSCOMP004: dynamic shape for tensor {tensor_name!r}")
        total *= value
    return total

def _symbol(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)

def build_hls_composition_plan(graph: Any, *, selected_contracts: Mapping[str, Any], selection_reports: Mapping[str, Mapping[str, str]] | None = None) -> HLSCompositionPlan:
    bindings: list[ExternalNodeBinding] = []
    current_tensor = graph.inputs[0] if getattr(graph, "inputs", None) else None
    try:
        from fpgai.ir.liveness import analyze_tensor_liveness
        has_branching = bool(analyze_tensor_liveness(graph).get("has_branching", False))
    except Exception:
        has_branching = False
    constants = set((getattr(graph, "constants", {}) or {}).keys())
    for op in getattr(graph, "ops", ()):
        provenance = op.attrs.get("_fpgai_external_operator") if isinstance(getattr(op, "attrs", None), Mapping) else None
        if provenance is not None:
            contract = selected_contracts.get(op.name)
            if contract is None:
                raise HLSCompositionError(f"HLSCOMP005: no selected implementation for node {op.name!r}")
            abi = parse_hls_abi(contract)
            runtime_inputs = tuple(str(x) for x in op.inputs if str(x) not in constants)
            outputs = tuple(str(x) for x in op.outputs)
            if isinstance(abi, HLSFlatArrayABI):
                if len(runtime_inputs) != 1 or len(outputs) != 1:
                    raise HLSCompositionError("HLSCOMP002: flat_array_v1 requires one runtime input and one output")
                if not has_branching and runtime_inputs[0] != current_tensor:
                    raise HLSCompositionError(f"HLSCOMP003: external node {op.name!r} does not consume the current sequential tensor")
                words = _shape_words(graph, runtime_inputs[0])
                out_words = _shape_words(graph, outputs[0])
                if words != out_words:
                    raise HLSCompositionError("HLSCOMP006: flat_array_v1 requires equal flattened input/output sizes")
            else:
                if len(runtime_inputs) != len(abi.inputs) or len(outputs) != len(abi.outputs):
                    raise HLSCompositionError(
                        f"HLSCOMP012: tensor_ports_v1 node {op.name!r} expects {len(abi.inputs)} inputs/{len(abi.outputs)} outputs"
                    )
                input_port_words = {port.name: _shape_words(graph, tensor) for port, tensor in zip(abi.inputs, runtime_inputs)}
                output_port_words = {port.name: _shape_words(graph, tensor) for port, tensor in zip(abi.outputs, outputs)}
                all_sizes = [*input_port_words.values(), *output_port_words.values()]
                if abi.count_mode == "shared" and len(set(all_sizes)) != 1:
                    raise HLSCompositionError(
                        "HLSCOMP013: tensor_ports_v1 with count_mode=shared requires equal flattened sizes; "
                        "use count_mode=per_port for heterogeneous tensor sizes"
                    )
                words = next(iter(input_port_words.values()))
                out_words = next(iter(output_port_words.values()))
            public_attrs = {k: v for k, v in op.attrs.items() if not str(k).startswith("_fpgai_")}
            bindings.append(ExternalNodeBinding(
                node_name=op.name, op_type=op.op_type,
                operator_id=str(provenance.get("operator_id", "")),
                operator_package_id=str(provenance.get("package_id", "")),
                operator_package_version=str(provenance.get("package_version", "")),
                operator_manifest_hash=str(provenance.get("manifest_sha256", provenance.get("manifest_hash", ""))),
                contract=contract, attributes=public_attrs,
                input_tensor=runtime_inputs[0], output_tensor=outputs[0],
                input_words=words, output_words=out_words,
                wrapper_symbol=f"fpgai_external_{_symbol(op.name)}",
                conversion_buffers=(abi.scalar_type != "float"),
                input_tensors=runtime_inputs, output_tensors=outputs,
                port_words=(words if not isinstance(abi, HLSTensorPortsABI) or abi.count_mode == "shared" else None),
                input_port_words=(input_port_words if isinstance(abi, HLSTensorPortsABI) else {}),
                output_port_words=(output_port_words if isinstance(abi, HLSTensorPortsABI) else {}),
            ))
        if getattr(op, "outputs", None):
            current_tensor = op.outputs[0]
    return HLSCompositionPlan(tuple(bindings), selection_reports or {}, graph_mode="dag_mixed_graph" if has_branching else "sequential_mixed_graph")
