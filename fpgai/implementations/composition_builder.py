from __future__ import annotations

import re
from typing import Any, Mapping

from fpgai.implementations.hls_integration import parse_flat_array_abi

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


def build_hls_composition_plan(
    graph: Any,
    *,
    selected_contracts: Mapping[str, Any],
    selection_reports: Mapping[str, Mapping[str, str]] | None = None,
) -> HLSCompositionPlan:
    bindings: list[ExternalNodeBinding] = []
    current_tensor = graph.inputs[0] if getattr(graph, "inputs", None) else None
    for op in getattr(graph, "ops", ()):
        provenance = op.attrs.get("_fpgai_external_operator") if isinstance(getattr(op, "attrs", None), Mapping) else None
        if provenance is not None:
            if len(op.inputs) != 1 or len(op.outputs) != 1:
                raise HLSCompositionError("HLSCOMP002: flat_array_v1 requires one runtime input and one output")
            if op.inputs[0] != current_tensor:
                raise HLSCompositionError(
                    f"HLSCOMP003: external node {op.name!r} does not consume the current sequential tensor"
                )
            contract = selected_contracts.get(op.name)
            if contract is None:
                raise HLSCompositionError(f"HLSCOMP005: no selected implementation for node {op.name!r}")
            abi = parse_flat_array_abi(contract)
            input_words = _shape_words(graph, op.inputs[0])
            output_words = _shape_words(graph, op.outputs[0])
            if input_words != output_words:
                raise HLSCompositionError("HLSCOMP006: flat_array_v1 requires equal flattened input/output sizes")
            public_attrs = {key: value for key, value in op.attrs.items() if not str(key).startswith("_fpgai_")}
            bindings.append(ExternalNodeBinding(
                node_name=op.name,
                op_type=op.op_type,
                operator_id=str(provenance.get("operator_id", "")),
                operator_package_id=str(provenance.get("package_id", "")),
                operator_package_version=str(provenance.get("package_version", "")),
                operator_manifest_hash=str(provenance.get("manifest_sha256", provenance.get("manifest_hash", ""))),
                contract=contract,
                attributes=public_attrs,
                input_tensor=op.inputs[0],
                output_tensor=op.outputs[0],
                input_words=input_words,
                output_words=output_words,
                wrapper_symbol=f"fpgai_external_{_symbol(op.name)}",
                conversion_buffers=(abi.scalar_type != "float"),
            ))
        if getattr(op, "outputs", None):
            current_tensor = op.outputs[0]
    return HLSCompositionPlan(tuple(bindings), selection_reports or {})
