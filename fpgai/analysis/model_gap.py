from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping

import numpy as np

from fpgai.analysis.training_capability import audit_training_capabilities
from fpgai.capabilities.capabilities import capability_for


def _provider(op: Any) -> Dict[str, Any]:
    attrs = getattr(op, "attrs", {}) or {}
    ext = attrs.get("_fpgai_external_operator")
    if isinstance(ext, Mapping):
        return {"kind": "ecosystem_operator", **dict(ext)}
    comp = attrs.get("_fpgai_composite_provider")
    if isinstance(comp, Mapping):
        provider = str(comp.get("provider", "fpgai"))
        return {
            "kind": "builtin_composite" if provider == "fpgai" else "ecosystem_composite",
            **dict(comp),
        }
    return {"kind": "builtin"}


def _shape_record(graph: Any, tensor_name: str) -> Dict[str, Any]:
    tensor = graph.get_tensor(str(tensor_name))
    if tensor is None:
        return {"name": str(tensor_name), "shape": [], "dtype": "unknown", "static": False}
    shape = [int(x) for x in (getattr(tensor, "shape", ()) or ())]
    semantics = getattr(tensor, "semantics", None)
    def _semantics_dict(name: str) -> Dict[str, Any]:
        value = getattr(semantics, name, None) if semantics is not None else None
        if value is None:
            return {}
        to_dict = getattr(value, "to_dict", None)
        return dict(to_dict()) if callable(to_dict) else (dict(value) if isinstance(value, Mapping) else {})
    return {
        "name": str(tensor_name),
        "shape": shape,
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "static": all(dim > 0 for dim in shape),
        "memory": _semantics_dict("memory"),
        "transport": _semantics_dict("transport"),
        "training": _semantics_dict("training"),
        "state": _semantics_dict("state"),
    }


def _dtype_kind(dtype: str) -> str:
    value = str(dtype or "unknown").strip().lower()
    if value == "bool":
        return "boolean"
    if value.startswith("int") or value.startswith("uint"):
        return "integer"
    if value.startswith("float") or value.startswith("bfloat"):
        return "floating"
    return "unknown"


def _index_tensor_names(graph: Any) -> set[str]:
    names: set[str] = set()
    for op in getattr(graph, "ops", ()) or ():
        if str(getattr(op, "op_type", "")) == "Gather" and len(getattr(op, "inputs", ()) or ()) >= 2:
            names.add(str(op.inputs[1]))
        if str(getattr(op, "op_type", "")) in {"Slice", "Squeeze", "Unsqueeze", "Resize"}:
            for name in list(getattr(op, "inputs", ()) or ())[1:]:
                names.add(str(name))
    return names


def _constant_bytes(graph: Any, op: Any) -> int:
    constants = getattr(graph, "constants", {}) or {}
    total = 0
    for name in getattr(op, "inputs", ()) or ():
        if str(name) in constants:
            total += int(np.asarray(constants[str(name)]).nbytes)
    return total


def audit_model_gaps(graph: Any, *, pipeline_mode: str) -> Dict[str, Any]:
    """Create a model-agnostic layer/operator compilation gap report.

    The audit never identifies a model family or selects a model-specific path.
    It reports only the source graph's operators, tensor/state requirements,
    training readiness, and implementation/mechanism contracts. This is the
    report used to audit arbitrary YOLO/LLM/custom graphs without hard-coding
    those model names into the compiler.
    """

    mode = str(pipeline_mode or "inference")
    index_names = _index_tensor_names(graph)
    layers: list[Dict[str, Any]] = []
    unsupported: set[str] = set()
    limited: set[str] = set()
    dynamic_ops: set[str] = set()
    provider_counts: Counter[str] = Counter()
    postprocess_partition_candidates: list[Dict[str, Any]] = []

    for index, op in enumerate(getattr(graph, "ops", ()) or ()):
        inference = capability_for(str(op.op_type), "inference")
        training = capability_for(str(op.op_type), "training_on_device")
        selected = training if mode == "training_on_device" else inference
        if selected.status == "unsupported":
            unsupported.add(str(op.op_type))
        elif selected.status == "limited":
            limited.add(str(op.op_type))

        if str(op.op_type) == "NonMaxSuppression":
            postprocess_partition_candidates.append({
                "index": index,
                "name": str(op.name),
                "op_type": "NonMaxSuppression",
                "recommended_partition": "ps_or_host_postprocess",
                "reason": "Detection post-processing may remain outside the PL graph while preserving an explicit deployment boundary; a selectable PL implementation can be supplied through the FPGAI Ecosystem later.",
                "compiler_behavior": "explicit_gap_not_silent_fallback",
            })

        input_records = [_shape_record(graph, name) for name in (op.inputs or ())]
        output_records = [_shape_record(graph, name) for name in (op.outputs or ())]
        for record in input_records + output_records:
            record["dtype_kind"] = _dtype_kind(record.get("dtype", "unknown"))
            record["index_tensor"] = record["name"] in index_names
        static_shapes = all(item["static"] for item in input_records + output_records)
        if not static_shapes:
            dynamic_ops.add(str(op.op_type))

        provider = _provider(op)
        provider_counts[str(provider.get("kind", "unknown"))] += 1
        op_semantics = getattr(op, "semantics", None)
        raw_candidates = getattr(op_semantics, "implementation_candidates", ()) if op_semantics is not None else ()
        candidates = [candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate) for candidate in (raw_candidates or ())]
        selected_backend = getattr(op_semantics, "selected_backend", None) if op_semantics is not None else None
        selected_implementation_id = getattr(op_semantics, "selected_implementation_id", None) if op_semantics is not None else None
        schedule = dict(getattr(op_semantics, "schedule", {}) or {}) if op_semantics is not None else {}
        buffering = dict(getattr(op_semantics, "buffering", {}) or {}) if op_semantics is not None else {}
        training_semantics = dict(getattr(op_semantics, "training", {}) or {}) if op_semantics is not None else {}
        layers.append(
            {
                "index": index,
                "name": str(op.name),
                "op_type": str(op.op_type),
                "provider": provider,
                "inputs": input_records,
                "outputs": output_records,
                "static_shapes": static_shapes,
                "constant_parameter_bytes": _constant_bytes(graph, op),
                "inference": inference.to_dict(),
                "training_on_device": training.to_dict(),
                "selected_pipeline_capability": selected.to_dict(),
                "selected_backend": selected_backend,
                "selected_implementation_id": selected_implementation_id,
                "implementation_candidates": candidates,
                "schedule": schedule,
                "buffering": buffering,
                "training_semantics": training_semantics,
            }
        )

    state_tensors = []
    integer_tensors = []
    index_tensors = []
    for name, tensor in (getattr(graph, "tensors", {}) or {}).items():
        semantics = getattr(tensor, "semantics", None)
        state_obj = getattr(semantics, "state", None) if semantics is not None else None
        state = state_obj.to_dict() if hasattr(state_obj, "to_dict") else {}
        record = _shape_record(graph, str(name))
        record["dtype_kind"] = _dtype_kind(record.get("dtype", "unknown"))
        record["index_tensor"] = str(name) in index_names
        if state.get("kind") not in {None, "", "stateless"} or state.get("mutable") or state.get("persistent_across_invocations"):
            state_tensors.append(record)
        if record["dtype_kind"] in {"integer", "boolean"}:
            integer_tensors.append(record)
        if record["index_tensor"]:
            index_tensors.append(record)

    runtime_state_requirements = []
    runtime_state_blockers = []
    for item in state_tensors:
        state = dict(item.get("state") or {})
        memory = dict(item.get("memory") or {})
        storage = str(memory.get("storage") or "unspecified").lower()
        policy = str(state.get("update_policy") or "none").lower()
        overflow = str(state.get("overflow_policy") or "saturate").lower()
        implemented_on_chip = storage in {"", "auto", "unspecified", "default", "bram", "uram"} and policy in {"none", "append"} and overflow == "saturate"
        implemented_external = storage in {"ddr", "host", "external"} and policy in {"none", "append"} and overflow == "saturate"
        record = {
            "name": item["name"],
            "kind": state.get("kind"),
            "owner": state.get("owner"),
            "state_group": state.get("state_group"),
            "storage": storage,
            "update_policy": policy,
            "overflow_policy": overflow,
            "hls_on_chip_state_supported": bool(implemented_on_chip),
            "hls_external_state_supported": bool(implemented_external),
            "hls_state_supported": bool(implemented_on_chip or implemented_external),
        }
        runtime_state_requirements.append(record)
        if not (implemented_on_chip or implemented_external):
            runtime_state_blockers.append(record)

    runtime_contract = dict(getattr(getattr(graph, "semantics", None), "runtime_contract", {}) or {})
    detection_contract = dict(runtime_contract.get("detection_output", {}) or {})
    detection_decode_contract = dict(runtime_contract.get("detection_decode", {}) or {})
    autoregressive_contract = dict(runtime_contract.get("autoregressive_session", {}) or {})
    training_audit = audit_training_capabilities(graph)
    report = {
        "schema": "fpgai.model-gap-audit/v1",
        "graph": str(getattr(graph, "name", "main")),
        "pipeline_mode": mode,
        "source_ir": getattr(getattr(graph, "semantics", None), "source_ir", None),
        "source_metadata": dict(getattr(getattr(graph, "semantics", None), "source_metadata", {}) or {}),
        "operator_count": len(layers),
        "operator_counts": dict(sorted(Counter(item["op_type"] for item in layers).items())),
        "unsupported_operator_types": sorted(unsupported),
        "limited_operator_types": sorted(limited),
        "dynamic_shape_operator_types": sorted(dynamic_ops),
        "state_tensors": state_tensors,
        "integer_tensors": integer_tensors,
        "index_tensors": index_tensors,
        "runtime_state_requirements": runtime_state_requirements,
        "runtime_state_blockers": runtime_state_blockers,
        "detection_output_contract": detection_contract or None,
        "detection_decode_contract": detection_decode_contract or None,
        "autoregressive_runtime_contract": autoregressive_contract or None,
        "typed_tensor_blockers": [item["name"] for item in index_tensors if item.get("dtype_kind") != "integer"],
        "postprocess_partition_candidates": postprocess_partition_candidates,
        "provider_counts": dict(sorted(provider_counts.items())),
        "training_capability_audit": training_audit,
        "layers": layers,
        "compilation_gap_count": len(unsupported),
        "compilation_ready_at_operator_contract_level": not unsupported,
        "policy": {
            "model_specific_compiler_path": False,
            "layerwise_operator_compilation": True,
            "ecosystem_layers_use_same_contract": True,
            "training_is_first_class": True,
            "architecture_choices_are_user_resolved": True,
        },
    }
    metadata = getattr(graph, "metadata", None)
    if isinstance(metadata, dict):
        metadata["model_gap_audit"] = report
    return report


__all__ = ["audit_model_gaps"]
