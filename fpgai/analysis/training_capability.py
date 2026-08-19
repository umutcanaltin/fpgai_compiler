from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from fpgai.engine.training import OP_TRAINING_CAPS, OP_TRAINING_REFERENCE_CAPS, TrainingOpCaps


@dataclass(frozen=True)
class TrainingCapabilityEntry:
    name: str
    op_type: str
    provider: Dict[str, Any]
    forward: bool
    backward_input: bool
    parameter_gradients: bool
    optimizer_update: bool
    reference_status: str
    hardware_status: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "provider": dict(self.provider),
            "training": {
                "forward": self.forward,
                "backward_input": self.backward_input,
                "parameter_gradients": self.parameter_gradients,
                "optimizer_update": self.optimizer_update,
            },
            "reference_status": self.reference_status,
            "hardware_status": self.hardware_status,
            "detail": self.detail,
        }


def _external_caps(op: Any) -> Mapping[str, Any] | None:
    attrs = getattr(op, "attrs", {}) or {}
    ext = attrs.get("_fpgai_external_operator")
    if not isinstance(ext, Mapping):
        return None
    caps = ext.get("capabilities")
    return caps if isinstance(caps, Mapping) else None


def _provider(op: Any) -> Dict[str, Any]:
    attrs = getattr(op, "attrs", {}) or {}
    ext = attrs.get("_fpgai_external_operator")
    if isinstance(ext, Mapping):
        return {"kind": "ecosystem_operator", **dict(ext)}
    comp = attrs.get("_fpgai_composite_provider")
    if isinstance(comp, Mapping):
        kind = "builtin_composite" if str(comp.get("provider", "fpgai")) == "fpgai" else "ecosystem_composite"
        return {"kind": kind, **dict(comp)}
    return {"kind": "builtin"}


def audit_training_capabilities(graph: Any) -> Dict[str, Any]:
    """Create a layerwise training-completeness report for built-in and ecosystem layers.

    This audit deliberately distinguishes semantic/reference readiness from
    on-device HLS/VHDL training implementation status. It prevents inference
    support from being reported as training support merely because a forward
    kernel exists.
    """
    layers: list[Dict[str, Any]] = []
    incomplete = 0
    for op in getattr(graph, "ops", ()) or ():
        external = _external_caps(op)
        if external is not None:
            forward = bool(external.get("training_forward", False))
            backward = bool(external.get("backward_input", False))
            params = bool(external.get("parameter_gradients", False))
            update = bool(external.get("parameter_gradients", False))
            reference_status = "declared_by_ecosystem_contract" if forward and backward else "incomplete"
            hardware_status = "implementation_contract_required"
            detail = "External operator capability is taken from its OperatorContract; selected hardware implementation must independently declare training support."
        else:
            ref_caps: TrainingOpCaps = OP_TRAINING_REFERENCE_CAPS.get(str(op.op_type), TrainingOpCaps(False, False, False, False))
            hw_caps: TrainingOpCaps = OP_TRAINING_CAPS.get(str(op.op_type), TrainingOpCaps(False, False, False, False))
            forward = bool(ref_caps.forward)
            backward = bool(ref_caps.backward_input)
            params = bool(ref_caps.backward_params)
            update = bool(ref_caps.update)
            reference_status = "supported" if forward and backward else "incomplete"
            hardware_status = "supported" if hw_caps.forward and hw_caps.backward_input else "incomplete"
            detail = "Built-in reference and on-device training coverage are tracked independently; hardware status is never inferred from reference support."

        if not (forward and backward):
            incomplete += 1
        entry = TrainingCapabilityEntry(
            name=str(op.name),
            op_type=str(op.op_type),
            provider=_provider(op),
            forward=forward,
            backward_input=backward,
            parameter_gradients=params,
            optimizer_update=update,
            reference_status=reference_status,
            hardware_status=hardware_status,
            detail=detail,
        ).to_dict()
        layers.append(entry)
        semantics = getattr(op, "semantics", None)
        training_semantics = getattr(semantics, "training", None) if semantics is not None else None
        if isinstance(training_semantics, dict):
            training_semantics["capability_audit"] = entry["training"]

    hardware_incomplete = sum(1 for item in layers if item["hardware_status"] != "supported")
    report = {
        "schema": "fpgai.training-capability-audit/v1",
        "layer_count": len(layers),
        "incomplete_layer_count": incomplete,
        "semantic_incomplete_count": incomplete,
        "hardware_incomplete_layer_count": hardware_incomplete,
        "hardware_incomplete_count": hardware_incomplete,
        "complete": incomplete == 0,
        "hardware_complete": hardware_incomplete == 0,
        "layers": layers,
        "policy": "every layer must declare forward/backward semantics; ecosystem layers use the same contract and hardware implementations must separately declare training capability",
    }
    metadata = getattr(graph, "metadata", None)
    if isinstance(metadata, dict):
        metadata["training_capability_audit"] = report
    return report
