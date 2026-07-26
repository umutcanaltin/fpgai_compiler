"""Backend-neutral numerical capture contracts for FPGAI validation.

The schema deliberately separates semantic tensors from the mechanism or
language that produced them. Python reference execution, HLS CSim, VHDL/RTL
simulation, Vivado hardware, and community implementations can therefore emit
compatible capture manifests and use the same comparison engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

CAPTURE_SCHEMA_VERSION = 2
SUPPORTED_PRODUCER_KINDS = {
    "python_reference",
    "hls_csim",
    "rtl_simulation",
    "vivado_hardware",
    "board_runtime",
    "community_adapter",
}

SEMANTIC_CAPTURE_ROLES = (
    "pre_update_loss",
    "post_update_loss",
    "logits",
    "parameter_gradients",
    "weights_before",
    "weights_after",
    "biases_before",
    "biases_after",
    "optimizer_m_before",
    "optimizer_m_after",
    "optimizer_v_before",
    "optimizer_v_after",
    "optimizer_step_before",
    "optimizer_step_after",
)


@dataclass(frozen=True)
class NumericCaptureContract:
    workload_fingerprint_sha256: str
    implementation_stack_fingerprint_sha256: str
    producer_kind: str
    producer_id: str
    captures: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        producer_kind = str(self.producer_kind).strip().lower()
        if producer_kind not in SUPPORTED_PRODUCER_KINDS:
            raise ValueError(
                f"Unsupported numeric capture producer_kind={self.producer_kind!r}; "
                f"expected one of {sorted(SUPPORTED_PRODUCER_KINDS)}."
            )
        normalized_captures: Dict[str, Dict[str, Any]] = {}
        for role, spec in sorted(self.captures.items()):
            if role not in SEMANTIC_CAPTURE_ROLES:
                raise ValueError(f"Unknown semantic capture role: {role}")
            if not isinstance(spec, dict):
                raise TypeError(f"Capture specification for {role} must be a mapping.")
            normalized_captures[role] = {
                "path": None if spec.get("path") is None else str(spec.get("path")),
                "dtype": str(spec.get("dtype", "float32")),
                "layout": str(spec.get("layout", "flat_canonical_parameter_order")),
                "required": bool(spec.get("required", True)),
                "status": str(spec.get("status", "capture_pending")),
            }
            for key in ("shape", "count", "tensor_map", "units", "parameter_layout", "source_path"):
                if key in spec:
                    normalized_captures[role][key] = spec[key]
        payload = {
            "artifact_kind": "fpgai_numeric_capture_contract",
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "workload_fingerprint_sha256": self.workload_fingerprint_sha256,
            "implementation_stack_fingerprint_sha256": self.implementation_stack_fingerprint_sha256,
            "producer": {"kind": producer_kind, "id": self.producer_id},
            "captures": normalized_captures,
            "metadata": dict(self.metadata),
        }
        payload["capture_contract_fingerprint_sha256"] = fingerprint_payload(payload)
        return payload


def fingerprint_payload(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_training_capture_requirements(*, optimizer_type: str, export_gradients: bool) -> Dict[str, Dict[str, Any]]:
    optimizer = str(optimizer_type).lower().replace("-", "_")
    required = {
        "pre_update_loss": {"required": True, "dtype": "float32", "layout": "scalar"},
        "post_update_loss": {"required": True, "dtype": "float32", "layout": "scalar"},
        "weights_after": {"required": True, "dtype": "float32", "layout": "flat_canonical_parameter_order"},
        "biases_after": {"required": True, "dtype": "float32", "layout": "flat_canonical_parameter_order"},
        "parameter_gradients": {
            "required": bool(export_gradients),
            "dtype": "float32",
            "layout": "flat_canonical_parameter_order",
        },
        "optimizer_step_after": {
            "required": optimizer == "adam",
            "dtype": "float32",
            "layout": "scalar",
        },
    }
    if optimizer == "adam":
        required["optimizer_m_after"] = {
            "required": True,
            "dtype": "float32",
            "layout": "flat_canonical_parameter_order",
        }
        required["optimizer_v_after"] = {
            "required": True,
            "dtype": "float32",
            "layout": "flat_canonical_parameter_order",
        }
    return {role: {**spec, "path": None, "status": "capture_pending"} for role, spec in required.items()}


def compare_capture_contracts(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Validate comparability before numeric file comparison.

    File-level numeric metrics remain owned by ``fpgai.validation.numeric``.
    This function establishes whether two producers describe the same workload
    and whether required semantic captures are available.
    """
    same_workload = (
        reference.get("workload_fingerprint_sha256")
        == candidate.get("workload_fingerprint_sha256")
    )
    reference_captures = reference.get("captures", {}) if isinstance(reference.get("captures"), dict) else {}
    candidate_captures = candidate.get("captures", {}) if isinstance(candidate.get("captures"), dict) else {}
    role_status: Dict[str, Any] = {}
    missing_required: list[str] = []
    for role in sorted(set(reference_captures) | set(candidate_captures)):
        ref = reference_captures.get(role, {}) or {}
        got = candidate_captures.get(role, {}) or {}
        required = bool(ref.get("required", False))
        ref_ready = bool(ref.get("path")) and str(ref.get("status")) not in {"capture_pending", "missing"}
        got_ready = bool(got.get("path")) and str(got.get("status")) not in {"capture_pending", "missing"}
        if required and not (ref_ready and got_ready):
            missing_required.append(role)
        role_status[role] = {
            "required": required,
            "reference_ready": ref_ready,
            "candidate_ready": got_ready,
            "comparable": bool(ref_ready and got_ready),
        }
    if not same_workload:
        status = "workload_mismatch"
    elif missing_required:
        status = "capture_pending"
    else:
        status = "ready_for_numeric_comparison"
    return {
        "artifact_kind": "fpgai_numeric_capture_comparability",
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "status": status,
        "same_workload": same_workload,
        "missing_required_captures": missing_required,
        "roles": role_status,
        "reference_producer": reference.get("producer"),
        "candidate_producer": candidate.get("producer"),
    }


def write_capture_contract(path: str | Path, contract: NumericCaptureContract) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out
