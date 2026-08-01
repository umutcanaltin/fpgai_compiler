from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .operator_errors import OperatorIssue
from .operator_schema import AttributeContract, OnnxBinding, TensorPortContract, freeze_mapping

_OPERATOR_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")


@dataclass(frozen=True)
class OperatorCapabilities:
    inference: bool
    training_forward: bool = False
    backward_input: bool = False
    parameter_gradients: bool = False
    bias_gradients: bool = False
    shape_inference: bool = False
    type_inference: bool = False
    numeric_reference: bool = False
    canonicalization: bool = False
    resource_estimation: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "inference": self.inference,
            "training_forward": self.training_forward,
            "backward_input": self.backward_input,
            "parameter_gradients": self.parameter_gradients,
            "bias_gradients": self.bias_gradients,
            "shape_inference": self.shape_inference,
            "type_inference": self.type_inference,
            "numeric_reference": self.numeric_reference,
            "canonicalization": self.canonicalization,
            "resource_estimation": self.resource_estimation,
        }


@dataclass(frozen=True)
class OperatorEntrypoints:
    shape_inference: str | None = None
    type_inference: str | None = None
    legality: str | None = None
    numeric_reference: str | None = None
    canonicalization: str | None = None
    training_reference: str | None = None
    gradient: str | None = None
    resource_estimator: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {name: value for name, value in self.__dict__.items() if value is not None}


@dataclass(frozen=True)
class OperatorContract:
    operator_id: str
    canonical_op_type: str
    version: int
    category: str
    inputs: tuple[TensorPortContract, ...]
    outputs: tuple[TensorPortContract, ...]
    attributes: tuple[AttributeContract, ...] = ()
    onnx_bindings: tuple[OnnxBinding, ...] = ()
    aliases: tuple[str, ...] = ()
    capabilities: OperatorCapabilities = field(default_factory=lambda: OperatorCapabilities(inference=False))
    entrypoints: OperatorEntrypoints = field(default_factory=OperatorEntrypoints)
    implementation_requirements: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "implementation_requirements", MappingProxyType(dict(self.implementation_requirements)))
        issues = validate_operator_contract(self)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            raise ValueError("; ".join(f"{issue.code}: {issue.message}" for issue in errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.operator-contract/v1",
            "operator_id": self.operator_id,
            "canonical_op_type": self.canonical_op_type,
            "version": self.version,
            "category": self.category,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "attributes": [item.to_dict() for item in self.attributes],
            "onnx_bindings": [item.to_dict() for item in self.onnx_bindings],
            "aliases": list(self.aliases),
            "capabilities": self.capabilities.to_dict(),
            "entrypoints": self.entrypoints.to_dict(),
            "implementation_requirements": dict(self.implementation_requirements),
            "notes": list(self.notes),
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }


def validate_operator_contract(contract: OperatorContract) -> tuple[OperatorIssue, ...]:
    issues: list[OperatorIssue] = []
    if not _OPERATOR_ID_RE.fullmatch(contract.operator_id):
        issues.append(OperatorIssue("OPCON001", "operator_id", "Operator ID must be namespace-qualified and lowercase"))
    if not contract.canonical_op_type.strip():
        issues.append(OperatorIssue("OPCON002", "canonical_op_type", "Canonical op type must not be empty"))
    if contract.version < 1:
        issues.append(OperatorIssue("OPCON003", "version", "Operator contract version must be positive"))
    if not contract.inputs:
        issues.append(OperatorIssue("OPCON004", "inputs", "At least one input contract is required"))
    if not contract.outputs:
        issues.append(OperatorIssue("OPCON005", "outputs", "At least one output contract is required"))

    attribute_names = [item.name for item in contract.attributes]
    if len(attribute_names) != len(set(attribute_names)):
        issues.append(OperatorIssue("OPCON006", "attributes", "Attribute names must be unique"))

    bindings = [(item.domain, item.op_type, item.opset_min, item.opset_max) for item in contract.onnx_bindings]
    if len(bindings) != len(set(bindings)):
        issues.append(OperatorIssue("OPCON007", "onnx_bindings", "ONNX bindings must be unique"))

    caps = contract.capabilities
    if caps.backward_input and not caps.training_forward:
        issues.append(OperatorIssue("OPCON008", "capabilities.backward_input", "Backward-input support requires training-forward support"))
    if caps.parameter_gradients and not caps.training_forward:
        issues.append(OperatorIssue("OPCON009", "capabilities.parameter_gradients", "Parameter-gradient support requires training-forward support"))
    if caps.bias_gradients and not caps.parameter_gradients:
        issues.append(OperatorIssue("OPCON010", "capabilities.bias_gradients", "Bias-gradient support requires parameter-gradient support"))
    if caps.shape_inference and not contract.entrypoints.shape_inference:
        issues.append(OperatorIssue("OPCONW001", "entrypoints.shape_inference", "Shape inference is declared but no external entrypoint is recorded", "warning"))
    if caps.numeric_reference and not contract.entrypoints.numeric_reference:
        issues.append(OperatorIssue("OPCONW002", "entrypoints.numeric_reference", "Numeric reference is declared but no external entrypoint is recorded", "warning"))
    return tuple(issues)
