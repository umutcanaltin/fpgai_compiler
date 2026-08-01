from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class TensorArity(str, Enum):
    ONE = "one"
    OPTIONAL = "optional"
    VARIADIC = "variadic"


class TrainingCapability(str, Enum):
    UNSUPPORTED = "unsupported"
    FORWARD = "forward"
    BACKWARD_INPUT = "backward_input"
    PARAMETER_GRADIENTS = "parameter_gradients"
    BIAS_GRADIENTS = "bias_gradients"


@dataclass(frozen=True)
class TensorPortContract:
    name: str
    arity: TensorArity = TensorArity.ONE
    description: str = ""
    tensor_role: str = "activation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arity": self.arity.value,
            "description": self.description,
            "tensor_role": self.tensor_role,
        }


@dataclass(frozen=True)
class AttributeContract:
    name: str
    value_type: str
    required: bool = False
    default: Any = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "value_type": self.value_type,
            "required": self.required,
            "description": self.description,
        }
        if self.default is not None:
            payload["default"] = self.default
        return payload


@dataclass(frozen=True)
class OnnxBinding:
    domain: str
    op_type: str
    opset_min: int | None = None
    opset_max: int | None = None

    def __post_init__(self) -> None:
        if not self.op_type.strip():
            raise ValueError("ONNX op_type must not be empty")
        if self.opset_min is not None and self.opset_min < 1:
            raise ValueError("opset_min must be positive")
        if self.opset_max is not None and self.opset_max < 1:
            raise ValueError("opset_max must be positive")
        if self.opset_min is not None and self.opset_max is not None and self.opset_min > self.opset_max:
            raise ValueError("opset_min must not exceed opset_max")

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "op_type": self.op_type,
            "opset_min": self.opset_min,
            "opset_max": self.opset_max,
        }


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))
