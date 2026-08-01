from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import numpy as np

@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"

@dataclass(frozen=True)
class OnnxImportContext:
    domain: str
    op_type: str
    opset: int
    node_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, Any]
    tensors: Mapping[str, TensorDescriptor] = field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

@dataclass(frozen=True)
class ImportedOperator:
    canonical_op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, Any]
    def __post_init__(self): object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

@dataclass(frozen=True)
class ShapeInferenceContext:
    attributes: Mapping[str, Any]
    inputs: tuple[TensorDescriptor, ...]

@dataclass(frozen=True)
class ShapeInferenceResult:
    output_shapes: tuple[tuple[int, ...], ...]

@dataclass(frozen=True)
class TypeInferenceContext:
    attributes: Mapping[str, Any]
    input_dtypes: tuple[str, ...]

@dataclass(frozen=True)
class TypeInferenceResult:
    output_dtypes: tuple[str, ...]

@dataclass(frozen=True)
class ReferenceExecutionContext:
    attributes: Mapping[str, Any]
    inputs: tuple[np.ndarray, ...]

@dataclass(frozen=True)
class ReferenceExecutionResult:
    outputs: tuple[np.ndarray, ...]
