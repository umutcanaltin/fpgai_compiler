"""Immutable progressive Model IR contracts for the standalone HLS research compiler."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    data: Optional[tuple[float, ...]] = None

    def __post_init__(self):
        if not self.name or any(type(d) is not int or d <= 0 for d in self.shape):
            raise ValueError("Named tensors require positive static dimensions")
        if self.dtype != "float32":
            raise ValueError("Only float32 ONNX tensors are supported in this research baseline")
        if self.data is not None:
            from math import prod, isfinite
            if len(self.data) != prod(self.shape):
                raise ValueError("Invalid constant tensor payload")
            # Normalize NumPy scalar constants to stable JSON-compatible values
            # so generated source hashes and Model IR artifacts agree.
            try:
                numbers = tuple(float(x) for x in self.data)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("Invalid constant tensor payload") from exc
            if not all(isfinite(x) for x in numbers):
                raise ValueError("Invalid constant tensor payload")
            object.__setattr__(self, "data", numbers)


@dataclass(frozen=True)
class Operator:
    name: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: tuple[tuple[str, object], ...] = ()

    def attr(self, key: str, default=None):
        return dict(self.attributes).get(key, default)


@dataclass(frozen=True)
class FunctionalModel:
    tensors: tuple[Tensor, ...]
    operators: tuple[Operator, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    version: int = 1

    def tensor(self, name: str) -> Tensor:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    def validate(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported Functional Model IR schema version")
        names = [t.name for t in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate tensor identity")
        if len({op.name for op in self.operators}) != len(self.operators):
            raise ValueError("Duplicate operator identity")
        all_names = set(names)
        constants = {t.name for t in self.tensors if t.data is not None}
        produced = set(self.inputs) | constants
        if not set(self.inputs) <= all_names or not set(self.outputs) <= all_names:
            raise ValueError("Unknown boundary tensor")
        if len(set(self.inputs)) != len(self.inputs) or constants & set(self.inputs):
            raise ValueError("Duplicate input or constant/input collision")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Duplicate model output identity")
        if not self.operators or not self.inputs or not self.outputs:
            raise ValueError("A supported model needs input, output and at least one operator")
        for op in self.operators:
            if not op.name or len(op.outputs) != 1 or not op.outputs[0]:
                raise ValueError("Only named, single-output operators are supported")
            if not op.kind or any(not name for name in op.inputs):
                raise ValueError("Invalid operator kind or missing input name")
            keys = [key for key, _ in op.attributes]
            if len(keys) != len(set(keys)) or any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("Duplicate or invalid operator attributes")
            if not set(op.inputs) <= produced:
                raise ValueError(f"Missing dependency or non-topological operators: {op.name}")
            if not set(op.outputs) <= all_names or set(op.outputs) & produced:
                raise ValueError(f"Invalid or duplicate produced tensor: {op.name}")
            produced.update(op.outputs)
        if not set(self.outputs) <= produced:
            raise ValueError("Unproduced model output")
        undefined = all_names - produced
        if undefined:
            raise ValueError(f"Declared tensor has no input, initializer or producer: {sorted(undefined)}")
        operator_outputs = {name for op in self.operators for name in op.outputs}
        if not set(self.outputs) <= operator_outputs:
            raise ValueError("Model output must be written by an operator, not alias a runtime input or initializer")


@dataclass(frozen=True)
class ArchitectureRequest:
    precision_bits: int = 16
    integer_bits: int = 6
    parallelism: int = 1
    pes: int = 1
    simd: int = 1
    serialization: int = 1
    tile: tuple[int, ...] = ()
    pipeline_ii: Optional[int] = None


@dataclass(frozen=True)
class ArchitecturePoint:
    operator: str
    requested: ArchitectureRequest
    effective: ArchitectureRequest
    iteration_shape: tuple[int, ...]
    source_scope: str = "model"
    kind: str = ""
    decision_sources: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ArchitecturalModel:
    functional: FunctionalModel
    points: tuple[ArchitecturePoint, ...]


@dataclass(frozen=True)
class LoweredOperator:
    name: str
    kind: str
    iteration_shape: tuple[int, ...]
    architecture: ArchitectureRequest
    source_scope: str
    tile_shape: tuple[int, ...] = ()
    replica_count: int = 1
    pe_count: int = 1
    simd_lanes: int = 1
    fold_factor: int = 1
    pipeline_ii: Optional[int] = None
    tail_policy: str = "bounds-guard"
    numerical_type: str = ""
    decision_sources: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LoweredModel:
    architectural: ArchitecturalModel
    operators: tuple[LoweredOperator, ...]
