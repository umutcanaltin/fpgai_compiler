from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from fpgai.operators import OperatorContract
from .external_api import (
    ImportedOperator, OnnxImportContext, ShapeInferenceContext, ShapeInferenceResult,
    TypeInferenceContext, TypeInferenceResult, ReferenceExecutionContext, ReferenceExecutionResult,
)

@dataclass(frozen=True)
class ExternalOperatorDefinition:
    api_version: int
    contract: OperatorContract
    onnx_import: Callable[[OnnxImportContext], ImportedOperator]
    shape_inference: Callable[[ShapeInferenceContext], ShapeInferenceResult] | None = None
    type_inference: Callable[[TypeInferenceContext], TypeInferenceResult] | None = None
    numeric_reference: Callable[[ReferenceExecutionContext], ReferenceExecutionResult] | None = None
    def __post_init__(self):
        if self.api_version != 1: raise ValueError("Only external operator API version 1 is supported")
        if not callable(self.onnx_import): raise TypeError("onnx_import must be callable")
