from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from fpgai.operators import OperatorContract
from .external_api import (
    ImportedOperator, OnnxImportContext, ShapeInferenceContext, ShapeInferenceResult,
    TypeInferenceContext, TypeInferenceResult, ReferenceExecutionContext, ReferenceExecutionResult,
    BackwardInputReferenceContext, BackwardInputReferenceResult,
)

@dataclass(frozen=True)
class ExternalOperatorDefinition:
    api_version: int
    contract: OperatorContract
    onnx_import: Callable[[OnnxImportContext], ImportedOperator]
    shape_inference: Callable[[ShapeInferenceContext], ShapeInferenceResult] | None = None
    type_inference: Callable[[TypeInferenceContext], TypeInferenceResult] | None = None
    numeric_reference: Callable[[ReferenceExecutionContext], ReferenceExecutionResult] | None = None
    backward_input_reference: Callable[[BackwardInputReferenceContext], BackwardInputReferenceResult] | None = None
    def __post_init__(self):
        if self.api_version != 1: raise ValueError("Only external operator API version 1 is supported")
        if not callable(self.onnx_import): raise TypeError("onnx_import must be callable")
        if self.contract.capabilities.backward_input and not callable(self.backward_input_reference):
            raise TypeError("Operator declares backward_input support but no backward_input_reference callback was provided")
