"""Standalone research implementation of explicit space--time Model IR for HLS."""
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel, ArchitectureRequest, ArchitecturePoint, ArchitecturalModel, LoweredModel
from fpgai_research.architecture.resolver import resolve, ArchitectureError, TargetLimits, coordinate
from fpgai_research.model_ir.lowering import lower
from fpgai_research.architecture.scopes import resolve_scoped

__all__ = ["Tensor", "Operator", "FunctionalModel", "ArchitectureRequest", "ArchitecturePoint", "ArchitecturalModel", "LoweredModel", "resolve", "lower", "coordinate", "ArchitectureError", "TargetLimits", "resolve_scoped"]
