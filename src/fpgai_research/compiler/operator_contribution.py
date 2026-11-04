"""Researcher-facing checklist for safely expanding supported Model IR ops.

This module DOES NOT register a new operator based solely on flags: every
operator needs real ONNX normalization, semantic evaluation, architecture
mapping, lowering, HLS emission and numerical/native validation. Incomplete
contributions must fail closed in the actual compiler.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorContributionChecklist:
    name: str
    frontend_shape_and_attributes: bool
    functional_model_ir: bool
    reference_numerical_semantics: bool
    architecture_legality: bool
    lowering_preserves_refinement: bool
    hls_source_emitter: bool
    source_and_csim_tests: bool
    independently_audited_native_hls_example: bool

    def missing(self) -> tuple[str, ...]:
        return tuple(
            key for key in (
                "frontend_shape_and_attributes",
                "functional_model_ir",
                "reference_numerical_semantics",
                "architecture_legality",
                "lowering_preserves_refinement",
                "hls_source_emitter",
                "source_and_csim_tests",
                "independently_audited_native_hls_example",
            ) if not getattr(self, key))

    def assert_native_ready(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("Operator name must be a valid identifier")
        missing = self.missing()
        if missing:
            raise ValueError(
                f"Operator {self.name} is not qualified for native support: "
                + ", ".join(missing))
