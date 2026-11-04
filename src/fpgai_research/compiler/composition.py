"""Compiler entry point for the opt-in compositional space-time tensor IR.

The original compile_architecture remains untouched for all source-pinned
research experiments and for unmodified operator-local architecture requests.
"""
from __future__ import annotations

from dataclasses import dataclass

from fpgai_research.codegen.hls.emitter import Emission, emit_hls
from fpgai_research.model_ir.composition import (
    ComposedModelIR, composition_receipt, resolve_composition,
)
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.exchange_witness import check_element_exchange
from fpgai_research.model_ir.semantics import verify_refinement


@dataclass(frozen=True)
class CompositionalCompilation:
    composed_ir: ComposedModelIR
    emission: Emission
    evidence: dict


def compile_composed_architecture(
        architecture: ArchitecturalModel, *,
        intermediate_layouts: dict[str, str] | None = None,
        loop_codegen: str | None = None) -> CompositionalCompilation:
    """Compile validated 2D tensor exchange contracts to changed HLS C++.

    This emits source only. Caller must run host/numerical/Vitis evidence
    separately; the emitted contracts do not prove achieved hardware layout,
    pipeline schedule, timing closure, or physical resource mappings.
    """
    composed = resolve_composition(architecture, layouts=intermediate_layouts)
    refinement = verify_refinement(composed.lowered)
    options = {"tensor_layouts": composed.layout_by_tensor()}
    if loop_codegen is not None:
        options["loop_codegen"] = loop_codegen
    emission = emit_hls(composed.lowered, **options)
    if emission.manifest["functional_sha256"] != refinement["functional_sha256"]:
        raise ValueError("Generated C++ is not bound to Functional Model IR")
    receipt = composition_receipt(composed)
    exchange = check_element_exchange(composed)
    return CompositionalCompilation(
        composed_ir=composed,
        emission=emission,
        evidence={
            **receipt,
            "cpp_sha256": emission.manifest["cpp_sha256"],
            "emitter_layouts": emission.manifest.get("tensor_physical_layouts", {}),
            "operator_to_boundary_exchange_witness": exchange,
            "strongest_evidence": "E1-generated-source-only",
            "host_executed": False,
            "vitis_csim_executed": False,
            "vitis_csynth_executed": False,
            "not_a_hardware_schedule_proof": True,
        },
    )
