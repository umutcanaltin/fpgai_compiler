"""One semantic-preserving architecture-to-HLS compilation sequence.

Adding an operator requires extending *each* real compiler subsystem, not
only this facade. No native vendor calls, synthetic QoR or case-folder writes.
"""
from __future__ import annotations

from dataclasses import dataclass

from fpgai_research.model_ir.contracts import ArchitecturalModel, LoweredModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import verify_refinement
from fpgai_research.codegen.hls.emitter import Emission, emit_hls


@dataclass(frozen=True)
class CompilationArtifact:
    lowered: LoweredModel
    refinement_receipt: dict
    emission: Emission


def compile_architecture(
        architecture: ArchitecturalModel, *, loop_codegen: str | None = None
        ) -> CompilationArtifact:
    """Produce E1 source only after verifying Model IR semantic refinement."""
    lowered = lower(architecture)
    receipt = verify_refinement(lowered)
    if receipt.get("stage_invariants_verified") is not True:
        raise ValueError("Compiler Model IR refinement failed")
    if loop_codegen is None:
        emitted = emit_hls(lowered)
    else:
        emitted = emit_hls(lowered, loop_codegen=loop_codegen)
    if emitted.manifest.get("functional_sha256") != receipt.get("functional_sha256"):
        raise ValueError("Emitter changed Functional Model IR identity")
    return CompilationArtifact(
        lowered=lowered, refinement_receipt=receipt, emission=emitted)
