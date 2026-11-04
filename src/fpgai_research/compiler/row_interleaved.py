"""Opt-in compiler for the row-lifetime composition subset.

No changes to the original FPGAI compile_architecture or source-pinned
research/Gym build pathway.
"""
from __future__ import annotations

from dataclasses import dataclass

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.row_interleaved import emit_row_interleaved_hls
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.row_lifetime import (
    RowInterleavedModelIR, resolve_row_interleaving,
    row_lifetime_receipt,
)


@dataclass(frozen=True)
class RowInterleavedCompilation:
    ir: RowInterleavedModelIR
    emission: Emission
    evidence: dict


def compile_row_interleaved(architecture: ArchitecturalModel) -> RowInterleavedCompilation:
    ir = resolve_row_interleaving(architecture)
    emission = emit_row_interleaved_hls(ir)
    receipt = row_lifetime_receipt(ir)
    return RowInterleavedCompilation(
        ir=ir, emission=emission,
        evidence={
            **receipt,
            "cpp_sha256": emission.manifest["cpp_sha256"],
            "source_row_interleaving_materialized": True,
            "strongest_evidence": "E1-generated-source-only",
            "host_cpp_execution_verified": False,
            "vitis_csim_executed": False,
            "vitis_csynth_executed": False,
            "post_route_clock_verified": False,
        },
    )
