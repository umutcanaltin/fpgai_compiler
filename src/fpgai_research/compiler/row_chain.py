"""Opt-in compiler for the multi-operator CST row-chain representation."""
from __future__ import annotations

from dataclasses import dataclass

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.row_chain import emit_row_chain_hls
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.row_chain import RowChainIR, resolve_row_chain, row_chain_receipt
from fpgai_research.model_ir.row_chain_liveness import verify_row_chain_liveness
from fpgai_research.model_ir.temporal_schedule import blocked_schedule, check_schedule


@dataclass(frozen=True)
class RowChainCompilation:
    ir: RowChainIR
    emission: Emission
    evidence: dict


def compile_row_chain(architecture: ArchitecturalModel, *, row_block: int = 1) -> RowChainCompilation:
    ir = resolve_row_chain(architecture)
    proof = verify_row_chain_liveness(ir)
    if proof["status"] != "EXHAUSTIVE_BOUNDED_LIVENESS_VERIFIED":
        raise ValueError("CST row-chain lifetime requires bounded independent verification")
    proposed = blocked_schedule(ir.row_count, len(ir.chain), row_block)
    schedule_check = check_schedule(ir, proposed, row_slots=row_block)
    if schedule_check["status"] != "VERIFIED":
        raise ValueError(
            "CST candidate schedule rejected: " +
            str(schedule_check.get("counterexample") or schedule_check["status"])
        )
    emission = emit_row_chain_hls(ir, row_block=row_block)
    return RowChainCompilation(
        ir=ir, emission=emission,
        evidence={
            **row_chain_receipt(ir),
            "source_sha256": emission.manifest["cpp_sha256"],
            "independent_liveness_witness": proof,
            "schedule_witness": schedule_check,
            "row_block": row_block,
            "new_hls_source_emitted": True,
            "host_executed": False,
            "vitis_calls": 0,
            "evidence": "E1-generated-row-chain-source-only",
        },
    )
