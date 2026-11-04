"""Single research-compiler entry point for all implemented CST Model IR policies.

Keeps the established compiler and source-pinned RL calls unchanged:
this is an *opt-in* compiler-IR interface for paper experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.compiler.composition import compile_composed_architecture
from fpgai_research.compiler.row_interleaved import compile_row_interleaved
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.model_ir.contracts import ArchitecturalModel


SUPPORTED_EXCHANGE_POLICIES = (
    "whole_tensor",
    "column_major",
    "row_lifetime",
    "row_chain",
)


@dataclass(frozen=True)
class CSTCompilation:
    policy: str
    ir: object
    emission: Emission
    evidence: dict


def compile_cst(architecture: ArchitecturalModel, *,
                exchange_policy: str = "whole_tensor",
                intermediate_tensor: str | None = None,
                row_block: int = 1) -> CSTCompilation:
    """Compile one explicitly chosen, actually supported tensor policy.

    No automatic search, implicit cast, streaming FIFO or materialization of
    unsupported producer/consumer programs is performed. Native tool execution
    must always be requested separately through the existing Vitis runner.
    """
    if exchange_policy not in SUPPORTED_EXCHANGE_POLICIES:
        raise ValueError("Unsupported CST IR exchange policy")
    if exchange_policy != "row_chain" and row_block != 1:
        raise ValueError("Row-block scheduling applies only to row_chain policy")
    if exchange_policy == "whole_tensor":
        if intermediate_tensor is not None:
            raise ValueError("Whole-tensor baseline cannot request a tensor adapter")
        result = compile_composed_architecture(architecture)
        return CSTCompilation(
            policy=exchange_policy, ir=result.composed_ir,
            emission=result.emission, evidence=result.evidence)
    if exchange_policy == "column_major":
        if not isinstance(intermediate_tensor, str) or not intermediate_tensor:
            raise ValueError("Column-major composition requires a tensor identity")
        result = compile_composed_architecture(
            architecture,
            intermediate_layouts={intermediate_tensor: "column_major"})
        return CSTCompilation(
            policy=exchange_policy, ir=result.composed_ir,
            emission=result.emission, evidence=result.evidence)
    if intermediate_tensor is not None:
        raise ValueError("Row-lifetime mode derives its intermediate tensor contracts")
    if exchange_policy == "row_chain":
        result = compile_row_chain(architecture, row_block=row_block)
        return CSTCompilation(
            policy=exchange_policy, ir=result.ir,
            emission=result.emission, evidence=result.evidence)
    result = compile_row_interleaved(architecture)
    return CSTCompilation(
        policy=exchange_policy, ir=result.ir,
        emission=result.emission, evidence=result.evidence)
