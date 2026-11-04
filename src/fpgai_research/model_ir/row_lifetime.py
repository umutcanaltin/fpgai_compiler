"""Typed row-lifetime refinement for a two-MatMul compositional IR.

Source semantics: a producer fully reduces each intermediate row, writes
all row elements, then a consumer uses that row. The same finite scratch
addresses are reused for *different logical tensor rows in disjoint epochs*.

Unlike a globally bijective static array layout, storage aliasing here is
legal because the live value sets of different row epochs do not overlap.
This is a symbolic, source-execution contract, NOT a Vitis cycle schedule.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
import json

from fpgai_research.model_ir.composition import ComposedModelIR, resolve_composition
from fpgai_research.model_ir.contracts import ArchitecturalModel


@dataclass(frozen=True)
class RowLifetimeContract:
    tensor: str
    shape: tuple[int, int]
    physical_row_buffer_elements: int
    logical_tensor_elements: int
    producer_operator: str
    consumer_operator: str
    availability: str
    physical_address_rule: str
    disjoint_lifetime_epochs: str
    producer_phi_sha256: str
    consumer_phi_sha256: str


@dataclass(frozen=True)
class RowInterleavedModelIR:
    base: ComposedModelIR
    row_contract: RowLifetimeContract


def resolve_row_interleaving(arch: ArchitecturalModel) -> RowInterleavedModelIR:
    """Fail-closed opt-in row-epoch architecture transformation."""
    base = resolve_composition(arch)
    graph = arch.functional
    if len(graph.operators) != 2 or len(base.boundaries) != 1:
        raise ValueError("Row interleaving currently requires exactly two MatMul operations")
    first, second = graph.operators
    if first.kind != "MatMul" or second.kind != "MatMul":
        raise ValueError("Row interleaving supports MatMul -> MatMul only")
    shared = first.outputs[0]
    if second.inputs[0] != shared or shared in graph.outputs:
        raise ValueError("Second MatMul must consume the sole first output as its lhs")
    if len(second.inputs) != 2 or len(first.inputs) != 2:
        raise ValueError("Row interleaving requires two binary MatMul operators")
    if (graph.tensor(first.inputs[1]).data is None or
            graph.tensor(second.inputs[1]).data is None):
        raise ValueError("Row interleaved research backend needs constant MatMul weights")
    if len(graph.inputs) != 1 or first.inputs[0] != graph.inputs[0]:
        raise ValueError("Row interleaving requires one live model input")
    if len(graph.outputs) != 1 or second.outputs[0] != graph.outputs[0]:
        raise ValueError("Row interleaving requires one model output")
    boundary = base.boundaries[0]
    if boundary.tensor != shared or boundary.consumers != (second.name,):
        raise ValueError("Row interleaving currently supports a single-use intermediate")
    a, b = arch.points
    # The existing Phi mapping defines a tile-assigned row coordinate. A
    # nontrivial requested M tile cannot currently be materialized while
    # switching producer/consumer at individual row boundaries.
    if a.effective.tile and a.effective.tile[0] != 1:
        raise ValueError("Producer requested M tiling is not compatible with row interleaving")
    if b.effective.tile and b.effective.tile[0] != 1:
        raise ValueError("Consumer requested M tiling is not compatible with row interleaving")
    m, n = boundary.shape
    # Override whole-operator availability with an actual per-row lifetime
    # exchange contract; do not retain contradictory boundary semantics.
    row_offsets = [col for row in range(m) for col in range(n)]
    row_boundary = replace(
        boundary,
        storage_layout="row_buffer",
        exchange_policy="producer_row_complete_then_consumer_row",
        physical_elements=n,
        logical_to_physical_sha256=sha256(json.dumps(
            {"row_epochs": m, "row_offsets": row_offsets},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )
    row_base = ComposedModelIR(lowered=base.lowered, boundaries=(row_boundary,))
    return RowInterleavedModelIR(
        base=row_base,
        row_contract=RowLifetimeContract(
            tensor=shared, shape=(m, n),
            physical_row_buffer_elements=n,
            logical_tensor_elements=m*n,
            producer_operator=first.name,
            consumer_operator=second.name,
            availability="producer_entire_row_reduction_complete_before_consumer_row",
            physical_address_rule="buffer[col], scoped to logical row epoch",
            disjoint_lifetime_epochs="row(r) producer; row(r) consumer; then row(r+1) producer",
            producer_phi_sha256=boundary.producer_mapping_sha256,
            consumer_phi_sha256=boundary.consumer_mapping_sha256[0][1],
        ),
    )


def row_lifetime_receipt(ir: RowInterleavedModelIR) -> dict:
    if not isinstance(ir, RowInterleavedModelIR):
        raise TypeError("Expected RowInterleavedModelIR")
    c = ir.row_contract
    raw = {
        "tensor": c.tensor, "shape": list(c.shape),
        "scratch_elements": c.physical_row_buffer_elements,
        "logical_elements": c.logical_tensor_elements,
        "availability": c.availability,
        "epoch_order": c.disjoint_lifetime_epochs,
        "producer_phi_sha256": c.producer_phi_sha256,
        "consumer_phi_sha256": c.consumer_phi_sha256,
    }
    return {
        "schema": "fpgai-research/cst-row-lifetime-ir/v1",
        "contract": raw,
        "contract_sha256": sha256(json.dumps(
            raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "evidence": "E0-typed-row-epoch-lifetime-only",
        "host_executed": False, "vitis_calls": 0,
        "limitations": "No cycle schedule, FIFO, RTL proof or derived physical area",
    }
