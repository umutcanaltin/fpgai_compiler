"""Typed row-epoch pipeline for 2..8 sequential mapped operators.

Supported stage subset: MatMul, constant-weight Gemm, Relu and optional
first-stage Flatten(axis=1). No operator elimination or hidden fusion.

Each tensor boundary has a symbolic per-row lifetime and a concrete row
scratch array. The graph executes row-ordered producer then consumer
computations; there is no assumption of concurrent RTL streaming, achieved
cycles, or physical BRAM savings. Fail closed for branches, mixed precision,
unsupported operator kinds, shape/layout, or row tiling.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import prod

from fpgai_research.model_ir.composition import ComposedModelIR, TensorExchange, resolve_composition
from fpgai_research.model_ir.contracts import ArchitecturalModel


@dataclass(frozen=True)
class RowChainIR:
    composed: ComposedModelIR
    chain: tuple[str, ...]
    row_count: int
    scratch_by_tensor: tuple[tuple[str, int], ...]
    physical_scratch_elements: int
    full_intermediate_elements: int
    schedule: str = "for each row: compute all chained operators in topological order"


def resolve_row_chain(arch: ArchitecturalModel, *, max_operators: int = 8) -> RowChainIR:
    if type(max_operators) is not int or not 2 <= max_operators <= 32:
        raise ValueError("Invalid bounded row-chain operator limit")
    base = resolve_composition(arch)
    graph = arch.functional
    ops = graph.operators
    if not 2 <= len(ops) <= max_operators:
        raise ValueError("Row-chain IR requires 2..max_operators MatMul operations")
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError("Row chain requires one functional model input/output")
    accepted = ("MatMul", "Gemm", "Relu", "Flatten")
    if any(op.kind not in accepted or len(op.outputs) != 1 for op in ops):
        raise ValueError("Row chain supports MatMul/Gemm/Relu and leading Flatten only")
    if any(op.kind == "Flatten" and i != 0 for i, op in enumerate(ops)):
        raise ValueError("Flatten is supported only as the first row-wise stage")
    if not any(op.kind in ("MatMul", "Gemm") for op in ops):
        raise ValueError("Row chain requires at least one matrix operator")
    if graph.inputs[0] != ops[0].inputs[0] or graph.outputs[0] != ops[-1].outputs[0]:
        raise ValueError("Row chain must start with model input and end at model output")
    rows = graph.tensor(graph.inputs[0]).shape[0]
    boundaries = {edge.tensor: edge for edge in base.boundaries}
    if len(boundaries) != len(ops) - 1:
        raise ValueError("Row-chain requires exactly one internal tensor per operator link")
    new_boundaries = []
    scratch = []
    previous = graph.inputs[0]
    for i, op in enumerate(ops):
        if op.inputs[0] != previous:
            raise ValueError("Row-chain requires sequential MatMul lhs tensor dependencies")
        if op.kind in ("MatMul", "Gemm"):
            if len(op.inputs) not in ((2, 3) if op.kind == "Gemm" else (2,)):
                raise ValueError("Unsupported matrix operator arity")
            if graph.tensor(op.inputs[1]).data is None:
                raise ValueError("Row-chain requires constant MatMul/Gemm weights")
            if len(op.inputs) == 3 and graph.tensor(op.inputs[2]).data is None:
                raise ValueError("Row-chain requires constant Gemm bias")
            if op.kind == "Gemm" and (
                    float(op.attr("alpha", 1)) != 1.0
                    or float(op.attr("beta", 1)) != 1.0
                    or int(op.attr("transA", 0)) != 0
                    or int(op.attr("transB", 0)) not in (0, 1)):
                raise ValueError("Unsupported Gemm attributes in row chain")
        elif op.kind == "Relu" and len(op.inputs) != 1:
            raise ValueError("Row-chain Relu requires one input")
        elif op.kind == "Flatten" and (
                len(op.inputs) != 1 or int(op.attr("axis", 1)) != 1):
            raise ValueError("Row-chain Flatten requires axis=1")
        shape = graph.tensor(op.inputs[0]).shape
        if shape[0] != rows or (
                len(shape) != 2 and not (op.kind == "Flatten" and i == 0)):
            raise ValueError("Row-chain requires invariant row dimension and 2D stages")
        output_shape = graph.tensor(op.outputs[0]).shape
        if len(output_shape) != 2 or output_shape[0] != rows:
            raise ValueError("Row-chain output must retain the two-dimensional row geometry")
        point = arch.points[i]
        if point.operator != op.name:
            raise ValueError("Resolved operator order disagrees with Functional Model IR")
        if point.effective.tile and point.effective.tile[0] != 1:
            raise ValueError("Row-chain requires row tile extent 1 for each operator")
        if i < len(ops)-1:
            edge = boundaries.get(op.outputs[0])
            if edge is None or edge.producer != op.name or edge.consumers != (ops[i+1].name,):
                raise ValueError("Row-chain requires one producer and one next-operator consumer")
            if edge.shape[0] != rows or len(edge.shape) != 2:
                raise ValueError("Unsupported row scratch geometry")
            width = edge.shape[1]
            address_rule = {
                "tensor": edge.tensor,
                "row_count": rows,
                "physical_offsets": list(range(width)),
                "live_epochs": "producer(row) then consumer(row) before producer(row+1)",
            }
            new_boundaries.append(replace(
                edge,
                storage_layout="row_buffer",
                exchange_policy="producer_row_complete_then_consumer_row",
                physical_elements=width,
                logical_to_physical_sha256=sha256(
                    json.dumps(address_rule, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            ))
            scratch.append((edge.tensor, width))
        previous = op.outputs[0]
    return RowChainIR(
        composed=ComposedModelIR(lowered=base.lowered, boundaries=tuple(new_boundaries)),
        chain=tuple(op.name for op in ops), row_count=rows,
        scratch_by_tensor=tuple(scratch),
        physical_scratch_elements=sum(size for _,size in scratch),
        full_intermediate_elements=sum(rows*size for _,size in scratch),
    )


def row_chain_receipt(ir: RowChainIR) -> dict:
    payload = {
        "chain": list(ir.chain), "row_count": ir.row_count,
        "scratch_by_tensor": [list(x) for x in ir.scratch_by_tensor],
        "boundaries": [{
            "tensor": b.tensor,
            "producer": b.producer,
            "consumers": list(b.consumers),
            "policy": b.exchange_policy,
            "owner_phi_sha256": b.producer_mapping_sha256,
            "consumer_phi_sha256": [list(x) for x in b.consumer_mapping_sha256],
        } for b in ir.composed.boundaries],
    }
    return {
        "schema": "fpgai-research/cst-row-chain-ir/v1",
        **payload,
        "contract_sha256": sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_scratch_elements": ir.physical_scratch_elements,
        "source_full_intermediate_elements": ir.full_intermediate_elements,
        "original_vitis_calls": 0,
        "evidence": "E0-bounded-symbolic-row-lifetime-only",
        "limits": "No hardware concurrency, on-chip BRAM or latency guarantee",
    }
