"""Compositional space-time Model IR: typed tensor-element exchange contracts.

This is an initial, source-backed research subset. Each contract joins a
producer/consumer dependency to the already resolved operator-local Phi and a
real intermediate-array address relation. Actual inter-operator exchange is
currently *whole-operator-complete* (not streaming or achieved-cycle timing).

Only bounded static, 2D, MatMul/Gemm-to-MatMul/Gemm intermediate tensors can
choose a noncanonical column-major array layout. All other graph connections
retain the canonical row-major source policy. Unsupported layout requests
fail closed rather than silently inserting a cast, FIFO or adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod
from typing import Mapping

from fpgai_research.model_ir.contracts import ArchitecturalModel, LoweredModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import verify_refinement


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TensorExchange:
    tensor: str
    producer: str
    consumers: tuple[str, ...]
    shape: tuple[int, ...]
    storage_layout: str
    exchange_policy: str
    producer_mapping_sha256: str
    consumer_mapping_sha256: tuple[tuple[str, str], ...]
    producer_completed_before_consumers: bool
    logical_to_physical_sha256: str
    physical_elements: int


@dataclass(frozen=True)
class ComposedModelIR:
    """Retains existing IR as-is while extending it with typed boundaries."""

    lowered: LoweredModel
    boundaries: tuple[TensorExchange, ...]

    def layout_by_tensor(self) -> dict[str, str]:
        return {edge.tensor: edge.storage_layout
                for edge in self.boundaries if edge.storage_layout != "row_major"}


def physical_offset(shape: tuple[int, int], indices: tuple[int, int],
                    layout: str) -> int:
    """Total, injective logical element address mapping for supported 2D data."""
    if len(shape) != 2 or len(indices) != 2 or any(
            type(v) is not int or v <= 0 for v in shape):
        raise ValueError("Physical layout requires a positive static 2D shape")
    rows, cols = shape
    i, j = indices
    if type(i) is not int or type(j) is not int or not (0 <= i < rows and 0 <= j < cols):
        raise ValueError("Tensor logical index out of bounds")
    if layout == "row_major":
        return i * cols + j
    if layout == "column_major":
        return j * rows + i
    raise ValueError("Unsupported concrete tensor layout")


def resolve_composition(architecture: ArchitecturalModel, *,
                        layouts: Mapping[str, str] | None = None,
                        max_elements: int = 200_000) -> ComposedModelIR:
    """Build a checkable graph boundary relation without invoking vendor tools.

    The source-level index-bijection check is exhaustive for each declared
    2D internal tensor up to max_elements. It is *not* RTL equivalence.
    """
    if not isinstance(architecture, ArchitecturalModel):
        raise TypeError("Expected typed ArchitecturalModel")
    if not isinstance(max_elements, int) or max_elements < 1:
        raise ValueError("Positive finite verification bound required")
    if layouts is not None and not isinstance(layouts, Mapping):
        raise TypeError("Layouts must be a tensor-name mapping")
    layouts = dict(layouts or {})
    if any(type(k) is not str or type(v) is not str for k, v in layouts.items()):
        raise ValueError("Layout names and values must be strings")
    lowered = lower(architecture)
    verify_refinement(lowered)
    f = architecture.functional
    if len(f.inputs) != 1 or len(f.outputs) != 1:
        raise ValueError("Current composed HLS top supports one input and one output")
    numeric = {(p.effective.precision_bits, p.effective.integer_bits)
               for p in architecture.points}
    if len(numeric) != 1:
        raise ValueError("Mixed per-operator fixed-point formats need implemented casts")
    producer = {op.outputs[0]: (i, op)
                for i, op in enumerate(f.operators)}
    uses: dict[str, list[tuple[int, object, int]]] = {}
    for idx, op in enumerate(f.operators):
        for pos, name in enumerate(op.inputs):
            uses.setdefault(name, []).append((idx, op, pos))
    for name in layouts:
        if name not in producer or name in f.outputs or name in f.inputs:
            raise ValueError("Layout override requires an internal produced tensor: " + name)
        if layouts[name] not in ("row_major", "column_major"):
            raise ValueError("Unsupported physical tensor layout: " + layouts[name])

    points = {p.operator: p for p in architecture.points}
    contracts: list[TensorExchange] = []
    for tensor in f.tensors:
        if tensor.name not in producer or tensor.name in f.outputs:
            continue
        downstream = uses.get(tensor.name, [])
        if not downstream:
            continue
        producer_index, upstream_op = producer[tensor.name]
        layout = layouts.get(tensor.name, "row_major")
        # This research slice lowers a true alternate layout only for
        # intermediate 2D matrices written by one MatMul/Gemm and read as the
        # lhs of MatMul/Gemm. Other operator types are not silently rewritten.
        if layout == "column_major":
            if len(tensor.shape) != 2 or min(tensor.shape) <= 1:
                raise ValueError("Column-major prototype requires nondegenerate 2D internal tensor")
            if upstream_op.kind not in ("MatMul", "Gemm") or any(
                    op.kind not in ("MatMul", "Gemm") or position != 0
                    for _idx, op, position in downstream):
                raise ValueError("Column-major lowering supports MatMul/Gemm output to lhs MatMul/Gemm only")
        if any(idx <= producer_index for idx, _op, _pos in downstream):
            raise ValueError("Tensor consumer must follow producer in materialized source")
        if len(tensor.shape) == 2:
            size = prod(tensor.shape)
            if size > max_elements:
                raise ValueError("Boundary bijection exceeded source verification bound")
            offset_set = {
                physical_offset(tensor.shape, (i, j), layout)
                for i in range(tensor.shape[0]) for j in range(tensor.shape[1])
            }
            if len(offset_set) != size or min(offset_set) != 0 or max(offset_set) != size - 1:
                raise AssertionError("Tensor layout is not an exact physical-address bijection")
            physical_hash = _digest([
                physical_offset(tensor.shape, (i, j), layout)
                for i in range(tensor.shape[0]) for j in range(tensor.shape[1])
            ])
        else:
            if layout != "row_major":
                raise ValueError("Unsupported noncanonical tensor rank")
            physical_hash = _digest({"shape": tensor.shape,
                                     "layout": "row_major",
                                     "rule": "C-contiguous-row-major"})
        contracts.append(TensorExchange(
            tensor=tensor.name, producer=upstream_op.name,
            consumers=tuple(op.name for _idx, op, _pos in downstream),
            shape=tensor.shape, storage_layout=layout,
            exchange_policy="whole_producer_completed_then_static_array_read",
            producer_mapping_sha256=_digest(asdict(points[upstream_op.name])),
            consumer_mapping_sha256=tuple(
                (op.name, _digest(asdict(points[op.name])))
                for _idx, op, _pos in downstream
            ),
            producer_completed_before_consumers=True,
            logical_to_physical_sha256=physical_hash,
            physical_elements=prod(tensor.shape),
        ))
    return ComposedModelIR(lowered=lowered, boundaries=tuple(contracts))


def composition_receipt(composed: ComposedModelIR) -> dict:
    """E0 typed IR+boundary manifest; no claim of HLS/source execution."""
    if not isinstance(composed, ComposedModelIR):
        raise TypeError("Expected composed IR")
    data = [asdict(b) for b in composed.boundaries]
    return {
        "schema": "fpgai-research/cst-model-ir/v1",
        "functional_graph_preserved": True,
        "boundary_contracts": data,
        "boundary_sha256": _digest(data),
        "evidence": "E0-typed-static-array-boundary-contract",
        "original_vitis_calls": 0,
        "limits": "No physical schedule, FIFO, streaming, cast, bit-exactness or RTL proof",
    }
