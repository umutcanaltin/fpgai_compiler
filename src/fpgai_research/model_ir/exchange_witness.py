"""Bounded cross-operator space-time/value-identity witness for CST Model IR.

Links each completed MatMul output element to its producer Phi spatial owner,
its selected concrete array address, and each downstream MatMul reduction
iteration that reads that logical element. All timing relationships are only
symbolic whole-operator barriers in the current sequential HLS source.

Unlike a hash-only representation receipt, this independent E0 checker
enumerates the concrete producer and consumer relationships. It is not
a formal proof of generated C++ source, HLS hardware or achieved cycles.
"""
from __future__ import annotations
from hashlib import sha256
import json

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.composition import ComposedModelIR, physical_offset


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_element_exchange(composed: ComposedModelIR, *, max_events: int = 50_000) -> dict:
    """Validate bounded element ownership/read relations using resolved Phi."""
    if not isinstance(composed, ComposedModelIR):
        raise TypeError("Expected ComposedModelIR")
    if type(max_events) is not int or max_events < 1:
        raise ValueError("Positive maximum event count required")
    model = composed.lowered.architectural.functional
    operators = {op.name: op for op in model.operators}
    points = {p.operator: p for p in composed.lowered.architectural.points}
    results = []
    for b in composed.boundaries:
        if (operators[b.producer].kind not in ("MatMul", "Gemm")
                or any(operators[c].kind not in ("MatMul", "Gemm")
                       for c in b.consumers)):
            results.append({"tensor": b.tensor, "status": "UNSUPPORTED_OPERATOR_RELATION"})
            continue
        m, n = b.shape
        writer_shape = points[b.producer].iteration_shape
        if len(writer_shape) != 3 or writer_shape[:2] != (m, n):
            raise AssertionError("Producer Phi iteration/output geometry mismatch")
        consumer_shapes = {c: points[c].iteration_shape for c in b.consumers}
        if any(len(s) != 3 or s[0] != m or s[2] != n
               for s in consumer_shapes.values()):
            raise AssertionError("Consumer Phi reduction/input geometry mismatch")
        required = m * n * (1 + sum(s[1] for s in consumer_shapes.values()))
        if required > max_events:
            results.append({
                "tensor": b.tensor, "status": "BOUND_EXCEEDED",
                "expected_producer_consumer_events": required,
            })
            continue
        events = []
        addresses = set()
        for row in range(m):
            for col in range(n):
                physical = physical_offset(b.shape, (row, col), b.storage_layout)
                if physical in addresses:
                    raise AssertionError("Two logical values share one physical address")
                addresses.add(physical)
                # Full reduction is completed before the produced output is
                # materialized. k-1 identifies its final logical reduction
                # contribution, not a Vitis cycle number.
                owner = coordinate(points[b.producer],
                                   (row, col, writer_shape[2] - 1))
                events.append({
                    "kind": "producer_completed_after_reduction",
                    "tensor_element": [row, col],
                    "physical_address": physical,
                    "owner_spatial": [owner["replica"], owner["pe"], owner["lane"]],
                    "owner_reuse_step": list(owner["reuse_step"]),
                })
                for consumer in b.consumers:
                    for output_col in range(consumer_shapes[consumer][1]):
                        read = coordinate(points[consumer], (row, output_col, col))
                        events.append({
                            "kind": "downstream_reduction_read",
                            "consumer": consumer,
                            "source_tensor_element": [row, col],
                            "consumer_output_col": output_col,
                            "physical_address": physical,
                            "consumer_spatial": [read["replica"], read["pe"], read["lane"]],
                            "consumer_reuse_step": list(read["reuse_step"]),
                            "after_whole_producer_completion": True,
                        })
        if len(addresses) != m * n or len(events) != required:
            raise AssertionError("Incomplete producer/consumer element relation")
        results.append({
            "tensor": b.tensor, "status": "BOUNDED_PHI_EXCHANGE_VERIFIED",
            "elements": m * n,
            "logical_producer_consumer_events": required,
            "storage_layout": b.storage_layout,
            "all_logical_values_distinct": True,
            "producer_before_consumers_symbolically": True,
            "event_trace_sha256": _digest(events),
            "preview": events[:3],
        })
    return {
        "schema": "fpgai-research/phi-to-tensor-exchange-witness/v1",
        "relations": results,
        "evidence": "E0-bounded-independent-operator-Phi-to-tensor-address-relation",
        "executed_cpp": False,
        "native_vitis_calls": 0,
        "limits": "No stream/FIFO, temporal cycle proof, C++ AST proof or RTL guarantee",
    }
