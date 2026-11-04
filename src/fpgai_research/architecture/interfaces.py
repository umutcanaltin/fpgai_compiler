"""Conservative E0 graph-boundary audit for the existing static HLS-array backend.

This is a source-backed inventory of *current* interface behavior, NOT a new
streaming protocol, cast adapter, bit-exact proof, or physical netlist. The
current emitter uses a single ap_fixed type for the entire generated model;
it materializes intermediate tensors as flat local C++ arrays. Distinct
producer/consumer PE, SIMD, and reuse settings do not themselves require a
hardware adapter under this materialized-array source contract.
"""
from __future__ import annotations

from typing import Any

from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint,
    functional_fingerprint,
    verify_refinement,
)


def inspect_array_interfaces(architecture: ArchitecturalModel) -> dict[str, Any]:
    """Report current typed graph-edge boundaries, fail closed on mixed format.

    All claims are E0 about typed Model IR and the declared current HLS source
    interface policy. This routine deliberately runs no HLS emitter, GCC,
    CSim, CSynth, DRL code, or vendor tools. Returned diagnostics cannot imply
    a physical FIFO, concurrent stream, achieved II, or verified RTL wiring.
    """
    if not isinstance(architecture, ArchitecturalModel):
        raise TypeError("A typed ArchitecturalModel is required")

    # Do not accept a forged topological graph, missing operator point, or
    # incorrectly restored architecture on the strength of a hash alone.
    lowered = lower(architecture)
    verified = verify_refinement(lowered)
    functional = architecture.functional
    points = {p.operator: p.effective for p in architecture.points}

    # The source emitter currently allocates exactly one data_t for the
    # complete generated function. A different per-op numeric format is a
    # hard source-backend blocker, not a cast to be inserted by this routine.
    formats = {
        op.name: (points[op.name].precision_bits, points[op.name].integer_bits)
        for op in functional.operators
    }
    unique_formats = set(formats.values())
    blockers: list[dict[str, Any]] = []
    if len(functional.inputs) != 1 or len(functional.outputs) != 1:
        blockers.append({
            "code": "UNSUPPORTED_HLS_TOP_BOUNDARY_ARITY",
            "inputs": list(functional.inputs),
            "outputs": list(functional.outputs),
            "detail": "Current HLS emitter supports exactly one graph input and one output",
        })
    if len(unique_formats) != 1:
        blockers.append({
            "code": "MIXED_OPERATOR_NUMERIC_FORMAT_UNSUPPORTED",
            "detail": "The current HLS emitter requires one ap_fixed type for all operators; no cast lowering exists",
            "formats_by_operator": {
                name: {"precision_bits": value[0], "integer_bits": value[1]}
                for name, value in formats.items()
            },
        })

    producer = {}
    for name in functional.inputs:
        producer[name] = "runtime_input"
    for tensor in functional.tensors:
        if tensor.data is not None:
            producer[tensor.name] = "initializer"
    for op in functional.operators:
        producer[op.outputs[0]] = op.name

    consumers: dict[str, list[str]] = {
        t.name: [] for t in functional.tensors
    }
    edges = []
    for op in functional.operators:
        for input_position, name in enumerate(op.inputs):
            tensor = functional.tensor(name)
            upstream = producer[name]
            consumers[name].append(op.name)
            src_format = formats.get(upstream)
            dst_format = formats[op.name]
            edge_compatible = src_format is None or src_format == dst_format
            if not edge_compatible:
                blockers.append({
                    "code": "INTER_OPERATOR_NUMERIC_FORMAT_MISMATCH",
                    "tensor": name,
                    "producer": upstream,
                    "consumer": op.name,
                    "producer_format": list(src_format),
                    "consumer_format": list(dst_format),
                })
            edges.append({
                "tensor": name,
                "logical_shape": list(tensor.shape),
                "functional_dtype": tensor.dtype,
                "producer": upstream,
                "consumer": op.name,
                "consumer_input_position": input_position,
                "edge_class": (
                    "inter_operator" if src_format is not None
                    else ("initializer" if upstream == "initializer" else "runtime_input")
                ),
                "numerical_format_compatible": edge_compatible,
                "source_layout": "flat_contiguous_static_array",
                "source_transfer_semantics": "ordered_tensor_access_in_one_HLS_function",
                "separate_stream_adapter_implemented": False,
            })

    tensors = [{
        "tensor": tensor.name,
        "shape": list(tensor.shape),
        "functional_dtype": tensor.dtype,
        "producer": producer[tensor.name],
        "consumers": consumers[tensor.name],
        "is_model_output": tensor.name in functional.outputs,
    } for tensor in functional.tensors]

    return {
        "schema": "fpgai-research/static-array-architecture-interface-audit/v1",
        "status": "SOURCE_BACKEND_BLOCKED" if blockers else "SOURCE_ARRAY_CONTRACT_COMPATIBLE",
        "functional_sha256": functional_fingerprint(functional),
        "effective_architecture_sha256": architecture_fingerprint(architecture),
        "refinement_e0_verified": verified["stage_invariants_verified"],
        "source_array_global_format": (
            {"precision_bits": next(iter(unique_formats))[0],
             "integer_bits": next(iter(unique_formats))[1]}
            if len(unique_formats) == 1 else None
        ),
        "operators": [op.name for op in functional.operators],
        "edges": edges,
        "tensors": tensors,
        "blockers": blockers,
        "evidence": "E0-typed-IR-and-existing-HLS-source-interface-policy-only",
        "compiled_cpp_executed": False,
        "original_vitis_calls": 0,
        "source_backend_one_input_one_output_required": True,
        "limitations": (
            "The HLS C++ emitter currently requires exactly one model input and "
            "one output and one common ap_fixed type; local tensor arrays and "
            "sequentially emitted operators do not establish a physical SRAM, "
            "FIFO, adapter, concurrent dataflow, achieved II, or routed timing. "
            "Different local PE/SIMD/reuse choices are legal under this source "
            "array interface when other compiler contracts pass."
        ),
    }
