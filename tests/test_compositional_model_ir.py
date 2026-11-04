"""Compiler-first compositional IR tests: no DRL or vendor calls."""
from __future__ import annotations

from dataclasses import replace
from math import prod
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.compiler.composition import compile_composed_architecture
from fpgai_research.model_ir.composition import (
    composition_receipt, physical_offset, resolve_composition,
)
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.host import validate_host


def two_matmuls(*, rows=3, inner=3, middle=5, output=2):
    w1 = tuple(0.03 * ((i % 7) - 3) for i in range(inner * middle))
    w2 = tuple(0.04 * ((i % 5) - 2) for i in range(middle * output))
    return FunctionalModel(
        tensors=(
            Tensor("x", (rows, inner)),
            Tensor("w1", (inner, middle), data=w1),
            Tensor("hidden", (rows, middle)),
            Tensor("w2", (middle, output), data=w2),
            Tensor("y", (rows, output)),
        ),
        operators=(
            Operator("first", "MatMul", ("x", "w1"), ("hidden",)),
            Operator("second", "MatMul", ("hidden", "w2"), ("y",)),
        ),
        inputs=("x",), outputs=("y",),
    )


def architecture(model):
    base = ArchitectureRequest()
    return resolve(model, base, {
        "first": replace(base, pes=2, simd=3, serialization=2,
                         tile=(2, 3, 2), pipeline_ii=2),
        "second": replace(base, parallelism=2, pes=2, simd=2,
                          serialization=3, tile=(2, 2, 3)),
    })


def test_address_bijection_and_explicit_iteration_to_boundary_mapping():
    assert [
        physical_offset((3, 5), (i, j), "row_major")
        for i in range(3) for j in range(5)
    ] == list(range(15))
    addresses = [
        physical_offset((3, 5), (i, j), "column_major")
        for i in range(3) for j in range(5)
    ]
    assert len(set(addresses)) == 15
    assert addresses != list(range(15))
    with pytest.raises(ValueError, match="Unsupported"):
        physical_offset((3, 5), (0, 0), "blocked_unknown")
    with pytest.raises(ValueError, match="out of bounds"):
        physical_offset((3, 5), (3, 0), "column_major")


def test_real_hls_producer_consumer_addresses_change_and_functional_identity_is_stable():
    arch = architecture(two_matmuls())
    baseline = compile_composed_architecture(arch)
    changed = compile_composed_architecture(
        arch, intermediate_layouts={"hidden": "column_major"})
    assert baseline.evidence["functional_graph_preserved"]
    assert changed.evidence["functional_graph_preserved"]
    assert baseline.emission.manifest["functional_sha256"] == changed.emission.manifest["functional_sha256"]
    assert baseline.emission.manifest["cpp_sha256"] != changed.emission.manifest["cpp_sha256"]
    assert "tensor_physical_layouts" not in baseline.emission.manifest
    assert changed.emission.manifest["tensor_physical_layouts"] == {"hidden": "column_major"}
    assert "tensor_2[ni*3+mi] = acc;" in changed.emission.cpp
    assert "tensor_2[kk*3+mi]" in changed.emission.cpp
    assert "tensor_2[mi*5+ni] = acc;" in baseline.emission.cpp
    assert "tensor_2[mi*5+kk]" in baseline.emission.cpp
    assert changed.evidence["strongest_evidence"] == "E1-generated-source-only"
    assert changed.evidence["vitis_csynth_executed"] is False
    contracts = changed.evidence["boundary_contracts"]
    assert len(contracts) == 1
    assert contracts[0]["tensor"] == "hidden"
    assert contracts[0]["producer"] == "first"
    assert contracts[0]["consumers"] == ("second",) or contracts[0]["consumers"] == ["second"]
    assert contracts[0]["storage_layout"] == "column_major"
    assert contracts[0]["producer_completed_before_consumers"]


def test_unmodified_legacy_source_is_identical_to_default_opt_in_composer():
    arch = architecture(two_matmuls())
    raw = emit_hls(lower(arch))
    composed = compile_composed_architecture(arch)
    assert raw.cpp == composed.emission.cpp
    assert raw.manifest == composed.emission.manifest
    assert composition_receipt(resolve_composition(arch))["original_vitis_calls"] == 0


@pytest.mark.parametrize("rows,inner,middle,output", [
    (2, 3, 5, 2), (3, 3, 5, 2), (4, 5, 3, 3), (5, 4, 7, 2),
])
def test_typed_intermediate_layouts_preserve_numerical_output_in_actual_cpp(
        rows, inner, middle, output):
    if shutil.which("g++") is None:
        pytest.skip("g++ not installed")
    model = two_matmuls(rows=rows, inner=inner, middle=middle, output=output)
    # Changing tensor dimensions can invalidate an explicitly fixed tiling
    # request in some backend revisions; baseline uses legal scalar mapping.
    arch = resolve(model, ArchitectureRequest())
    before = compile_composed_architecture(arch)
    after = compile_composed_architecture(
        arch, intermediate_layouts={"hidden": "column_major"})
    x = np.linspace(-0.15, 0.15, rows * inner, dtype=np.float32).reshape(rows, inner)
    baseline = validate_host(model, before.emission, x)
    variant = validate_host(model, after.emission, x)
    assert baseline["accepted"] and variant["accepted"]
    assert variant["max_absolute_error"] < 1e-5
    assert variant["cpp_sha256"] != baseline["cpp_sha256"]


def test_unsupported_layout_or_unimplemented_tensor_exchange_fails_closed():
    model = two_matmuls()
    arch = architecture(model)
    with pytest.raises(ValueError, match="Unsupported physical tensor layout"):
        resolve_composition(arch, layouts={"hidden": "streaming"})
    with pytest.raises(ValueError, match="internal produced"):
        resolve_composition(arch, layouts={"x": "column_major"})
    with pytest.raises(ValueError, match="internal produced"):
        resolve_composition(arch, layouts={"y": "column_major"})
    with pytest.raises(ValueError, match="verification bound"):
        resolve_composition(arch, layouts={"hidden": "column_major"}, max_elements=2)
    with pytest.raises(ValueError, match="not an exact|positive|out of bounds"):
        physical_offset((3, 0), (0, 0), "row_major")


def test_non_matmul_consumer_is_not_silently_rewritten():
    original = two_matmuls()
    # Replace downstream MatMul by a shape-preserving Relu.
    model = FunctionalModel(
        tensors=(
            original.tensors[0], original.tensors[1], original.tensors[2],
            Tensor("y", original.tensors[2].shape),
        ),
        operators=(
            original.operators[0],
            Operator("relu", "Relu", ("hidden",), ("y",)),
        ),
        inputs=("x",), outputs=("y",),
    )
    arch = resolve(model, ArchitectureRequest())
    with pytest.raises(ValueError, match="MatMul/Gemm output to lhs"):
        resolve_composition(arch, layouts={"hidden": "column_major"})


def test_numeric_mismatch_is_blocked_in_compositional_ir():
    model = two_matmuls()
    base = ArchitectureRequest()
    arch = resolve(model, base, {
        "second": replace(base, precision_bits=12, integer_bits=4)})
    with pytest.raises(ValueError, match="Mixed per-operator"):
        compile_composed_architecture(
            arch, intermediate_layouts={"hidden": "column_major"})


def test_one_producer_two_independent_matmul_consumers_share_real_layout():
    """Fan-out uses one typed value relation and two separately mapped readers."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    rows, inner, middle, output = 3, 3, 5, 2
    model = FunctionalModel(
        tensors=(
            Tensor("x", (rows, inner)),
            Tensor("w1", (inner, middle), data=tuple(
                0.02*((i % 7)-3) for i in range(inner*middle))),
            Tensor("hidden", (rows, middle)),
            Tensor("wa", (middle, output), data=tuple(
                0.03*((i % 5)-2) for i in range(middle*output))),
            Tensor("wb", (middle, output), data=tuple(
                0.04*((i % 3)-1) for i in range(middle*output))),
            Tensor("ya", (rows, output)),
            Tensor("yb", (rows, output)),
            Tensor("y", (rows, output)),
        ),
        operators=(
            Operator("producer", "MatMul", ("x", "w1"), ("hidden",)),
            Operator("consumer_a", "MatMul", ("hidden", "wa"), ("ya",)),
            Operator("consumer_b", "MatMul", ("hidden", "wb"), ("yb",)),
            Operator("join", "Add", ("ya", "yb"), ("y",)),
        ),
        inputs=("x",), outputs=("y",),
    )
    base = ArchitectureRequest()
    arch = resolve(model, base, {
        "producer": replace(base, pes=2, simd=3),
        "consumer_a": replace(base, pes=2, simd=2),
        "consumer_b": replace(base, pes=3, simd=3),
    })
    default = compile_composed_architecture(arch)
    variant = compile_composed_architecture(
        arch, intermediate_layouts={"hidden": "column_major"})
    boundary = next(b for b in variant.composed_ir.boundaries
                    if b.tensor == "hidden")
    assert boundary.consumers == ("consumer_a", "consumer_b")
    assert boundary.physical_elements == rows*middle
    assert "tensor_2[kk*3+mi]" in variant.emission.cpp
    assert variant.emission.cpp.count("tensor_2[kk*3+mi]") >= 2
    from fpgai_research.model_ir.exchange_witness import check_element_exchange
    witness = check_element_exchange(variant.composed_ir)
    edge = next(c for c in witness["relations"] if c["tensor"] == "hidden")
    assert edge["status"] == "BOUNDED_PHI_EXCHANGE_VERIFIED"
    assert edge["logical_producer_consumer_events"] == rows*middle*(1+2+2)
    x = np.linspace(-0.2, 0.2, rows*inner,
                    dtype=np.float32).reshape(rows, inner)
    original = validate_host(model, default.emission, x)
    modified = validate_host(model, variant.emission, x)
    assert original["accepted"] and modified["accepted"]
    assert default.emission.manifest["functional_sha256"] == variant.emission.manifest["functional_sha256"]
    assert default.emission.manifest["cpp_sha256"] != variant.emission.manifest["cpp_sha256"]
