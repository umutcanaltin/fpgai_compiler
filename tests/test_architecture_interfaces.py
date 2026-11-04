"""E0 compiler array-interface contracts: positive and fail-closed graph cases.

No RL, native vendor tools or generated-case directories are used.
"""
from dataclasses import replace

from fpgai_research.architecture.interfaces import inspect_array_interfaces
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)


def chain():
    return FunctionalModel(
        tensors=(
            Tensor("x", (1, 3)),
            Tensor("w1", (3, 4), data=(0.125,) * 12),
            Tensor("h", (1, 4)),
            Tensor("w2", (4, 2), data=(0.125,) * 8),
            Tensor("y", (1, 2)),
        ),
        operators=(
            Operator("mm1", "MatMul", ("x", "w1"), ("h",)),
            Operator("mm2", "MatMul", ("h", "w2"), ("y",)),
        ),
        inputs=("x",),
        outputs=("y",),
    )


def test_independent_local_spatial_architectures_are_compatible_across_array_boundary():
    default = ArchitectureRequest()
    arch = resolve(chain(), default, {
        "mm1": replace(default, pes=2, simd=3, serialization=2),
        "mm2": replace(default, pes=3, simd=2, parallelism=2),
    })
    report = inspect_array_interfaces(arch)
    assert report["status"] == "SOURCE_ARRAY_CONTRACT_COMPATIBLE"
    assert report["blockers"] == []
    assert report["original_vitis_calls"] == 0
    assert report["refinement_e0_verified"]
    edge = next(x for x in report["edges"] if x["tensor"] == "h")
    assert edge["producer"] == "mm1" and edge["consumer"] == "mm2"
    assert edge["logical_shape"] == [1, 4]
    assert edge["numerical_format_compatible"]
    assert not edge["separate_stream_adapter_implemented"]
    assert report["source_array_global_format"] == {
        "precision_bits": 16, "integer_bits": 6
    }


def test_mixed_operator_numerical_formats_fail_closed_and_do_not_claim_casts():
    default = ArchitectureRequest()
    arch = resolve(chain(), default, {
        "mm2": replace(default, precision_bits=12, integer_bits=5),
    })
    report = inspect_array_interfaces(arch)
    assert report["status"] == "SOURCE_BACKEND_BLOCKED"
    codes = {x["code"] for x in report["blockers"]}
    assert "MIXED_OPERATOR_NUMERIC_FORMAT_UNSUPPORTED" in codes
    assert "INTER_OPERATOR_NUMERIC_FORMAT_MISMATCH" in codes
    edge = next(x for x in report["edges"] if x["tensor"] == "h")
    assert not edge["numerical_format_compatible"]
    assert report["source_array_global_format"] is None
    assert not report["compiled_cpp_executed"]


def test_fanout_and_merge_preserve_producer_consumer_tensor_identity():
    model = FunctionalModel(
        tensors=(
            Tensor("x", (1, 3)),
            Tensor("wa", (3, 2), data=(0.1,) * 6),
            Tensor("wb", (3, 2), data=(0.2,) * 6),
            Tensor("a", (1, 2)),
            Tensor("b", (1, 2)),
            Tensor("y", (1, 2)),
        ),
        operators=(
            Operator("left", "MatMul", ("x", "wa"), ("a",)),
            Operator("right", "MatMul", ("x", "wb"), ("b",)),
            Operator("merge", "Add", ("a", "b"), ("y",)),
        ),
        inputs=("x",),
        outputs=("y",),
    )
    report = inspect_array_interfaces(resolve(model, ArchitectureRequest()))
    assert report["status"] == "SOURCE_ARRAY_CONTRACT_COMPATIBLE"
    assert len([e for e in report["edges"] if e["tensor"] == "x"]) == 2
    assert next(x for x in report["tensors"] if x["tensor"] == "x")["consumers"] == [
        "left", "right"
    ]
    assert next(x for x in report["tensors"] if x["tensor"] == "a")["consumers"] == [
        "merge"
    ]


def test_source_top_boundary_arity_is_not_mistaken_for_supported_hls():
    model = FunctionalModel(
        tensors=(
            Tensor("x", (1, 3)), Tensor("z", (1, 3)),
            Tensor("w", (3, 2), data=(0.1,) * 6),
            Tensor("a", (1, 2)), Tensor("b", (1, 2)),
            Tensor("y", (1, 2)),
        ),
        operators=(
            Operator("mm_a", "MatMul", ("x", "w"), ("a",)),
            Operator("mm_b", "MatMul", ("z", "w"), ("b",)),
            Operator("join", "Add", ("a", "b"), ("y",)),
        ),
        inputs=("x", "z"),
        outputs=("y",),
    )
    report = inspect_array_interfaces(resolve(model, ArchitectureRequest()))
    assert report["status"] == "SOURCE_BACKEND_BLOCKED"
    assert "UNSUPPORTED_HLS_TOP_BOUNDARY_ARITY" in {b["code"] for b in report["blockers"]}
