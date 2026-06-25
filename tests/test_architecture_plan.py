from __future__ import annotations

from dataclasses import replace

from fpgai.analysis.hls_architecture import build_hls_architecture
from fpgai.engine.models import (
    ArchitecturePlan,
    BufferingPlan,
    CompilePlan,
    LayerDescriptor,
    LayerMemoryPlan,
    LayerPlan,
    ParallelismPlan,
    PartitionPlan,
    PipelinePlan,
    PrecisionPlan,
    TilingPlan,
)


def _architecture() -> ArchitecturePlan:
    return ArchitecturePlan(
        precision=PrecisionPlan(
            mode="fixed",
            activation_bits=12,
            weight_bits=10,
            bias_bits=20,
            accumulator_bits=24,
            activation_int_bits=4,
            weight_int_bits=3,
            bias_int_bits=8,
            accumulator_int_bits=10,
        ),
        pipeline=PipelinePlan(ii=2, style="aggressive"),
        parallelism=ParallelismPlan(
            pe=2,
            simd=4,
            unroll={"out": 2, "in": 4},
        ),
        partitioning=PartitionPlan(
            factor=4,
            mode="cyclic",
            targets={
                "input": 4,
                "output": 2,
                "weight": 8,
            },
        ),
        tiling=TilingPlan(
            sizes={"in": 32, "out": 8}
        ),
        buffering=BufferingPlan(mode="double"),
        memory=LayerMemoryPlan(
            weight_mode="ddr",
            activation_mode="buffer",
            weight_region="DDR",
            activation_region="BRAM",
        ),
    )


def test_architecture_signature_is_stable() -> None:
    first = _architecture()
    second = _architecture()

    assert first.signature == second.signature
    assert len(first.signature) == 64


def test_each_architecture_section_changes_signature() -> None:
    base = _architecture()
    variants = [
        replace(
            base,
            precision=replace(
                base.precision,
                activation_bits=14,
            ),
        ),
        replace(
            base,
            pipeline=replace(base.pipeline, ii=1),
        ),
        replace(
            base,
            parallelism=replace(
                base.parallelism,
                pe=4,
            ),
        ),
        replace(
            base,
            partitioning=replace(
                base.partitioning,
                factor=2,
            ),
        ),
        replace(
            base,
            tiling=TilingPlan(
                sizes={"in": 16, "out": 8}
            ),
        ),
        replace(
            base,
            buffering=BufferingPlan(mode="single"),
        ),
        replace(
            base,
            memory=replace(
                base.memory,
                weight_mode="embedded",
            ),
        ),
    ]

    assert all(
        variant.signature != base.signature
        for variant in variants
    )
    assert len(
        {variant.signature for variant in variants}
    ) == len(variants)


def test_legacy_layer_plan_builds_typed_contract() -> None:
    layer = LayerPlan(
        node_name="dense0",
        op_type="Dense",
        precision_mode="fixed",
        act_bits=12,
        weight_bits=10,
        tile={"in": 32, "out": 8},
        unroll={"in": 4, "out": 2},
        pipeline_ii=2,
        weight_mode="ddr",
        activation_mode="buffer",
        buffering="double",
        notes={
            "partition_factor": 4,
            "partition_mode": "cyclic",
            "requested_bias_bits": 20,
            "requested_accum_bits": 24,
        },
    )

    assert layer.architecture is not None
    assert layer.architecture.pipeline.ii == 2
    assert layer.architecture.parallelism.pe == 2
    assert layer.architecture.parallelism.simd == 4
    assert layer.architecture.partitioning.factor == 4
    assert layer.architecture.tiling.sizes["in"] == 32
    assert layer.architecture.buffering.double_buffer is True
    assert layer.architecture.memory.weight_mode == "ddr"


def test_hls_architecture_uses_typed_effective_values() -> None:
    descriptor = LayerDescriptor(
        node_name="dense0",
        op_type="Dense",
        inputs=["input", "weights", "bias"],
        outputs=["output"],
        input_shapes=[
            (1, 64),
            (8, 64),
            (8,),
        ],
        output_shapes=[(1, 8)],
        attrs={
            "in_features": 64,
            "out_features": 8,
        },
    )
    layer_plan = LayerPlan(
        node_name="dense0",
        op_type="Dense",
        precision_mode="fixed",
        act_bits=16,
        weight_bits=16,
        tile={"in": 64, "out": 8},
        unroll={"in": 1, "out": 1},
        pipeline_ii=1,
        weight_mode="embedded",
        buffering="single",
        architecture=_architecture(),
    )
    compile_plan = CompilePlan(
        clock_mhz=175.0,
        layer_plans=[layer_plan],
    )
    config = {
        "optimization": {
            "parallel_policy": "Balanced",
            "parallel": {
                "pe": 1,
                "simd": 1,
                "partition_factor": 1,
            },
        },
        "numerics": {
            "defaults": {
                "activation": {
                    "type": "ap_fixed",
                    "total_bits": 16,
                    "int_bits": 6,
                },
                "weight": {
                    "type": "ap_fixed",
                    "total_bits": 16,
                    "int_bits": 6,
                },
                "bias": {
                    "type": "ap_fixed",
                    "total_bits": 24,
                    "int_bits": 10,
                },
                "accum": {
                    "type": "ap_fixed",
                    "total_bits": 24,
                    "int_bits": 10,
                },
            }
        },
    }

    result = build_hls_architecture(
        [descriptor],
        config,
        compile_plan,
    )
    layer = result.layers[0]

    assert result.clock_mhz == 175.0
    assert result.architecture_signature == (
        compile_plan.architecture_signature
    )
    assert layer.pipeline_ii == 2
    assert layer.unroll == {"out": 2, "in": 4}
    assert layer.arithmetic["activation_bits"] == 12
    assert layer.arithmetic["weight_bits"] == 10
    assert layer.memory["partition_factor"] == 4
    assert layer.memory["weight_mode"] == "ddr"
    assert layer.tiling == {"in": 32, "out": 8}
    assert layer.buffering["double_buffer"] is True
    assert layer.architecture_signature == (
        layer_plan.architecture_signature
    )


def test_compile_signature_changes_with_effective_architecture() -> None:
    first_layer = LayerPlan(
        node_name="dense0",
        op_type="Dense",
        architecture=_architecture(),
    )
    second_layer = LayerPlan(
        node_name="dense0",
        op_type="Dense",
        architecture=replace(
            _architecture(),
            pipeline=PipelinePlan(
                ii=1,
                style="aggressive",
            ),
        ),
    )

    first = CompilePlan(
        target_board="kv260",
        target_part="xck26",
        clock_mhz=200.0,
        layer_plans=[first_layer],
    )
    duplicate = CompilePlan(
        target_board="kv260",
        target_part="xck26",
        clock_mhz=200.0,
        layer_plans=[
            LayerPlan(
                node_name="renamed_dense",
                op_type="Dense",
                architecture=_architecture(),
            )
        ],
    )
    changed = CompilePlan(
        target_board="kv260",
        target_part="xck26",
        clock_mhz=200.0,
        layer_plans=[second_layer],
    )

    assert first.architecture_signature == (
        duplicate.architecture_signature
    )
    assert first.architecture_signature != (
        changed.architecture_signature
    )






def test_memory_first_policy_is_planned_as_memory_safe():
    from fpgai.engine.planner import POLICIES

    memory = POLICIES["Memory-First"]
    bram = POLICIES["BRAM-Saver"]

    assert memory.name == "Memory-First"

    for field in ["pe", "simd", "unroll_factor", "partition_factor"]:
        assert getattr(memory, field) == getattr(bram, field)

    # Memory-First is a distinct, HLS-safe memory policy:
    # use BRAM before URAM for initialized embedded parameter arrays.
    assert memory.weight_region_preference[0] == "BRAM"
    assert memory.activation_region_preference[0] == "BRAM"
    assert "URAM" in memory.weight_region_preference
