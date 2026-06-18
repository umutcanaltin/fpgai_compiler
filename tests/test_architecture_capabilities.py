from __future__ import annotations

import pytest

from fpgai.compiler.architecture_capabilities import (
    ArchitectureCapabilityError,
    IMPLEMENTED,
    LIMITED,
    PLANNING_ONLY,
    UNSUPPORTED,
    validate_architecture_capabilities,
)
from fpgai.engine.models import (
    ArchitecturePlan,
    BufferingPlan,
    CompilePlan,
    LayerMemoryPlan,
    LayerPlan,
    ParallelismPlan,
    PartitionPlan,
    PipelinePlan,
    PrecisionPlan,
    TilingPlan,
)


def _layer(
    *,
    name: str = "dense0",
    ii: int = 1,
    tile: bool = False,
    partition: int = 1,
    buffering: str = "single",
    weight_mode: str = "embedded",
    unroll_in: int = 2,
) -> LayerPlan:
    return LayerPlan(
        node_name=name,
        op_type="Dense",
        architecture=ArchitecturePlan(
            precision=PrecisionPlan(
                mode="fixed",
                activation_bits=12,
                weight_bits=10,
                bias_bits=20,
                accumulator_bits=24,
            ),
            pipeline=PipelinePlan(ii=ii),
            parallelism=ParallelismPlan(
                pe=2,
                simd=unroll_in,
                unroll={
                    "out": 2,
                    "in": unroll_in,
                },
            ),
            partitioning=PartitionPlan(
                factor=partition,
                mode="cyclic",
                targets={
                    "input": partition,
                    "output": partition,
                    "weight": partition,
                },
            ),
            tiling=TilingPlan(
                sizes=(
                    {"in": 32, "out": 8}
                    if tile
                    else {}
                )
            ),
            buffering=BufferingPlan(mode=buffering),
            memory=LayerMemoryPlan(
                weight_mode=weight_mode,
                activation_mode="buffer",
            ),
        ),
    )


def _status_map(report) -> dict[str, str]:
    return {
        issue.feature: issue.status
        for issue in report.issues
    }


def test_supported_baseline_has_no_blocking_issues() -> None:
    report = validate_architecture_capabilities(
        CompilePlan(layer_plans=[_layer()])
    )
    statuses = _status_map(report)

    assert report.valid is True
    assert statuses["precision"] == IMPLEMENTED
    assert statuses["pipeline"] == IMPLEMENTED
    assert statuses["parallelism"] == IMPLEMENTED
    assert statuses["partitioning"] == IMPLEMENTED
    assert statuses["tiling"] == IMPLEMENTED
    assert statuses["buffering"] == IMPLEMENTED
    assert statuses["memory"] == IMPLEMENTED


def test_planning_only_features_are_reported() -> None:
    report = validate_architecture_capabilities(
        CompilePlan(
            layer_plans=[
                _layer(
                    tile=True,
                    partition=4,
                    buffering="double",
                )
            ]
        )
    )
    statuses = _status_map(report)

    assert report.valid is False
    assert statuses["tiling"] == IMPLEMENTED
    assert statuses["partitioning"] == IMPLEMENTED
    assert statuses["buffering"] == PLANNING_ONLY


def test_ddr_memory_is_unsupported() -> None:
    report = validate_architecture_capabilities(
        CompilePlan(
            layer_plans=[
                _layer(weight_mode="ddr")
            ]
        )
    )

    assert _status_map(report)["memory"] == UNSUPPORTED
    assert report.valid is False


def test_per_layer_ii_difference_is_implemented_for_inference() -> None:
    report = validate_architecture_capabilities(
        CompilePlan(
            layer_plans=[
                _layer(name="dense0", ii=1),
                _layer(name="dense1", ii=2),
            ]
        )
    )

    pipeline_issues = [
        issue
        for issue in report.issues
        if issue.feature == "pipeline"
    ]
    assert all(
        issue.status == IMPLEMENTED
        for issue in pipeline_issues
    )


def test_training_specialization_is_implemented() -> None:
    report = validate_architecture_capabilities(
        CompilePlan(
            layer_plans=[
                _layer(
                    tile=False,
                    partition=4,
                )
            ]
        ),
        pipeline_mode="training_on_device",
    )
    statuses = _status_map(report)

    assert statuses["pipeline"] == IMPLEMENTED
    assert statuses["parallelism"] == IMPLEMENTED
    assert statuses["partitioning"] == IMPLEMENTED


def test_strict_mode_rejects_planning_only_options() -> None:
    with pytest.raises(
        ArchitectureCapabilityError,
        match="buffering",
    ):
        validate_architecture_capabilities(
            CompilePlan(
                layer_plans=[_layer(buffering="double")]
            ),
            strict=True,
        )
