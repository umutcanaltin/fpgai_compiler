from __future__ import annotations

from types import SimpleNamespace

from fpgai.compiler.architecture_capabilities import (
    IMPLEMENTED,
    PLANNING_ONLY,
    validate_architecture_capabilities,
)


class _Section(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def _architecture_with_tiling():
    return SimpleNamespace(
        precision=_Section(bits=16, integer_bits=6),
        pipeline=_Section(ii=2),
        parallelism=_Section(unroll={"out": 2, "in": 4}),
        partitioning=_Section(factor=1, targets={}),
        tiling=_Section(sizes={"input": 16, "output": 8}),
        buffering=_Section(double_buffer=False),
        memory=_Section(
            weight_mode="embedded",
            activation_mode="buffer",
        ),
    )


def _compile_plan(op_type: str = "Dense"):
    return SimpleNamespace(
        layer_plans=[
            SimpleNamespace(
                node_name=f"{op_type.lower()}0",
                op_type=op_type,
                architecture=_architecture_with_tiling(),
            )
        ]
    )


def _status(report, feature: str) -> str:
    matches = [
        issue
        for issue in report.issues
        if issue.feature == feature
    ]
    assert len(matches) == 1
    return matches[0].status


def test_dense_inference_tiling_is_implemented() -> None:
    report = validate_architecture_capabilities(
        _compile_plan("Dense"),
        pipeline_mode="inference",
    )

    assert _status(report, "tiling") == IMPLEMENTED
    issue = [
        item
        for item in report.issues
        if item.feature == "tiling"
    ][0]
    assert issue.effective == {
        "sizes": {
            "input": 16,
            "output": 8,
        }
    }


def test_conv_inference_tiling_is_implemented() -> None:
    report = validate_architecture_capabilities(
        _compile_plan("Conv"),
        pipeline_mode="inference",
    )

    assert _status(report, "tiling") == IMPLEMENTED


def test_training_tiling_remains_planning_only_until_training_emitters_are_tiled() -> None:
    report = validate_architecture_capabilities(
        _compile_plan("Dense"),
        pipeline_mode="training_on_device",
    )

    assert _status(report, "tiling") == PLANNING_ONLY
