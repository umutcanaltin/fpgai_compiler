from __future__ import annotations

import numpy as np

from fpgai.analysis.model_inspection import inspect_graph
from fpgai.ir import Graph


def _graph_with(
    op_type: str,
) -> Graph:
    graph = Graph("inspection_test")

    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        "input",
        (1, 4),
    )
    graph.add_tensor(
        "output",
        (1, 2),
    )

    graph.constants["weights"] = np.ones(
        (2, 4),
        dtype=np.float32,
    )

    graph.add_op(
        op_type,
        ["input", "weights"],
        ["output"],
        name="op0",
    )

    return graph


def test_inspection_counts_parameters_and_supported_ops() -> None:
    report = inspect_graph(
        _graph_with("Dense"),
        model_path="model.onnx",
        pipeline_mode="inference",
        allowed_operators=["Dense"],
    )

    assert report.compilation_ready is True
    assert report.operator_counts == {
        "Dense": 1,
    }
    assert report.parameter_values == 8
    assert report.parameter_bytes == 32
    assert report.unsupported_operators == []
    assert report.disallowed_operators == []


def test_inspection_reports_disallowed_and_unsupported_ops() -> None:
    report = inspect_graph(
        _graph_with("CustomOp"),
        model_path="model.onnx",
        pipeline_mode="inference",
        allowed_operators=["Dense"],
    )

    assert report.compilation_ready is False
    assert report.disallowed_operators == [
        "CustomOp",
    ]
    assert report.unsupported_operators == [
        "CustomOp",
    ]


def test_inspection_marks_limited_ops_without_rejecting_model() -> None:
    report = inspect_graph(
        _graph_with("Add"),
        model_path="model.onnx",
        pipeline_mode="inference",
        allowed_operators=["Add"],
    )

    assert report.compilation_ready is True
    assert report.limited_operators == [
        "Add",
    ]