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

def test_write_model_inspection_report_outputs_profile_and_summary(tmp_path):
    from fpgai.analysis.model_inspection import (
        ModelInspection,
        write_model_inspection_report,
    )

    inspection = ModelInspection(
        model_path="model.onnx",
        pipeline_mode="inference",
        graph_name="main",
        inputs=[],
        outputs=[],
        operator_counts={"Dense": 1},
        operators=[],
        constants=[],
        parameter_values=10,
        parameter_bytes=40,
        disallowed_operators=[],
        unsupported_operators=[],
        limited_operators=[],
    )

    paths = write_model_inspection_report(
        inspection,
        tmp_path,
    )

    profile = tmp_path / "model_profile.json"
    summary = tmp_path / "prediction_summary.md"

    assert paths["model_profile_json"] == str(profile)
    assert paths["prediction_summary_md"] == str(summary)
    assert profile.exists()
    assert summary.exists()
    assert "compilation_ready" in profile.read_text(encoding="utf-8")
    assert "Resource and timing prediction artifacts" in summary.read_text(encoding="utf-8")

