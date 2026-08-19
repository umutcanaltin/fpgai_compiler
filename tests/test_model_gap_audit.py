from __future__ import annotations

from pathlib import Path

from fpgai.analysis.model_gap import audit_model_gaps
from fpgai.analysis.model_inspection import inspect_graph, write_model_inspection_report
from fpgai.ir import Graph


def test_model_gap_audit_is_layerwise_and_model_agnostic(tmp_path: Path) -> None:
    g = Graph("custom_changed_model")
    g.inputs = ["x"]
    g.outputs = ["z"]
    g.add_tensor("x", (1, 4))
    g.add_tensor("y", (1, 4))
    g.add_tensor("z", (1, 4))
    g.add_op("SiLU", ["x"], ["y"], name="activation")
    g.add_op("UserResearchLayer", ["y"], ["z"], name="custom")

    report = audit_model_gaps(g, pipeline_mode="training_on_device")
    assert report["policy"]["model_specific_compiler_path"] is False
    assert report["policy"]["layerwise_operator_compilation"] is True
    assert report["unsupported_operator_types"] == ["UserResearchLayer"]
    assert report["operator_counts"] == {"SiLU": 1, "UserResearchLayer": 1}
    assert report["layers"][0]["training_on_device"]["status"] in {"supported", "limited"}

    inspection = inspect_graph(
        g,
        model_path="user_model.onnx",
        pipeline_mode="training_on_device",
        allowed_operators=["SiLU", "UserResearchLayer"],
    )
    paths = write_model_inspection_report(inspection, tmp_path)
    assert Path(paths["model_gap_audit_json"]).is_file()


def test_model_gap_audit_preserves_ecosystem_provider() -> None:
    g = Graph("ecosystem_graph")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor("x", (1, 4))
    g.add_tensor("y", (1, 4))
    op = g.add_op("SiLU", ["x"], ["y"], name="external_silu")
    op.attrs["_fpgai_external_operator"] = {
        "package_id": "university_x.silu",
        "operator_id": "fpgai.operator.silu",
        "capabilities": {
            "training_forward": True,
            "backward_input": True,
            "parameter_gradients": False,
        },
    }
    report = audit_model_gaps(g, pipeline_mode="training_on_device")
    assert report["layers"][0]["provider"]["kind"] == "ecosystem_operator"
    assert report["layers"][0]["provider"]["package_id"] == "university_x.silu"
    assert report["provider_counts"]["ecosystem_operator"] == 1


def test_model_gap_audit_marks_nms_as_explicit_postprocess_partition_candidate() -> None:
    g = Graph("detector_with_nms")
    g.inputs = ["boxes"]
    g.outputs = ["selected"]
    g.add_tensor("boxes", (1, 100, 4))
    g.add_tensor("scores", (1, 1, 100))
    g.add_tensor("selected", (10, 3), "int64")
    g.add_op("NonMaxSuppression", ["boxes", "scores"], ["selected"], name="nms")
    report = audit_model_gaps(g, pipeline_mode="inference")
    assert "NonMaxSuppression" in report["unsupported_operator_types"]
    assert report["postprocess_partition_candidates"] == [
        {
            "index": 0,
            "name": "nms",
            "op_type": "NonMaxSuppression",
            "recommended_partition": "ps_or_host_postprocess",
            "reason": "Detection post-processing may remain outside the PL graph while preserving an explicit deployment boundary; a selectable PL implementation can be supplied through the FPGAI Ecosystem later.",
            "compiler_behavior": "explicit_gap_not_silent_fallback",
        }
    ]
