from types import SimpleNamespace

import pytest

from fpgai.engine.network_execution import build_network_execution_plan, requested_network_execution_mode


def _desc(name, op_type, in_shape=(1, 4, 8), out_shape=(1, 4, 8)):
    return SimpleNamespace(
        node_name=name,
        op_type=op_type,
        input_shapes=[in_shape],
        output_shapes=[out_shape],
        attrs={},
    )


def test_network_mode_prefers_architecture_network_over_legacy_hls_key():
    raw = {
        "architecture": {"network": {"execution": {"mode": "dataflow"}}},
        "hls": {"execution_mode": "sequential"},
    }
    assert requested_network_execution_mode(raw) == "dataflow"


def test_streamed_alias_resolves_to_dataflow():
    assert requested_network_execution_mode({"architecture": {"network": {"execution": {"mode": "streamed"}}}}) == "dataflow"


def test_invalid_network_mode_is_rejected():
    with pytest.raises(ValueError, match="NETEXEC001"):
        requested_network_execution_mode({"architecture": {"network": {"execution": {"mode": "magic"}}}})


def test_phase_shared_builds_only_compatible_reuse_groups():
    descs = [
        _desc("q", "MatMul"),
        _desc("k", "MatMul"),
        _desc("v", "MatMul"),
        _desc("other", "MatMul", out_shape=(1, 4, 16)),
        _desc("relu", "Relu"),
    ]
    plan = build_network_execution_plan(
        {"architecture": {"network": {"execution": {"mode": "phase_shared"}}}},
        descs,
        pipeline_mode="training_on_device",
    )
    assert plan.resolved_mode == "phase_shared"
    assert plan.physical_status == "planning_only"
    matmul_groups = [g for g in plan.reuse_groups if g["op_type"] == "MatMul"]
    assert len(matmul_groups) == 1
    assert matmul_groups[0]["members"] == ["q", "k", "v"]


def test_inference_dataflow_is_physical_pragma_mode():
    plan = build_network_execution_plan(
        {"architecture": {"network": {"execution": {"mode": "dataflow"}}}},
        [],
        pipeline_mode="inference",
    )
    assert plan.dataflow_pragma is True
    assert plan.physical_status == "implemented"


def test_training_dataflow_is_not_overclaimed():
    plan = build_network_execution_plan(
        {"architecture": {"network": {"execution": {"mode": "dataflow"}}}},
        [],
        pipeline_mode="training_on_device",
    )
    assert plan.dataflow_pragma is False
    assert plan.physical_status == "planning_only"


def test_compile_plan_signature_can_distinguish_network_execution_modes():
    from fpgai.engine.models import CompilePlan
    sequential = CompilePlan(notes={"network_execution": {"resolved_mode": "sequential"}})
    dataflow = CompilePlan(notes={"network_execution": {"resolved_mode": "dataflow"}})
    assert sequential.architecture_signature != dataflow.architecture_signature
