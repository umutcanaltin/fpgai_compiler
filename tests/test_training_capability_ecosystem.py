from __future__ import annotations

from fpgai.analysis.training_capability import audit_training_capabilities
from fpgai.ir import Graph


def test_training_audit_separates_reference_from_hardware_for_new_layers():
    g = Graph("audit")
    g.add_tensor("x", (4,)); g.add_tensor("y", (4,))
    g.add_op("SiLU", ["x"], ["y"], name="silu")
    report = audit_training_capabilities(g)
    layer = report["layers"][0]
    assert layer["reference_status"] == "supported"
    assert layer["hardware_status"] == "supported"
    assert report["complete"] is True
    assert report["hardware_complete"] is True


def test_external_operator_training_contract_is_used_by_same_audit():
    g = Graph("external")
    g.add_tensor("x", (4,)); g.add_tensor("y", (4,))
    g.add_op("CommunityOp", ["x"], ["y"], name="community", attrs={
        "_fpgai_external_operator": {
            "package_id": "community.operator_pkg",
            "operator_id": "community.operator.foo",
            "capabilities": {
                "inference": True,
                "training_forward": True,
                "backward_input": True,
                "parameter_gradients": False,
            },
        }
    })
    report = audit_training_capabilities(g)
    layer = report["layers"][0]
    assert layer["provider"]["kind"] == "ecosystem_operator"
    assert layer["reference_status"] == "declared_by_ecosystem_contract"
    assert layer["training"]["backward_input"] is True
