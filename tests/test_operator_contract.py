import pytest

from fpgai.operators import (
    AttributeContract,
    OnnxBinding,
    OperatorCapabilities,
    OperatorContract,
    OperatorEntrypoints,
    TensorPortContract,
    validate_operator_contract,
)


def _contract(**overrides):
    values = {
        "operator_id": "community.operator.grid_sample",
        "canonical_op_type": "GridSample",
        "version": 1,
        "category": "sampling",
        "inputs": (TensorPortContract("input"), TensorPortContract("grid")),
        "outputs": (TensorPortContract("output"),),
        "attributes": (AttributeContract("mode", "string", default="bilinear"),),
        "onnx_bindings": (OnnxBinding("ai.onnx", "GridSample", 16, None),),
        "capabilities": OperatorCapabilities(
            inference=True,
            shape_inference=True,
            type_inference=True,
            numeric_reference=True,
        ),
        "entrypoints": OperatorEntrypoints(
            shape_inference="python/operator.py:infer_shape",
            type_inference="python/operator.py:infer_type",
            numeric_reference="python/reference.py:run",
        ),
    }
    values.update(overrides)
    return OperatorContract(**values)


def test_operator_contract_is_research_scoped_and_serializable():
    payload = _contract().to_dict()
    assert payload["schema"] == "fpgai.operator-contract/v1"
    assert payload["usage"] == {"platform_scope": "research", "production_path": "morfics"}
    assert payload["onnx_bindings"][0]["op_type"] == "GridSample"


def test_operator_contract_rejects_non_namespaced_id():
    with pytest.raises(ValueError, match="OPCON001"):
        _contract(operator_id="GridSample")


def test_training_capabilities_are_consistent():
    with pytest.raises(ValueError, match="OPCON008"):
        _contract(capabilities=OperatorCapabilities(inference=True, backward_input=True))


def test_declared_shape_inference_without_entrypoint_is_warning_only():
    contract = _contract(
        capabilities=OperatorCapabilities(inference=True, shape_inference=True),
        entrypoints=OperatorEntrypoints(),
    )
    issues = validate_operator_contract(contract)
    assert any(issue.code == "OPCONW001" and issue.severity == "warning" for issue in issues)


def test_onnx_binding_rejects_invalid_opset_range():
    with pytest.raises(ValueError, match="opset_min"):
        OnnxBinding("ai.onnx", "GridSample", 18, 16)
