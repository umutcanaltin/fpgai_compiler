from __future__ import annotations

from fpgai.implementations.compatibility import CompatibilityRequest, evaluate_implementation_compatibility
from fpgai.implementations.implementation_contract import ImplementationContract, TrainingImplementationCapabilities


def _contract(training: TrainingImplementationCapabilities) -> ImplementationContract:
    return ImplementationContract(
        package_id="community.matmul_hls",
        version="1.0.0",
        operator_id="community.operator.matmul",
        language="hls_cpp",
        backend="vitis_hls",
        top="matmul",
        sources=("matmul.cpp",),
        training=training,
        weight_storage=("bram", "ddr"),
        activation_storage=("bram",),
        gradient_storage=("uram", "ddr"),
        optimizer_state_storage=("ddr",),
    )


def test_training_implementation_contract_enforces_backward_update_and_storage():
    contract = _contract(TrainingImplementationCapabilities(forward=True, backward_input=True, parameter_gradients=True, optimizer_update=False))
    request = CompatibilityRequest(
        mode="training",
        backend="vitis_hls",
        weight_storage="bram",
        activation_storage="bram",
        gradient_storage="uram",
        optimizer_state_storage="ddr",
        require_backward_input=True,
        require_parameter_gradients=True,
        require_optimizer_update=True,
    )
    result = evaluate_implementation_compatibility(contract, request)
    assert result.compatible is False
    assert "training_optimizer_update_not_supported" in result.reasons


def test_training_implementation_contract_accepts_full_capability_match():
    contract = _contract(TrainingImplementationCapabilities(forward=True, backward_input=True, parameter_gradients=True, optimizer_update=True))
    request = CompatibilityRequest(
        mode="training",
        gradient_storage="ddr",
        optimizer_state_storage="ddr",
        require_backward_input=True,
        require_parameter_gradients=True,
        require_optimizer_update=True,
    )
    assert evaluate_implementation_compatibility(contract, request).compatible is True
