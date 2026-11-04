"""Real-network coverage audit must never silently omit nonlinear operators."""
import pytest

from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.experiments.cst_trained_model_preflight import (
    audit_trained_mnist, inspect_model,
)
from fpgai_research.model_ir.contracts import FunctionalModel, Operator, Tensor


def test_matmul_chain_admitted_as_current_restricted_compiler_subdomain():
    graph = make_chain(3, (3, 5, 2))
    result = inspect_model(graph)
    assert result["row_lifetime_status"] == "SUPPORTED_STATIC_MATMUL_CHAIN"
    assert result["supported_compositional_boundaries"] == 1
    assert result["cst_hls_source_emitted"] is False
    assert result["original_vitis_calls"] == 0


def test_relu_in_realistic_mlp_graph_must_not_be_dropped():
    existing = make_chain(3, (3, 5, 2))
    op0, op1 = existing.operators
    graph = FunctionalModel(
        tensors=existing.tensors + (Tensor("activation", (3, 5)),),
        operators=(
            op0,
            Operator("activation_relu", "Relu", ("h0",), ("activation",)),
            Operator(op1.name, "MatMul", ("activation", "w1"), ("y",)),
        ),
        inputs=existing.inputs,
        outputs=existing.outputs,
    )
    result = inspect_model(graph)
    assert result["architecture_resolved"]
    assert result["operators"] == ["MatMul", "Relu", "MatMul"]
    assert result["non_matmul_operators"] == ["Relu"]
    assert result["row_lifetime_status"] == "SUPPORTED_SEQUENTIAL_CST_ROW_CHAIN"
    assert result["first_blocker"] is None
    assert result["supported_compositional_boundaries"] == 2
    assert not result["independent_numerical_reference_executed"]


def test_no_asset_means_no_claimed_trained_model_evidence(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        audit_trained_mnist(trained_directory=tmp_path, kind="synthetic",
                            output=tmp_path/"out")
    with pytest.raises(ValueError, match="Missing or linked trained"):
        audit_trained_mnist(trained_directory=tmp_path/"nonexistent",
                            kind="mlp", output=tmp_path/"out")
    assert not (tmp_path/"out").exists()
