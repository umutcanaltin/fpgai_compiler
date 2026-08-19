from fpgai.operators.builtin_operator_contracts import get_builtin_operator_contract


def test_attention_operator_contracts_have_onnx_bindings_and_current_hls_capabilities() -> None:
    matmul = get_builtin_operator_contract("MatMul")
    layernorm = get_builtin_operator_contract("LayerNormalization")
    transpose = get_builtin_operator_contract("Transpose")
    mul = get_builtin_operator_contract("Mul")
    rmsnorm = get_builtin_operator_contract("RMSNorm")

    assert [item.op_type for item in matmul.onnx_bindings] == ["MatMul"]
    assert [item.op_type for item in layernorm.onnx_bindings] == ["LayerNormalization"]
    assert layernorm.onnx_bindings[0].opset_min == 17
    assert [item.op_type for item in rmsnorm.onnx_bindings] == ["RMSNormalization"]
    assert rmsnorm.onnx_bindings[0].opset_min == 23
    assert [item.op_type for item in transpose.onnx_bindings] == ["Transpose"]
    assert [item.op_type for item in mul.onnx_bindings] == ["Mul"]

    # These capabilities now reflect real DAG-HLS implementations. Limited means the
    # backend still enforces the documented static-shape/axis/scalar restrictions.
    assert matmul.capabilities.inference is True
    assert layernorm.capabilities.inference is True
    assert rmsnorm.capabilities.inference is True
    assert transpose.capabilities.inference is True
    assert mul.capabilities.inference is True

    # Training-forward and backward-input coverage now exist for MatMul; parameter-gradient
    # coverage remains explicit rather than being inferred from inference support.
    assert matmul.capabilities.training_forward is True
    assert matmul.capabilities.backward_input is True
    assert matmul.capabilities.parameter_gradients is False
    assert layernorm.capabilities.training_forward is True
    assert layernorm.capabilities.backward_input is True
    assert layernorm.capabilities.parameter_gradients is True
    assert rmsnorm.capabilities.training_forward is True
    assert rmsnorm.capabilities.backward_input is True
    assert rmsnorm.capabilities.parameter_gradients is True
