from fpgai.layers.registry import layer_registry
from fpgai.operators import builtin_operator_contracts, get_builtin_operator_contract


def test_builtin_contracts_cover_existing_layer_registry():
    existing = layer_registry()
    contracts = builtin_operator_contracts()
    by_id = {contract.operator_id: contract for contract in contracts}
    assert len(contracts) == len(existing)
    for op_type, capability in existing.items():
        package_id = "fpgai.operator." + op_type.lower().replace("_", "")
        contract = by_id[package_id]
        assert contract.capabilities.inference == capability["inference"]["supported"]
        assert contract.capabilities.training_forward == capability["training"]["supported"]
        assert contract.implementation_requirements["has_weights"] == capability["has_weights"]


def test_builtin_dense_contract_declares_onnx_and_canonicalization_metadata():
    contract = get_builtin_operator_contract("Dense")
    assert {binding.op_type for binding in contract.onnx_bindings} == {"Gemm", "MatMul"}
    assert contract.capabilities.canonicalization is True
    assert contract.entrypoints.canonicalization


def test_alias_contract_preserves_canonical_target():
    contract = get_builtin_operator_contract("Conv2D")
    assert contract.canonical_op_type == "Conv"
