from fpgai.implementations.builtin_implementation_contracts import builtin_implementation_contracts
from fpgai.implementations.implementation_registry_adapter import implementation_contract_to_registry_entry


def test_builtin_contracts_are_descriptive_and_registry_compatible() -> None:
    contracts = builtin_implementation_contracts()
    assert {item.operator_id for item in contracts} == {"fpgai.operator.dense", "fpgai.operator.conv2d"}
    entries = [implementation_contract_to_registry_entry(item) for item in contracts]
    assert all(entry.asset_type == "implementation" for entry in entries)
    assert all(entry.metadata["operator_id"].startswith("fpgai.operator.") for entry in entries)
