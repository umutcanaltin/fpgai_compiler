from fpgai.operators import builtin_operator_contracts, builtin_operator_entries, operator_contract_to_registry_entry
from fpgai.registries import BaseRegistry


def test_operator_contract_adapter_creates_research_registry_entry():
    contract = builtin_operator_contracts()[0]
    entry = operator_contract_to_registry_entry(contract)
    assert entry.asset_type == "operator"
    assert entry.usage["platform_scope"] == "research"
    assert entry.usage["production_path"] == "morfics"
    assert entry.metadata["operator_contract"]["operator_id"] == contract.operator_id


def test_builtin_operator_entries_register_without_conflicts():
    registry = BaseRegistry("operator")
    entries = builtin_operator_entries()
    for entry in entries:
        assert registry.register(entry).ok
    assert len(registry.list_entries()) == len(entries)


def test_operator_entry_hash_is_deterministic():
    first = builtin_operator_entries()
    second = builtin_operator_entries()
    assert [(entry.package_id, entry.manifest_hash) for entry in first] == [
        (entry.package_id, entry.manifest_hash) for entry in second
    ]
