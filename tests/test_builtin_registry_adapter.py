from fpgai.layers.registry import layer_registry
from fpgai.registries.builtin_layers import builtin_layer_entries


def test_builtin_layer_adapter_matches_existing_layer_registry():
    existing=layer_registry()
    entries=builtin_layer_entries()
    assert len(entries) == len(existing)
    by_name={entry.metadata["name"]: entry for entry in entries}
    for name, capability in existing.items():
        assert by_name[name].capabilities["inference"] == capability["inference"]["supported"]
        assert by_name[name].capabilities["training"] == capability["training"]["supported"]
