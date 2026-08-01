import json
from fpgai.registries import build_builtin_catalogue
from fpgai.registries.registry_inventory import inventory_payload, write_registry_inventory


def test_registry_inventory_is_deterministic_and_machine_readable(tmp_path):
    catalogue=build_builtin_catalogue()
    first=inventory_payload(catalogue)
    second=inventory_payload(catalogue)
    assert first == second
    assert first["schema"] == "fpgai.registry-inventory/v1"
    json_path, markdown_path=write_registry_inventory(catalogue,tmp_path)
    assert json.loads(json_path.read_text()) == first
    assert markdown_path.exists()
