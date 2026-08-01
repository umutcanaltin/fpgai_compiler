from pathlib import Path
from fpgai.registries import RegistryCatalogue


def test_catalogue_routes_manifest_by_asset_type():
    root=Path("examples/packages/model")
    if not root.exists():
        return
    catalogue=RegistryCatalogue()
    result=catalogue.register_package(root, "project_local")
    assert result.ok
    assert catalogue.models.list_entries()[0].package_id == "community.mnist_mlp_model"
