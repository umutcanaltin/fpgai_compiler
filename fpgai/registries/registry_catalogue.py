from __future__ import annotations
from pathlib import Path
from fpgai.contracts.package_manifest import load_package_manifest
from fpgai.contracts.package_types import AssetType
from .base_registry import BaseRegistry, RegistryResult
from .registry_entry import RegistrySource

class RegistryCatalogue:
    def __init__(self) -> None:
        self._registries={asset.value:BaseRegistry(asset.value) for asset in AssetType}
    def registry_for(self, asset_type: str) -> BaseRegistry: return self._registries[AssetType(asset_type).value]
    def register_package(self, package_root: str|Path, source: str|RegistrySource=RegistrySource.PROJECT_LOCAL) -> RegistryResult:
        try: manifest=load_package_manifest(package_root); registry=self.registry_for(manifest.asset_type)
        except (OSError,ValueError) as exc:
            from .registry_errors import RegistryIssue
            return RegistryResult(errors=(RegistryIssue("PKGREG001",str(package_root),str(exc)),))
        return registry.register_manifest(package_root,source)
    def find_by_package_id(self, package_id: str):
        return tuple(e for r in self._registries.values() for e in r.list_entries() if e.package_id==package_id)
    def find_by_asset_type(self, asset_type: str): return self.registry_for(asset_type).list_entries()
    def inventory(self): return tuple(sorted((e for r in self._registries.values() for e in r.list_entries()),key=lambda e:(e.asset_type,e.package_id,e.version,e.source.value)))
    @property
    def operators(self): return self.registry_for("operator")
    @property
    def implementations(self): return self.registry_for("implementation")
    @property
    def boards(self): return self.registry_for("board")
    @property
    def backends(self): return self.registry_for("backend")
    @property
    def models(self): return self.registry_for("model")
