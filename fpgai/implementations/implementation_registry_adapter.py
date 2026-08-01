from __future__ import annotations

from fpgai.registries.registry_entry import RegistryEntry, RegistrySource

from .implementation_contract import ImplementationContract


def implementation_contract_to_registry_entry(
    contract: ImplementationContract,
    *,
    source: RegistrySource = RegistrySource.PROJECT_LOCAL,
) -> RegistryEntry:
    return RegistryEntry(
        package_id=contract.package_id,
        version=contract.version,
        asset_type="implementation",
        provider=contract.package_id.split(".", 1)[0],
        source=source,
        source_path=contract.package_root,
        manifest_hash=contract.manifest_hash,
        capabilities={"inference": contract.inference, "training": contract.training.to_dict()},
        compatibility={
            "boards": list(contract.boards),
            "toolchains": {key: list(value) for key, value in contract.toolchains.items()},
            "precisions": list(contract.precisions),
            "backend": contract.backend,
            "language": contract.language,
        },
        validation_level=contract.validation_level,
        license_category=contract.license_category,
        usage={"platform_scope": "research", "production_path": "morfics"},
        metadata={
            **dict(contract.metadata),
            "operator_id": contract.operator_id,
            "top": contract.top,
        },
    )
