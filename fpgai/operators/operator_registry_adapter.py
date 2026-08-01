from __future__ import annotations

import hashlib
import json

from fpgai.registries import RegistryEntry, RegistrySource

from .builtin_operator_contracts import builtin_operator_contracts
from .operator_contract import OperatorContract


def operator_contract_to_registry_entry(
    contract: OperatorContract,
    *,
    source: RegistrySource = RegistrySource.BUILTIN,
) -> RegistryEntry:
    payload = json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return RegistryEntry(
        package_id=contract.operator_id,
        version=f"{contract.version}.0.0",
        asset_type="operator",
        provider=contract.operator_id.split(".", 1)[0],
        source=source,
        source_path=None,
        manifest_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        capabilities=contract.capabilities.to_dict(),
        compatibility={"fpgai_contract": ">=1.0,<2.0", "onnx_bindings": [item.to_dict() for item in contract.onnx_bindings]},
        validation_level="reference_tested",
        license_category="open_source",
        usage={"platform_scope": "research", "production_path": "morfics"},
        metadata={
            "name": contract.canonical_op_type,
            "category": contract.category,
            "operator_contract": contract.to_dict(),
        },
    )


def builtin_operator_entries(*, pipeline_mode: str = "inference") -> tuple[RegistryEntry, ...]:
    return tuple(
        operator_contract_to_registry_entry(contract)
        for contract in builtin_operator_contracts(pipeline_mode=pipeline_mode)
    )
