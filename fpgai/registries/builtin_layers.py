from __future__ import annotations
import hashlib, json
from fpgai.layers.registry import layer_registry
from .registry_entry import RegistryEntry, RegistrySource

def builtin_layer_entries(*, pipeline_mode: str="inference") -> tuple[RegistryEntry,...]:
    entries=[]
    for op_type, capability in sorted(layer_registry(pipeline_mode=pipeline_mode).items()):
        package_id="fpgai.operator."+op_type.lower().replace("_","")
        payload=json.dumps(capability,sort_keys=True,separators=(",",":")).encode()
        entries.append(RegistryEntry(
            package_id=package_id,version="1.0.0",asset_type="operator",provider="fpgai",source=RegistrySource.BUILTIN,
            source_path=None,manifest_hash="sha256:"+hashlib.sha256(payload).hexdigest(),
            capabilities={"inference":capability["inference"]["supported"],"training":capability["training"]["supported"],"layer":capability},
            compatibility={"fpgai_contract":">=1.0,<2.0"},validation_level="reference_tested",license_category="open_source",
            usage={"platform_scope":"research","production_path":"morfics"},metadata={"name":op_type,"description":capability["inference"]["detail"]},
        ))
    return tuple(entries)
