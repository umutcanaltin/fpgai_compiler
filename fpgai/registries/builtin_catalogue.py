from __future__ import annotations
import hashlib, json
from .registry_catalogue import RegistryCatalogue
from .registry_entry import RegistryEntry, RegistrySource
from .builtin_layers import builtin_layer_entries

def _entry(package_id,asset_type,capabilities=None):
    payload=json.dumps({"id":package_id,"asset_type":asset_type},sort_keys=True).encode()
    return RegistryEntry(package_id=package_id,version="1.0.0",asset_type=asset_type,provider="fpgai",source=RegistrySource.BUILTIN,source_path=None,
        manifest_hash="sha256:"+hashlib.sha256(payload).hexdigest(),capabilities=capabilities or {},compatibility={"fpgai_contract":">=1.0,<2.0"},
        validation_level="reference_tested",license_category="open_source",usage={"platform_scope":"research","production_path":"morfics"},metadata={"name":package_id})

def build_builtin_catalogue() -> RegistryCatalogue:
    c=RegistryCatalogue()
    for e in builtin_layer_entries(): c.operators.register(e)
    for pid,typ,caps in [
      ("fpgai.backend.vitis_hls","backend",{"language":"hls_cpp"}), ("fpgai.backend.vivado","backend",{"implementation":True}),
      ("fpgai.optimizer.sgd","optimizer",{"training":True}), ("fpgai.optimizer.momentum","optimizer",{"training":True}),
      ("fpgai.optimizer.adam","optimizer",{"training":True}), ("fpgai.loss.mse","loss",{"training":True}),
      ("fpgai.loss.cross_entropy","loss",{"training":True})]: c.registry_for(typ).register(_entry(pid,typ,caps))
    return c
