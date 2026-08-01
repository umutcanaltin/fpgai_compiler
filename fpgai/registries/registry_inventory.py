from __future__ import annotations
import json
from pathlib import Path
from .registry_catalogue import RegistryCatalogue

def inventory_payload(catalogue: RegistryCatalogue) -> dict:
    return {"schema":"fpgai.registry-inventory/v1","entries":[e.to_dict() for e in catalogue.inventory()]}
def write_registry_inventory(catalogue: RegistryCatalogue,out_dir: str|Path) -> tuple[Path,Path]:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); payload=inventory_payload(catalogue)
    jp=out/'registry_inventory.json'; mp=out/'registry_inventory.md'
    jp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding='utf-8')
    lines=["# FPGAI Registry Inventory","",f"Entries: {len(payload['entries'])}","","| Package | Version | Asset type | Source | Validation |","|---|---:|---|---|---|"]
    for e in payload['entries']: lines.append(f"| {e['package_id']} | {e['version']} | {e['asset_type']} | {e['source']} | {e['validation_level']} |")
    mp.write_text("\n".join(lines)+"\n",encoding='utf-8'); return jp,mp
