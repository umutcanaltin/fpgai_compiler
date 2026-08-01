from __future__ import annotations

import json
from pathlib import Path


def write_composition_report(plan, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "mixed_hls_composition.json"
    md_path = reports_dir / "mixed_hls_composition.md"
    payload = plan.to_dict()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Mixed HLS composition", "", f"External nodes: {len(plan.bindings)}", ""]
    for item in plan.bindings:
        lines.append(f"- `{item.node_name}` → `{item.contract.package_id}` (`{item.contract.backend}`)")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
