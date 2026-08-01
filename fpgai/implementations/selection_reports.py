from __future__ import annotations

import json
from pathlib import Path

from .selection_result import ImplementationSelectionResult


def write_implementation_selection_report(result: ImplementationSelectionResult, output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "implementation_selection.json"
    md_path = root / "implementation_selection.md"
    payload = result.to_dict()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# FPGAI implementation selection",
        "",
        f"- Operator: `{result.operator_id}`",
        f"- Policy: `{result.policy}`",
        f"- Status: `{'selected' if result.ok else 'failed'}`",
        f"- Selected: `{result.selected.package_id if result.selected else 'none'}`",
        "",
        "## Candidates",
        "",
    ]
    for candidate in result.candidates:
        reasons = ", ".join(candidate.reasons) or "none"
        lines.append(f"- `{candidate.contract.package_id}@{candidate.contract.version}` — **{candidate.status}** — {reasons}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
