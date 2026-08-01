from __future__ import annotations

import json
from pathlib import Path

from .discovery_types import DiscoveryResult


def render_discovery_markdown(result: DiscoveryResult) -> str:
    summary = result.summary()
    lines = [
        "# FPGAI Package Discovery",
        "",
        f"Status: **{'passed' if result.ok else 'failed'}**",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Packages", "", "| Package | Version | Type | Source | Status |", "|---|---:|---|---|---|"])
    packages = sorted(
        (*result.discovered, *result.quarantined, *result.deduplicated, *result.conflicts),
        key=lambda item: (item.package_id, item.version, item.source.value, str(item.package_root)),
    )
    for item in packages:
        lines.append(f"| {item.package_id or '-'} | {item.version or '-'} | {item.asset_type or '-'} | {item.source.value} | {item.status} |")
    return "\n".join(lines) + "\n"


def write_discovery_report(result: DiscoveryResult, output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "package_discovery.json"
    markdown_path = root / "package_discovery.md"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_discovery_markdown(result), encoding="utf-8")
    return json_path, markdown_path
