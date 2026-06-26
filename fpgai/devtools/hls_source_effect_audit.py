
"""Audit whether hardware knobs are visible in generated HLS source artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _read_texts(root: Path) -> Dict[str, str]:
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".cpp", ".h", ".hpp", ".tcl"}:
            try:
                files[str(path.relative_to(root))] = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                pass
    return files


def _search(files: Dict[str, str], patterns: List[str]) -> Dict[str, Any]:
    hits = []
    for rel, text in files.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append({"file": rel, "pattern": pat})
    return {"present": bool(hits), "hits": hits[:30]}


def _case_audit(case_root: Path) -> Dict[str, Any]:
    hls_root = case_root / "hls"
    files = _read_texts(hls_root)

    checks = {
        "pipeline_pragmas": [
            r"#pragma\s+HLS\s+PIPELINE",
            r"PIPELINE\s+II\s*=",
            r"II\s*=\s*1",
            r"II\s*=\s*2",
        ],
        "array_partition_pragmas": [
            r"#pragma\s+HLS\s+ARRAY_PARTITION",
            r"ARRAY_PARTITION",
            r"complete",
            r"cyclic",
        ],
        "unroll_pragmas": [
            r"#pragma\s+HLS\s+UNROLL",
            r"UNROLL",
            r"factor\s*=\s*2",
            r"factor\s*=\s*8",
        ],
        "tiling_constants_or_comments": [
            r"tile",
            r"TILE",
            r"tm",
            r"tn",
            r"tk",
            r"tr",
            r"tc",
            r"tile_in",
            r"tile_out",
        ],
        "precision_or_axis_types": [
            r"ap_fixed",
            r"ap_uint",
            r"AXI",
            r"axis",
        ],
        "memory_storage_markers": [
            r"bram",
            r"uram",
            r"BIND_STORAGE",
            r"RESOURCE",
        ],
        "training_markers": [
            r"train",
            r"gradient",
            r"loss",
            r"optimizer",
            r"backward",
            r"weight_delta",
        ],
    }

    return {
        "case": case_root.name,
        "hls_root": str(hls_root),
        "hls_root_exists": hls_root.exists(),
        "source_file_count": len(files),
        "files": sorted(files.keys()),
        "checks": {name: _search(files, pats) for name, pats in checks.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="paper_experiments/full_pipeline_gate/sprint25_e2e_audit",
    )
    args = ap.parse_args()

    root = Path(args.root)
    cases = [
        p for p in root.iterdir()
        if p.is_dir() and p.name not in {"configs", "logs"}
    ]

    payload = {
        "format": "fpgai.hls_source_effect_audit.v1",
        "root": str(root),
        "cases": [_case_audit(case) for case in sorted(cases)],
    }

    out_json = root / "hls_source_effect_audit.json"
    out_md = root / "hls_source_effect_audit.md"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# HLS source effect audit", ""]
    for case in payload["cases"]:
        lines.append(f"## {case['case']}")
        lines.append(f"- source files: `{case['source_file_count']}`")
        lines.append(f"- hls root exists: `{case['hls_root_exists']}`")
        lines.append("- files:")
        for file_name in case["files"]:
            lines.append(f"  - `{file_name}`")
        lines.append("- checks:")
        for check_name, result in case["checks"].items():
            lines.append(f"  - {check_name}: `{result['present']}`")
            for hit in result["hits"][:8]:
                lines.append(f"    - `{hit['file']}` matched `{hit['pattern']}`")
        lines.append("")

    out_md.write_text("\\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
