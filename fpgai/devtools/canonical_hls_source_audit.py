"""Audit YAML hardware effects in canonical FPGAI-generated HLS source only.

This intentionally ignores Vitis-generated/internal directories such as:
- fpgai_hls_proj/
- .autopilot/
- impl/
- syn/
- csim/

It only scans:
- hls/src
- hls/include
- hls/run_hls.tcl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


CANONICAL_REL_ROOTS = [
    Path("hls/src"),
    Path("hls/include"),
]

CANONICAL_REL_FILES = [
    Path("hls/run_hls.tcl"),
]


def _iter_canonical_files(case_root: Path) -> Iterable[Path]:
    for rel_root in CANONICAL_REL_ROOTS:
        root = case_root / rel_root
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in {".cpp", ".h", ".hpp", ".tcl"}:
                    yield path

    for rel_file in CANONICAL_REL_FILES:
        path = case_root / rel_file
        if path.exists() and path.is_file():
            yield path


def _read_texts(case_root: Path) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for path in _iter_canonical_files(case_root):
        try:
            files[str(path.relative_to(case_root))] = path.read_text(
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
    files = _read_texts(case_root)

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
        "tiling_markers": [
            r"tile",
            r"TILE",
            r"tile_in",
            r"tile_out",
            r"conv2d_tiled",
            r"dense_out_in_tiled",
        ],
        "precision_or_axis_types": [
            r"ap_fixed",
            r"ap_uint",
            r"hls::stream",
            r"axis",
            r"ap_axiu",
        ],
        "memory_storage_markers": [
            r"bram",
            r"uram",
            r"BIND_STORAGE",
            r"RESOURCE",
        ],
        "training_markers": [
            r"gradient",
            r"loss",
            r"optimizer",
            r"backward",
            r"weight_delta",
            r"sgd_update",
        ],
    }

    return {
        "case": case_root.name,
        "canonical_file_count": len(files),
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
        "format": "fpgai.canonical_hls_source_audit.v1",
        "root": str(root),
        "scope": {
            "included": ["hls/src", "hls/include", "hls/run_hls.tcl"],
            "excluded": ["hls/fpgai_hls_proj", ".autopilot", "impl", "syn", "csim"],
        },
        "cases": [_case_audit(case) for case in sorted(cases)],
    }

    out_json = root / "canonical_hls_source_audit.json"
    out_md = root / "canonical_hls_source_audit.md"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Canonical HLS source effect audit", ""]
    lines.append("Scope: only FPGAI-generated canonical source files.")
    lines.append("")
    for case in payload["cases"]:
        lines.append(f"## {case['case']}")
        lines.append(f"- canonical files: `{case['canonical_file_count']}`")
        lines.append("- files:")
        for file_name in case["files"]:
            lines.append(f"  - `{file_name}`")
        lines.append("- checks:")
        for check_name, result in case["checks"].items():
            lines.append(f"  - {check_name}: `{result['present']}`")
            for hit in result["hits"][:8]:
                lines.append(f"    - `{hit['file']}` matched `{hit['pattern']}`")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")

    for case in payload["cases"]:
        print("CASE", case["case"])
        print("canonical_file_count", case["canonical_file_count"])
        for name, result in case["checks"].items():
            print(" ", name, result["present"], "hits", len(result["hits"]))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
