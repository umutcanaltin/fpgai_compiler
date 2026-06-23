#!/usr/bin/env python3
"""Extract memory binding artifacts from generated HLS artifacts."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except FileNotFoundError:
        return ""


def _bindings(source: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    pat = re.compile(
        r"#pragma\s+HLS\s+BIND_STORAGE\s+variable=(?P<var>\w+)\s+type=(?P<typ>\w+)\s+impl=(?P<impl>\w+)",
    )
    for m in pat.finditer(source):
        out.append(m.groupdict())
    return out


def _summary(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    metrics = path / "build" / "bench" / "metrics.json"
    if metrics.exists():
        try:
            data["benchmark"] = json.loads(metrics.read_text())
        except Exception:
            data["benchmark"] = {"parse_error": str(metrics)}
    cpp = path / "build" / "hls" / "src" / "deeplearn.cpp"
    src = _read(cpp)
    binds = _bindings(src)
    data["num_bind_storage_pragmas"] = len(binds)
    data["weight_bindings"] = [
        b for b in binds if re.match(r"[WB]\d+", b.get("var", ""))
    ]
    data["all_bindings"] = binds
    data["has_uram_weight_binding"] = any(
        b.get("impl") == "uram" for b in data["weight_bindings"]  # type: ignore[index]
    )
    data["has_bram_weight_binding"] = any(
        b.get("impl") == "bram" for b in data["weight_bindings"]  # type: ignore[index]
    )
    return data


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m fpgai.reporting.memory_binding_artifacts <experiment_dir>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    artifacts = root / "artifacts"
    if not artifacts.exists():
        print(f"[ERROR] artifacts directory not found: {artifacts}", file=sys.stderr)
        return 1

    rows = []
    for design_dir in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        s = _summary(design_dir)
        bench = s.get("benchmark", {})
        rows.append(
            {
                "design": design_dir.name,
                "passed": bench.get("passed") if isinstance(bench, dict) else None,
                "cosine_similarity": bench.get("cosine_similarity") if isinstance(bench, dict) else None,
                "num_weight_bindings": len(s["weight_bindings"]),
                "has_bram_weight_binding": s["has_bram_weight_binding"],
                "has_uram_weight_binding": s["has_uram_weight_binding"],
                "weight_bindings": s["weight_bindings"],
            }
        )

    out_dir = root / "memory_binding_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_binding_artifacts.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "| design | passed | cosine | #W/B bindings | BRAM | URAM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['design']} | {row['passed']} | {row['cosine_similarity']} | "
            f"{row['num_weight_bindings']} | {row['has_bram_weight_binding']} | {row['has_uram_weight_binding']} |"
        )
    (out_dir / "memory_binding_artifacts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[OK] Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
