#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def short_hash(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def count_tokens(path: Path) -> dict[str, int | str]:
    if not path.exists():
        return {"hash": "MISSING", "pipeline": 0, "ii1": 0, "ii2": 0, "ii3": 0, "ii4": 0, "unroll": 0, "partition": 0}
    txt = path.read_text(errors="ignore")
    return {
        "hash": short_hash(path),
        "pipeline": txt.count("#pragma HLS PIPELINE"),
        "ii1": txt.count("II=1") + txt.count("II = 1"),
        "ii2": txt.count("II=2") + txt.count("II = 2"),
        "ii3": txt.count("II=3") + txt.count("II = 3"),
        "ii4": txt.count("II=4") + txt.count("II = 4"),
        "unroll": txt.count("#pragma HLS UNROLL"),
        "partition": txt.count("#pragma HLS ARRAY_PARTITION"),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python -m fpgai.reporting.inspect_pipeline_policy_artifacts <experiment_dir>")
        return 2
    exp = Path(argv[1])
    root = exp / "artifacts"
    if not root.exists():
        raise SystemExit(f"Missing artifacts dir: {root}")

    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        candidates = [
            d / "build/hls/src/deeplearn.cpp",
            d / "build/hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.pp.0.cpp",
        ]
        stats = [count_tokens(p) for p in candidates]
        rows.append((d.name, candidates, stats))

    headers = ["design", "file", "hash", "PIPELINE", "II1", "II2", "II3", "II4", "UNROLL", "PARTITION"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for name, candidates, stats in rows:
        for path, st in zip(candidates, stats):
            rel = path.relative_to(root / name) if path.exists() else path.name
            print(
                f"| {name} | {rel} | {st['hash']} | {st['pipeline']} | {st['ii1']} | {st['ii2']} | {st['ii3']} | {st['ii4']} | {st['unroll']} | {st['partition']} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
