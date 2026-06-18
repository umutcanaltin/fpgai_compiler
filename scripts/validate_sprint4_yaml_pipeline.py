from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path) -> None:
    print("$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser("Validate Sprint 4 through YAML/main pipeline")
    parser.add_argument("--config", default="fpgai.yml")
    parser.add_argument("--action", choices=["compile", "auto", "benchmark"], default="compile")
    parser.add_argument("--skip-vitis", action="store_true", help="Set toolchain.vitis_hls.enabled=false by using compile only is not supported here; this flag only skips external post-check wording.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = root / args.config
    if not cfg.exists():
        raise SystemExit(f"Config not found: {cfg}")

    out_dir = root / "build" / "fpgai_example_dense"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    if args.action == "auto":
        run([sys.executable, "main.py", "--config", str(cfg)], cwd=root)
    else:
        run([sys.executable, "main.py", args.action, "--config", str(cfg)], cwd=root)

    hls_dir = out_dir / "hls"
    top = hls_dir / "src" / "deeplearn.cpp"
    params = hls_dir / "include" / "fpgai_params.h"
    meta = hls_dir / "codegen_meta.json"
    reports = hls_dir / "reports"

    required = [top, params, meta, reports / "tiling_analysis.json", reports / "tiling_resource_estimate.json", reports / "tiling_performance_estimate.json"]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing Sprint 4 artifacts: " + ", ".join(missing))

    top_text = top.read_text(encoding="utf-8")
    if "dense_out_in_tiled" not in top_text and "conv2d_tiled" not in top_text:
        raise SystemExit("No tiled kernel found in generated top C++")

    params_text = params.read_text(encoding="utf-8")
    bad = re.findall(r"(?<![A-Za-z0-9_.])[-+]?[0-9]+f\b", params_text)
    if bad:
        raise SystemExit("Invalid C++ float literals found: " + ", ".join(sorted(set(bad))))

    meta_json = read_json(meta)
    print("[OK] YAML pipeline generated Sprint 4 HLS artifacts")
    print(f"[OK] top_name={meta_json.get('top_name')} pipeline_mode={meta_json.get('pipeline_mode')}")
    print(f"[OK] hls_dir={hls_dir.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
