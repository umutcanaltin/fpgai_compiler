#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from fpgai.backends.vivado.tcl_generator import emit_vivado_tcl


DEFAULT_BOARD_PARTS = {
    "kv260": "xilinx.com:kv260_som:part0:1.4",
}


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config root in {path}")
    return data


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fpgai.yml")
    parser.add_argument("--vivado-exe", default="vivado")
    parser.add_argument("--project-name", default="fpgai_vivado_proj")
    parser.add_argument("--board-part", default=None)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)

    out_dir = Path(cfg["project"]["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    board_name = (
        cfg.get("targets", {})
        .get("platform", {})
        .get("board", "kv260")
    )
    part = cfg["targets"]["platform"]["part"]
    board_part = args.board_part or DEFAULT_BOARD_PARTS.get(board_name, DEFAULT_BOARD_PARTS["kv260"])

    hls_impl_dir = out_dir / "hls" / "fpgai_hls_proj" / "sol1" / "impl"
    ip_repo = hls_impl_dir / "ip"

    if not ip_repo.exists():
        print(f"[ERROR] IP repository not found: {ip_repo}", file=sys.stderr)
        print("[HINT] Run HLS export first so Vivado can consume the packaged IP.", file=sys.stderr)
        return 1

    tcl_content = emit_vivado_tcl(
        args.project_name,
        str(out_dir),
        part,
        board_part,
        str(ip_repo),
    )

    tcl_path = out_dir / "build_bitstream.tcl"
    tcl_path.write_text(tcl_content, encoding="utf-8")

    vivado_meta = {
        "project_name": args.project_name,
        "out_dir": str(out_dir),
        "part": part,
        "board_name": board_name,
        "board_part": board_part,
        "ip_repo": str(ip_repo),
        "tcl_path": str(tcl_path),
    }
    (out_dir / "vivado_build_meta.json").write_text(json.dumps(vivado_meta, indent=2), encoding="utf-8")

    print("\n[FPGAI] Vivado TCL written:")
    print(f"  {tcl_path}")
    print("[FPGAI] Build metadata written:")
    print(f"  {out_dir / 'vivado_build_meta.json'}")

    if args.skip_run:
        print("[FPGAI] --skip-run set, not launching Vivado.")
        return 0

    if shutil.which(args.vivado_exe) is None:
        print(f"[ERROR] Vivado executable not found in PATH: {args.vivado_exe}", file=sys.stderr)
        return 1

    print("\n[FPGAI] Starting Vivado bitstream generation...")
    cmd = [
        args.vivado_exe,
        "-mode",
        "batch",
        "-source",
        str(tcl_path),
    ]
    proc = subprocess.run(cmd, cwd=str(out_dir), check=False)

    if proc.returncode != 0:
        print("[ERROR] Vivado build failed.", file=sys.stderr)
        return proc.returncode

    print("[OK] Vivado build completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())