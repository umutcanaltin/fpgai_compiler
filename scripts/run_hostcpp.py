from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

from fpgai.config.loader import load_config
from fpgai.engine.compiler import Compiler


def run(cmd, *, cwd: Path, verbose: bool):
    if verbose:
        print("[CMD]", " ".join(str(x) for x in cmd))
        print("[CWD]", cwd)
    subprocess.check_call(cmd, cwd=str(cwd))


def parse_args():
    p = argparse.ArgumentParser("Build+run FPGAI hostcpp reference")
    p.add_argument("--config", required=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--regen", action="store_true", help="Run compiler first (regenerate artifacts)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.raw.get("project", {}).get("out_dir", "build/fpgai")).resolve()
    top = cfg.raw.get("pipeline", {}).get("outputs", {}).get("top_kernel_name", "deeplearn")

    if args.regen:
        if args.verbose:
            print("[FPGAI] Regenerating artifacts via compiler...")
        Compiler(cfg).compile()

    host_dir = out_dir / "hostcpp"
    if not host_dir.exists():
        print("ERROR: hostcpp dir not found:", host_dir, file=sys.stderr)
        return 2

    input_bin = out_dir / "input.bin"
    if not input_bin.exists():
        print("ERROR: input.bin not found:", input_bin, file=sys.stderr)
        print("Hint: compiler should emit <out_dir>/input.bin", file=sys.stderr)
        return 2

    exe = host_dir / "host_ref.exe"
    include = host_dir / "include"
    src = host_dir / "src"

    compile_cmd = [
        "g++", "-O2", "-std=c++17",
        "-I", str(include),
        str(src / "run.cpp"),
        str(src / f"{top}_host.cpp"),
        str(src / f"{top}_params.cpp"),
        "-o", str(exe),
    ]
    print("[FPGAI] Building host reference...")
    run(compile_cmd, cwd=host_dir, verbose=args.verbose)

    print("[FPGAI] Running:", exe)
    run([str(exe), str(input_bin)], cwd=host_dir, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
