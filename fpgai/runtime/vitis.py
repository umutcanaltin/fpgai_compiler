from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_vitis_csim(project_dir: str | Path, script: str = "run_hls.tcl", timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run Vitis HLS C simulation from a project directory when Vitis is available."""
    project = Path(project_dir)
    cmd = ["vitis_hls", "-f", script]
    return subprocess.run(cmd, cwd=str(project), text=True, capture_output=True, timeout=timeout, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Vitis HLS C simulation for an FPGAI project.")
    parser.add_argument("project_dir")
    parser.add_argument("--script", default="run_hls.tcl")
    parser.add_argument("--timeout-sec", type=int)
    ns = parser.parse_args(argv)

    result = run_vitis_csim(ns.project_dir, script=ns.script, timeout=ns.timeout_sec)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
