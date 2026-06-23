from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_bitstream(project_dir: str | Path, command: list[str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a bitstream build command in a generated project directory."""
    project = Path(project_dir)
    cmd = command or ["make", "bitstream"]
    return subprocess.run(cmd, cwd=str(project), text=True, capture_output=True, timeout=timeout, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a bitstream for a generated FPGAI project.")
    parser.add_argument("project_dir")
    parser.add_argument("--timeout-sec", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Optional command after --, default: make bitstream")
    ns = parser.parse_args(argv)

    command = ns.command if ns.command else None
    if command and command[0] == "--":
        command = command[1:]

    result = build_bitstream(ns.project_dir, command=command, timeout=ns.timeout_sec)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
