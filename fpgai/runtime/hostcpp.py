from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_host_executable(executable: str | Path, *args: str, cwd: str | Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a generated host C++ executable and capture text output."""
    cmd = [str(executable), *map(str, args)]
    return subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, text=True, capture_output=True, timeout=timeout, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a generated FPGAI host C++ executable.")
    parser.add_argument("executable")
    parser.add_argument("args", nargs="*")
    parser.add_argument("--cwd")
    parser.add_argument("--timeout-sec", type=int)
    ns = parser.parse_args(argv)

    result = run_host_executable(ns.executable, *ns.args, cwd=ns.cwd, timeout=ns.timeout_sec)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
