from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import subprocess


@dataclass
class HLSRunResult:
    ok: bool
    returncode: int
    command: str
    workdir: str
    stdout_log: str
    stderr_log: str
    csynth_report: Optional[str] = None


def _find_csynth_report(hls_dir: Path) -> Optional[str]:
    candidates = [
        hls_dir / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt",
        hls_dir / "sol1" / "syn" / "report" / "csynth.rpt",
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


def run_vitis_hls(
    *,
    hls_dir: Path,
    vitis_hls_exe: str = "vitis_hls",
    settings64: str | None = None,
    tcl_name: str = "run_hls.tcl",
) -> HLSRunResult:
    hls_dir = Path(hls_dir).resolve()
    tcl_path = hls_dir / tcl_name

    if not tcl_path.exists():
        raise FileNotFoundError(f"TCL script not found: {tcl_path}")

    logs_dir = hls_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = logs_dir / "vitis_hls_stdout.log"
    stderr_log = logs_dir / "vitis_hls_stderr.log"

    if settings64:
        cmd = (
            f'bash -lc "source \\"{settings64}\\" && '
            f'\\"{vitis_hls_exe}\\" -f \\"{tcl_path.name}\\""'
        )
        shell = True
        run_args = cmd
    else:
        shell = False
        run_args = [vitis_hls_exe, "-f", tcl_path.name]

    env = os.environ.copy()

    with open(stdout_log, "w", encoding="utf-8") as so, open(stderr_log, "w", encoding="utf-8") as se:
        proc = subprocess.run(
            run_args,
            cwd=str(hls_dir),
            shell=shell,
            stdout=so,
            stderr=se,
            env=env,
            text=True,
        )

    return HLSRunResult(
        ok=(proc.returncode == 0),
        returncode=proc.returncode,
        command=cmd if settings64 else " ".join(run_args),
        workdir=str(hls_dir),
        stdout_log=str(stdout_log.resolve()),
        stderr_log=str(stderr_log.resolve()),
        csynth_report=_find_csynth_report(hls_dir),
    )