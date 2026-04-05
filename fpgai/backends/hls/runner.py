from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import shlex
import subprocess
import threading
import time


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


def _stream_reader(pipe, sink, prefix: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            sink.write(line)
            sink.flush()
            print(f"{prefix}{line}", end="")
    finally:
        try:
            pipe.close()
        except Exception:
            pass


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

    env = os.environ.copy()
    timeout_sec_raw = env.get("FPGAI_VITIS_HLS_TIMEOUT_SEC", "").strip()
    timeout_sec = int(timeout_sec_raw) if timeout_sec_raw else None

    if settings64:
        cmd = (
            f'source "{settings64}" && '
            f'"{vitis_hls_exe}" -f "{tcl_path.name}"'
        )
        run_args = ["bash", "-lc", cmd]
        command_str = f'bash -lc {shlex.quote(cmd)}'
    else:
        run_args = [vitis_hls_exe, "-f", tcl_path.name]
        command_str = " ".join(shlex.quote(x) for x in run_args)

    start = time.time()
    last_heartbeat = start

    with open(stdout_log, "w", encoding="utf-8") as so, open(stderr_log, "w", encoding="utf-8") as se:
        proc = subprocess.Popen(
            run_args,
            cwd=str(hls_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        t_out = threading.Thread(target=_stream_reader, args=(proc.stdout, so, "[vitis_hls][stdout] "), daemon=True)
        t_err = threading.Thread(target=_stream_reader, args=(proc.stderr, se, "[vitis_hls][stderr] "), daemon=True)
        t_out.start()
        t_err.start()

        timed_out = False
        try:
            while True:
                rc = proc.poll()
                now = time.time()

                if rc is not None:
                    break

                if now - last_heartbeat >= 30:
                    elapsed = int(now - start)
                    print(f"[vitis_hls] still running... elapsed={elapsed}s log={stdout_log}")
                    last_heartbeat = now

                if timeout_sec is not None and (now - start) > timeout_sec:
                    timed_out = True
                    proc.kill()
                    print(f"[vitis_hls] timeout after {timeout_sec}s, process killed.")
                    break

                time.sleep(1.0)
        finally:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

            t_out.join(timeout=5)
            t_err.join(timeout=5)

    returncode = proc.returncode if proc.returncode is not None else (-9 if timed_out else -1)

    return HLSRunResult(
        ok=(returncode == 0),
        returncode=returncode,
        command=command_str,
        workdir=str(hls_dir),
        stdout_log=str(stdout_log.resolve()),
        stderr_log=str(stderr_log.resolve()),
        csynth_report=_find_csynth_report(hls_dir),
    )