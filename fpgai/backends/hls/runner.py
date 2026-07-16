from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import shlex
import subprocess
import threading
import time

from fpgai.toolchain import build_xilinx_tool_command


@dataclass
class HLSRunResult:
    ok: bool
    returncode: int
    command: str
    workdir: str
    stdout_log: str
    stderr_log: str
    csynth_report: Optional[str] = None
    csim_ran: bool | None = None
    csim_ok: bool | None = None
    csynth_ran: bool | None = None
    csynth_ok: bool | None = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None


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

    run_args, tool_info = build_xilinx_tool_command(
        "vitis_hls",
        ["-f", tcl_path.name],
        executable=vitis_hls_exe,
        settings64=settings64,
        env=env,
    )
    command_str = str(tool_info.get("command") or " ".join(shlex.quote(str(x)) for x in run_args))

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

    raw_returncode = proc.returncode if proc.returncode is not None else (-9 if timed_out else -1)
    stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace") if stdout_log.exists() else ""
    stderr_text = stderr_log.read_text(encoding="utf-8", errors="replace") if stderr_log.exists() else ""
    lowered = f"{stdout_text}\n{stderr_text}".lower()

    csim_ran = "csim start" in lowered or "running: csim_design" in lowered
    csim_failed = (
        "'csim_design' failed" in lowered
        or "error: [sim" in lowered
        or "err: [sim" in lowered
    )
    csim_ok = (not csim_failed) if csim_ran else None

    csynth_ran = "running: csynth_design" in lowered
    csynth_report = _find_csynth_report(hls_dir)
    csynth_failed = "'csynth_design' failed" in lowered or "error: [syn" in lowered
    csynth_ok = (not csynth_failed and csynth_report is not None) if csynth_ran else None

    failure_stage = None
    failure_reason = None
    if timed_out:
        failure_stage = "tool_timeout"
        failure_reason = f"Vitis HLS timed out after {timeout_sec}s."
    elif csim_failed:
        failure_stage = "csim"
        failure_reason = "Vitis HLS C simulation failed; see vitis_hls_stdout.log and the CSim report."
    elif csynth_failed:
        failure_stage = "csynth"
        failure_reason = "Vitis HLS synthesis failed; see vitis_hls_stdout.log."
    elif raw_returncode != 0:
        failure_stage = "tool_process"
        failure_reason = f"Vitis HLS exited with return code {raw_returncode}."

    effective_returncode = raw_returncode
    if effective_returncode == 0 and failure_stage is not None:
        effective_returncode = 1

    return HLSRunResult(
        ok=(effective_returncode == 0 and failure_stage is None),
        returncode=effective_returncode,
        command=command_str,
        workdir=str(hls_dir),
        stdout_log=str(stdout_log.resolve()),
        stderr_log=str(stderr_log.resolve()),
        csynth_report=csynth_report,
        csim_ran=csim_ran,
        csim_ok=csim_ok,
        csynth_ran=csynth_ran,
        csynth_ok=csynth_ok,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
    )
