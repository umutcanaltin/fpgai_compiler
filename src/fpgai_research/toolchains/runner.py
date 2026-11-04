"""Explicit local tool execution; a zero exit code alone never proves synthesis."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import subprocess
import signal
import time

def digest_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def rtl_sha256(case_directory: str | Path) -> dict[str, str]:
    """Hash the exact Vitis-generated Verilog inputs for Vivado attribution."""
    root = Path(case_directory)/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    if not root.is_dir():
        return {}
    candidates = sorted((*root.glob("*.v"), *root.glob("*.sv")))
    if len(candidates)>4096:
        raise ValueError("Excessive RTL source count")
    if any(not path.is_file() or not path.resolve().is_relative_to(root.resolve()) for path in candidates):
        raise ValueError("RTL must remain inside generated Verilog directory")
    return {path.name:digest_file(path) for path in candidates}

def rtl_memory_sha256(case_directory: str | Path) -> dict[str, str]:
    """Hash original local Verilog ROM initialization inputs."""
    root=Path(case_directory)/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    if root.is_symlink():
        raise ValueError("ROM source directory must not be symlinked")
    if not root.is_dir():
        return {}
    candidates=sorted((*root.glob("*.dat"),*root.glob("*.mem")))
    if len(candidates)>4096 or any(p.is_symlink() or not p.is_file()
                                   for p in candidates):
        raise ValueError("Unsafe original HLS ROM payload")
    return {p.name:digest_file(p) for p in candidates}

def _report_candidates(tool: str, cwd: Path) -> dict[str, Path]:
    if tool == "vitis_hls":
        base = cwd/"fpgai_hls_project"/"solution1"
        return {"csynth_xml":base/"syn"/"report"/"fpgai_model_csynth.xml",
                # Archive human-readable scheduling diagnostics when Vitis
                # actually generates this report in the current invocation.
                # Presence/exit-0 alone cannot establish a loop II.
                "csynth_rpt":base/"syn"/"report"/"fpgai_model_csynth.rpt",
                "csim_log":base/"csim"/"report"/"fpgai_model_csim.log",
                "csim_results":base/"csim"/"build"/"csim_results.log"}
    return {name:cwd/"reports"/name for name in
            ("route_status.rpt","timing_summary.rpt","utilization.rpt","drc.rpt")}


def _fingerprints(paths: dict[str, Path]) -> dict[str, tuple[int,str]]:
    return {key:(file.stat().st_mtime_ns,digest_file(file))
            for key,file in paths.items() if file.is_file()}


def run_tool(*, tool: str, argv: list[str], directory: str | Path,
             timeout_seconds: int = 7200, executable: str | None = None,
             allow_verified_readmem_post_absence: bool = False,
             environment: dict[str, str] | None = None) -> dict:
    if tool not in ("vitis_hls", "vivado"):
        raise ValueError("Unknown vendor tool")
    if type(allow_verified_readmem_post_absence) is not bool or (
            tool!="vivado" and allow_verified_readmem_post_absence):
        raise ValueError("Post-run readmem absence policy only applies to Vivado")
    if not (1 <= timeout_seconds <= 172800):
        raise ValueError("timeout_seconds must be 1..172800")
    cwd = Path(directory).resolve(strict=True)
    resolved = shutil.which(executable or tool)
    if not resolved or not Path(resolved).is_file():
        raise FileNotFoundError(f"{tool} executable not found: {executable or tool}")
    expected = (["-f", "run_hls.tcl"] if tool == "vitis_hls"
                else ["-mode", "batch", "-source", "run_vivado.tcl", "-notrace"])
    if argv != expected:
        raise ValueError("Only generated tool workflow scripts may be executed")
    script = cwd / ("run_hls.tcl" if tool == "vitis_hls" else "run_vivado.tcl")
    if not script.is_file():
        raise FileNotFoundError(script)
    logs = cwd / "logs"
    logs.mkdir(exist_ok=True)
    prefix = "vitis" if tool == "vitis_hls" else "vivado"
    report_paths=_report_candidates(tool,cwd)
    old_reports=_fingerprints(report_paths)
    case = cwd if tool=="vitis_hls" else cwd.parent
    rtl_before=rtl_sha256(case)
    rom_before=rtl_memory_sha256(case)
    if tool=="vivado":
        from fpgai_research.toolchains.vivado.rtl_inputs import compare_staged_roms
        staged_rom_before=compare_staged_roms(cwd,rom_before)
        if not staged_rom_before["matches_original_roms"]:
            raise ValueError("Original Vivado staged ROMs differ from source before execution")
    else:
        staged_rom_before=None
    inputs = {"model_cpp":case/"model.cpp", "testbench_cpp":case/"tb.cpp", "tcl":script}
    input_before = {name:digest_file(path) for name,path in inputs.items() if path.is_file()}
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    env = os.environ.copy()
    if environment is not None:
        if (not isinstance(environment, dict)
                or any(not isinstance(k, str) or not isinstance(v, str)
                       or not k or "=" in k or "\0" in k or "\0" in v
                       for k, v in environment.items())):
            raise ValueError("Invalid explicit native tool environment")
        env.update(environment)
    env["LC_ALL"] = "C"
    # Stream directly to the same hash-attested log paths while the vendor
    # process is active. capture_output=True previously buffered hours of
    # output and only wrote these files after the process returned.
    # A separate process session lets timeout/interrupt clean up the entire
    # AMD shell-wrapper/loader/tee process tree, not only its shell parent.
    timed_out = False
    code = None
    def stop_group(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # A shell can exit before its loader and grandchildren. The private
        # process group belongs only to this run; terminate any survivors.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    stdout_file = logs / f"{prefix}.stdout.log"
    stderr_file = logs / f"{prefix}.stderr.log"
    with stdout_file.open("wb") as out, stderr_file.open("wb") as err:
        process = subprocess.Popen([resolved, *expected], cwd=cwd,
                                   stdout=out, stderr=err,
                                   start_new_session=True, env=env)
        deadline = before + timeout_seconds
        print(f"[FPGAI] {prefix}: started (pid={process.pid}, "
              f"timeout={timeout_seconds}s)", flush=True)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 and process.poll() is None:
                    timed_out = True
                    stop_group(process)
                    break
                try:
                    process.wait(timeout=min(60, max(0.001, remaining)))
                    break
                except subprocess.TimeoutExpired:
                    elapsed = round(time.monotonic() - before)
                    print(f"[FPGAI] {prefix}: active for {elapsed}s; "
                          f"stdout={stdout_file.stat().st_size} bytes, "
                          f"stderr={stderr_file.stat().st_size} bytes; "
                          f"live logs in {logs}", flush=True)
            code = None if timed_out else process.returncode
        except BaseException:
            stop_group(process)
            raise
    print(f"[FPGAI] {prefix}: finished after "
          f"{round(time.monotonic()-before)}s; "
          f"{'TIMEOUT' if timed_out else 'exit='+str(code)}", flush=True)
    case = cwd if tool == "vitis_hls" else cwd.parent
    source = case/"model.cpp"
    bench = case/"tb.cpp"
    version = None
    try:
        probe = subprocess.run([resolved,"-version"],cwd=cwd,text=True,
                 capture_output=True,timeout=30,check=False,env=env)
        if probe.returncode == 0:
            version = (probe.stdout or probe.stderr)[:4096]
    except (OSError,subprocess.TimeoutExpired):
        pass
    updated_reports=_fingerprints(report_paths)
    input_after = {name:digest_file(path) for name,path in inputs.items() if path.is_file()}
    inputs_stable = input_before == input_after
    rtl_after=rtl_sha256(case)
    rom_after=rtl_memory_sha256(case)
    rom_inputs_stable=rom_before==rom_after
    staged_rom_after=(compare_staged_roms(cwd,rom_before)
                      if tool=="vivado" else None)
    staged_rom_inputs_stable=(
        staged_rom_before is not None and
        staged_rom_before["matches_original_roms"] and
        staged_rom_after["matches_original_roms"]
        if tool=="vivado" else None)
    readmem_absence_witness=None
    if (tool=="vivado" and allow_verified_readmem_post_absence and
            not staged_rom_after["matches_original_roms"]):
        from fpgai_research.toolchains.vivado.rtl_inputs import inventory_vitis_rtl
        from fpgai_research.toolchains.vivado.rom_read_attestation import (
            verify_post_run_readmem_absence)
        # The exact initial hash comes from a pre-tool snapshot; Vivado
        # stdout is hashed from the original process, never rewritten.
        readmem_absence_witness=verify_post_run_readmem_absence(
            expected=rom_before,
            literal_refs=inventory_vitis_rtl(case)["literal_readmem_references"],
            before=staged_rom_before,after=staged_rom_after,
            stdout_file=stdout_file)
    qualified_rom_inputs=(rom_inputs_stable and (
        staged_rom_inputs_stable is True or
        (readmem_absence_witness is not None and
         readmem_absence_witness["qualified_postrun_absence"] is True)
    )) if tool=="vivado" else None
    changed_rtl=sorted(name for name,digest in rtl_after.items() if rtl_before.get(name)!=digest)
    fresh_reports={name:entry[1] for name,entry in updated_reports.items()
                   if name not in old_reports or old_reports[name] != entry}
    receipt = {"schema":"fpgai-research/vendor-invocation/v1",
               "tool_version_output":version,
               "source_sha256":digest_file(source) if source.is_file() else None,
               "testbench_sha256":digest_file(bench) if bench.is_file() else None,
               "tool":tool,"executable":str(Path(resolved).resolve()),
               "argv":[resolved,*expected],"cwd":str(cwd),
               "script_sha256":digest_file(script),"started_utc":started,
               "duration_seconds":round(time.monotonic()-before,3),
               "returncode":code,"timed_out":timed_out,
               "input_hashes_before_execution":input_before,
               "input_hashes_after_execution":input_after,
               "inputs_stable_during_execution":inputs_stable,
               "completed_successfully":(code==0 and not timed_out and inputs_stable
                                          and (tool!="vivado" or
                                               qualified_rom_inputs)),
               "stdout_sha256":digest_file(logs/f"{prefix}.stdout.log"),
               "stderr_sha256":digest_file(logs/f"{prefix}.stderr.log"),
               "fresh_report_sha256":fresh_reports,
               "rtl_input_sha256":rtl_before if tool=="vivado" else None,
               "rtl_output_sha256":rtl_after if tool=="vitis_hls" else None,
               "rom_input_sha256":rom_before if tool=="vivado" else None,
               "rom_output_sha256":rom_after if tool=="vitis_hls" else None,
               "rom_inputs_stable_during_execution":(
                    rom_inputs_stable if tool=="vivado" else None),
               "staged_rom_input_integrity_before":staged_rom_before,
               "staged_rom_input_integrity_after":staged_rom_after,
               "staged_rom_inputs_stable_during_execution":staged_rom_inputs_stable,
               "allow_verified_readmem_post_absence":allow_verified_readmem_post_absence,
               "verified_readmem_post_absence_witness":readmem_absence_witness,
               "rtl_new_or_changed_files":changed_rtl if tool=="vitis_hls" else None,
               "evidence_level":None,
               "note":"Tool process completion does not establish the HLS or physical evidence level."}
    (cwd/f"{prefix}_invocation.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt
