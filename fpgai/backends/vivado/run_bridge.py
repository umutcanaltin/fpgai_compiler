#!/usr/bin/env python3
"""Generate Vivado bridge artifacts and optionally run Vitis HLS/Vivado automatically."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fpgai.backends.vivado import generate_vivado_bridge_for_experiment
from fpgai.runtime.package import emit_runtime_package


def _load_json(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _iter_artifacts(exp: Path) -> Iterable[Path]:
    arts = exp / "artifacts"
    if arts.exists():
        for p in sorted(arts.iterdir()):
            if p.is_dir():
                yield p
    else:
        yield exp


def _limited(items: Iterable[Path], max_designs: Optional[int]) -> List[Path]:
    vals = list(items)
    if max_designs is not None and max_designs >= 0:
        return vals[:max_designs]
    return vals


def _bridge_dir(artifact: Path) -> Path:
    build = artifact / "build" if (artifact / "build").exists() else artifact
    return build / "vivado_bridge"


def _hls_impl_ip_dir(bridge: Path) -> Path:
    return bridge.parent / "hls" / "fpgai_hls_proj" / "sol1" / "impl" / "ip"


def _clear_exported_ip(bridge: Path) -> None:
    """Remove stale exported IP locations before a forced re-export.

    This prevents stale component.xml files from making a failed HLS export look
    successful and from allowing Vivado to continue with incompatible old IP.
    """
    for path in [bridge / "hls_ip", _hls_impl_ip_dir(bridge)]:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

def _mirror_hls_ip_to_bridge(bridge: Path) -> None:
    """Copy canonical Vitis HLS impl/ip contents into vivado_bridge/hls_ip.

    Vitis HLS may place component.xml under build/hls/.../impl/ip while the
    Vivado bridge expects a stable IP repo at build/vivado_bridge/hls_ip. This
    mirror makes both manual inspection and Vivado ip_repo_paths deterministic.
    """
    src = _hls_impl_ip_dir(bridge)
    dst = bridge / "hls_ip"
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _component_xml_paths(bridge: Path) -> List[str]:
    paths: List[str] = []
    for root in [bridge / "hls_ip", _hls_impl_ip_dir(bridge)]:
        if root.exists():
            paths.extend([p.as_posix() for p in root.glob("**/component.xml")])
    return sorted(set(paths))


def _component_xml_exists(bridge: Path) -> bool:
    return bool(_component_xml_paths(bridge))


def _exported_ip_artifacts(bridge: Path) -> List[str]:
    paths = _component_xml_paths(bridge)
    for root in [bridge / "hls_ip", _hls_impl_ip_dir(bridge)]:
        if root.exists():
            paths.extend([p.as_posix() for p in root.glob("**/*.zip")])
    return sorted(set(paths))


def _load_compile_manifest(project_dir: Path) -> dict:
    manifest_path = project_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fit_policy_gate_from_manifest(project_dir: Path) -> dict:
    manifest = _load_compile_manifest(project_dir)
    gate = manifest.get("fit_policy_gate", {})
    return gate if isinstance(gate, dict) else {}


def _manifest_path(project_dir: Path) -> Path:
    return project_dir / "manifest.json"


def _compile_output_dir_from_project_dir(project_dir: Path) -> Path:
    """Return the original compiler output directory.

    In paper bridge wrappers, build/manifest.json may be a symlink to the real
    build/paper/<design>/manifest.json. Resolving it lets the bridge update the
    original build reports/package instead of only the temporary wrapper.
    """
    manifest = _manifest_path(project_dir)
    try:
        if manifest.exists():
            return manifest.resolve().parent
    except OSError:
        pass
    return project_dir.resolve()


def _deep_find_dict(obj: Any, key: str) -> Dict[str, Any]:
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
        for value in obj.values():
            found = _deep_find_dict(value, key)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _deep_find_dict(value, key)
            if found:
                return found
    return {}


def _tool_cfg_from_manifest(project_dir: Path, tool: str) -> Dict[str, Any]:
    manifest = _load_compile_manifest(project_dir)
    toolchain = manifest.get("toolchain", {}) if isinstance(manifest.get("toolchain", {}), dict) else {}
    if not toolchain:
        toolchain = _deep_find_dict(manifest, "toolchain")
    cfg = toolchain.get(tool, {}) if isinstance(toolchain.get(tool, {}), dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def _resolved_tool_command(project_dir: Path, tool: str, args: List[str]) -> tuple[List[str], Dict[str, Any]]:
    """Resolve Vivado/Vitis commands from manifest/YAML/env/PATH.

    Priority:
    1. manifest/resolved config toolchain.<tool>.settings64 + executable/exe/path
    2. environment variables
    3. PATH fallback
    """
    cfg = _tool_cfg_from_manifest(project_dir, tool)
    env_prefix = "VIVADO" if tool == "vivado" else "VITIS_HLS"
    default_exe = "vivado" if tool == "vivado" else "vitis_hls"

    executable = (
        cfg.get("executable")
        or cfg.get("exe")
        or cfg.get("path")
        or os.environ.get(f"FPGAI_{env_prefix}_EXECUTABLE")
        or os.environ.get(f"{env_prefix}_EXECUTABLE")
        or os.environ.get(f"FPGAI_{env_prefix}")
        or os.environ.get(env_prefix)
        or default_exe
    )
    settings64 = (
        cfg.get("settings64")
        or cfg.get("settings")
        or os.environ.get(f"FPGAI_{env_prefix}_SETTINGS64")
        or os.environ.get(f"{env_prefix}_SETTINGS64")
    )

    executable = str(executable)
    settings64 = str(settings64) if settings64 else ""

    if settings64:
        shell_cmd = "source {settings} && exec {exe} {args}".format(
            settings=shlex.quote(settings64),
            exe=shlex.quote(executable),
            args=" ".join(shlex.quote(str(a)) for a in args),
        )
        cmd = ["bash", "-lc", shell_cmd]
        resolved = shutil.which("bash")
    else:
        cmd = [executable] + list(args)
        resolved = shutil.which(executable)

    return cmd, {
        "tool": tool,
        "executable": executable,
        "settings64": settings64 or None,
        "uses_settings64": bool(settings64),
        "launcher": cmd[0],
        "resolved_launcher": resolved,
        "path_available_without_settings": shutil.which(executable) is not None,
        "source": "manifest_or_env" if (cfg or settings64 or executable != default_exe) else "path_default",
    }


def _same_path(src: Path, dst: Path) -> bool:
    """Return True when src and dst refer to the same filesystem path."""
    try:
        return src.resolve() == dst.resolve()
    except OSError:
        return src.absolute() == dst.absolute()


def _copy_dir_contents(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        return
    if _same_path(src, dst):
        # Direct build-directory bridge mode already writes into the compiler
        # output's vivado_bridge directory. Treat syncing a directory onto
        # itself as a no-op, not as a tool failure.
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if _same_path(child, target):
            continue
        if child.is_dir():
            if target.exists() and not target.is_symlink():
                shutil.rmtree(target)
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _first_existing_file(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.exists() and path.is_file():
                return path
    return None


def _write_md_report(path: Path, payload: Dict[str, Any], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"- status: `{payload.get('status')}`",
        f"- stage: `{payload.get('stage')}`",
        f"- requested: `{payload.get('requested')}`",
        f"- claimed_success: `{payload.get('claimed_success')}`",
    ]
    reason = payload.get("reason")
    if reason:
        lines.append(f"- reason: {reason}")
    artifact = payload.get("artifact")
    if artifact:
        lines.extend(["", "## Artifact", f"- `{artifact}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sync_bridge_result_to_compile_output(artifact: Path, bridge: Path, row: Dict[str, Any]) -> None:
    """Synchronize successful external Vivado bridge evidence into the real build.

    This replaces stale compile-time placeholder reports such as tool_missing only
    after the bridge has real Vivado reports/bitstream/XSA artifacts.
    """
    wrapper_build_dir = bridge.parent
    compile_dir = _compile_output_dir_from_project_dir(wrapper_build_dir)
    if not compile_dir.exists():
        return

    # Mirror only deployment/report-facing bridge outputs, not necessarily the full Vivado project.
    target_bridge = compile_dir / "vivado_bridge"
    for subdir in ["bitstream", "reports", "logs", "scripts", "hls_ip"]:
        _copy_dir_contents(bridge / subdir, target_bridge / subdir)
    for filename in ["vivado_bridge_manifest.json", "README_VIVADO.md"]:
        src = bridge / filename
        if src.exists():
            target_bridge.mkdir(parents=True, exist_ok=True)
            target = target_bridge / filename
            if not _same_path(src, target):
                shutil.copy2(src, target)

    reports_dir = compile_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    bit = _first_existing_file(target_bridge, ["bitstream/*.bit", "project/**/*.bit"])
    hwh = _first_existing_file(target_bridge, ["bitstream/*.hwh", "project/**/*.hwh"])
    xsa = _first_existing_file(target_bridge, ["bitstream/*.xsa", "project/**/*.xsa"])
    util = _first_existing_file(target_bridge, ["reports/utilization_impl.rpt", "project/**/*utilization*impl*.rpt"])
    timing = _first_existing_file(target_bridge, ["reports/timing_impl.rpt", "project/**/*timing*impl*.rpt"])
    power = _first_existing_file(target_bridge, ["reports/power_impl.rpt", "project/**/*power*impl*.rpt"])

    vivado_ok = bool(row.get("vivado_ok"))
    bit_ok = bool(bit is not None and xsa is not None)

    tool_info = row.get("vivado_tool") or {}
    vivado_run = row.get("vivado_run") or {}

    impl_payload = {
        "format": "fpgai.vivado_implementation_report.v1",
        "stage": "vivado_implementation",
        "requested": bool(row.get("vivado_impl_requested")),
        "status": "passed" if vivado_ok and (util or timing or power or bit or xsa) else "failed",
        "reason": "Vivado bridge implementation completed with real artifacts." if vivado_ok else (row.get("vivado_error") or "Vivado bridge implementation failed."),
        "claimed_success": bool(vivado_ok and (util or timing or power or bit or xsa)),
        "vivado_ok": vivado_ok,
        "vivado_ran": bool(row.get("vivado_ran")),
        "returncode": row.get("vivado_returncode"),
        "vivado_tool": tool_info,
        "vivado_run": vivado_run,
        "vivado_bridge_dir": target_bridge.as_posix(),
        "artifact": str(util or timing or power or bit or xsa) if (util or timing or power or bit or xsa) else None,
        "utilization_report": str(util) if util else None,
        "timing_report": str(timing) if timing else None,
        "power_report": str(power) if power else None,
        "stdout_log": row.get("vivado_stdout_log"),
        "stderr_log": row.get("vivado_stderr_log"),
        "truth_boundary": "Vivado implementation success is claimed only from a completed bridge run and real Vivado reports or implementation artifacts.",
    }

    bit_payload = {
        "format": "fpgai.bitstream_report.v1",
        "stage": "bitstream",
        "requested": bool(row.get("vivado_impl_requested")),
        "status": "passed" if vivado_ok and bit_ok else "failed",
        "reason": "Bitstream/XSA artifacts were generated by the Vivado bridge." if vivado_ok and bit_ok else "Bitstream/XSA artifacts are missing after Vivado bridge run.",
        "claimed_success": bool(vivado_ok and bit_ok),
        "requires_vivado_implementation_passed": True,
        "vivado_implementation_status": impl_payload["status"],
        "vivado_tool": tool_info,
        "artifact": str(bit or xsa) if (bit or xsa) else None,
        "bitstream_exists": bit is not None,
        "hwh_exists": hwh is not None,
        "xsa_exists": xsa is not None,
        "bitstream_path": str(bit) if bit else None,
        "hwh_path": str(hwh) if hwh else None,
        "xsa_path": str(xsa) if xsa else None,
        "vivado_bridge_dir": target_bridge.as_posix(),
        "truth_boundary": "Bitstream success is claimed only when real .bit and XSA artifacts exist after a completed Vivado bridge implementation run.",
    }

    _write_json(reports_dir / "vivado_implementation_report.json", impl_payload)
    _write_json(reports_dir / "bitstream_report.json", bit_payload)
    _write_md_report(reports_dir / "vivado_implementation_report.md", impl_payload, "Vivado implementation report")
    _write_md_report(reports_dir / "bitstream_report.md", bit_payload, "Bitstream report")

    # Refresh runtime package so package_manifest sees vivado_bridge/bitstream.
    try:
        emit_runtime_package(compile_dir)
    except Exception as exc:
        _write_json(reports_dir / "runtime_package_refresh_error.json", {"error": str(exc)})


def _vivado_gate_block_reason(project_dir: Path, *, run_vivado_impl: bool) -> str:
    if not run_vivado_impl:
        return ""

    gate = _fit_policy_gate_from_manifest(project_dir)
    if not bool(gate.get("blocked")):
        return ""

    blocked_stages = gate.get("blocked_stages") or []
    if "vivado_impl" not in blocked_stages and "bitstream" not in blocked_stages:
        return ""

    policy = gate.get("policy", "unknown")
    status = gate.get("board_fit_status", "unknown")
    limiting = gate.get("board_fit_limiting_dimension", "unknown")
    reason = gate.get("reason", "fit_policy_gate blocked Vivado implementation.")
    return (
        f"fit_policy_gate blocked Vivado implementation: "
        f"policy={policy}, board_fit_status={status}, limiting={limiting}. {reason}"
    )


def _bitstream_exists(bridge: Path) -> bool:
    return bool(list((bridge / "bitstream").glob("*.bit"))) if (bridge / "bitstream").exists() else False


def _xsa_exists(bridge: Path) -> bool:
    return bool(list((bridge / "bitstream").glob("*.xsa"))) if (bridge / "bitstream").exists() else False


def _reports_present(bridge: Path) -> bool:
    reports = bridge / "reports"
    return any(reports.glob("*.rpt")) if reports.exists() else False


def _run_cmd(cmd: List[str], cwd: Path, log_prefix: str, timeout_sec: int | None, env: Dict[str, str] | None = None) -> Dict[str, Any]:
    logs = cwd / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{log_prefix}_stdout.log"
    stderr_path = logs / f"{log_prefix}_stderr.log"
    meta_path = logs / f"{log_prefix}_run.json"
    started = datetime.now().isoformat(timespec="seconds")

    if shutil.which(cmd[0]) is None:
        result = {
            "requested": True,
            "ran": False,
            "ok": False,
            "returncode": None,
            "cmd": cmd,
            "cwd": cwd.as_posix(),
            "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "error": f"Executable not found: {cmd[0]}",
            "stdout_log": stdout_path.as_posix(),
            "stderr_log": stderr_path.as_posix(),
        }
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(result["error"] + "\n", encoding="utf-8")
        _write_json(meta_path, result)
        return result

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            env=env,
        )
        stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="ignore")
        result = {
            "requested": True,
            "ran": True,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "cmd": cmd,
            "cwd": cwd.as_posix(),
            "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "stdout_log": stdout_path.as_posix(),
            "stderr_log": stderr_path.as_posix(),
        }
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8", errors="ignore")
        stderr_path.write_text((exc.stderr or "") + f"\nTIMEOUT after {timeout_sec}s\n", encoding="utf-8", errors="ignore")
        result = {
            "requested": True,
            "ran": True,
            "ok": False,
            "returncode": None,
            "cmd": cmd,
            "cwd": cwd.as_posix(),
            "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "error": f"timeout after {timeout_sec}s",
            "stdout_log": stdout_path.as_posix(),
            "stderr_log": stderr_path.as_posix(),
        }
    _write_json(meta_path, result)
    return result


def _refresh_manifest(bridge: Path, updates: Dict[str, Any] | None = None) -> Dict[str, Any]:
    path = bridge / "vivado_bridge_manifest.json"
    man = _load_json(path) or {}
    if updates:
        man.update(updates)
    component_paths = _component_xml_paths(bridge)
    man["hls_ip_exported"] = bool(component_paths)
    man["component_xml_exists"] = bool(component_paths)
    man["component_xml_count"] = len(component_paths)
    man["component_xml_paths"] = component_paths
    man["hls_ip_artifacts"] = _exported_ip_artifacts(bridge)
    man["bitstream_exists"] = _bitstream_exists(bridge)
    man["xsa_exists"] = _xsa_exists(bridge)
    man["vivado_reports_present"] = _reports_present(bridge)
    _write_json(path, man)
    return man


def _classify_vivado_failure(bridge_dir: Path, res: Dict[str, Any]) -> str:
    """Classify common Vivado failures from generated text logs/reports."""
    text_parts: List[str] = [str(res.get("error") or "")]

    for root in [
        bridge_dir / "logs",
        bridge_dir / "reports",
        bridge_dir / "project" / "fpgai_vivado.runs" / "impl_1",
    ]:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".log", ".jou", ".rpt", ".txt", ".rst"}:
                    try:
                        text_parts.append(path.read_text(errors="ignore"))
                    except Exception:
                        pass

    low = "\n".join(text_parts).lower()

    if (
        "utlz-1" in low
        and "resource utilization" in low
        and (
            "lut as logic over-utilized" in low
            or "slice luts over-utilized" in low
            or ("slice lut" in low and "over-utilized" in low)
        )
    ):
        return "vivado_impl_failed_board_capacity_lut_overutilized"

    if "utlz-1" in low and "over-utilized" in low:
        return "vivado_impl_failed_board_capacity_overutilized"

    if "place_design failed" in low:
        return "vivado_impl_failed_place_design"

    if "route_design failed" in low:
        return "vivado_impl_failed_route_design"

    if "write_bitstream" in low and "failed" in low:
        return "vivado_impl_failed_write_bitstream"

    if not bool(res.get("ok")):
        return "vivado_failed"

    return ""



def _run_for_artifact(
    artifact: Path,
    export_hls_ip: bool,
    run_vivado_synth: bool,
    run_vivado_impl: bool,
    timeout_sec: int | None,
    force_hls_export: bool = False,
) -> Dict[str, Any]:
    bridge = _bridge_dir(artifact)
    bridge.mkdir(parents=True, exist_ok=True)
    row: Dict[str, Any] = {
        "design": artifact.name,
        "vivado_bridge_dir": bridge.as_posix(),
        "hls_ip_export_requested": bool(export_hls_ip),
        "hls_ip_export_ran": False,
        "hls_ip_export_tool_ok": False,
        "hls_ip_export_returncode": None,
        "vivado_synth_requested": bool(run_vivado_synth or run_vivado_impl),
        "vivado_impl_requested": bool(run_vivado_impl),
        "vivado_ran": False,
        "vivado_ok": False,
        "vivado_returncode": None,
    }

    if export_hls_ip:
        if force_hls_export:
            _clear_exported_ip(bridge)
        else:
            _mirror_hls_ip_to_bridge(bridge)
        existing_components = _component_xml_paths(bridge)
        if existing_components and not force_hls_export:
            res = {
                "requested": True,
                "ran": False,
                "ok": True,
                "returncode": 0,
                "reused_existing_ip": True,
                "component_xml_count": len(existing_components),
                "component_xml_paths": existing_components,
                "error": "",
            }
        else:
            cmd, tool_info = _resolved_tool_command(bridge.parent, "vitis_hls", ["-f", "scripts/export_hls_ip.tcl"])
            res = _run_cmd(cmd, bridge, "vitis_hls_export_ip", timeout_sec)
            res["resolved_tool"] = tool_info
            _mirror_hls_ip_to_bridge(bridge)
        row.update({
            "hls_ip_export_run": res,
            "hls_ip_export_ran": bool(res.get("ran")),
            "hls_ip_export_reused_existing_ip": bool(res.get("reused_existing_ip", False)),
            "hls_ip_export_tool_ok": bool(res.get("ok")),
            "hls_ip_export_returncode": res.get("returncode"),
            "hls_ip_export_error": res.get("error", ""),
            "hls_ip_export_stdout_log": res.get("stdout_log", ""),
            "hls_ip_export_stderr_log": res.get("stderr_log", ""),
        })
        row["hls_ip_export_ok"] = bool(res.get("ok")) and _component_xml_exists(bridge)
        _refresh_manifest(bridge, {
            "hls_ip_export_run": res,
            "hls_ip_export_ok": row["hls_ip_export_ok"],
            "hls_ip_export_error": row.get("hls_ip_export_error", ""),
        })

    compile_project_dir = bridge.parent
    gate_block_reason = _vivado_gate_block_reason(compile_project_dir, run_vivado_impl=run_vivado_impl)
    if gate_block_reason:
        bridge.mkdir(parents=True, exist_ok=True)
        manifest_path = bridge / "manifest.json"
        blocked_manifest = {
            "design": artifact.name,
            "vivado_bridge_generated": False,
            "fit_policy_gate_blocked": True,
            "fit_policy_gate_reason": gate_block_reason,
            "hls_ip_export_requested": bool(export_hls_ip or run_vivado_synth or run_vivado_impl),
            "hls_ip_export_ran": False,
            "hls_ip_export_reused_existing_ip": False,
            "hls_ip_export_tool_ok": False,
            "hls_ip_export_returncode": None,
            "hls_ip_exported": False,
            "component_xml_exists": False,
            "component_xml_count": 0,
            "vivado_synth_requested": bool(run_vivado_synth or run_vivado_impl),
            "vivado_impl_requested": bool(run_vivado_impl),
            "bitstream_requested": bool(run_vivado_impl),
            "vivado_ran": False,
            "vivado_ok": False,
            "vivado_returncode": None,
            "vivado_reports_present": False,
            "bitstream_exists": False,
            "xsa_exists": False,
            "error": gate_block_reason,
        }
        _write_json(manifest_path, blocked_manifest)
        print(f"[FPGAI] {gate_block_reason}")
        return blocked_manifest

    if run_vivado_synth or run_vivado_impl:
        if export_hls_ip and not row.get("hls_ip_export_ok", False):
            res = {
                "requested": True,
                "ran": False,
                "ok": False,
                "returncode": None,
                "error": "Cannot run Vivado: HLS IP export failed; inspect vitis_hls_export_ip logs",
                "stdout_log": row.get("hls_ip_export_stdout_log", ""),
                "stderr_log": row.get("hls_ip_export_stderr_log", ""),
            }
        else:
            _mirror_hls_ip_to_bridge(bridge)
            if not _component_xml_exists(bridge):
                res = {
                    "requested": True,
                    "ran": False,
                    "ok": False,
                    "returncode": None,
                    "error": "Cannot run Vivado: no exported HLS IP component.xml found under vivado_bridge/hls_ip",
                    "stdout_log": "",
                    "stderr_log": "",
                }
            else:
                env = os.environ.copy()
                if run_vivado_impl:
                    env["FPGAI_VIVADO_RUN_IMPL"] = "1"
                cmd, tool_info = _resolved_tool_command(bridge.parent, "vivado", ["-mode", "batch", "-source", "scripts/run_vivado.tcl"])
                res = _run_cmd(cmd, bridge, "vivado_build", timeout_sec, env=env)
                res["resolved_tool"] = tool_info

        failure_class = "" if bool(res.get("ok")) else _classify_vivado_failure(bridge, res)
        if failure_class:
            res["failure_class"] = failure_class
            if not res.get("error"):
                res["error"] = failure_class

        row.update({
            "vivado_run": res,
            "vivado_ran": bool(res.get("ran")),
            "vivado_ok": bool(res.get("ok")),
            "vivado_returncode": res.get("returncode"),
            "vivado_error": res.get("error", ""),
            "vivado_failure_class": res.get("failure_class", ""),
            "vivado_stdout_log": res.get("stdout_log", ""),
            "vivado_stderr_log": res.get("stderr_log", ""),
            "vivado_tool": res.get("resolved_tool", {}),
        })
        _refresh_manifest(
            bridge,
            {
                "vivado_run": res,
                "vivado_ok": row["vivado_ok"],
                "vivado_returncode": row.get("vivado_returncode"),
                "vivado_error": row.get("vivado_error", ""),
                "vivado_failure_class": row.get("vivado_failure_class", ""),
            },
        )

    _mirror_hls_ip_to_bridge(bridge)
    man = _refresh_manifest(bridge)
    row["hls_ip_exported"] = bool(man.get("hls_ip_exported"))
    row["component_xml_exists"] = bool(man.get("component_xml_exists"))
    row["component_xml_count"] = int(man.get("component_xml_count") or 0)
    row["component_xml_paths"] = man.get("component_xml_paths", [])
    row["hls_ip_artifacts"] = man.get("hls_ip_artifacts", [])
    row["bitstream_exists"] = bool(man.get("bitstream_exists"))
    row["xsa_exists"] = bool(man.get("xsa_exists"))
    row["vivado_reports_present"] = bool(man.get("vivado_reports_present"))
    if run_vivado_synth or run_vivado_impl:
        _sync_bridge_result_to_compile_output(artifact, bridge, row)
    return row


def _cell(v: Any) -> str:
    if v is None:
        return ""
    return str(v).replace("|", "/").replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="Experiment directory containing artifacts/<design>/build/hls")
    ap.add_argument("--board", default="pynq_z2", help="Board key: pynq_z2, kv260, kr260")
    ap.add_argument("--run-impl-default", action="store_true", help="Generate run_vivado.tcl with implementation/bitstream enabled")
    ap.add_argument("--export-hls-ip", action="store_true", help="Ensure an exported HLS IP repo exists for each selected design; reuses existing IP unless --force-hls-export is set")
    ap.add_argument("--force-hls-export", action="store_true", help="Force rerunning vitis_hls export even if vivado_bridge/hls_ip/component.xml already exists")
    ap.add_argument("--run-vivado-synth", action="store_true", help="Automatically run vivado -mode batch -source scripts/run_vivado.tcl")
    ap.add_argument("--run-vivado-impl", action="store_true", help="Generate implementation-enabled Vivado script and run it")
    ap.add_argument("--max-designs", type=int, default=None, help="Limit tool execution to the first N designs for smoke testing")
    ap.add_argument("--timeout-sec", type=int, default=3600, help="Timeout per tool command")
    args = ap.parse_args()

    exp = Path(args.experiment).resolve()
    run_impl_default = bool(args.run_impl_default or args.run_vivado_impl)

    gen_rows = generate_vivado_bridge_for_experiment(exp, board_name=args.board, run_impl_default=run_impl_default)

    run_rows: List[Dict[str, Any]] = []
    tool_artifacts = _limited(_iter_artifacts(exp), args.max_designs)
    if args.export_hls_ip or args.run_vivado_synth or args.run_vivado_impl:
        for art in tool_artifacts:
            try:
                run_rows.append(_run_for_artifact(
                    art,
                    export_hls_ip=args.export_hls_ip or args.run_vivado_synth or args.run_vivado_impl,
                    run_vivado_synth=args.run_vivado_synth,
                    run_vivado_impl=args.run_vivado_impl,
                    timeout_sec=args.timeout_sec,
                    force_hls_export=args.force_hls_export,
                ))
            except Exception as exc:
                run_rows.append({"design": art.name, "error": str(exc)})

    out = exp / "vivado_bridge_run_artifacts.json"
    _write_json(out, {"generated": gen_rows, "tool_runs": run_rows})

    print("# Automated Vivado bridge")
    print()
    print("| design | generated | hls_req | hls_ran | hls_reused | hls_tool_ok | hls_returncode | hls_ip_exported | component_xml_count | vivado_req | vivado_ran | vivado_ok | vivado_returncode | reports | bitstream | xsa | error | stdout_log | stderr_log |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    by_design = {r.get("design"): r for r in run_rows}
    for g in gen_rows:
        d = g.get("design", "")
        r = by_design.get(d, {})
        bridge = Path(g.get("vivado_bridge_dir", "")) if g.get("vivado_bridge_dir") else None
        if bridge and bridge.exists():
            _mirror_hls_ip_to_bridge(bridge)
            man = _refresh_manifest(bridge)
            fallback_ip = bool(man.get("hls_ip_exported"))
            fallback_count = int(man.get("component_xml_count") or 0)
        else:
            fallback_ip = False
            fallback_count = 0
        err = r.get("error") or r.get("hls_ip_export_error") or r.get("vivado_error") or g.get("error", "")
        stdout_log = r.get("vivado_stdout_log") or r.get("hls_ip_export_stdout_log") or ""
        stderr_log = r.get("vivado_stderr_log") or r.get("hls_ip_export_stderr_log") or ""
        print(
            "| {design} | {generated} | {hls_req} | {hls_ran} | {hls_reused} | {hls_ok} | {hls_rc} | {ip} | {count} | {vivado_req} | {vivado_ran} | {vivado_ok} | {vivado_rc} | {reports} | {bit} | {xsa} | {error} | {stdout} | {stderr} |".format(
                design=_cell(d),
                generated=_cell(g.get("vivado_bridge_generated", False)),
                hls_req=_cell(r.get("hls_ip_export_requested", False)),
                hls_ran=_cell(r.get("hls_ip_export_ran", False)),
                hls_reused=_cell(r.get("hls_ip_export_reused_existing_ip", False)),
                hls_ok=_cell(r.get("hls_ip_export_tool_ok", False)),
                hls_rc=_cell(r.get("hls_ip_export_returncode")),
                ip=_cell(r.get("hls_ip_exported", fallback_ip)),
                count=_cell(r.get("component_xml_count", fallback_count)),
                vivado_req=_cell(r.get("vivado_synth_requested", False) or r.get("vivado_impl_requested", False)),
                vivado_ran=_cell(r.get("vivado_ran", False)),
                vivado_ok=_cell(r.get("vivado_ok", False)),
                vivado_rc=_cell(r.get("vivado_returncode")),
                reports=_cell(r.get("vivado_reports_present", False)),
                bit=_cell(r.get("bitstream_exists", False)),
                xsa=_cell(r.get("xsa_exists", False)),
                error=_cell(err),
                stdout=_cell(stdout_log),
                stderr=_cell(stderr_log),
            )
        )
    requested_tool_run = bool(args.export_hls_ip or args.run_vivado_synth or args.run_vivado_impl)
    failed_rows = []

    if requested_tool_run:
        for row in run_rows:
            design = row.get("design", "")
            if row.get("error"):
                failed_rows.append((design, row.get("error")))
                continue

            if args.export_hls_ip or args.run_vivado_synth or args.run_vivado_impl:
                if not row.get("hls_ip_export_ok", False):
                    failed_rows.append((design, row.get("hls_ip_export_error") or "HLS IP export failed"))
                    continue

            if args.run_vivado_synth or args.run_vivado_impl:
                if not row.get("vivado_ok", False):
                    failed_rows.append((design, row.get("vivado_error") or f"Vivado failed with returncode={row.get('vivado_returncode')}"))
                    continue

            if args.run_vivado_impl:
                if not row.get("bitstream_exists", False):
                    failed_rows.append((design, "Vivado implementation requested but bitstream was not produced"))
                    continue
                if not row.get("xsa_exists", False):
                    failed_rows.append((design, "Vivado implementation requested but XSA was not produced"))
                    continue

    print()
    if failed_rows:
        print(f"[ERROR] Wrote {out}, but {len(failed_rows)} requested tool run(s) failed:")
        for design, reason in failed_rows:
            print(f"  - {design}: {reason}")
        return 1

    print(f"[OK] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
