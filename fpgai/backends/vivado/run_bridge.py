#!/usr/bin/env python3
"""Generate Vivado bridge artifacts and optionally run Vitis HLS/Vivado automatically."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fpgai.backends.vivado import generate_vivado_bridge_for_experiment


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
            res = _run_cmd(["vitis_hls", "-f", "scripts/export_hls_ip.tcl"], bridge, "vitis_hls_export_ip", timeout_sec)
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
                res = _run_cmd(["vivado", "-mode", "batch", "-source", "scripts/run_vivado.tcl"], bridge, "vivado_build", timeout_sec, env=env)
        row.update({
            "vivado_run": res,
            "vivado_ran": bool(res.get("ran")),
            "vivado_ok": bool(res.get("ok")),
            "vivado_returncode": res.get("returncode"),
            "vivado_error": res.get("error", ""),
            "vivado_stdout_log": res.get("stdout_log", ""),
            "vivado_stderr_log": res.get("stderr_log", ""),
        })
        _refresh_manifest(bridge, {"vivado_run": res, "vivado_ok": row["vivado_ok"], "vivado_error": row.get("vivado_error", "")})

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
    print()
    print(f"[OK] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
