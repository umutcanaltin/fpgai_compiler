from __future__ import annotations

"""Vivado bridge execution and compile-manifest update helpers."""

import json
from pathlib import Path
from typing import Any, Dict

from fpgai.config.access import get_path
from fpgai.backends.vivado.run_bridge import run_vivado_bridge_flow
from fpgai.benchmark.experiment_artifacts import emit_experiment_artifact_reports
from fpgai.runtime.package import emit_runtime_package
from fpgai.util.fs import write_text

_cfg_get = get_path

def _yaml_requested_vivado_bridge(build_stages: Dict[str, Any]) -> bool:
    return bool(
        build_stages.get("vivado_project")
        or build_stages.get("vivado_implementation")
        or build_stages.get("bitstream")
    )


def _vivado_bridge_timeout_sec(raw: Dict[str, Any]) -> int | None:
    for path in (
        "toolchain.vivado.timeout_sec",
        "toolchain.timeout_sec",
        "build.vivado_timeout_sec",
        "build.tool_timeout_sec",
        "build.timeout_sec",
    ):
        value = _cfg_get(raw, path, None)
        if value in (None, ""):
            continue
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            continue
        return ivalue if ivalue > 0 else None
    return 3600




def _existing_hls_ip_component_exists(out_dir: Path) -> bool:
    """Return True when a concrete HLS IP repo is already available.

    ``build.existing_hls_ip=true`` allows Vivado handoff generation without
    rerunning HLS synthesis. It does not identify a concrete IP repository by
    itself. When no component.xml is present, the compiler must keep the stage
    at the generated-artifact/reporting boundary instead of invoking Vitis HLS
    or running Vivado against a nonexistent IP.
    """
    roots = [
        out_dir / "vivado_bridge" / "hls_ip",
        out_dir / "hls" / "fpgai_hls_proj" / "sol1" / "impl" / "ip",
    ]
    for root in roots:
        if root.exists() and any(root.glob("**/component.xml")):
            return True
    return False


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _runtime_package_manifest_summary(out_dir: Path) -> Dict[str, Any] | None:
    package = _read_json_file(out_dir / "runtime_package" / "package_manifest.json")
    if not package:
        return None
    hardware = package.get("hardware", {}) if isinstance(package.get("hardware", {}), dict) else {}
    return {
        "status": package.get("status", "created"),
        "path": "runtime_package/package_manifest.json",
        "deployable_overlay_present": bool(hardware.get("deployable_overlay_present")),
        "bitstream_present": bool((hardware.get("bitstream") or {}).get("present")) if isinstance(hardware.get("bitstream"), dict) else False,
        "hwh_present": bool((hardware.get("hwh") or {}).get("present")) if isinstance(hardware.get("hwh"), dict) else False,
        "xsa_present": bool((hardware.get("xsa") or {}).get("present")) if isinstance(hardware.get("xsa"), dict) else False,
        "file_count": len(package.get("files", {})) if isinstance(package.get("files"), dict) else package.get("file_count"),
        "hardware": hardware,
    }


def _update_manifest_after_vivado_bridge(out_dir: Path, bridge_payload: Dict[str, Any]) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = _read_json_file(manifest_path)
    if not manifest:
        return

    generated = bridge_payload.get("generated", [])
    tool_runs = bridge_payload.get("tool_runs", [])
    first_generated = generated[0] if generated and isinstance(generated[0], dict) else {}
    first_run = tool_runs[0] if tool_runs and isinstance(tool_runs[0], dict) else {}

    bridge_summary = dict(first_generated)
    if first_run:
        bridge_summary.update({
            "hls_ip_export_requested": bool(first_run.get("hls_ip_export_requested")),
            "hls_ip_export_ok": bool(first_run.get("hls_ip_export_ok")),
            "hls_ip_export_reused_existing_ip": bool(first_run.get("hls_ip_export_reused_existing_ip")),
            "component_xml_count": first_run.get("component_xml_count"),
            "vivado_synth_requested": bool(first_run.get("vivado_synth_requested")),
            "vivado_impl_requested": bool(first_run.get("vivado_impl_requested")),
            "bitstream_requested": bool(first_run.get("bitstream_requested")),
            "vivado_ran": bool(first_run.get("vivado_ran")),
            "vivado_ok": bool(first_run.get("vivado_ok")),
            "vivado_returncode": first_run.get("vivado_returncode"),
            "vivado_failure_class": first_run.get("vivado_failure_class", ""),
            "bitstream_exists": bool(first_run.get("bitstream_exists")),
            "xsa_exists": bool(first_run.get("xsa_exists")),
            "vivado_reports_present": bool(first_run.get("vivado_reports_present")),
            "stdout_log": first_run.get("vivado_stdout_log") or first_run.get("hls_ip_export_stdout_log"),
            "stderr_log": first_run.get("vivado_stderr_log") or first_run.get("hls_ip_export_stderr_log"),
        })
    bridge_summary["artifacts_json"] = bridge_payload.get("artifacts_json")
    bridge_summary["ok"] = bool(bridge_payload.get("ok", not bridge_payload.get("failed_rows")))
    bridge_summary["failed_rows"] = bridge_payload.get("failed_rows", [])
    manifest["vivado_bridge"] = bridge_summary

    runtime_summary = _runtime_package_manifest_summary(out_dir)
    if runtime_summary is not None:
        manifest["runtime_package"] = runtime_summary

    stages = manifest.get("pipeline_stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            name = stage.get("name")
            if name == "vivado_project" and manifest.get("build_stages", {}).get("vivado_project"):
                stage["status"] = "done" if bridge_summary.get("vivado_bridge_generated") else "failed"
                stage["detail"] = "Vivado bridge/project artifacts were generated from the YAML-requested build stage."
            elif name == "bitstream" and manifest.get("build_stages", {}).get("bitstream"):
                if bridge_summary.get("bitstream_exists") and bridge_summary.get("xsa_exists"):
                    stage["status"] = "done"
                    stage["detail"] = "Bitstream/XSA artifacts were generated by YAML-driven Vivado execution."
                elif bridge_summary.get("failed_rows"):
                    stage["status"] = "failed"
                    stage["detail"] = "YAML-driven Vivado execution was requested but did not produce required bitstream/XSA artifacts."
    manifest["pipeline_stages"] = stages

    write_text(manifest_path, json.dumps(manifest, indent=2))


def _run_yaml_requested_vivado_bridge(
    out_dir: Path,
    raw: Dict[str, Any],
    build_stages: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Execute YAML-requested Vivado stages through the existing bridge backend."""
    if not _yaml_requested_vivado_bridge(build_stages):
        return None

    board = str(
        _cfg_get(
            raw,
            "targets.platform.board",
            _cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "pynq_z2")),
        )
        or "pynq_z2"
    )
    run_impl_requested = bool(build_stages.get("vivado_implementation") or build_stages.get("bitstream"))
    run_bitstream_requested = bool(build_stages.get("bitstream"))

    existing_hls_ip = bool(_cfg_get(raw, "build.existing_hls_ip", False))
    hls_synthesis_requested = bool(build_stages.get("hls_synthesis"))
    concrete_existing_ip = _existing_hls_ip_component_exists(out_dir)

    # Export IP only when this compile actually owns an HLS synthesis flow.
    # For ``build.existing_hls_ip=true`` report-only/unit-test flows, do not
    # invoke Vitis HLS just because a Vivado project was requested.  Vivado can
    # be executed only if a concrete component.xml already exists.
    export_hls_ip = bool(
        (build_stages.get("vivado_project") or run_impl_requested)
        and hls_synthesis_requested
        and not existing_hls_ip
    )
    run_impl = run_impl_requested
    run_bitstream = run_bitstream_requested
    if existing_hls_ip and run_impl_requested and not concrete_existing_ip:
        run_impl = False
        run_bitstream = False

    target_clock_mhz = float(
        _cfg_get(raw, "targets.platform.clocks.0.target_mhz", 100.0) or 100.0
    )
    payload = run_vivado_bridge_flow(
        out_dir,
        board=board,
        export_hls_ip=export_hls_ip,
        run_vivado_synth=False,
        run_vivado_impl=run_impl,
        run_bitstream=run_bitstream,
        timeout_sec=_vivado_bridge_timeout_sec(raw),
        target_clock_mhz=target_clock_mhz,
    )
    _update_manifest_after_vivado_bridge(out_dir, payload)
    if build_stages.get("runtime_package"):
        emit_runtime_package(out_dir)
    emit_experiment_artifact_reports(out_dir)

    failed_rows = payload.get("failed_rows") or []
    if failed_rows:
        reason = "; ".join(f"{design}: {why}" for design, why in failed_rows)
        raise RuntimeError(f"YAML-requested Vivado bridge flow failed: {reason}")
    return payload
