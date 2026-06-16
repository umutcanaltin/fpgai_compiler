from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from fpgai.backends.hls.runner import HLSRunResult, run_vitis_hls


TILED_KERNEL_NAMES = (
    "dense_out_in_tiled",
    "conv2d_tiled",
)


def detect_tiled_kernels(
    hls_dir: str | Path,
    *,
    top_name: str | None = None,
) -> dict[str, bool]:
    """Detect whether generated HLS source contains tiled kernels."""
    hls_path = Path(hls_dir)
    src_dir = hls_path / "src"

    candidates: list[Path] = []
    if top_name:
        candidates.append(src_dir / f"{top_name}.cpp")

    candidates.extend(
        [
            src_dir / "deeplearn.cpp",
            *sorted(src_dir.glob("*.cpp")),
        ]
    )

    seen: set[Path] = set()
    source_text = ""
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)

        if candidate.exists():
            source_text += candidate.read_text(
                encoding="utf-8",
                errors="ignore",
            )
            source_text += "\n"

    return {
        kernel_name: kernel_name in source_text
        for kernel_name in TILED_KERNEL_NAMES
    }


def _safe_path(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def build_hls_validation_report(
    *,
    hls_dir: str | Path,
    result: HLSRunResult | None = None,
    top_name: str | None = None,
    requested: bool = True,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a serializable HLS validation report."""
    hls_path = Path(hls_dir)
    tiled_kernels = detect_tiled_kernels(
        hls_path,
        top_name=top_name,
    )

    ok = bool(result.ok) if result is not None else False
    returncode = int(result.returncode) if result is not None else None
    csynth_report = (
        _safe_path(result.csynth_report)
        if result is not None
        else None
    )

    return {
        "format": "fpgai.hls_validation.v1",
        "requested": bool(requested),
        "ok": ok and error is None,
        "returncode": returncode,
        "command": result.command if result is not None else None,
        "workdir": result.workdir if result is not None else str(hls_path),
        "stdout_log": result.stdout_log if result is not None else None,
        "stderr_log": result.stderr_log if result is not None else None,
        "csynth_report": csynth_report,
        "csynth_report_present": bool(
            csynth_report and Path(csynth_report).exists()
        ),
        "tiled_kernels_detected": tiled_kernels,
        "tiling_enabled": any(tiled_kernels.values()),
        "error": error,
    }


def write_hls_validation_json(
    path: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    serializable = dict(report)
    output_path.write_text(
        json.dumps(
            serializable,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return serializable


def hls_validation_manifest_entry(
    report: Mapping[str, Any],
    *,
    path: str = "hls_validation.json",
) -> dict[str, Any]:
    tiled = report.get("tiled_kernels_detected", {})
    if not isinstance(tiled, Mapping):
        tiled = {}

    return {
        "format": report.get(
            "format",
            "fpgai.hls_validation.v1",
        ),
        "path": path,
        "requested": bool(report.get("requested", False)),
        "ok": bool(report.get("ok", False)),
        "returncode": report.get("returncode"),
        "csynth_report_present": bool(
            report.get("csynth_report_present", False)
        ),
        "tiling_enabled": bool(
            report.get("tiling_enabled", False)
        ),
        "dense_out_in_tiled": bool(
            tiled.get("dense_out_in_tiled", False)
        ),
        "conv2d_tiled": bool(
            tiled.get("conv2d_tiled", False)
        ),
    }


def attach_hls_validation_to_manifest(
    manifest: MutableMapping[str, Any],
    report: Mapping[str, Any],
    *,
    path: str = "hls_validation.json",
) -> MutableMapping[str, Any]:
    manifest["hls_validation"] = hls_validation_manifest_entry(
        report,
        path=path,
    )
    return manifest


def run_and_write_hls_validation(
    hls_dir: str | Path,
    *,
    manifest: MutableMapping[str, Any] | None = None,
    run_hls_fn: Callable[..., HLSRunResult] = run_vitis_hls,
    vitis_hls_exe: str = "vitis_hls",
    settings64: str | None = None,
    tcl_name: str = "run_hls.tcl",
    top_name: str | None = None,
    filename: str = "hls_validation.json",
) -> tuple[dict[str, Any], MutableMapping[str, Any]]:
    """Run Vitis HLS, write hls_validation.json, and update manifest."""
    hls_path = Path(hls_dir)
    reports_dir = hls_path / "reports"

    updated_manifest: MutableMapping[str, Any]
    if manifest is None:
        updated_manifest = {}
    else:
        updated_manifest = manifest

    try:
        result = run_hls_fn(
            hls_dir=hls_path,
            vitis_hls_exe=vitis_hls_exe,
            settings64=settings64,
            tcl_name=tcl_name,
        )
        report = build_hls_validation_report(
            hls_dir=hls_path,
            result=result,
            top_name=top_name,
            requested=True,
        )
    except Exception as exc:
        report = build_hls_validation_report(
            hls_dir=hls_path,
            result=None,
            top_name=top_name,
            requested=True,
            error=str(exc),
        )

    write_hls_validation_json(
        reports_dir / filename,
        report,
    )
    attach_hls_validation_to_manifest(
        updated_manifest,
        report,
        path=filename,
    )

    return report, updated_manifest
