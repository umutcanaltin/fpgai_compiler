from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _copy_if_exists(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": src.as_posix(),
        "package_path": dst.as_posix(),
        "bytes": dst.stat().st_size,
    }


def _first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        for hit in hits:
            if hit.is_file():
                return hit
    return None


def _collect_existing(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend([p for p in sorted(root.glob(pattern)) if p.is_file()])
    return out


def _artifact_status(path: Path | None) -> dict[str, Any]:
    return {
        "present": bool(path is not None and path.exists()),
        "path": path.as_posix() if path is not None else None,
    }


def emit_runtime_package(
    out_dir: str | Path,
    *,
    board: str | None = None,
    pipeline_mode: str | None = None,
    top_name: str | None = None,
    hls_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-describing runtime package from existing compile artifacts.

    This function does not run Vivado, deploy to a board, or infer that hardware
    artifacts exist. It packages files that are already present and records
    bitstream/XSA/HWH status truthfully.
    """

    root = Path(out_dir).resolve()
    package_dir = root / "runtime_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}

    copy_plan = {
        "compile_manifest": (root / "manifest.json", package_dir / "manifest.json"),
        "input_bin": (root / "input.bin", package_dir / "inputs" / "input.bin"),
        "output_bin": (root / "output.bin", package_dir / "outputs" / "output.bin"),
        "hls_artifact_metadata": (
            root / "hls_artifact_metadata.json",
            package_dir / "hls" / "hls_artifact_metadata.json",
        ),
        "hls_schedule_summary": (
            root / "hls_schedule_summary.json",
            package_dir / "hls" / "hls_schedule_summary.json",
        ),
        "hls_ii_comparison": (
            root / "hls_ii_comparison.json",
            package_dir / "hls" / "hls_ii_comparison.json",
        ),
    }

    for name, (src, dst) in copy_plan.items():
        copied = _copy_if_exists(src, dst)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            files[name] = copied

    # Capture HLS run logs when present.
    hls_logs = _collect_existing(root, ["hls/logs/*.log", "hls/logs/*.json"])
    copied_logs: list[dict[str, Any]] = []
    for src in hls_logs:
        copied = _copy_if_exists(src, package_dir / "hls" / "logs" / src.name)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            copied_logs.append(copied)
    if copied_logs:
        files["hls_logs"] = copied_logs

    # Runtime hardware handoff/status. These are presence checks only.
    bitstream = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.bit",
            "vivado_bridge/project/**/*.bit",
            "**/*.bit",
        ],
    )
    hwh = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.hwh",
            "vivado_bridge/project/**/*.hwh",
            "**/*.hwh",
        ],
    )
    xsa = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.xsa",
            "vivado_bridge/project/**/*.xsa",
            "**/*.xsa",
        ],
    )

    hardware = {
        "bitstream": _artifact_status(bitstream),
        "hwh": _artifact_status(hwh),
        "xsa": _artifact_status(xsa),
        "deployable_overlay_present": bool(bitstream is not None and (hwh is not None or xsa is not None)),
    }

    for name, src in {"bitstream": bitstream, "hwh": hwh, "xsa": xsa}.items():
        if src is None:
            continue
        copied = _copy_if_exists(src, package_dir / "hardware" / src.name)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            files[name] = copied

    payload: dict[str, Any] = {
        "schema_version": 1,
        "package_kind": "fpgai_runtime_package",
        "status": "created",
        "package_dir": package_dir.as_posix(),
        "source_out_dir": root.as_posix(),
        "board": board,
        "pipeline_mode": pipeline_mode,
        "top_name": top_name,
        "hls_artifacts": dict(hls_artifacts or {}),
        "hardware": hardware,
        "files": files,
        "notes": [
            "Runtime package records and copies existing artifacts only.",
            "It does not run Vivado, deploy to hardware, or infer missing bitstream/XSA/HWH files.",
        ],
    }

    manifest_path = package_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    readme = package_dir / "README_RUNTIME.md"
    readme.write_text(
        "\n".join(
            [
                "# FPGAI Runtime Package",
                "",
                "This package contains runtime-facing artifacts copied from an FPGAI compile output.",
                "",
                f"- board: `{board}`",
                f"- pipeline_mode: `{pipeline_mode}`",
                f"- top_name: `{top_name}`",
                f"- bitstream present: `{hardware['bitstream']['present']}`",
                f"- hwh present: `{hardware['hwh']['present']}`",
                f"- xsa present: `{hardware['xsa']['present']}`",
                f"- deployable overlay present: `{hardware['deployable_overlay_present']}`",
                "",
                "The package is truthful: missing hardware handoff files are recorded as missing.",
                "Use the Vivado bridge flow to generate bitstream/XSA artifacts before board deployment.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "path": "runtime_package/package_manifest.json",
        "package_dir": "runtime_package",
        "status": payload["status"],
        "deployable_overlay_present": hardware["deployable_overlay_present"],
        "bitstream_present": hardware["bitstream"]["present"],
        "hwh_present": hardware["hwh"]["present"],
        "xsa_present": hardware["xsa"]["present"],
        "file_count": len(files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an FPGAI runtime package from a compile output directory.")
    parser.add_argument("out_dir")
    parser.add_argument("--board")
    parser.add_argument("--pipeline-mode")
    parser.add_argument("--top-name")
    ns = parser.parse_args(argv)

    result = emit_runtime_package(
        ns.out_dir,
        board=ns.board,
        pipeline_mode=ns.pipeline_mode,
        top_name=ns.top_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
