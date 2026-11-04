"""Read-only diagnostic for optional AMD FPGA toolchain environments.

No imports from vendor tools, no vendor execution, no secrets read, and
no license/availability promise. The optional runner commands must be
invoked by the researcher explicitly after verification.

Examples:
  python scripts/check_amd_environment.py
  python scripts/check_amd_environment.py --require-hls --require-vivado
  python scripts/check_amd_environment.py --require-platform --require-xrt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def _is_file_executable(value: str | None) -> bool:
    if not value:
        return False
    path = Path(value).expanduser()
    if "/" in value or path.is_absolute():
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(value) is not None


def _tool(name: str, preferred: str | None, fallback: str) -> dict:
    choice = preferred if preferred else fallback
    # Use only an explicitly configured executable, or the PATH.
    located = shutil.which(choice)
    if located is None and "/" in choice:
        p = Path(choice).expanduser()
        located = str(p) if p.is_file() and os.access(p, os.X_OK) else None
    return {
        "name": name,
        "configured_executable": choice,
        "resolved_executable": located,
        "available": bool(located and _is_file_executable(located)),
    }


def inspect_environment(env: dict[str, str] | None = None) -> dict:
    """Inspect settings; never spawn binaries or read license files."""
    if env is None:
        env = dict(os.environ)

    # Vendor variables are paths, not secret license values; license keys
    # are never included in the output.
    vitis = _tool(
        "vitis_hls", env.get("FPGAI_VITIS_HLS_EXECUTABLE"), "vitis_hls")
    vivado = _tool("vivado", env.get("FPGAI_VIVADO_EXECUTABLE"), "vivado")
    platform_paths = [
        path for path in env.get("PLATFORM_REPO_PATHS", "").split(os.pathsep)
        if path
    ]
    platforms = [
        {"path": str(Path(path).expanduser()),
         "exists": Path(path).expanduser().is_dir()}
        for path in platform_paths
    ]
    rootfs=env.get("ROOTFS")
    sysroot=env.get("SYSROOT") or env.get("SDKTARGETSYSROOT")
    embedded={
        "ROOTFS": rootfs,
        "ROOTFS_EXISTS": bool(rootfs and Path(rootfs).expanduser().exists()),
        "SYSROOT": sysroot,
        "SYSROOT_EXISTS": bool(sysroot and Path(sysroot).expanduser().is_dir()),
    }
    xrt_setup = env.get("FPGAI_XRT_SETUP", "/opt/xilinx/xrt/setup.sh")
    xrt_setup_present = Path(xrt_setup).expanduser().is_file()
    xrt_env = env.get("XILINX_XRT") or env.get("XRT_ROOT")
    return {
        "schema": "fpgai-research/amd-environment-check/v1",
        "vendor_commands_executed": False,
        "hardware_programmed": False,
        "settings": {
            "FPGAI_AMD_ROOT": env.get("FPGAI_AMD_ROOT"),
            "FPGAI_AMD_VERSION": env.get("FPGAI_AMD_VERSION", "2023.2"),
            "XILINX_VITIS": env.get("XILINX_VITIS"),
            "XILINX_VIVADO": env.get("XILINX_VIVADO"),
            "PLATFORM_REPO_PATHS": platforms,
            "XRT_SETUP": xrt_setup,
            "XRT_SETUP_PRESENT": xrt_setup_present,
            "XRT_ENV_PRESENT": bool(xrt_env),
            "EMBEDDED_LINUX": embedded,
        },
        "tools": {"vitis_hls": vitis, "vivado": vivado},
        "caveats": [
            "Executable discovery does not verify vendor version, licenses, target support or board compatibility.",
            "Generated HLS source is not a board bitstream or runnable FPGA design.",
            "XRT is not required for source-only HLS/Vivado out-of-context studies.",
        ],
    }


def required_missing(
    report: dict, *, hls: bool, vivado: bool,
    platform: bool, xrt: bool, rootfs: bool = False, sysroot: bool = False,
) -> tuple[str, ...]:
    missing = []
    if hls and not report["tools"]["vitis_hls"]["available"]:
        missing.append("vitis_hls")
    if vivado and not report["tools"]["vivado"]["available"]:
        missing.append("vivado")
    if platform and not any(
        entry["exists"] for entry in report["settings"]["PLATFORM_REPO_PATHS"]
    ):
        missing.append("PLATFORM_REPO_PATHS")
    if xrt and not (
        report["settings"]["XRT_SETUP_PRESENT"]
        or report["settings"]["XRT_ENV_PRESENT"]
    ):
        missing.append("XRT_SETUP")
    if rootfs and not report["settings"]["EMBEDDED_LINUX"]["ROOTFS_EXISTS"]:
        missing.append("ROOTFS")
    if sysroot and not report["settings"]["EMBEDDED_LINUX"]["SYSROOT_EXISTS"]:
        missing.append("SYSROOT")
    return tuple(missing)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-hls", action="store_true")
    ap.add_argument("--require-vivado", action="store_true")
    ap.add_argument("--require-platform", action="store_true")
    ap.add_argument("--require-xrt", action="store_true")
    ap.add_argument("--require-rootfs", action="store_true")
    ap.add_argument("--require-sysroot", action="store_true")
    args = ap.parse_args(argv)
    report = inspect_environment()
    missing = required_missing(
        report, hls=args.require_hls, vivado=args.require_vivado,
        platform=args.require_platform, xrt=args.require_xrt,
        rootfs=args.require_rootfs, sysroot=args.require_sysroot,
    )
    report["requested_requirements_missing"] = list(missing)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
