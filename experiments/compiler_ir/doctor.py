"""Read-only workstation readiness report for compiler-paper experiment series.

This checks the *local* machine at command time; GitHub CI cannot attest the
user's workstation, license, tool installation, available disk, model assets,
or resource contention. No vendor synthesis, RL or filesystem mutation.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

def _probe_command(executable: str, args: tuple[str, ...], timeout: int = 15) -> dict:
    path = shutil.which(executable)
    if not path:
        return {"available": False, "resolved_path": None, "version": None}
    version = None
    try:
        p = subprocess.run([path, *args], text=True, capture_output=True,
                           timeout=timeout, check=False)
        lines = (p.stdout or p.stderr).strip().splitlines()
        if lines:
            version = "\n".join(lines[:5])[:1024]
        if p.returncode != 0:
            version = ("version-check-exit=" + str(p.returncode) + "; " +
                       (version or "no output"))
    except (OSError, subprocess.TimeoutExpired) as exc:
        version = type(exc).__name__ + " during version inspection"
    return {"available": True, "resolved_path": path, "version": version}


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def inspect(root: str | Path, *, with_native: bool = False,
            native_executable: str = "vitis_hls",
            trained_mlp: str | None = None,
            trained_cnn: str | None = None) -> dict:
    base = Path(root)
    if not base.is_dir() or base.is_symlink():
        raise ValueError("Expected real compiler checkout folder")
    requirements = ("numpy", "pytest", "onnx", "onnxruntime")
    modules = {m: _module(m) for m in requirements}
    executables = {
        "git": _probe_command("git", ("--version",)),
        "g++": _probe_command("g++", ("--version",)),
    }
    status = subprocess.run(
        ["git","-C",str(base),"status","--porcelain","--untracked-files=no"],
        capture_output=True, text=True, timeout=10)
    head = subprocess.run(["git","-C",str(base),"rev-parse","HEAD"],
                          capture_output=True, text=True, timeout=10)
    version_ok = sys.version_info >= (3, 10)
    clean = status.returncode == 0 and not status.stdout.strip()
    git_revision = head.stdout.strip() if head.returncode == 0 else None
    disk = shutil.disk_usage(base)
    assets = {}
    for kind, loc in (("mlp", trained_mlp), ("cnn", trained_cnn)):
        if loc is None:
            assets[kind] = {"requested":False, "ready":False,
                            "reason":"Not supplied; no trained-model run"}
            continue
        folder = Path(loc)
        try:
            # Uses the original study's strict hash/provenance validator.
            from fpgai_research.experiments.mnist_assets import _validate
            checked = _validate(folder, kind, check_onnx=True)
            assets[kind] = {
                "requested":True, "ready":True,
                "sha256": checked["files_sha256"],
                "samples": checked["sample_count"],
            }
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            assets[kind] = {
                "requested":True, "ready":False,
                "reason":type(exc).__name__ + ": " + str(exc),
            }
    native = _probe_command(native_executable, ("-version",)) if with_native else {
        "available": False, "resolved_path": None, "version": None,
        "reason":"Not requested; no native tool will be started"}
    cpu_ready = bool(version_ok and clean and len(git_revision or "") == 40
                     and all(modules.values())
                     and all(x["available"] for x in executables.values()))
    original_native_ready = bool(
        cpu_ready and with_native and native["available"])
    # Detecting a binary does not establish tool licensing, exact part
    # support or timing/numerical success; they require original execution.
    return {
        "schema":"fpgai-research/compiler-ir-workstation-preflight/v1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "compiler_checkout": str(base),
        "git_revision": git_revision,
        "tracked_checkout_clean": clean,
        "python_modules": modules,
        "executables": executables,
        "disk_free_gib": round(disk.free/(1024**3), 2),
        "trained_assets": assets,
        "native_tool": native,
        "cpu_paper_series_ready": cpu_ready,
        "native_binary_detected": bool(native["available"]) if with_native else False,
        "native_execution_environment_ready_to_attempt": original_native_ready,
        "missing_for_cpu": [
            *([] if version_ok else ["Python 3.10+"]),
            *([] if clean else ["Clean committed checkout"]),
            *([] if len(git_revision or "") == 40 else ["Git SHA"]),
            *(m for m, enabled in modules.items() if not enabled),
            *(name for name, x in executables.items() if not x["available"]),
        ],
        "caveats": [
            "Vitis -version is only a binary check, not a license, part, clock or runtime guarantee",
            "An isolated checkout is required if any RL study is active",
            "No test can prove native QoR without an actual original Vitis run",
            "Paper novelty is unverified until independent prior-art comparison",
        ],
    }


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout",required=True)
    p.add_argument("--with-native", action="store_true")
    p.add_argument("--vitis-executable", default="vitis_hls")
    p.add_argument("--trained-mlp")
    p.add_argument("--trained-cnn")
    args=p.parse_args(argv)
    info=inspect(args.checkout,with_native=args.with_native,
                 native_executable=args.vitis_executable,
                 trained_mlp=args.trained_mlp,trained_cnn=args.trained_cnn)
    print(json.dumps(info,indent=2,sort_keys=True))
    # Native binary absence or requested model incompatibility is a useful
    # machine-readable hard stop for the user, not a silently skipped study.
    if not info["cpu_paper_series_ready"]:
        return 2
    if args.with_native and not info["native_execution_environment_ready_to_attempt"]:
        return 3
    if any(v["requested"] and not v["ready"]
           for v in info["trained_assets"].values()):
        return 4
    return 0


if __name__=="__main__":
    sys.exit(main())
