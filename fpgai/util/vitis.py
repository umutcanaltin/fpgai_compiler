from __future__ import annotations

from pathlib import Path
import os
import subprocess
from typing import Dict, Optional


def resolve_vitis_env(vitis_cfg: Dict) -> Dict[str, str]:
    """
    Returns an environment dict with Vitis HLS set up.
    Does NOT mutate os.environ.
    """

    root = vitis_cfg.get("root")
    if not root:
        raise RuntimeError("toolchain.vitis_hls.root is not set")

    root = Path(root)
    settings = vitis_cfg.get("settings_script", "settings64.sh")

    settings_path = root / settings
    if not settings_path.exists():
        raise FileNotFoundError(f"Vitis settings script not found: {settings_path}")

    # Source settings script and capture env
    cmd = [
        "bash",
        "-c",
        f"source {settings_path} >/dev/null && env",
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    env = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v

    return env


def vitis_hls_cmd(vitis_cfg: Dict) -> str:
    root = vitis_cfg.get("root")
    if not root:
        return "vitis_hls"  # fallback (PATH)

    return str(Path(root) / "bin" / "vitis_hls")
