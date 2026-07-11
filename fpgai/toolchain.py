"""Xilinx toolchain discovery helpers for FPGAI.

This module centralizes Vitis HLS/Vivado executable and settings64.sh
resolution so compiler, HLS, and Vivado bridge flows behave consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import os
import re
import shlex
import shutil


@dataclass(frozen=True)
class XilinxToolResolution:
    tool: str
    executable: str
    settings64: Optional[str]
    uses_settings64: bool
    executable_resolved: Optional[str]
    settings64_resolved: Optional[str]
    launcher: str
    resolved_launcher: Optional[str]
    source: str
    searched_roots: List[str]
    searched_candidates: List[str]

    @property
    def available(self) -> bool:
        if self.uses_settings64:
            return self.resolved_launcher is not None and self.settings64_resolved is not None
        return self.executable_resolved is not None

    def as_dict(self) -> Dict[str, object]:
        return {
            "tool": self.tool,
            "executable": self.executable,
            "settings64": self.settings64,
            "uses_settings64": self.uses_settings64,
            "executable_resolved": self.executable_resolved,
            "settings64_resolved": self.settings64_resolved,
            "launcher": self.launcher,
            "resolved_launcher": self.resolved_launcher,
            "available": self.available,
            "path_available_without_settings": shutil.which(self.executable) is not None,
            "source": self.source,
            "searched_roots": self.searched_roots,
            "searched_candidates": self.searched_candidates,
        }


_TOOL_DEFAULTS = {
    "vitis_hls": {
        "exe": "vitis_hls",
        "env_prefixes": ["VITIS_HLS", "XILINX_HLS"],
        "settings_envs": ["FPGAI_VITIS_HLS_SETTINGS64", "VITIS_HLS_SETTINGS64", "XILINX_HLS_SETTINGS64"],
        "exe_envs": ["FPGAI_VITIS_HLS_EXECUTABLE", "VITIS_HLS_EXECUTABLE", "FPGAI_VITIS_HLS", "VITIS_HLS"],
        "install_dirs": ["Vitis_HLS", "Vitis"],
    },
    "vivado": {
        "exe": "vivado",
        "env_prefixes": ["VIVADO", "XILINX_VIVADO"],
        "settings_envs": ["FPGAI_VIVADO_SETTINGS64", "VIVADO_SETTINGS64", "XILINX_VIVADO_SETTINGS64"],
        "exe_envs": ["FPGAI_VIVADO_EXECUTABLE", "VIVADO_EXECUTABLE", "FPGAI_VIVADO", "VIVADO"],
        "install_dirs": ["Vivado", "Vitis"],
    },
}


def _tool_spec(tool: str) -> Dict[str, object]:
    key = str(tool).lower().replace("-", "_")
    if key not in _TOOL_DEFAULTS:
        raise ValueError(f"Unsupported Xilinx tool: {tool!r}")
    return _TOOL_DEFAULTS[key]


def _split_paths(value: str | None) -> List[Path]:
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def _existing_file(path: str | Path | None) -> Optional[Path]:
    if not path:
        return None
    p = Path(str(path)).expanduser()
    return p.resolve() if p.exists() and p.is_file() else None


def _version_key(path: Path) -> Tuple[int, ...]:
    nums = [int(x) for x in re.findall(r"\d+", path.as_posix())]
    return tuple(nums) if nums else (0,)


def _existing_unique(paths: Iterable[Path], seen: set[Path] | None = None) -> List[Path]:
    """Return existing files in caller-defined priority order.

    Tool discovery must never let a global version/path sort make /tools/Xilinx
    override an explicit YAML/env/test root. The caller provides priority groups
    (explicit settings, environment install roots, common roots); this helper only
    filters missing/duplicate files while preserving that priority.
    """
    found: List[Path] = []
    seen = seen if seen is not None else set()
    for path in paths:
        try:
            p = path.expanduser()
            if not p.exists() or not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            found.append(rp)
        except OSError:
            continue
    return found


def _existing_sorted_by_version(paths: Iterable[Path], seen: set[Path] | None = None) -> List[Path]:
    """Return existing files sorted only within one priority group/root."""
    return sorted(_existing_unique(paths, seen), key=lambda p: (_version_key(p), p.as_posix()), reverse=True)


def xilinx_search_roots(env: Mapping[str, str] | None = None) -> List[Path]:
    """Return no implicit search roots.

    FPGAI intentionally does not scan common Xilinx installation directories.
    The toolchain path is a compiler input and must be provided explicitly in
    YAML, for example:

        toolchain:
          vitis_hls:
            settings64: /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
            executable: vitis_hls
          vivado:
            settings64: /tools/Xilinx/Vivado/2023.2/settings64.sh
            executable: vivado

    If no settings64 file is provided, FPGAI runs the configured executable as
    given and lets the subprocess result report whether it was available on
    PATH. This keeps toolchain resolution deterministic and traceable.
    """
    return []


def _settings_candidates(tool: str, roots: Sequence[Path], env: Mapping[str, str]) -> List[Path]:
    """Compatibility helper: implicit settings64 discovery is disabled."""
    return []


def _exe_candidates(tool: str, roots: Sequence[Path], env: Mapping[str, str]) -> List[Path]:
    """Compatibility helper: implicit executable discovery is disabled."""
    return []


def _source_label(explicit_exe: bool, explicit_settings: bool, settings: Optional[Path], exe_path: Optional[Path]) -> str:
    if explicit_settings:
        return "yaml_settings64"
    if explicit_exe:
        return "configured_executable"
    if exe_path:
        return "path_executable"
    return "path_default"


def resolve_xilinx_tool(
    tool: str,
    *,
    executable: str | None = None,
    settings64: str | None = None,
    env: Mapping[str, str] | None = None,
) -> XilinxToolResolution:
    """Resolve a Xilinx command without scanning the filesystem.

    FPGAI does not guess Xilinx installation locations. Users must provide
    settings64 in YAML when they want FPGAI to source Vivado/Vitis before
    running the tool. Without settings64, the resolver uses the configured
    executable name/path only.
    """
    env = env or os.environ
    spec = _tool_spec(tool)
    tool_key = str(tool).lower().replace("-", "_")
    default_exe = str(spec["exe"])

    explicit_settings = bool(settings64)
    explicit_exe = bool(executable and str(executable) != default_exe)

    settings_path = Path(str(settings64)).expanduser() if explicit_settings else None
    exe_value = str(executable or default_exe)
    exe_resolved = shutil.which(exe_value)
    exe_path = Path(exe_resolved).resolve() if exe_resolved else _existing_file(exe_value)

    if settings_path is not None:
        launcher = "bash"
        resolved_launcher = shutil.which("bash")
        executable_for_shell = exe_value if exe_value else default_exe
        return XilinxToolResolution(
            tool=tool_key,
            executable=executable_for_shell,
            settings64=settings_path.as_posix(),
            uses_settings64=True,
            executable_resolved=str(exe_path) if exe_path else exe_resolved,
            settings64_resolved=settings_path.resolve().as_posix() if settings_path.exists() else None,
            launcher=launcher,
            resolved_launcher=resolved_launcher,
            source=_source_label(explicit_exe, explicit_settings, settings_path, exe_path),
            searched_roots=[],
            searched_candidates=[],
        )

    executable_final = str(exe_path) if exe_path and not executable else exe_value
    return XilinxToolResolution(
        tool=tool_key,
        executable=executable_final,
        settings64=None,
        uses_settings64=False,
        executable_resolved=str(exe_path) if exe_path else exe_resolved,
        settings64_resolved=None,
        launcher=executable_final,
        resolved_launcher=str(exe_path) if exe_path else exe_resolved,
        source=_source_label(explicit_exe, explicit_settings, None, exe_path),
        searched_roots=[],
        searched_candidates=[],
    )


def build_xilinx_tool_command(
    tool: str,
    args: Sequence[str],
    *,
    executable: str | None = None,
    settings64: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Tuple[List[str], Dict[str, object]]:
    """Return subprocess args and resolution metadata for a Xilinx tool."""
    resolution = resolve_xilinx_tool(tool, executable=executable, settings64=settings64, env=env)
    if resolution.uses_settings64:
        shell_cmd = "source {settings} && exec {exe} {args}".format(
            settings=shlex.quote(str(resolution.settings64)),
            exe=shlex.quote(str(resolution.executable)),
            args=" ".join(shlex.quote(str(a)) for a in args),
        )
        cmd = ["bash", "-lc", shell_cmd]
        command = f"bash -lc {shlex.quote(shell_cmd)}"
    else:
        cmd = [str(resolution.executable)] + [str(a) for a in args]
        command = " ".join(shlex.quote(str(a)) for a in cmd)

    info = resolution.as_dict()
    info["command"] = command
    return cmd, info
