"""Side-effect-minimal, source-revision-aware paper environment manifest.

Inspection never runs synthesis, implementation, deployment or GitHub Actions.
Optional tool-version probes execute only the resolved binary with -version.
"""
from __future__ import annotations

import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

_PACKAGES = ('numpy', 'onnx', 'onnxruntime', 'torch', 'torchvision', 'hls4ml', 'pytest')
_TOOLS = ('g++', 'vitis_hls', 'vivado')


def _read_git_head_without_subprocess(root: Path) -> str | None:
    """Read HEAD from an ordinary checkout or linked worktree, without git.

    Also supports packed refs and detached HEAD. The manifest is only a
    provenance hint; the qualification runner independently checks Git state.
    """
    marker = root / '.git'
    try:
        if marker.is_dir():
            git_dir = marker
        elif marker.is_file():
            pointer = marker.read_text(encoding='utf-8').strip()
            if not pointer.startswith('gitdir: '):
                return None
            git_dir = (root / pointer[len('gitdir: '):].strip()).resolve()
        else:
            return None
        common_dir = git_dir
        common_file = git_dir / 'commondir'
        if common_file.is_file():
            common_dir = (git_dir / common_file.read_text(encoding='utf-8').strip()).resolve()
        head = (git_dir / 'HEAD').read_text(encoding='utf-8').strip()
        if re.fullmatch(r'[0-9a-f]{40}', head):
            return head
        if not head.startswith('ref: '):
            return None
        ref = head[len('ref: '):].strip()
        if (not re.fullmatch(r'refs/[A-Za-z0-9._/-]+', ref)
                or '..' in ref.split('/') or '//' in ref):
            return None
        for directory in (git_dir, common_dir):
            loose = directory / ref
            if loose.is_file():
                value = loose.read_text(encoding='utf-8').strip()
                if re.fullmatch(r'[0-9a-f]{40}', value):
                    return value
        for directory in (git_dir, common_dir):
            packed = directory / 'packed-refs'
            if packed.is_file():
                for line in packed.read_text(encoding='utf-8').splitlines():
                    value, separator, name = line.partition(' ')
                    if separator and name == ref and re.fullmatch(r'[0-9a-f]{40}', value):
                        return value
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def inspect_environment(*, repo_directory: str | Path = '.', probe_versions: bool = False) -> dict:
    if type(probe_versions) is not bool:
        raise ValueError('probe_versions must be boolean')
    root = Path(repo_directory).resolve(strict=True)
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    binaries = {}
    for name in _TOOLS:
        executable = shutil.which(name)
        item = {'available': executable is not None, 'path': str(Path(executable).resolve()) if executable else None,
                'version_output': None, 'version_probe_succeeded': None}
        if executable and probe_versions:
            try:
                argument = '--version' if name == 'g++' else '-version'
                result = subprocess.run([executable, argument], shell=False, cwd=root,
                                        capture_output=True, text=True, check=False, timeout=20)
                item['version_probe_succeeded'] = result.returncode == 0
                item['version_output'] = (result.stdout or result.stderr)[:4096]
            except (OSError, subprocess.TimeoutExpired) as exc:
                item['version_probe_succeeded'] = False
                item['version_output'] = f'{type(exc).__name__}: {exc}'
        binaries[name] = item
    git_head = _read_git_head_without_subprocess(root)
    return {'schema': 'fpgai-research/reproduction-environment/v1',
            'research_repo_revision': git_head,
            'python_version': sys.version.split()[0],
            'platform': '-'.join((platform.system(), platform.release(), platform.machine())),
            'packages': packages,
            'executables': binaries,
            'native_tools_ready': all(binaries[k]['available'] for k in ('vitis_hls', 'vivado')),
            'vendor_tools_executed': False,
            'version_probes_only': probe_versions,
            'note': 'Availability and -version probes do not prove licensing, simulation, synthesis or place-and-route.'}
