"""Regression for the installed RL CLI, which pytest.pythonpath cannot prove.

An editable or wheel install must distribute BOTH src/fpgai_research and the
root-level rl_environment. The original failure only happened outside pytest
when the virtualenv executable imported an undistributed RL package.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_setuptools_discovers_both_root_rl_and_src_compiler_packages():
    setuptools = pytest.importorskip("setuptools")
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[tool.setuptools.package-dir]' in config
    assert 'fpgai_research = "src/fpgai_research"' in config
    assert 'rl_environment = "rl_environment"' in config
    assert 'where = ["src", "."]' in config
    assert 'include = ["fpgai_research*", "rl_environment*"]' in config
    assert 'fpgai-rl-hardware-space = "rl_environment.hardware_space:main"' in config
    packages = set(
        setuptools.find_packages(where="src", include=("fpgai_research*",))
    ) | set(
        setuptools.find_packages(where=".", include=("rl_environment*",))
    )
    assert "fpgai_research" in packages
    assert "fpgai_research.toolchains" in packages
    assert "rl_environment" in packages
    assert "rl_environment.agents" in packages
    assert "rl_environment.environments" in packages
    assert "rl_environment.evaluation" in packages
    assert "rl_environment.experiments" in packages
    assert "rl_environment.infrastructure" in packages


def test_installed_rl_console_help_runs_outside_repo_without_pythonpath(tmp_path):
    """Exercise actual venv entry point; not the pytest working-directory import."""
    exe = Path(sysconfig.get_path("scripts")) / (
        "fpgai-rl-hardware-space.exe" if os.name == "nt"
        else "fpgai-rl-hardware-space"
    )
    if not exe.is_file():
        pytest.skip(
            "Editable/wheel install not present; install with "
            "'python -m pip install --no-deps -e .' for installed CLI test"
        )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    assert tmp_path.resolve() != ROOT.resolve()
    process = subprocess.run(
        [str(exe), "--help"], cwd=tmp_path, env=env,
        text=True, capture_output=True, timeout=30,
        check=False,
    )
    assert process.returncode == 0, (
        "Installed CLI not importable outside repo\n"
        + process.stdout + "\n" + process.stderr
    )
    assert "--onnx" in process.stdout
    assert "--part" in process.stdout
    assert "--max-parallel-units" in process.stdout


def test_isolated_installed_rl_module_imports_without_repository_root(tmp_path):
    """Isolated Python hides cwd and PYTHONPATH; editable mapping must work."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    program = (
        "from rl_environment.hardware_space import initial_state;"
        "from fpgai_research.architecture.resolver import TargetLimits;"
        "print('INSTALLED_RL_PACKAGE_IMPORT_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        check=False,
    )
    # The source-only workflow may run without installing the project at all;
    # that environment has no CLI. Keep this test strict whenever the CLI is
    # installed, rather than implicitly depending on pytest's sys.path.
    exe = Path(sysconfig.get_path("scripts")) / (
        "fpgai-rl-hardware-space.exe" if os.name == "nt"
        else "fpgai-rl-hardware-space"
    )
    if not exe.is_file():
        pytest.skip("Installed CLI not present; run editable-install preflight")
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "INSTALLED_RL_PACKAGE_IMPORT_OK" in proc.stdout
