"""No-vendor regression for refactored hardware-RL package imports."""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import pytest


MODULES = (
    ("hardware_space", "environments.hardware_space"),
    ("hardware_native", "infrastructure.hardware_native"),
    ("hardware_gym", "environments.hardware_gym"),
    ("hardware_agent", "agents.hardware_agent"),
    ("hardware_algorithms", "agents.hardware_algorithms"),
    ("hardware_campaign", "experiments.hardware_campaign"),
    ("hardware_checkpoint", "infrastructure.hardware_checkpoint"),
    ("hardware_compare", "evaluation.hardware_compare"),
)


@pytest.mark.parametrize("legacy,canonical", MODULES)
def test_hardware_legacy_and_canonical_import_share_module_identity(legacy, canonical):
    old = importlib.import_module("rl_environment." + legacy)
    new = importlib.import_module("rl_environment." + canonical)
    assert old is new
    assert sys.modules["rl_environment." + legacy] is new
    assert old.__name__ == "rl_environment." + canonical
    assert Path(new.__file__).name == canonical.rsplit(".", 1)[-1] + ".py"


def test_hardware_legacy_monkeypatch_updates_canonical_module(monkeypatch):
    old = importlib.import_module("rl_environment.hardware_campaign")
    canonical = importlib.import_module("rl_environment.experiments.hardware_campaign")
    marker = object()
    monkeypatch.setattr(old, "_source_revision", marker)
    assert canonical._source_revision is marker


@pytest.mark.parametrize("module", (
    "rl_environment.hardware_campaign",
    "rl_environment.experiments.hardware_campaign",
))
def test_hardware_legacy_and_canonical_module_cli_help_source_only(module):
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--tasks" in proc.stdout
    assert "--execute-native" in proc.stdout
