"""Dependency failures must be actionable and fail before native side effects.

This module intentionally does NOT skip when torch/ONNX/Gymnasium are absent.
The full CPU suite must exercise the missing-PyTorch path as well.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rl_environment import hardware_agent


def test_hardware_gnn_reports_rl_extra_when_torch_is_missing(monkeypatch):
    monkeypatch.setattr(hardware_agent, "torch", None)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[rl,onnx\]'") as exc:
        hardware_agent.HardwareGraphTrainer()
    assert "--no-deps" in str(exc.value)
    assert "hardware GNN" in str(exc.value)


def test_hardware_agent_cli_preflights_torch_before_model_or_native(
        monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setattr(hardware_agent, "torch", None)
    root = tmp_path / "must-not-create"
    result = hardware_agent.main([
        "--onnx", str(tmp_path / "missing.onnx"),
        "--samples", str(tmp_path / "missing.npy"),
        "--part", "xc7z020clg400-1",
        "--resource-caps", "53200", "106400", "220", "280", "0",
        "--max-absolute-error", "0.1",
        "--output-root", str(root),
    ])
    assert result == 2
    captured = capsys.readouterr()
    assert "Hardware GNN stopped: RuntimeError" in captured.err
    assert "pip install -e '.[rl,onnx]'" in captured.err
    assert not root.exists()
