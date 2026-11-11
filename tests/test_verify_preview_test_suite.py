"""Safety and evidence-scope regression for curated isolated preview pytest."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from shutil import copytree
import subprocess

import pytest

from scripts import prepare_public_preview as staged
from scripts import verify_preview_test_suite as sut


def _case(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    build = root / "build"
    build.mkdir(parents=True)
    monkeypatch.setattr(sut, "ROOT", root)
    preview = build / "staged-preview"
    preview.mkdir()
    required = {
        "README.md", "pyproject.toml", "src/fpgai_research/cli.py",
        "examples/make_toy_onnx.py", "examples/make_neural_network_onnx.py",
        "examples/compile_neural_network.py",
        *staged.PUBLIC_TESTS,
    }
    for path in required:
        target = preview / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic-only\n", encoding="utf-8")
    manifest = preview / "PREVIEW_NOT_FOR_PUBLICATION.json"
    manifest.write_text(json.dumps({
        "schema": staged.SCHEMA,
        "profile": "compiler",
        "ready_to_publish": False,
        "selected_paths": sorted(required),
        "selected_file_count": len(required),
    }), encoding="utf-8")
    workspace = build / "isolated"
    workspace.mkdir()
    copytree(preview, workspace / "source")
    python = workspace / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("synthetic test interpreter\n")
    installed_module = (
        workspace / "venv/lib/python3.12/site-packages/fpgai_research/__init__.py"
    )
    installed_module.parent.mkdir(parents=True)
    installed_module.write_text("synthetic package\n")
    (workspace / "installed_preview_smoke_receipt.json").write_text(json.dumps({
        "schema": "fpgai-research/installed-preview-smoke/v1",
        "status": "INSTALLED_PREVIEW_SOURCE_SMOKE_VERIFIED",
        "installed_package_tested": True,
        "ready_to_publish": False,
        "workspace": str(workspace),
        "preview": str(preview),
        "preview_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "isolated_package_module": str(installed_module),
    }))
    return workspace


def test_plan_has_no_pytest_or_vendor_side_effects(tmp_path, monkeypatch):
    workspace = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (
        _ for _ in ()).throw(AssertionError("Plan may not run processes")))
    result = sut.plan(workspace)
    assert result["source_copy_matches_stage"] is True
    assert result["curated_preview_pytest_executed"] is False
    assert result["ready_to_publish"] is False
    assert result["checkout_only_test_deselected"] == sut.GIT_CHECKOUT_TEST
    assert set(result["curated_test_modules"]) == staged.PUBLIC_TESTS
    assert not (workspace / sut.RECEIPT_FILE).exists()


def test_selected_source_mutation_and_unqualified_receipt_fail(tmp_path, monkeypatch):
    workspace = _case(tmp_path, monkeypatch)
    (workspace / "source" / "README.md").write_text("tampered\n")
    with pytest.raises(ValueError, match="differs from stage"):
        sut.plan(workspace)
    (workspace / "source" / "README.md").write_text("synthetic-only\n")
    file = workspace / "installed_preview_smoke_receipt.json"
    report = json.loads(file.read_text())
    report["installed_package_tested"] = False
    file.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="receipt"):
        sut.plan(workspace)


def test_curated_execution_is_explicit_create_only_and_keeps_one_deselection(
        tmp_path, monkeypatch):
    workspace = _case(tmp_path, monkeypatch)
    calls = []

    def pretend_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "14 passed, 1 deselected in 1.2s\n", "")
    monkeypatch.setattr(subprocess, "run", pretend_run)
    result = sut.execute(workspace)
    assert result["status"] == "CURATED_PREVIEW_SOURCE_TESTS_PASSED"
    assert result["curated_passed"] == 14
    assert result["checkout_only_tests_deselected"] == 1
    assert result["full_development_suite_executed"] is False
    assert result["ready_to_publish"] is False
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == [
        str(workspace / "venv/bin/python"), "-m", "pytest"]
    assert "--deselect=" + sut.GIT_CHECKOUT_TEST in command
    assert set(sut.MODULES) <= set(command)
    assert kwargs["cwd"] == workspace / "source"
    assert "PYTHONPATH" not in kwargs["env"]
    assert (workspace / sut.RECEIPT_FILE).is_file()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.execute(workspace)


def test_executed_pytest_without_required_deselection_is_not_certified(
        tmp_path, monkeypatch):
    workspace = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 0, "14 passed\n", ""))
    with pytest.raises(AssertionError, match="one Git-only"):
        sut.execute(workspace)
    assert not (workspace / sut.RECEIPT_FILE).exists()


def test_cli_does_not_execute_without_explicit_switch(tmp_path, monkeypatch, capsys):
    workspace = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(sut, "execute", lambda _: (
        _ for _ in ()).throw(AssertionError("Execution requires --execute")))
    sut.main(["--workspace", str(workspace), "--plan"])
    assert '"curated_preview_pytest_executed": false' in capsys.readouterr().out
