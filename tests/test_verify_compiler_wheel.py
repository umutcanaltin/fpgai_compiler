"""Synthetic, non-network regression for nonpublishable compiler wheel gate."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import verify_compiler_wheel as gate


def _wheel(path: Path, *, omitted=(), extra=None):
    modules = sorted(gate.REQUIRED_MODULES)
    with ZipFile(path, "w") as z:
        for module in modules:
            if module not in omitted:
                z.writestr(module, "# synthetic\n")
        root = "fpgai_research-0.3.0.dev0.dist-info"
        z.writestr(root + "/METADATA",
                   "Metadata-Version: 2.3\nName: fpgai-research\nVersion: 0.3.0.dev0\n")
        z.writestr(root + "/WHEEL",
                   "Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\n"
                   "Tag: py3-none-any\n")
        z.writestr(root + "/entry_points.txt",
                   "[console_scripts]\nfpgai-research = fpgai_research.cli:main\n")
        z.writestr(root + "/RECORD", "synthetic test\n")
        if extra is not None:
            z.writestr(extra, "private payload")
    return path


def _selected():
    return {"src/" + name for name in gate.REQUIRED_MODULES}


def test_pure_python_wheel_members_match_curated_source_exactly(tmp_path):
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl")
    report = gate.inspect_wheel(wheel, _selected())
    assert report["wheel_unexpected_members"] == 0
    assert report["wheel_all_staged_modules_present"] is True
    assert report["wheel_python_module_count"] == len(gate.REQUIRED_MODULES)
    assert report["wheel_sha256"]


@pytest.mark.parametrize("extra", [
    "paper/internal_notes.md", "tests/test_private.py",
    "fpgai_research/private_weights.onnx",
    "../outside.txt", "rl_environment/secret.py",
])
def test_wheel_rejects_unlisted_and_unsafe_members(tmp_path, extra):
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl",
                   extra=extra)
    with pytest.raises(ValueError, match="Unsafe|Unallowlisted|non-Python"):
        gate.inspect_wheel(wheel, _selected())


def test_wheel_missing_staged_module_fails_closed(tmp_path):
    name = next(iter(gate.REQUIRED_MODULES))
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl",
                   omitted=(name,))
    with pytest.raises(ValueError, match="differ"):
        gate.inspect_wheel(wheel, _selected())


def test_plan_only_requires_curated_verification_and_fresh_output(
        tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    workspace = root / "build" / "installed"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(gate, "ROOT", root)
    stage = "original manifest digest"
    monkeypatch.setattr(gate, "curated_plan", lambda w: {
        "original_preview_manifest_sha256": stage})
    (workspace / gate.RECEIPT_FILE).write_text(json.dumps({
        "schema": "fpgai-research/curated-installed-preview-tests/v1",
        "status": "CURATED_PREVIEW_SOURCE_TESTS_PASSED",
        "curated_preview_pytest_executed": True,
        "curated_passed": 65,
        "curated_skipped": 0,
        "checkout_only_tests_deselected": 1,
        "ready_to_publish": False,
        "workspace": str(workspace),
        "original_preview_manifest_sha256": stage,
    }))
    output = root / "build" / "new-wheel"
    result = gate.plan(workspace, output)
    assert result["wheel_built"] is False
    assert result["ready_to_publish"] is False
    assert result["curated_preview_passed"] == 65
    assert not output.exists()
    output.mkdir()
    with pytest.raises(ValueError, match="create-only"):
        gate.plan(workspace, output)
    (workspace / gate.RECEIPT_FILE).unlink()
    with pytest.raises(ValueError, match="receipt"):
        gate.plan(workspace, root / "build" / "another")


def test_cli_requires_explicit_execute_or_plan(tmp_path):
    with pytest.raises(SystemExit):
        gate.main(["--workspace", str(tmp_path),
                   "--output", str(tmp_path / "new")])
