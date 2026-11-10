"""Source-only safety checks for the opt-in isolated preview installer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import prepare_public_preview
from scripts import verify_compiler_preview as sut


REQUIRED = (
    "README.md",
    "pyproject.toml",
    "src/fpgai_research/cli.py",
    "examples/make_toy_onnx.py",
    "examples/make_neural_network_onnx.py",
    "examples/compile_neural_network.py",
)


def _staged(tmp_path: Path) -> Path:
    preview = tmp_path / "preview"
    preview.mkdir()
    for name in REQUIRED:
        file = preview / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("synthetic fixture\n")
    (preview / "PREVIEW_NOT_FOR_PUBLICATION.json").write_text(
        json.dumps({
            "schema": prepare_public_preview.SCHEMA,
            "profile": "compiler",
            "ready_to_publish": False,
            "selected_paths": list(REQUIRED),
            "selected_file_count": len(REQUIRED),
        })
    )
    return preview


def test_plan_validates_staged_manifest_without_installing(tmp_path, monkeypatch):
    preview = _staged(tmp_path)
    monkeypatch.setattr(sut, "execute", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("No installation in plan")))
    report = sut.inspect(preview)
    assert report["selected_file_count"] == len(REQUIRED)
    assert report["installed_package_tested"] is False
    assert report["ready_to_publish"] is False
    sut.main(["--preview", str(preview), "--plan"])


def test_broken_manifest_and_extra_file_rejected_before_execution(tmp_path):
    preview = _staged(tmp_path)
    (preview / "unexpected-private.txt").write_text("never allowed")
    with pytest.raises(ValueError, match="inventory mismatch"):
        sut.inspect(preview)
    (preview / "unexpected-private.txt").unlink()
    doc = preview / "PREVIEW_NOT_FOR_PUBLICATION.json"
    manifest = json.loads(doc.read_text())
    manifest["ready_to_publish"] = True
    doc.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonpublishable"):
        sut.inspect(preview)


def test_create_only_workspace_must_be_outside_preview_under_build(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    build = root / "build"
    build.mkdir(parents=True)
    monkeypatch.setattr(sut, "ROOT", root)
    preview = build / "original-preview"
    preview.mkdir()
    output = build / "installed-preview-check"
    assert sut.require_new_workspace(preview, output) == output
    with pytest.raises(ValueError, match="create-only"):
        sut.require_new_workspace(preview, preview)
    with pytest.raises(ValueError, match="create-only"):
        sut.require_new_workspace(preview, root / "not-under-build")
    output.mkdir()
    with pytest.raises(ValueError, match="create-only"):
        sut.require_new_workspace(preview, output)


def test_execute_requires_explicit_flag_and_destination(tmp_path):
    preview = _staged(tmp_path)
    with pytest.raises(SystemExit):
        sut.main(["--preview", str(preview), "--execute"])
    with pytest.raises(SystemExit):
        sut.main(["--preview", str(preview), "--plan",
                  "--output", str(tmp_path / "unwanted")])
    assert not (tmp_path / "unwanted").exists()
