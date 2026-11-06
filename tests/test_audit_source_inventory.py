"""Repository source inventory: static module owner/import graph and SHA tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import audit_source_inventory as sut


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _sample(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.org")
    _git(root, "config", "user.name", "Test Fixture")
    files = {
        "src/fpgai_research/__init__.py": "",
        "src/fpgai_research/model_ir/__init__.py": "",
        "src/fpgai_research/model_ir/contracts.py":
            "class Contract:\n    pass\n\ndef check():\n    return 1\n",
        "src/fpgai_research/compiler/main.py":
            "from fpgai_research.model_ir.contracts import Contract\n"
            "def run():\n    return Contract()\n",
        "src/fpgai_research/compiler/relative.py":
            "from .main import run\n"
            "from ..model_ir import contracts\n",
        "tests/test_compiler.py":
            "from fpgai_research.compiler.main import run\n"
            "def test_run():\n    assert run()\n",
        "docs/README.md": "A synthetic research doc.\n",
    }
    for path, source in files.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(source)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "synthetic files for inventory")
    return root, files


def test_tracked_file_owner_sha_and_static_internal_importers(tmp_path):
    root, files = _sample(tmp_path)
    report = sut.inspect(root)
    assert report["schema"] == sut.SCHEMA
    assert report["working_tree_clean"] is True
    assert report["file_count"] == len(files)
    assert report["python_parse_errors"] == []
    by_name = {item["path"]: item for item in report["entries"]}
    source = "src/fpgai_research/model_ir/contracts.py"
    main = "src/fpgai_research/compiler/main.py"
    relative = "src/fpgai_research/compiler/relative.py"
    test = "tests/test_compiler.py"
    assert by_name[source]["owner"] == "compiler/model_ir"
    assert by_name[source]["top_level_definitions"] == ["Contract", "check"]
    assert main in by_name[source]["direct_importers"]
    assert relative in by_name[source]["direct_importers"]
    assert main in by_name[test]["internal_imports"]
    assert main in by_name[relative]["internal_imports"]
    assert len(by_name[source]["sha256"]) == 64
    assert not by_name[source]["parse_error"]
    assert "docs/README.md" in by_name


def test_modified_worktree_is_identified_not_claimed_clean(tmp_path):
    root, _ = _sample(tmp_path)
    (root / "docs/README.md").write_text("A changed doc.\n")
    report = sut.inspect(root)
    assert report["working_tree_clean"] is False
    assert report["working_tree_has_changes_not_identical_to_head"] is True


def test_create_only_inventory_is_checksum_pinned(tmp_path):
    root, _ = _sample(tmp_path)
    report = sut.inspect(root)
    output = tmp_path / "new_inventory"
    receipt = sut.export_create_only(report, output=output)
    assert len(receipt["inventory_sha256"]) == 64
    assert json.loads((output / "inventory.json").read_text())["schema"] == sut.SCHEMA
    with pytest.raises(FileExistsError, match="create-only"):
        sut.export_create_only(report, output=output)
    assert sorted(p.name for p in output.iterdir()) == [
        "inventory.json", "sha256.json"]


def test_syntax_error_recorded_without_hiding_or_deleting_file(tmp_path):
    root, _ = _sample(tmp_path)
    p = root / "src/fpgai_research/compiler/main.py"
    p.write_text("def bad(\n")
    report = sut.inspect(root)
    entry = next(e for e in report["entries"] if e["path"] == (
        "src/fpgai_research/compiler/main.py"))
    assert entry["parse_error"].startswith("SyntaxError:")
    assert report["python_parse_errors"] == [entry["path"]]
    assert p.read_text() == "def bad(\n"


def test_symlinked_tracked_source_fails_closed(tmp_path):
    root, _ = _sample(tmp_path)
    target = root / "docs/README.md"
    target.unlink()
    target.symlink_to(root / "tests/test_compiler.py")
    with pytest.raises(ValueError, match="symlinked"):
        sut.inspect(root)


def test_source_inventory_cannot_run_outside_git(tmp_path):
    with pytest.raises(ValueError, match="Git checkout"):
        sut.inspect(tmp_path)
