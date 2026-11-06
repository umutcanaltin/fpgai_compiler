"""Curated public documentation links must not point to excluded research files."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_public_docs as sut


def test_relative_target_resolution_excludes_external_web_and_anchors():
    assert sut.target_of("docs/README.md","../README.md")=="README.md"
    assert sut.target_of("examples/README.md","../docs/USING_FPGAI.md")==(
        "docs/USING_FPGAI.md")
    assert sut.target_of("docs/README.md","../paper/compiler_ir/README.md")==(
        "paper/compiler_ir/README.md")
    assert sut.target_of("docs/README.md","https://github.com/org/repo") is None
    assert sut.target_of("docs/README.md","#architecture") is None
    assert sut.target_of("README.md","mailto:maintainer@example.org") is None
    assert sut.target_of("docs/README.md","../../outside.md")==(
        "!outside-release")


def test_synthetic_source_docs_fail_on_unlisted_research_reference(
    tmp_path,monkeypatch
):
    (tmp_path/"docs").mkdir()
    (tmp_path/"README.md").write_text(
        "# Public repo\n\n[Read the manual](docs/README.md)\n")
    (tmp_path/"docs/README.md").write_text(
        "# Manual\n\n[Back](../README.md)\n"
        "[Internal notes](../paper/compiler_ir/SPRINTS.md)\n")
    monkeypatch.setattr(sut,"plan",lambda root,profile:{
        "selected_paths":["README.md","docs/README.md"],
    })
    out=sut.audit_source_links(tmp_path,"compiler")
    assert out["all_local_public_links_resolve"] is False
    assert out["public_release_authorized"] is False
    assert out["checked_local_markdown_links"]==3
    assert out["unresolved_public_links"]==[{
        "source":"docs/README.md",
        "target":"paper/compiler_ir/SPRINTS.md",
        "line":4,
    }]


def test_synthetic_source_docs_success_requires_whitelisted_targets(
    tmp_path,monkeypatch
):
    (tmp_path/"docs").mkdir()
    (tmp_path/"README.md").write_text(
        "[Guide](docs/README.md) [Web](https://example.com)\n")
    (tmp_path/"docs/README.md").write_text("[Start](../README.md)\n")
    monkeypatch.setattr(sut,"plan",lambda root,profile:{
        "selected_paths":["README.md","docs/README.md"],
    })
    out=sut.audit_source_links(tmp_path,"compiler")
    assert out["all_local_public_links_resolve"] is True
    assert out["checked_local_markdown_links"]==2
    assert out["unresolved_public_links"]==[]


def _synthetic_git_free_preview(root):
    import json
    from scripts import prepare_public_preview as preview

    (root/"docs").mkdir(parents=True)
    (root/"README.md").write_text("[Guide](docs/README.md)\n")
    (root/"docs/README.md").write_text("[Start](../README.md)\n")
    (root/"PREVIEW_NOT_FOR_PUBLICATION.json").write_text(json.dumps({
        "schema":preview.SCHEMA,
        "profile":"compiler",
        "ready_to_publish":False,
        "selected_paths":["README.md","docs/README.md"],
    }))


def test_audit_staged_preview_without_git_history(tmp_path,monkeypatch):
    root=tmp_path/"git-free"
    _synthetic_git_free_preview(root)
    def no_git(*args,**kwargs):
        raise AssertionError("Staged source checker must never consult parent Git history")
    monkeypatch.setattr(sut,"plan",no_git)
    report=sut.audit_source_links(root,"compiler")
    assert report["all_local_public_links_resolve"] is True
    assert report["checked_local_markdown_links"]==2
    assert report["public_release_authorized"] is False


def test_audit_staged_preview_rejects_unlisted_file(tmp_path,monkeypatch):
    root=tmp_path/"git-free"
    _synthetic_git_free_preview(root)
    monkeypatch.setattr(sut,"plan",lambda *args,**kwargs:None)
    (root/"paper").mkdir()
    (root/"paper"/"internal_sprint.md").write_text("private notebook")
    with pytest.raises(ValueError,match="inventory mismatch"):
        sut.audit_source_links(root,"compiler")


def test_audit_staged_preview_rejects_deleted_listed_file(tmp_path):
    root=tmp_path/"git-free"
    _synthetic_git_free_preview(root)
    (root/"docs/README.md").unlink()
    with pytest.raises(ValueError,match="inventory mismatch"):
        sut.audit_source_links(root,"compiler")


def test_audit_staged_preview_rejects_forged_approval(tmp_path):
    import json
    root=tmp_path/"git-free"
    _synthetic_git_free_preview(root)
    file=root/"PREVIEW_NOT_FOR_PUBLICATION.json"
    data=json.loads(file.read_text())
    data["ready_to_publish"]=True
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="Invalid staged preview"):
        sut.audit_source_links(root,"compiler")


def test_direct_script_and_module_invocations_match_from_checkout():
    """Regression: Makefile release-links used to fail with ModuleNotFoundError."""
    import json
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    commands = (
        [sys.executable, "scripts/check_public_docs.py", "--profile", "compiler"],
        [sys.executable, "-m", "scripts.check_public_docs", "--profile", "compiler"],
    )
    receipts = []
    for command in commands:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True,
            timeout=25, check=False,
        )
        assert result.returncode == 0, (
            f"Public docs entrypoint failed: {command}\n{result.stderr}\n{result.stdout}"
        )
        receipt = json.loads(result.stdout)
        assert receipt["all_local_public_links_resolve"] is True
        assert receipt["public_release_authorized"] is False
        receipts.append(receipt)
    assert receipts[0] == receipts[1]
