"""Release preview is nonpublishable, allowlisted and non-destructive."""
from __future__ import annotations

import json

import pytest

from scripts import prepare_public_preview as sut


def test_source_preview_excludes_internal_notes_native_workflows_and_assets():
    permitted=(
        "README.md",
        "CONTRIBUTING.md",
        "docs/USING_FPGAI.md",
        "examples/make_toy_onnx.py",
        "examples/make_neural_network_onnx.py",
        "examples/compile_neural_network.py",
        "examples/make_mlp_onnx.py",
        "examples/NETWORK_GALLERY.md",
        "src/fpgai_research/cli.py",
    )
    assert all(sut.allowed(path,profile="compiler") for path in permitted)
    denied=(
        "AGENTS.md","SCIENTIFIC_REVIEW_README.md",
        "paper/compiler_ir/RESEARCH_SPRINT_MASTER_PLAN.md",
        "paper/compiler_ir/REPRODUCIBILITY_LEDGER.md",
        "docs/maintainers/README_RESEARCH_CHRONOLOGY_20261104.md",
        ".github/workflows/compiler-ir-native-vitis.yml",
        ".git/config","build/study/summary.json",
        "src/fpgai_research/__pycache__/cache.pyc",
        "rl_environment/gym_env.py","tests/test_gym_env.py",
        "weights/private.pt","a/../README.md",
        "src/fpgai_research/hidden_model.onnx",
    )
    assert all(not sut.allowed(path,profile="compiler") for path in denied)
    assert sut.allowed("rl_environment/gym_env.py",profile="full-research")
    assert sut.allowed("tests/test_gym_env.py",profile="full-research")
    assert not sut.allowed("rl_environment/HANDOFF.md",profile="full-research")
    assert not sut.allowed("paper/compiler_ir/SPRINTS.md",profile="full-research")
    assert not sut.allowed("scripts/history_scan.py",profile="full-research")


def test_unknown_profile_rejected():
    with pytest.raises(ValueError,match="profile"):
        sut.allowed("README.md",profile="unknown")


def _fake_source(root,monkeypatch):
    files=set(sut.PUBLIC_FILES)
    files.update({
        "src/fpgai_research/__init__.py",
        "src/fpgai_research/model_ir/contracts.py",
        "rl_environment/agent.py",
        "paper/compiler_ir/EXPERIMENT_PLAN.md",
        ".github/workflows/compiler-ir-native-vitis.yml",
        "build/private.json","weights/model.pt",
        "tests/test_gym_graph_hls_contract.py",
        "experiments/compiler_ir/series.py",
    })
    for rel in files:
        path=root/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("only synthetic fixture\n")
    monkeypatch.setattr(sut,"tracked",lambda _:tuple(sorted(files)))
    return files


def test_create_only_preview_copies_only_public_compiler_files(tmp_path,monkeypatch):
    root=tmp_path/"source"
    root.mkdir()
    files=_fake_source(root,monkeypatch)
    report=sut.plan(root,"compiler")
    assert report["ready_to_publish"] is False
    assert report["redistribution_license_approved"] is False
    assert report["history_security_reviewed"] is False
    assert report["clean_room_tested"] is False
    assert report["selected_file_count"] < len(files)
    assert report["internal_paper_docs_included"] is False
    assert report["native_tool_workflow_included"] is False
    assert report["vendor_binaries_or_weights_included"] is False

    dest=tmp_path/"public-preview"
    result=sut.stage(root,dest,report)
    assert result["ready_to_publish"] is False
    assert (dest/"README.md").is_file()
    assert (dest/"examples/NETWORK_GALLERY.md").is_file()
    assert (dest/"examples/make_neural_network_onnx.py").is_file()
    assert (dest/"examples/compile_neural_network.py").is_file()
    assert (dest/"examples/make_mlp_onnx.py").is_file()
    assert (dest/"src/fpgai_research/model_ir/contracts.py").is_file()
    assert not (dest/"paper").exists()
    assert not (dest/"rl_environment").exists()
    assert (dest/".github/ISSUE_TEMPLATE/bug_report.md").is_file()
    assert (dest/".github/PULL_REQUEST_TEMPLATE.md").is_file()
    assert not (dest/".github/workflows").exists()
    assert not (dest/"weights").exists()
    assert not (dest/"build").exists()
    manifest=json.loads((dest/"PREVIEW_NOT_FOR_PUBLICATION.json").read_text())
    assert manifest["ready_to_publish"] is False
    assert manifest["selected_file_count"]==result["selected_file_count"]
    with pytest.raises(FileExistsError,match="create-only"):
        sut.stage(root,dest,report)


def test_full_research_preview_still_excludes_historical_private_docs(
    tmp_path,monkeypatch
):
    root=tmp_path/"source"
    root.mkdir()
    _fake_source(root,monkeypatch)
    report=sut.plan(root,"full-research")
    assert "rl_environment/agent.py" in report["selected_paths"]
    assert "experiments/compiler_ir/series.py" in report["selected_paths"]
    assert "tests/test_gym_graph_hls_contract.py" in report["selected_paths"]
    assert not any(path.startswith("paper/") for path in report["selected_paths"])


def test_source_tree_destination_is_forbidden(tmp_path,monkeypatch):
    root=tmp_path/"source"
    root.mkdir()
    _fake_source(root,monkeypatch)
    report=sut.plan(root,"compiler")
    with pytest.raises(ValueError,match="under build"):
        sut.stage(root,root/"src"/"new-preview",report)
    assert not (root/"src"/"new-preview").exists()


def test_symlinked_tracked_content_fails_closed(tmp_path,monkeypatch):
    root=tmp_path/"source"
    root.mkdir()
    _fake_source(root,monkeypatch)
    (root/"README.md").unlink()
    (root/"README.md").symlink_to(root/"CONTRIBUTING.md")
    with pytest.raises(ValueError,match="Unsafe"):
        sut.plan(root,"compiler")


def test_output_claiming_publication_authorized_fails_closed(
    tmp_path,monkeypatch
):
    root=tmp_path/"source"
    root.mkdir()
    _fake_source(root,monkeypatch)
    report={**sut.plan(root,"compiler"),"ready_to_publish":True}
    with pytest.raises(ValueError,match="mistaken publication"):
        sut.stage(root,tmp_path/"preview",report)
