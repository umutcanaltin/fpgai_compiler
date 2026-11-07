"""Public research publication hygiene: source-only, fail-closed checks.

Checks the CURRENT checkout only, NOT secrets/IP ownership across historical
Git objects or third-party licenses. Passing these tests NEVER authorizes
publication, licensing, a release, or a native vendor tool invocation.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

from scripts import check_public_docs
from scripts import prepare_public_preview


ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_protects_native_outputs_and_original_studies():
    path = ROOT / ".gitignore"
    assert path.is_file() and not path.is_symlink()
    text = path.read_text()
    for required in (
        "/build/", "/results/", "/data/", "/datasets/", "/weights/",
        ".venv/", "__pycache__/", "*.onnx", "*.npy", "*.pt",
        "*.dcp", "*.bit", "*.rpt", ".Xil/", "release-approval.json",
        ".env", "*.key",
    ):
        assert required in text, required
    # Don't ignore all source Python, Markdown, YAML, tests or Git metadata.
    for forbidden in ("*.py\n", "*.md\n", "*.yaml\n", "*.toml\n", "/src/\n"):
        assert forbidden not in text


def test_release_preview_explicitly_allowlists_linked_user_docs():
    planned = prepare_public_preview.plan(ROOT, "compiler")
    selected = set(planned["selected_paths"])
    assert planned["ready_to_publish"] is False
    assert planned["redistribution_license_approved"] is False
    assert planned["history_security_reviewed"] is False
    assert {
        ".gitignore", ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        "docs/RESEARCH_PUBLICATION_STATUS.md",
        "docs/PYNQ_Z2_BOARD_DMA_PILOT.md",
    } <= selected
    assert not any(p.startswith("paper/") or p.startswith("rl_environment/")
                   or p.startswith(".github/workflows/") for p in selected)


def test_compiler_preview_markdown_links_resolve_without_internal_paper_files():
    report = check_public_docs.audit_source_links(ROOT, "compiler")
    assert report["public_release_authorized"] is False
    assert report["all_local_public_links_resolve"] is True, (
        report["unresolved_public_links"])
    assert report["unresolved_public_links"] == []


def test_paper_release_status_does_not_claim_rights_or_physical_latency():
    readme = (ROOT / "README.md").read_text()
    status = (ROOT / "docs/RESEARCH_PUBLICATION_STATUS.md").read_text()
    license_note = (ROOT / "docs/LICENSE_DECISION.md").read_text()
    assert "docs/RESEARCH_PUBLICATION_STATUS.md" in readme
    assert "no operative source-code license" in readme.lower()
    assert "no operative source-code license" in license_note.lower()
    assert "NOT SUBMISSION" in status.upper() or "NOT SUBMISSION-READY" in status.upper()
    assert "physical compute latency" in status.lower()
    assert "unknown" in status.lower()
    assert "be2adaa4acf30196a2ccb17114f2caa0af3e6679dc2685872e27cb086ca86ec1" in status


def test_source_only_repo_has_no_tracked_github_actions_workflows():
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".github/workflows"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    assert not tracked, (
        "GitHub Actions workflows must not be introduced into the published "
        "compiler artifact without an explicit change of research policy")
