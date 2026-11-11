"""Synthetic, non-network regression for nonpublishable compiler wheel gate."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import verify_compiler_wheel as gate


def _wheel(path: Path, *, omitted=(), extra=None, notice=None,
           metadata_addendum=""):
    modules = sorted(gate.REQUIRED_MODULES)
    with ZipFile(path, "w") as z:
        for module in modules:
            if module not in omitted:
                z.writestr(module, "# synthetic\n")
        root = "fpgai_research-0.3.0.dev0.dist-info"
        z.writestr(root + "/METADATA",
                   "Metadata-Version: 2.4\nName: fpgai-research\nVersion: 0.3.0.dev0\n"
                   + ("License-File: NOTICE.md\n" if notice is not None else "")
                   + metadata_addendum)
        z.writestr(root + "/WHEEL",
                   "Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\n"
                   "Tag: py3-none-any\n")
        z.writestr(root + "/entry_points.txt",
                   "[console_scripts]\nfpgai-research = fpgai_research.cli:main\n")
        z.writestr(root + "/RECORD", "synthetic test\n")
        if notice is not None:
            z.writestr(root + "/licenses/NOTICE.md", notice)
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


def test_standard_notice_member_is_attested_against_staged_file(tmp_path):
    text = b"# Notices\nNo operative repository license.\n"
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl",
                   notice=text)
    selected = _selected() | {"NOTICE.md"}
    receipt = gate.inspect_wheel(
        wheel, selected, expected_notice=text)
    assert receipt["wheel_notice_matches_stage"] is True
    assert receipt["wheel_notice_is_operative_license"] is False
    assert receipt["staged_notice_sha256"]


def test_notice_member_without_staged_attestation_fails(tmp_path):
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl",
                   notice=b"unreviewed notice")
    with pytest.raises(ValueError, match="Unattested"):
        gate.inspect_wheel(wheel, _selected())


def test_notice_byte_mismatch_is_rejected(tmp_path):
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl",
                   notice=b"original")
    with pytest.raises(ValueError, match="differs from staged source"):
        gate.inspect_wheel(wheel, _selected() | {"NOTICE.md"},
                           expected_notice=b"tampered")


def test_missing_notice_when_staged_source_contains_one_is_rejected(tmp_path):
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl")
    with pytest.raises(ValueError, match="missing staged NOTICE"):
        gate.inspect_wheel(wheel, _selected() | {"NOTICE.md"},
                           expected_notice=b"staged")


def test_staged_notice_requires_allowlisted_original_path(tmp_path):
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl",
                   notice=b"staged")
    with pytest.raises(ValueError, match="source allowlist"):
        gate.inspect_wheel(wheel, _selected(), expected_notice=b"staged")


@pytest.mark.parametrize("extra", [
    "fpgai_research-0.3.0.dev0.dist-info/licenses/PRIVATE.md",
    "fpgai_research-0.3.0.dev0.dist-info/licenses/LICENSE",
    "fpgai_research-0.3.0.dev0.dist-info/third_party/secret.txt",
])
def test_only_exact_approved_notice_location_is_accepted(tmp_path, extra):
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl",
                   notice=b"approved", extra=extra)
    with pytest.raises(ValueError, match="Unexpected wheel metadata"):
        gate.inspect_wheel(wheel, _selected() | {"NOTICE.md"},
                           expected_notice=b"approved")


@pytest.mark.parametrize("metadata_addendum", [
    "License-File: OTHER.md\n",
    "License-Expression: MIT\n",
])
def test_wheel_rejects_unknown_licensing_metadata(tmp_path, metadata_addendum):
    wheel = _wheel(tmp_path / "fpgai_research-0.3.0.dev0-py3-none-any.whl",
                   notice=b"staged", metadata_addendum=metadata_addendum)
    with pytest.raises(ValueError, match="License-File|License-Expression"):
        gate.inspect_wheel(wheel, _selected() | {"NOTICE.md"},
                           expected_notice=b"staged")
