"""Opt-in curated pytest regression against an already installed compiler preview.

This uses the isolated venv and preview COPY made by verify_compiler_preview.
It does not install dependencies, use Git, invoke vendor tools, publish
anything, or mutate the original 478-file staged preview.

A single source-checkout integration test is deliberately deselected: the
staged preview intentionally has no Git repository/index. This exclusion
is recorded rather than counted as passing.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess

if __package__ in (None, ""):
    from verify_compiler_preview import inspect, ROOT
    from prepare_public_preview import PUBLIC_TESTS
else:
    from scripts.verify_compiler_preview import inspect, ROOT
    from scripts.prepare_public_preview import PUBLIC_TESTS

SCHEMA = "fpgai-research/curated-installed-preview-tests/v1"
GIT_CHECKOUT_TEST = (
    "tests/test_public_docs_links.py::"
    "test_direct_script_and_module_invocations_match_from_checkout"
)
RECEIPT_FILE = "curated_preview_pytest_receipt.json"
MODULES = tuple(sorted(PUBLIC_TESTS))


def plan(workspace: str | Path) -> dict:
    workspace = Path(workspace).absolute()
    build = ROOT / "build"
    if (workspace.is_symlink() or not workspace.is_dir()
            or build not in workspace.parents):
        raise ValueError("Existing isolated workspace under checkout build/ required")
    receipt_path = workspace / "installed_preview_smoke_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Verified installed-preview smoke receipt required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "fpgai-research/installed-preview-smoke/v1"
            or receipt.get("status") != "INSTALLED_PREVIEW_SOURCE_SMOKE_VERIFIED"
            or receipt.get("installed_package_tested") is not True
            or receipt.get("ready_to_publish") is not False
            or receipt.get("workspace") != str(workspace)):
        raise ValueError("Valid nonpublishable installed-preview receipt required")
    preview = Path(receipt["preview"]).absolute()
    verified = inspect(preview)
    if (verified["preview_manifest_sha256"]
            != receipt.get("preview_manifest_sha256")):
        raise ValueError("Original stage no longer matches installed receipt")
    source = workspace / "source"
    if source.is_symlink() or not source.is_dir() or (source / ".git").exists():
        raise ValueError("Isolated source copy must exist without Git history")
    manifest = json.loads(
        (preview / "PREVIEW_NOT_FOR_PUBLICATION.json").read_text(encoding="utf-8"))
    if not set(MODULES) <= set(manifest["selected_paths"]):
        raise ValueError("Staged preview does not contain every curated test")
    # The previous pip install can add .egg-info/ and Python caches in its
    # copy. Audit all *selected* file bytes, not those created runtime extras.
    for rel in manifest["selected_paths"]:
        first = preview / rel
        second = source / rel
        if (second.is_symlink() or not second.is_file()
                or sha256(first.read_bytes()).digest()
                != sha256(second.read_bytes()).digest()):
            raise ValueError("Selected source copy differs from stage: " + rel)
    python = workspace / "venv" / "bin" / "python"
    if not python.is_file():
        raise ValueError("Isolated virtualenv Python executable not found")
    package_path = Path(receipt["isolated_package_module"])
    if (not package_path.is_file()
            or (workspace / "venv").resolve()
            not in package_path.resolve().parents):
        raise ValueError("Smoke-tested Python package not isolated")
    return {
        "schema": SCHEMA,
        "status": "READY_FOR_CURATED_PREVIEW_TESTS",
        "workspace": str(workspace),
        "source_copy_matches_stage": True,
        "original_preview_manifest_sha256": verified["preview_manifest_sha256"],
        "curated_test_modules": list(MODULES),
        "checkout_only_test_deselected": GIT_CHECKOUT_TEST,
        "installed_smoke_verified": True,
        "curated_preview_pytest_executed": False,
        "native_vendor_calls": 0,
        "ready_to_publish": False,
    }


def execute(workspace: str | Path) -> dict:
    report = plan(workspace)
    workspace = Path(workspace).absolute()
    receipt_path = workspace / RECEIPT_FILE
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("Curated preview test receipt is create-only")
    source = workspace / "source"
    python = workspace / "venv" / "bin" / "python"
    cmd = [str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--deselect=" + GIT_CHECKOUT_TEST, *MODULES]
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                "PYTHONUSERBASE", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        cmd, cwd=source, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=600, check=False,
    )
    if completed.returncode:
        raise RuntimeError("Curated preview pytest failed; original stage unchanged.\n"
                           + completed.stdout[-4500:] + "\n"
                           + completed.stderr[-2500:])
    found = re.search(r"\b(\d+) passed\b", completed.stdout)
    if not found or int(found.group(1)) < 1:
        raise AssertionError("Curated pytest did not report a passing population")
    removed = re.search(r"\b(\d+) deselected\b", completed.stdout)
    if not removed or int(removed.group(1)) != 1:
        raise AssertionError("Exactly one Git-only test must be deselected")
    skipped = re.search(r"\b(\d+) skipped\b", completed.stdout)
    result = {
        **report,
        "status": "CURATED_PREVIEW_SOURCE_TESTS_PASSED",
        "curated_preview_pytest_executed": True,
        "curated_passed": int(found.group(1)),
        "curated_skipped": int(skipped.group(1)) if skipped else 0,
        "checkout_only_tests_deselected": 1,
        "source_checkout_git_required": False,
        "result_scope": "curated staged-source tests in isolated Python environment",
        "full_development_suite_executed": False,
        "installed_smoke_verified": True,
        "ready_to_publish": False,
    }
    with receipt_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, sort_keys=True, indent=2)
        file.write("\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = execute(args.workspace) if args.execute else plan(args.workspace)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
