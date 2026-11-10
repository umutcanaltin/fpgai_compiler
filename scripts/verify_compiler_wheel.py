"""Opt-in build-and-inspect gate for an UNLICENSED compiler-preview wheel.

Requires an existing staged preview, isolated source-install smoke and curated
source-preview test receipt. Planning is read-only. --execute builds a wheel
from the existing isolated source COPY, inspects every archive member and
checks that all selected package Python modules are present. It installs the
wheel into a *new* environment, invokes the wheel-installed compiler CLI on a
synthetic ONNX model and checks generated source identity. No upload, vendor
tools, original experiment writes, license decision or Git history access.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import venv
import zipfile

if __package__ in (None, ""):
    from verify_preview_test_suite import plan as curated_plan, RECEIPT_FILE
    from verify_compiler_preview import ROOT
else:
    from scripts.verify_preview_test_suite import plan as curated_plan, RECEIPT_FILE
    from scripts.verify_compiler_preview import ROOT

SCHEMA = "fpgai-research/compiler-preview-wheel-audit/v1"
TARGET_PACKAGE = "fpgai_research"
REQUIRED_MODULES = frozenset({
    "fpgai_research/__init__.py",
    "fpgai_research/cli.py",
    "fpgai_research/compiler/__init__.py",
    "fpgai_research/model_ir/contracts.py",
    "fpgai_research/architecture/resolver.py",
})


def plan(workspace: str | Path, output: str | Path) -> dict:
    workspace = Path(workspace).absolute()
    stage = curated_plan(workspace)
    receipt = workspace / RECEIPT_FILE
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("Author-verified curated-preview pytest receipt required")
    report = json.loads(receipt.read_text(encoding="utf-8"))
    if (report.get("schema") != "fpgai-research/curated-installed-preview-tests/v1"
            or report.get("status") != "CURATED_PREVIEW_SOURCE_TESTS_PASSED"
            or report.get("curated_preview_pytest_executed") is not True
            or report.get("curated_passed", 0) < 1
            or report.get("checkout_only_tests_deselected") != 1
            or report.get("curated_skipped") != 0
            or report.get("ready_to_publish") is not False
            or report.get("workspace") != str(workspace)
            or report.get("original_preview_manifest_sha256")
            != stage["original_preview_manifest_sha256"]):
        raise ValueError("Verified nonpublishable curated source receipt required")

    candidate = Path(output).absolute()
    build = ROOT / "build"
    if (candidate == build or build not in candidate.parents
            or candidate.exists() or candidate.is_symlink()
            or workspace in candidate.parents or candidate in workspace.parents):
        raise ValueError("Fresh create-only wheel workspace under build/ required")
    if not candidate.parent.is_dir() or candidate.parent.is_symlink():
        raise ValueError("Wheel destination parent must be an existing real directory")
    return {
        "schema": SCHEMA,
        "status": "READY_FOR_WHEEL_AUDIT",
        "workspace": str(workspace),
        "output": str(candidate),
        "stage_manifest_sha256": stage["original_preview_manifest_sha256"],
        "source_copy_matches_stage": True,
        "curated_preview_passed": report["curated_passed"],
        "wheel_built": False,
        "wheel_installed_and_cli_tested": False,
        "wheel_contents_checked": False,
        "native_vendor_calls": 0,
        "license_or_history_approved": False,
        "ready_to_publish": False,
    }


def inspect_wheel(wheel: Path, selected_source_paths: set[str]) -> dict:
    if wheel.is_symlink() or not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("Actual wheel archive required")
    expected = {
        rel.removeprefix("src/")
        for rel in selected_source_paths
        if rel.startswith("src/fpgai_research/") and rel.endswith(".py")
    }
    if not REQUIRED_MODULES <= expected:
        raise ValueError("Preview lacks essential Python implementation modules")
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        names = [info.filename for info in entries]
        if len(names) != len(set(names)) or not names:
            raise ValueError("Wheel archive has duplicate or missing members")
        if len(names) != len(set(x.casefold() for x in names)):
            raise ValueError("Wheel archive has conflicting member names")
        dist = sorted({
            name.split("/", 1)[0]
            for name in names if ".dist-info/" in name
        })
        if len(dist) != 1 or not re.fullmatch(
                r"fpgai_research-[A-Za-z0-9_.+-]+[.]dist-info", dist[0]):
            raise ValueError("Unexpected wheel metadata directory")
        allowed_meta = {
            "METADATA", "WHEEL", "RECORD", "entry_points.txt",
            "top_level.txt", "INSTALLER", "REQUESTED",
        }
        module_names = set()
        for info in entries:
            name = info.filename
            p = PurePosixPath(name)
            if (not name or name.startswith("/") or "\\" in name or
                    p.as_posix() != name or ".." in p.parts or
                    any(k in p.parts for k in ("paper", "tests", "rl_environment",
                                                "weights", "build", ".github"))):
                raise ValueError("Unsafe archive path: " + name)
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise ValueError("Symlink not allowed in wheel: " + name)
            if name.startswith(TARGET_PACKAGE + "/"):
                if not name.endswith(".py"):
                    raise ValueError("Unreviewed non-Python wheel payload: " + name)
                module_names.add(name)
            elif name.startswith(dist[0] + "/"):
                if name.removeprefix(dist[0] + "/") not in allowed_meta:
                    raise ValueError("Unexpected wheel metadata: " + name)
            else:
                raise ValueError("Unallowlisted wheel member: " + name)
        if module_names != expected:
            raise ValueError(
                "Wheel package modules differ from staged preview: " +
                str({"missing": sorted(expected - module_names)[:10],
                     "unexpected": sorted(module_names - expected)[:10]})
            )
        meta = Parser().parsestr(archive.read(dist[0] + "/METADATA").decode())
        if meta.get("Name", "").replace("_", "-").lower() != "fpgai-research":
            raise ValueError("Wrong wheel distribution identity")
        if "fpgai-research" not in archive.read(
                dist[0] + "/entry_points.txt").decode():
            raise ValueError("Missing public compiler CLI entry point")
        return {
            "wheel_file": wheel.name,
            "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
            "wheel_member_count": len(names),
            "wheel_python_module_count": len(module_names),
            "wheel_all_staged_modules_present": True,
            "wheel_unexpected_members": 0,
            "package_name": meta["Name"],
        }


def _run(command: list[str], cwd: Path, env: dict[str, str],
         timeout: int = 600) -> str:
    finished = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True,
        check=False, timeout=timeout,
    )
    if finished.returncode:
        raise RuntimeError(
            "Wheel smoke command failed: " + repr(command[:4]) +
            "\nstdout: " + finished.stdout[-2500:] +
            "\nstderr: " + finished.stderr[-2500:])
    return finished.stdout


def execute(workspace: str | Path, output: str | Path) -> dict:
    report = plan(workspace, output)
    workspace = Path(workspace).absolute()
    destination = Path(output).absolute()
    destination.mkdir(exist_ok=False)
    wheel_dir = destination / "wheel"
    wheel_dir.mkdir()
    source = workspace / "source"
    original_python = workspace / "venv" / "bin" / "python"
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                 "PYTHONUSERBASE", "VIRTUAL_ENV"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    _run([str(original_python), "-m", "pip", "wheel", "--no-input",
          "--no-cache-dir", "--no-deps", "--wheel-dir", str(wheel_dir),
          str(source)], destination, env, timeout=900)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Exactly one compiler wheel required")
    original_receipt = json.loads(
        (workspace / "installed_preview_smoke_receipt.json").read_text())
    preview = Path(original_receipt["preview"])
    manifest = json.loads(
        (preview / "PREVIEW_NOT_FOR_PUBLICATION.json").read_text())
    wheel_receipt = inspect_wheel(wheels[0], set(manifest["selected_paths"]))

    env_root = destination / "clean_venv"
    venv.create(env_root, with_pip=True, system_site_packages=False)
    python = env_root / "bin" / "python"
    # Explicit network-capable action. Does not modify the original venv.
    _run([str(python), "-m", "pip", "install", "--no-input",
          "--no-cache-dir", str(wheels[0]) + "[onnx]"],
         destination, env, timeout=900)
    work = destination / "work"
    work.mkdir()
    imported = _run([str(python), "-c",
        "import fpgai_research; print(fpgai_research.__file__)"], work, env).strip()
    if not (Path(imported).is_file()
            and env_root.resolve() in Path(imported).resolve().parents):
        raise AssertionError("Wheel package import escaped isolated virtualenv")
    onnx = work / "toy.onnx"
    _run([str(python), str(source / "examples/make_toy_onnx.py"), str(onnx)],
         work, env)
    out = work / "compiled"
    _run([str(env_root / "bin" / "fpgai-research"), str(onnx),
          "-o", str(out)], work, env)
    cpp = out / "model.cpp"
    man = json.loads((out / "manifest.json").read_text())
    if (not cpp.is_file() or
            sha256(cpp.read_bytes()).hexdigest() != man.get("cpp_sha256")):
        raise AssertionError("Wheel-installed compiler emitted invalid source digest")

    result = {**report, **wheel_receipt,
        "status": "WHEEL_CONTENTS_AND_INSTALLED_CLI_VERIFIED",
        "wheel_built": True,
        "wheel_contents_checked": True,
        "wheel_installed_and_cli_tested": True,
        "isolated_wheel_package_path": imported,
        "generated_cpp_sha256": man["cpp_sha256"],
        "full_original_research_studies_reproduced": False,
        "license_or_history_approved": False,
        "ready_to_publish": False,
    }
    with (destination / "WHEEL_NOT_FOR_PUBLICATION.json").open("x") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True, help="Fresh ignored build/ directory")
    control = parser.add_mutually_exclusive_group(required=True)
    control.add_argument("--plan", action="store_true")
    control.add_argument("--execute", action="store_true",
                         help="Opt in to wheel build and dependency installation")
    args = parser.parse_args(argv)
    result = execute(args.workspace, args.output) if args.execute else plan(
        args.workspace, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
