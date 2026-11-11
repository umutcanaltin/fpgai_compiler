"""Opt-in installed-package smoke test for a staged, nonpublishable compiler preview.

Planning is read-only. --execute copies the already audited staged preview
into a fresh ignored build/ workspace, creates a new virtual environment,
installs the package (may download third-party dependencies), and runs small
synthetic ONNX compilation tests from OUTSIDE the staged source tree.

Never starts Vitis/Vivado, rewrites original study outputs, uses Git history,
publishes code or grants redistribution rights.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

if __package__ in (None, ""):
    from check_public_docs import audit_source_links
else:
    from scripts.check_public_docs import audit_source_links

SCHEMA = "fpgai-research/installed-preview-smoke/v1"
ROOT = Path(__file__).resolve().parents[1]


def inspect(preview: str | Path) -> dict:
    path = Path(preview).absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError("A real staged preview directory is required")
    manifest = path / "PREVIEW_NOT_FOR_PUBLICATION.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Staged preview manifest required; Git index is not a fallback")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if (document.get("profile") != "compiler"
            or document.get("ready_to_publish") is not False):
        raise ValueError("Compiler-only, explicitly nonpublishable preview required")
    audit = audit_source_links(path, "compiler")
    if not audit["all_local_public_links_resolve"]:
        raise ValueError("Staged preview has unresolved relative documentation links")
    mandatory = {
        "pyproject.toml", "src/fpgai_research/cli.py",
        "examples/make_toy_onnx.py", "examples/make_neural_network_onnx.py",
        "examples/compile_neural_network.py",
    }
    if not mandatory <= set(document["selected_paths"]):
        raise ValueError("Staged preview missing mandatory compiler smoke inputs")
    return {
        "schema": SCHEMA,
        "preview": str(path),
        "preview_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "selected_file_count": document["selected_file_count"],
        "link_count": audit["checked_local_markdown_links"],
        "staged_inventory_and_links_verified": True,
        "installed_package_tested": False,
        "ready_to_publish": False,
    }


def require_new_workspace(preview: Path, output: str | Path) -> Path:
    candidate = Path(output).absolute()
    build = ROOT / "build"
    if (candidate == build or build not in candidate.parents
            or candidate.exists() or candidate.is_symlink()
            or preview == candidate or preview in candidate.parents
            or candidate in preview.parents):
        raise ValueError("New create-only workspace under checkout build/ required")
    parent = candidate.parent
    if (not parent.is_dir() or parent.is_symlink()
            or any(p.is_symlink() for p in parent.parents if p != ROOT)):
        raise ValueError("Workspace parent must be an existing non-symlink directory")
    return candidate


def _run(command: list[str], *, cwd: Path, env: dict[str, str],
         timeout: int = 300) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Clean-room command failed: " + repr(command[:3]) +
            "\nstdout: " + result.stdout[-3000:] +
            "\nstderr: " + result.stderr[-3000:])
    return result.stdout


def execute(preview: str | Path, output: str | Path) -> dict:
    planned = inspect(preview)
    source_preview = Path(preview).absolute()
    workspace = require_new_workspace(source_preview, output)
    workspace.mkdir(parents=False, exist_ok=False)
    source = workspace / "source"
    shutil.copytree(source_preview, source, symlinks=False)
    installed_audit = inspect(source)
    if installed_audit["preview_manifest_sha256"] != planned["preview_manifest_sha256"]:
        raise ValueError("Source-copy manifest changed during clean-room staging")

    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                "PYTHONUSERBASE", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    isolated = workspace / "venv"
    venv.create(isolated, with_pip=True, system_site_packages=False)
    python = isolated / "bin" / "python"
    compiler = isolated / "bin" / "fpgai-research"
    work = workspace / "work"
    work.mkdir()
    _run([str(python), "-m", "pip", "install", "--no-input",
          "--no-cache-dir", ".[onnx,test]"], cwd=source, env=env, timeout=900)

    # Exclude an accidental import of the developer checkout/preview source.
    installed_path = _run([str(python), "-c",
        "import fpgai_research;print(fpgai_research.__file__)"],
        cwd=work, env=env).strip()
    if not (Path(installed_path).is_file()
            and isolated.resolve() in Path(installed_path).resolve().parents):
        raise AssertionError("Installed package resolved outside isolated venv")

    toy = work / "toy.onnx"
    _run([str(python), str(source / "examples/make_toy_onnx.py"), str(toy)],
         cwd=work, env=env)
    toy_out = work / "toy_output"
    _run([str(compiler), str(toy), "-o", str(toy_out)], cwd=work, env=env)
    toy_cpp = toy_out / "model.cpp"
    toy_manifest = json.loads((toy_out / "manifest.json").read_text())
    if (not toy_cpp.is_file()
            or toy_manifest.get("cpp_sha256") != sha256(toy_cpp.read_bytes()).hexdigest()):
        raise AssertionError("Installed CLI generated invalid C++ source digest")

    cnn = work / "cnn_pool_dense.onnx"
    _run([str(python), str(source / "examples/make_neural_network_onnx.py"),
          "cnn_pool_dense", str(cnn)], cwd=work, env=env)
    cnn_out = work / "cnn_output"
    command = [str(python), str(source / "examples/compile_neural_network.py"),
               str(cnn), "--mapping", "independent", "--output", str(cnn_out)]
    host_available = shutil.which("g++") is not None
    if host_available:
        command.append("--verify-host")
    _run(command, cwd=work, env=env, timeout=300)
    summary = json.loads((cnn_out / "example_summary.json").read_text())
    cpp = cnn_out / "model.cpp"
    if (not summary.get("all_stages_lowered")
            or summary.get("canonical_cpp_sha256") != sha256(cpp.read_bytes()).hexdigest()
            or summary.get("native_vitis_or_vivado_calls") != 0
            or (host_available and summary.get("host_verification")
                != "ALL_INTERMEDIATE_TENSORS_AGREED")):
        raise AssertionError("Installed CNN source or optional host verification failed")

    result = {
        **planned,
        "installed_package_tested": True,
        "status": "INSTALLED_PREVIEW_SOURCE_SMOKE_VERIFIED",
        "workspace": str(workspace),
        "isolated_package_module": installed_path,
        "toy_cpp_sha256": toy_manifest["cpp_sha256"],
        "cnn_cpp_sha256": summary["canonical_cpp_sha256"],
        "cnn_all_stages_lowered": True,
        "cnn_host_gcc_checked": host_available,
        "native_vendor_calls": 0,
        "full_preview_pytest_executed": False,
        "clean_room_rights_or_history_reviewed": False,
        "ready_to_publish": False,
    }
    (workspace / "installed_preview_smoke_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", required=True, help="Existing staged compiler preview")
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true", help="No writes or installs")
    action.add_argument("--execute", action="store_true",
                        help="Create workspace and install third-party dependencies")
    ap.add_argument("--output", help="Fresh build/ path, mandatory with --execute")
    args = ap.parse_args(argv)
    if args.plan:
        if args.output:
            ap.error("--output is only valid with --execute")
        report = inspect(args.preview)
    else:
        if not args.output:
            ap.error("--execute requires --output")
        report = execute(args.preview, args.output)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
