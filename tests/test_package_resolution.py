from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from fpgai.contracts.package_lock import PACKAGE_LOCK_SCHEMA_V1, write_package_lock
from fpgai.contracts.package_resolution import resolve_package_set


BASE = {
    "schema": "fpgai.package/v1",
    "package": {
        "id": "community.placeholder",
        "name": "Package",
        "version": "1.0.0",
        "asset_type": "dataset",
        "provider": "community",
        "description": "Resolution fixture",
    },
    "usage": {
        "platform_scope": "research",
        "permitted_uses": ["research", "experimentation", "validation", "benchmarking"],
        "production_path": "morfics",
    },
    "license": {"category": "open_source", "identifier": "Apache-2.0"},
    "compatibility": {"fpgai_contract": ">=1.0,<2.0"},
    "capabilities": {"inference": False, "training": {"forward": False}},
    "validation": {"declared_level": "unvalidated"},
}


def _package(
    base: Path,
    package_id: str,
    version: str,
    *,
    dependencies: list[dict] | None = None,
    suffix: str = "",
    description: str = "Resolution fixture",
) -> Path:
    raw = copy.deepcopy(BASE)
    raw["package"].update({"id": package_id, "version": version, "description": description})
    if dependencies is not None:
        raw["dependencies"] = dependencies
    root = base / f"{package_id.replace('.', '_')}_{version.replace('.', '_')}{suffix}"
    root.mkdir(parents=True)
    (root / "README.md").write_text("# Package\n", encoding="utf-8")
    (root / "fpgai.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return root


def _codes(result) -> set[str]:
    return {issue.code for issue in result.errors}


def test_resolution_selects_highest_compatible_version_and_orders_dependencies(tmp_path: Path) -> None:
    operator_v1 = _package(tmp_path, "fpgai.operator.conv2d", "1.2.0")
    operator_v2 = _package(tmp_path, "fpgai.operator.conv2d", "2.0.0")
    implementation = _package(
        tmp_path,
        "community.conv2d_hls",
        "1.0.0",
        dependencies=[{"package": "fpgai.operator.conv2d", "version": ">=1.0,<2.0"}],
    )
    result = resolve_package_set(
        [operator_v2, implementation, operator_v1],
        root_package_ids=["community.conv2d_hls"],
    )
    assert result.ok, result.to_json()
    selected = {item.package_id: item.version for item in result.selected}
    assert selected == {
        "community.conv2d_hls": "1.0.0",
        "fpgai.operator.conv2d": "1.2.0",
    }
    assert result.resolution_order == ("fpgai.operator.conv2d", "community.conv2d_hls")


def test_missing_required_and_optional_dependencies_are_distinguished(tmp_path: Path) -> None:
    required = _package(
        tmp_path,
        "community.required_root",
        "1.0.0",
        dependencies=[{"package": "community.missing", "version": "^1.0", "required": True}],
    )
    optional = _package(
        tmp_path,
        "community.optional_root",
        "1.0.0",
        dependencies=[{"package": "community.optional_missing", "version": "^1.0", "required": False}],
    )
    required_result = resolve_package_set([required])
    optional_result = resolve_package_set([optional])
    assert "PKGR004" in _codes(required_result)
    assert optional_result.ok
    assert {issue.code for issue in optional_result.warnings} == {"PKGRW001"}


def test_unsatisfied_version_range_is_reported(tmp_path: Path) -> None:
    dependency = _package(tmp_path, "community.dependency", "2.0.0")
    root = _package(
        tmp_path,
        "community.root",
        "1.0.0",
        dependencies=[{"package": "community.dependency", "version": "<2.0"}],
    )
    result = resolve_package_set([dependency, root], root_package_ids=["community.root"])
    assert "PKGR003" in _codes(result)


def test_duplicate_identity_with_different_content_is_rejected(tmp_path: Path) -> None:
    first = _package(tmp_path, "community.duplicate", "1.0.0", suffix="_a", description="A")
    second = _package(tmp_path, "community.duplicate", "1.0.0", suffix="_b", description="B")
    result = resolve_package_set([first, second])
    assert "PKGR002" in _codes(result)


def test_dependency_cycle_is_reported(tmp_path: Path) -> None:
    a = _package(
        tmp_path,
        "community.a",
        "1.0.0",
        dependencies=[{"package": "community.b", "version": "1.0.0"}],
    )
    b = _package(
        tmp_path,
        "community.b",
        "1.0.0",
        dependencies=[{"package": "community.a", "version": "1.0.0"}],
    )
    result = resolve_package_set([a, b], root_package_ids=["community.a"])
    assert "PKGR005" in _codes(result)


def test_lock_manifest_is_deterministic_and_contains_hashes(tmp_path: Path) -> None:
    dependency = _package(tmp_path, "community.dependency", "1.0.0")
    root = _package(
        tmp_path,
        "community.root",
        "1.0.0",
        dependencies=[{"package": "community.dependency", "version": "^1.0"}],
    )
    first = resolve_package_set([root, dependency], root_package_ids=["community.root"])
    second = resolve_package_set([dependency, root], root_package_ids=["community.root"])
    assert first.ok and second.ok
    assert first.to_lock().to_json() == second.to_lock().to_json()
    payload = json.loads(first.to_lock().to_json())
    assert payload["schema"] == PACKAGE_LOCK_SCHEMA_V1
    assert all(item["manifest_sha256"].startswith("sha256:") for item in payload["packages"])
    output = write_package_lock(first.to_lock(), tmp_path / "fpgai.package-lock.yml")
    assert output.is_file()
    assert "fpgai.package-lock/v1" in output.read_text(encoding="utf-8")


def test_resolution_is_metadata_only(tmp_path: Path, monkeypatch) -> None:
    import importlib
    import socket
    import subprocess

    package = _package(tmp_path, "community.metadata_only", "1.0.0")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resolution attempted code execution or network access")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    assert resolve_package_set([package]).ok
