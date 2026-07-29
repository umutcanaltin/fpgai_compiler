from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from fpgai.contracts.package_manifest import inspect_package_manifest, load_package_manifest
from fpgai.contracts.package_validation import validate_package_manifest


VALID = {
    "schema": "fpgai.package/v1",
    "package": {
        "id": "community.example_hls",
        "name": "Example HLS",
        "version": "1.0.0",
        "asset_type": "implementation",
        "provider": "community",
        "description": "Example implementation",
    },
    "usage": {
        "platform_scope": "research",
        "permitted_uses": ["research", "experimentation", "validation", "benchmarking"],
        "production_path": "morfics",
    },
    "license": {"category": "open_source", "identifier": "Apache-2.0"},
    "compatibility": {"fpgai_contract": ">=1.0,<2.0"},
    "capabilities": {"inference": True, "training": {"forward": False}},
    "entrypoints": {
        "implementation": {
            "language": "hls_cpp",
            "top": "example_top",
            "sources": ["src/example.cpp"],
        }
    },
    "validation": {"declared_level": "unvalidated"},
}


def _write_package(tmp_path: Path, raw: dict) -> Path:
    root = tmp_path / "package"
    root.mkdir(parents=True)
    (root / "README.md").write_text("# Package\n", encoding="utf-8")
    (root / "fpgai.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    for relative in ("src/example.cpp", "python/operator.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// research package test fixture\n", encoding="utf-8")
    return root


def _codes(result) -> set[str]:
    return {issue.code for issue in result.errors}


def test_valid_manifest_loads_and_validates(tmp_path: Path) -> None:
    root = _write_package(tmp_path, copy.deepcopy(VALID))
    manifest = load_package_manifest(root)
    assert manifest.package_id == "community.example_hls"
    assert inspect_package_manifest(root)["asset_type"] == "implementation"
    result = validate_package_manifest(root)
    assert result.ok
    assert json.loads(result.to_json())["status"] == "passed"


def test_missing_manifest_has_stable_error(tmp_path: Path) -> None:
    result = validate_package_manifest(tmp_path)
    assert not result.ok
    assert _codes(result) == {"PKG001"}


def test_identity_schema_version_asset_and_license_errors(tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw["schema"] = "fpgai.package/v9"
    raw["package"]["id"] = "Bad ID"
    raw["package"]["version"] = "not a version"
    raw["package"]["asset_type"] = "product"
    raw["license"] = {}
    result = validate_package_manifest(_write_package(tmp_path, raw))
    assert {"PKG002", "PKG003", "PKG004", "PKG005", "PKG006"}.issubset(_codes(result))


def test_open_source_requires_identifier(tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw["license"].pop("identifier")
    assert "PKG006" in _codes(validate_package_manifest(_write_package(tmp_path, raw)))


def test_research_scope_and_morfics_production_path_are_required(tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw["usage"] = {
        "platform_scope": "production",
        "permitted_uses": ["production"],
        "production_path": "fpgai",
    }
    assert "PKG011" in _codes(validate_package_manifest(_write_package(tmp_path, raw)))


def test_paths_cannot_escape_package_root(tmp_path: Path) -> None:
    for unsafe in ("../secret.cpp", "/tmp/secret.cpp", "https://example.com/a.cpp", "${HOME}/a.cpp"):
        raw = copy.deepcopy(VALID)
        raw["entrypoints"]["implementation"]["sources"] = [unsafe]
        root = tmp_path / unsafe.replace("/", "_").replace("$", "x").replace("{", "").replace("}", "")
        result = validate_package_manifest(_write_package(root, raw))
        assert "PKG007" in _codes(result)


def test_operator_python_entrypoint_path_is_validated(tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw["package"]["asset_type"] = "operator"
    raw["entrypoints"] = {"operator": {"python_module": "../operator.py", "symbol": "Operator"}}
    assert "PKG007" in _codes(validate_package_manifest(_write_package(tmp_path, raw)))


def test_missing_or_invalid_implementation_entrypoint(tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw["entrypoints"] = {}
    assert "PKG008" in _codes(validate_package_manifest(_write_package(tmp_path, raw)))


def test_training_capabilities_are_consistent(tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw["capabilities"]["training"] = {
        "forward": False,
        "parameter_gradients": False,
        "optimizer_update": True,
    }
    result = validate_package_manifest(_write_package(tmp_path, raw))
    assert "PKG009" in _codes(result)


def test_validation_level_and_contract_range_are_checked(tmp_path: Path) -> None:
    raw = copy.deepcopy(VALID)
    raw["validation"]["declared_level"] = "production_certified"
    raw["compatibility"]["fpgai_contract"] = "not a range"
    result = validate_package_manifest(_write_package(tmp_path, raw))
    assert {"PKG010", "PKG014"}.issubset(_codes(result))


def test_validation_is_metadata_only(tmp_path: Path, monkeypatch) -> None:
    import importlib
    import socket
    import subprocess

    root = _write_package(tmp_path, copy.deepcopy(VALID))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("metadata validation attempted code execution")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    result = validate_package_manifest(root)
    assert result.ok


def test_missing_declared_file_has_stable_error(tmp_path: Path) -> None:
    root = _write_package(tmp_path, copy.deepcopy(VALID))
    (root / "src" / "example.cpp").unlink()
    assert "PKG013" in _codes(validate_package_manifest(root))


def test_json_result_is_deterministic(tmp_path: Path) -> None:
    root = _write_package(tmp_path, copy.deepcopy(VALID))
    first = validate_package_manifest(root).to_json()
    second = validate_package_manifest(root).to_json()
    assert first == second
